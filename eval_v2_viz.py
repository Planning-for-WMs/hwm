"""eval_v2_viz.py — single combined MP4: imagined LL rollout (top) vs the
agent's actual env-rendered trajectory (bottom). Uses the same closed-loop
unified CEM policy as eval_unified.py.

For each replan, we take the FIRST `replan_every` LL tokens of the imagined
rollout (= what the LL predictor expects up to the next replan), decode them
through `decoder_v4`, repeat each frame `action_block` times for env-step
resolution, and concatenate across replans. The bottom strip is the env
render captured at each env step.

Both strips are temporally aligned and identical in length (one frame per
env step). Watching the two strips side-by-side shows where the LL world
model's imagined dynamics diverge from the real env.

Output: <viz_dir>/imagined_vs_agent.mp4

Run:
  python eval_v2_viz.py policy_ckpt=$STABLEWM_HOME/hjepa_v12/hjepa_v12_epoch_13_object.ckpt eval.num_eval=1
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from collections import deque
from pathlib import Path

import hydra
import imageio.v3 as iio
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval_unified import UnifiedPolicy, img_transform, get_dataset
from train_decoder import ConvDecoder
from train_cost_model import CostHead


def latest_weights(folder, pattern="*_weights.ckpt"):
    cands = list(Path(folder).glob(pattern))
    if not cands:
        raise FileNotFoundError(f"no {pattern} under {folder}")
    def ep(p):
        s = p.stem
        if "_epoch_" in s:
            try: return int(s.split("_epoch_")[1].split("_")[0])
            except: return -1
        return -1
    epoched = [p for p in cands if ep(p) >= 0]
    return max(epoched, key=ep) if epoched else max(cands, key=lambda p: p.stat().st_mtime)


def to_uint8_chw(pix):
    pix = pix.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    return pix.permute(1, 2, 0).cpu().numpy()


def capture_agent_frame(pixels):
    """Coerce info_dict['pixels'] to uint8 HWC numpy."""
    if isinstance(pixels, np.ndarray):
        p = pixels
        while p.ndim > 3:
            p = p[0]
        return np.clip(p, 0, 255).astype(np.uint8) if p.dtype != np.uint8 else p
    if isinstance(pixels, torch.Tensor):
        p = pixels.detach()
        while p.dim() > 3:
            p = p[0]
        if p.dim() == 3 and p.shape[0] == 3 and p.shape[-1] != 3:
            p = p.permute(1, 2, 0)
        p = p.cpu().numpy()
        return np.clip(p, 0, 255).astype(np.uint8) if p.dtype != np.uint8 else p
    raise TypeError(f"unsupported pixels type: {type(pixels)}")


class V2VizPolicy(UnifiedPolicy):
    """Closed-loop unified policy + records per-replan imagined frames (first
    replan_every LL tokens, decoded via ll_decoder) and per-env-step agent
    frames (the rendered pixels seen by the env wrapper).

    If `cost_model` is provided, the CEM cost is rewritten to use the learned
    `(HL_target, LL_current) → ||LL_target − LL_current||²` head:
      - short_cost   = cost_model(s_first_hl_subgoal, ll_after_first_K_macro)
      - final-mode   = cost_model(s_hl_goal_p, ll_after_full_plan)
    long_cost (HL CEM rollout vs goal in HL space) is left unchanged, so the
    HL pathway still contributes to candidate ranking.
    """

    def __init__(self, *, ll_decoder, cost_model=None, **kw):
        super().__init__(**kw)
        self.ll_decoder = ll_decoder
        self.cost_model = cost_model
        self._imagined_frames: list[np.ndarray] = []   # H*W*3 uint8, env-step resolution
        self._agent_frames: list[np.ndarray] = []      # H*W*3 uint8

    @torch.inference_mode()
    def _cem(self, ll_curr, s_hl_goal_p, warm_mean=None, ll_goal=None, final=False):
        """Same CEM as UnifiedPolicy._cem, but `short_cost` (and final-mode cost)
        is replaced by `cost_model(HL_target, LL_current)` when a cost head was
        provided. Long_cost still uses HL CEM rollout vs goal."""
        if self.cost_model is None:
            return super()._cem(ll_curr, s_hl_goal_p, warm_mean=warm_mean,
                                ll_goal=ll_goal, final=final)
        # ---- cost-model variant (parallels UnifiedPolicy._cem) ----
        B = ll_curr.size(0); device = ll_curr.device
        T_HL = int(self.cfg["T_HL"]); K_macro = int(self.cfg["K_macro"])
        S = int(self.cfg["num_samples"]); n_steps = int(self.cfg["n_steps"])
        topk = int(self.cfg["topk"])
        var_scale = float(self.cfg["var_scale"])
        var_ema = float(self.cfg.get("var_ema", 0.9))
        short_w = float(self.cfg.get("short_weight", 1.0))
        action_dim = 2 * self.action_block
        H_plan = T_HL * K_macro
        hist = self.h.hlp.history

        mean = warm_mean.clone() if warm_mean is not None else torch.zeros(
            B, H_plan, action_dim, device=device)
        var = var_scale * torch.ones(B, H_plan, action_dim, device=device)
        init = ll_curr.unsqueeze(1)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, H_plan, action_dim, device=device, generator=self._cem_gen,
            )
            blk = cands.view(B, S, T_HL, K_macro, action_dim)
            macros = self.h.encode_macro(
                blk.reshape(B * S * T_HL, K_macro, action_dim)
            ).view(B, S, T_HL, -1)
            # HL rollout (same as parent)
            states_init = self._hl_states.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hist, -1)
            macros_init = self._hl_macros.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hist, -1)
            m_flat = macros.reshape(B * S, T_HL, -1)
            states = states_init; macros_buf = macros_init; s_first = None
            for t in range(T_HL):
                macros_buf = torch.cat([macros_buf[:, 1:], m_flat[:, t : t + 1]], dim=1)
                s_next = self.h.predict_hl(states, macros_buf)
                if t == 0:
                    s_first = s_next
                states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            s_final = s_next.view(B, S, -1)
            s_first = s_first.view(B, S, -1)
            long_cost = ((s_final - s_hl_goal_p.unsqueeze(1)) ** 2).sum(dim=-1)

            # LL rollout — first K_macro tokens used for short_cost; full plan
            # used for final-mode cost.
            ll_traj = self.h.ll_rollout_from_emb(init, cands)        # (B, S, H+1, ll_dim)
            ll_first = ll_traj[:, :, K_macro, :].float()             # (B, S, ll_dim)
            ll_final_full = ll_traj[:, :, -1, :].float()             # (B, S, ll_dim)

            # ----- cost-model short_cost: distance from K_macro-step LL terminal
            #       to the FIRST HL subgoal, in learned LL-distance units.
            cm_hl_target = s_first.reshape(B * S, -1)
            cm_ll_curr   = ll_first.reshape(B * S, -1)
            short_cost = self.cost_model(cm_hl_target, cm_ll_curr).view(B, S)

            if final:
                # Final mode: cost = learned LL-distance to the actual GOAL,
                # using ll_final_full so it reflects the full plan.
                hl_goal_tiled = s_hl_goal_p.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
                ll_final_flat = ll_final_full.reshape(B * S, -1)
                cost = self.cost_model(hl_goal_tiled, ll_final_flat).view(B, S)
                long_cost = cost
                short_cost = torch.zeros_like(cost)
            else:
                cost = long_cost + short_w * short_cost

            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, H_plan, action_dim))
            mean = elites.mean(dim=1)
            var = var * var_ema

        elite_long = torch.gather(long_cost, 1, idx).mean().item()
        elite_short = torch.gather(short_cost, 1, idx).mean().item()
        print(f"[replan step={self._total_steps}] long={elite_long:.4f} "
              f"short(cost_model)={elite_short:.4f} "
              f"total={elite_long + short_w * elite_short:.4f}")
        return mean

    @torch.inference_mode()
    def _capture_imagined(self, ll_init, mean):
        """Roll LL predictor for the first replan_every LL tokens of `mean`,
        decode each step, repeat each frame `action_block` times."""
        replan_every = int(self.cfg.get("replan_every", 1))
        # take exactly replan_every LL tokens (these are the ones we'll execute)
        actions = mean[:, :replan_every].unsqueeze(1)   # (B=1, 1, replan_every, A)
        init = ll_init.unsqueeze(1)                     # (B=1, 1, ll_dim)
        traj = self.h.ll_rollout_from_emb(init, actions)  # (B=1, 1, replan_every+1, ll_dim)
        # drop init slot; keep the replan_every predictions
        preds = traj.squeeze(0).squeeze(0).float()[1:]
        pix = self.ll_decoder(preds.to(next(self.ll_decoder.parameters()).dtype))
        frames = [to_uint8_chw(p) for p in pix]
        # repeat each LL-token frame `action_block` times -> env-step resolution
        expanded = np.repeat(np.stack(frames), self.action_block, axis=0)
        self._imagined_frames.extend(list(expanded))

    def get_action(self, info_dict, **kwargs):
        # Capture the env-render frame for the agent strip BEFORE parent mutates info_dict.
        if "pixels" in info_dict:
            try:
                self._agent_frames.append(capture_agent_frame(info_dict["pixels"]))
            except Exception as e:
                if not self._agent_frames:
                    print(f"[viz] could not capture agent frame: {e}")

        was_empty = len(self._action_buffer) == 0
        action = super().get_action(info_dict, **kwargs)

        if was_empty:
            # Just replanned. Use the parent's stored mean to capture imagined.
            mean = self._warm_mean
            device = next(self.h.parameters()).device
            pix_tx = info_dict["pixels"]
            if torch.is_tensor(pix_tx):
                pix_tx = pix_tx.to(device)
            with torch.inference_mode():
                ll_curr = self.h.encode_ll(pix_tx)[:, -1]
            self._capture_imagined(ll_curr, mean)
        return action

    def finalize(self, out_path: Path, fps: int = 10):
        if not self._agent_frames or not self._imagined_frames:
            print(f"[viz] nothing to write (agent={len(self._agent_frames)} "
                  f"imagined={len(self._imagined_frames)})")
            return
        # Equal length (trim to min). Agent strip = 1 frame per env step;
        # imagined strip = same after action_block repetition.
        n = min(len(self._agent_frames), len(self._imagined_frames))
        top = np.stack(self._imagined_frames[:n])      # (n, H, W, 3)
        bot = np.stack(self._agent_frames[:n])
        # Match heights / widths via resizing if they differ.
        if top.shape[1:3] != bot.shape[1:3]:
            import cv2
            new_h = min(top.shape[1], bot.shape[1])
            new_w = min(top.shape[2], bot.shape[2])
            top = np.stack([cv2.resize(f, (new_w, new_h)) for f in top])
            bot = np.stack([cv2.resize(f, (new_w, new_h)) for f in bot])
        # Stack vertically (top = imagined, bottom = agent).
        combined = np.concatenate([top, bot], axis=1)  # (n, 2H, W, 3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out_path, combined, fps=fps)
        print(f"[viz] saved {out_path}  ({n} env-step frames, "
              f"top=imagined LL rollouts, bottom=agent env render)")


def _load_cost_model(ckpt_path, device):
    """Load a CostHead from a checkpoint produced by train_cost_model.py."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    hl_dim = int(state.get("hl_dim", 96))
    ll_dim = int(state.get("ll_dim", 192))
    hidden_dim = int(state.get("hidden_dim", 192))
    depth = int(state.get("depth", 3))
    model = CostHead(hl_dim=hl_dim, ll_dim=ll_dim,
                     hidden_dim=hidden_dim, depth=depth).to(device).eval()
    sd = state.get("model", state.get("state_dict", state))
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[viz] CostHead <- {ckpt_path}  "
          f"(hl={hl_dim}, ll={ll_dim}, hidden={hidden_dim}, depth={depth})")
    return model


def _load_ll_decoder(folder, device):
    cfg_path = Path(folder) / "config.yaml"
    init_kwargs = dict(emb_dim=192, base=256, init_hw=7)
    if cfg_path.exists():
        c = OmegaConf.load(cfg_path)
        if "model" in c:
            init_kwargs.update(OmegaConf.to_container(c.model, resolve=True))
    dec = ConvDecoder(**init_kwargs).to(device).eval()
    w = latest_weights(folder)
    print(f"[viz] ConvDecoder <- {w}")
    sd = torch.load(w, map_location=device, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    dec.load_state_dict(sd, strict=False)
    return dec


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_unified")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = int(cfg.get("max_episode_steps_fixed", 1000))
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        proc = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        proc.fit(col_data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc

    device = torch.device("cuda")
    hjepa = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    hjepa = hjepa.to(device).eval(); hjepa.requires_grad_(False)
    hjepa.ll.predictor.to(torch.bfloat16)

    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    ll_decoder = _load_ll_decoder(Path(home) / "decoder_v4", device)

    # Optional learned cost head (replaces short_cost in unified CEM).
    # Path resolves from cfg.unified.cost_model_ckpt; null/empty → original cost.
    cost_model = None
    cm_path = cfg.unified.get("cost_model_ckpt", None)
    if cm_path:
        cost_model = _load_cost_model(cm_path, device)
    else:
        print("[viz] no cost_model_ckpt set → using stock unified CEM cost")

    ckpt_stem = Path(cfg.policy_ckpt).stem.replace("_object", "")
    out_dir = Path(home) / "eval_v2_viz" / ckpt_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    unified_cfg = OmegaConf.to_container(cfg.unified)
    unified_cfg["seed"] = int(cfg.seed)
    policy = V2VizPolicy(
        ll_decoder=ll_decoder, cost_model=cost_model,
        hjepa=hjepa, cfg=unified_cfg, action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        process=process, transform=transform,
    )

    def get_episodes_length(ds, eps):
        ep = ds.get_col_data(col_name); st = ds.get_col_data("step_idx")
        return np.array([np.max(st[ep == e]) + 1 for e in eps])
    ep_len = get_episodes_length(dataset, ep_indices)
    max_start = ep_len - cfg.eval.goal_offset_steps - 1
    start_dict = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_per_row = np.array([start_dict[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_per_row)[0]
    g = np.random.default_rng(cfg.seed)
    sel = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_eps = dataset.get_row_data(sel)[col_name].tolist()
    eval_starts = dataset.get_row_data(sel)["step_idx"].tolist()
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    world.set_policy(policy)
    print(f"[viz] writing to {out_dir}  ep={eval_eps[0]} start={eval_starts[0]}")
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_starts,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_eps,
            callables=callables,
            video_path=out_dir,
        )
    print(f"metrics: {metrics}")
    policy.finalize(out_dir / "imagined_vs_agent.mp4")


if __name__ == "__main__":
    run()
