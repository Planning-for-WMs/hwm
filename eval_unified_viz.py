"""eval_unified with visualization.

Inherits UnifiedPolicy from eval_unified. At every replan, additionally:
  1) Decodes the T_HL imagined subgoals (from the picked CEM plan) using
     HLDecoder -> ConvDecoder, and accumulates them as a row of a per-replan
     subgoal image table. Saves as <viz_dir>/subgoals_table.png at the end.
  2) Autoregressively rolls out the LL predictor from the current LL emb
     under the CEM-picked action plan, decodes each step via ConvDecoder,
     and saves <viz_dir>/imagined_replan_<idx>.mp4.
  3) Captures every env-rendered observation and saves <viz_dir>/expert.mp4.

Notes on space mismatch:
  - HLDecoder was trained on the pre-projector HL state (HLE output). At
    plan time we work in projected space (post hl_projector / hl_pred_proj).
    Feeding projected states to HLDecoder is mildly OOD; reconstructions
    are still typically informative.
  - ConvDecoder was trained on pre-projector LL CLS. The LL rollout chains
    post-projector embeddings. The decode is therefore also slightly OOD.

Intended for num_eval=1 (debugging / inspection). Multi-episode runs append
all replans into the same files.

Run:
  python eval_unified_viz.py policy_ckpt=$STABLEWM_HOME/hjepa_v5/hjepa_v5_epoch_80_object.ckpt \
      eval.num_eval=1
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import hydra
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval_unified import UnifiedPolicy, img_transform, get_dataset
from train_decoder import ConvDecoder
from train_hl_decoder import HLDecoder


def latest_weights(folder, pattern="*_weights.ckpt"):
    """Pick the highest-epoch weights.ckpt under `folder`."""
    cands = list(Path(folder).glob(pattern))
    if not cands:
        raise FileNotFoundError(f"no {pattern} under {folder}")
    def epoch_of(p):
        s = p.stem
        if "_epoch_" in s:
            try:
                return int(s.split("_epoch_")[1].split("_")[0])
            except Exception:
                return -1
        return -1
    epoched = [p for p in cands if epoch_of(p) >= 0]
    if epoched:
        return max(epoched, key=epoch_of)
    return max(cands, key=lambda p: p.stat().st_mtime)


def to_uint8_chw(pix):
    """(3, H, W) float in [-1, 1] -> uint8 HWC numpy."""
    pix = pix.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    return pix.permute(1, 2, 0).cpu().numpy()


def capture_expert_frame(pixels):
    """info_dict['pixels'] -> uint8 HWC numpy (env render). Tolerant to layouts."""
    if isinstance(pixels, np.ndarray):
        p = pixels
        # squeeze leading singleton dims (B, T) until 3-D
        while p.ndim > 3:
            p = p[0]
        if p.dtype != np.uint8:
            p = np.clip(p, 0, 255).astype(np.uint8)
        return p
    if isinstance(pixels, torch.Tensor):
        p = pixels.detach()
        while p.dim() > 3:
            p = p[0]
        # If CHW, transpose
        if p.dim() == 3 and p.shape[0] == 3 and p.shape[-1] != 3:
            p = p.permute(1, 2, 0)
        p = p.cpu().numpy()
        if p.dtype != np.uint8:
            p = np.clip(p, 0, 255).astype(np.uint8)
        return p
    raise TypeError(f"unsupported pixels type: {type(pixels)}")


class VizUnifiedPolicy(UnifiedPolicy):
    def __init__(self, hl_decoder, ll_decoder, viz_dir, **kw):
        super().__init__(**kw)
        self.hl_decoder = hl_decoder
        self.ll_decoder = ll_decoder
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self._subgoal_rows: list[np.ndarray] = []   # one (T_HL, H, W, 3) per replan
        self._obs_decoded: list[np.ndarray] = []    # one (H, W, 3) per replan — obs HL decoded
        self._replan_idx = 0
        self._expert_frames: list[np.ndarray] = []

    # ---- decoding helpers ----

    @torch.inference_mode()
    def _decode_hl_state(self, s_hl_proj):
        """(N, hl_dim) projected HL state -> (N, 3, H, W) pixels in [-1, 1]."""
        s = s_hl_proj.to(next(self.hl_decoder.parameters()).dtype)
        ll_emb = self.hl_decoder(s)
        pix = self.ll_decoder(ll_emb.to(next(self.ll_decoder.parameters()).dtype))
        return pix

    @torch.inference_mode()
    def _decode_ll_rollout_to_mp4(self, ll_init, action_plan, out_path,
                                  action_block=5, fps=10):
        """Imagined LL rollout decoded to MP4 using decoder_v4 (post-projector).
        Each frame is repeated `action_block` times so the clip duration matches
        the env-step horizon (T_HL * K_macro * action_block env steps)."""
        # decoder_v4 is trained on POST-projector LL emb. The LL rollout's chain
        # produces post-pred_proj embeddings (loss-equivalent to post-projector),
        # which is exactly the decoder's training distribution. Feed those.
        init = ll_init.unsqueeze(1)                                   # (1, 1, ll_dim) post-projector
        actions = action_plan.unsqueeze(1)                            # (1, 1, H_plan, A)
        traj = self.h.ll_rollout_from_emb(init, actions)              # (1, 1, H_plan+1, ll_dim)
        traj = traj.squeeze(0).squeeze(0).float()                     # (H_plan+1, ll_dim)
        all_pix = self.ll_decoder(traj.to(next(self.ll_decoder.parameters()).dtype))
        all_frames = [to_uint8_chw(p) for p in all_pix]
        # repeat each frame action_block times -> env-step-resolution duration
        expanded = np.repeat(np.stack(all_frames), action_block, axis=0)
        iio.imwrite(out_path, expanded, fps=fps)

    @torch.inference_mode()
    def _subgoals_from_plan(self, mean):
        """Run HL rollout once using `mean` (B=1, H_plan, A) with the maintained
        history buffer; return (T_HL, hl_dim) subgoals in **pre-hl_pred_proj**
        space (HLP's direct output, before hl_pred_proj). Chain still uses
        post-projector states for rollout consistency, but the decoded ones
        are the raw HLP outputs — closer to HLDecoder's training distribution
        if hl_pred_proj ≈ hl_projector."""
        T_HL = int(self.cfg["T_HL"])
        K_macro = int(self.cfg["K_macro"])
        action_dim = mean.size(-1)
        blk = mean.view(1, T_HL, K_macro, action_dim)
        macros = self.h.encode_macro(blk.reshape(T_HL, K_macro, action_dim))  # (T_HL, hl_dim)
        macros = macros.unsqueeze(0)                                          # (1, T_HL, hl_dim)
        states = self._hl_states.clone()
        macros_buf = self._hl_macros.clone()
        # hl_decoder_v2 is trained on POST-hl_projector HL state (= what predict_hl
        # produces via hl_pred_proj, loss-equivalent). Return post-pred_proj.
        subgoals = []
        for t in range(T_HL):
            macros_buf = torch.cat([macros_buf[:, 1:], macros[:, t : t + 1]], dim=1)
            s_next = self.h.predict_hl(states, macros_buf)
            states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            subgoals.append(s_next)
        return torch.cat(subgoals, dim=0)                                     # (T_HL, hl_dim) post-projector

    # ---- override get_action ----

    def get_action(self, info_dict, **kwargs):
        # Capture env-rendered frame for the expert MP4. info_dict here is raw
        # (the parent's _prepare_info returns a NEW dict; the outer copy stays raw).
        if "pixels" in info_dict:
            try:
                self._expert_frames.append(capture_expert_frame(info_dict["pixels"]))
            except Exception as e:
                if self._replan_idx == 0 and len(self._expert_frames) == 0:
                    print(f"[viz] could not capture expert frame: {e}")

        was_empty = (len(self._action_buffer) == 0)
        action = super().get_action(info_dict, **kwargs)

        if was_empty:
            # we just replanned — produce viz for this replan.
            mean = self._warm_mean                  # (B=1, H_plan, A) — picked plan
            # parent's _prepare_info has already mutated info_dict; pixels is now
            # a transformed tensor. Re-encode from it.
            device = next(self.h.parameters()).device
            pixels_tx = info_dict["pixels"]
            if torch.is_tensor(pixels_tx):
                pixels_tx = pixels_tx.to(device)
            with torch.inference_mode():
                ll_curr = self.h.encode_ll(pixels_tx)[:, -1]                # (1, ll_dim) post-projector
                # Control: decode the obs's POST-hl_projector HL state — the
                # same distribution hl_decoder_v2 was trained on AND the same
                # one predicted subgoals live in. If this looks like PushT,
                # the decoder pipeline is healthy and any failure in the
                # predicted-subgoal columns is purely a model issue.
                s_hl_obs_p = self.h.encode_hl_proj(ll_curr)
                obs_pix = self._decode_hl_state(s_hl_obs_p)                 # (1, 3, H, W)
                self._obs_decoded.append(to_uint8_chw(obs_pix[0]))
            # subgoals row
            subgoals = self._subgoals_from_plan(mean)                       # (T_HL, hl_dim)
            sg_pix = self._decode_hl_state(subgoals)                        # (T_HL, 3, H, W)
            self._subgoal_rows.append(np.stack([to_uint8_chw(p) for p in sg_pix]))
            # imagined LL rollout video, decoded via decoder_v4
            self._decode_ll_rollout_to_mp4(
                ll_curr, mean,
                self.viz_dir / f"imagined_replan_{self._replan_idx}.mp4",
                action_block=self.action_block,
            )
            print(f"[viz] saved replan {self._replan_idx}: "
                  f"{len(sg_pix)} subgoals + imagined rollout")
            self._replan_idx += 1
        return action

    # ---- finalization ----

    def finalize(self):
        # The env-rendered trajectory is the AGENT's behavior under our policy
        # (not the expert demo).  Saved as agent.mp4; expert.mp4 is dumped in
        # run() directly from the H5 pixel dataset.
        if self._expert_frames:
            iio.imwrite(self.viz_dir / "agent.mp4",
                        np.stack(self._expert_frames), fps=10)
            print(f"[viz] saved agent.mp4 ({len(self._expert_frames)} frames)")
        if self._subgoal_rows:
            R = len(self._subgoal_rows)
            C = max(len(r) for r in self._subgoal_rows)
            # Each row has 1 control column (decoded obs) + C predicted columns.
            ncols = C + 1
            fig, axes = plt.subplots(R, ncols, figsize=(2 * ncols, 2 * R), squeeze=False)
            for r in range(R):
                ax0 = axes[r][0]
                ax0.imshow(self._obs_decoded[r]); ax0.set_xticks([]); ax0.set_yticks([])
                if r == 0: ax0.set_title("obs (sanity)", fontsize=9)
                ax0.set_ylabel(f"replan {r}", fontsize=9)
                for c in range(C):
                    ax = axes[r][c + 1]
                    if c < len(self._subgoal_rows[r]):
                        ax.imshow(self._subgoal_rows[r][c])
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(f"subgoal {c+1}", fontsize=9)
            fig.suptitle("HL decoded (hl_decoder_v2 + decoder_v4, post-projector) — "
                         "col 0: obs HL state; cols 1..: predicted subgoals",
                         fontsize=10)
            fig.tight_layout()
            out = self.viz_dir / "subgoals_table.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[viz] saved {out}  ({R} replans × {C} subgoals + 1 control col)")


def _load_hl_decoder(folder, hjepa_arch_cfg, device):
    """Try resolving HLDecoder kwargs from <folder>/config.yaml; fall back to defaults."""
    cfg_path = Path(folder) / "config.yaml"
    kwargs = dict(hl_dim=96, ll_dim=192, hidden_dim=512, depth=4)
    if cfg_path.exists():
        c = OmegaConf.load(cfg_path)
        if "model" in c:
            kwargs.update(OmegaConf.to_container(c.model, resolve=True))
    dec = HLDecoder(**kwargs).to(device).eval()
    weights_path = latest_weights(folder)
    print(f"[viz] HLDecoder <- {weights_path}")
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    dec.load_state_dict(sd, strict=True)
    return dec


def _load_ll_decoder(folder, device):
    """ConvDecoder from decoder_v2/*.ckpt."""
    cfg_path = Path(folder) / "config.yaml"
    init_kwargs = dict(emb_dim=192, base=256, init_hw=7)
    if cfg_path.exists():
        c = OmegaConf.load(cfg_path)
        if "model" in c:
            init_kwargs.update(OmegaConf.to_container(c.model, resolve=True))
    dec = ConvDecoder(**init_kwargs).to(device).eval()
    weights_path = latest_weights(folder)
    print(f"[viz] ConvDecoder <- {weights_path}")
    sd = torch.load(weights_path, map_location=device, weights_only=True)
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

    # ---- load HJEPA and the two decoders ----
    device = torch.device("cuda")
    hjepa = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    hjepa = hjepa.to(device).eval()
    hjepa.requires_grad_(False)
    hjepa.ll.predictor.to(torch.bfloat16)

    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    hl_decoder = _load_hl_decoder(Path(home) / "hl_decoder_v2", None, device)
    ll_decoder = _load_ll_decoder(Path(home) / "decoder_v4", device)

    # ---- output dir ----
    ckpt_stem = Path(cfg.policy_ckpt).stem.replace("_object", "")
    viz_dir = Path(home) / "eval_viz" / ckpt_stem
    print(f"[viz] writing to {viz_dir}")

    unified_cfg = OmegaConf.to_container(cfg.unified)
    unified_cfg["seed"] = int(cfg.seed)
    policy = VizUnifiedPolicy(
        hl_decoder=hl_decoder, ll_decoder=ll_decoder, viz_dir=viz_dir,
        hjepa=hjepa, cfg=unified_cfg, action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        process=process, transform=transform,
    )

    # ---- episode selection (same as eval_unified) ----
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

    # ---- expert-action LL rollout sanity test ----
    # Take the expert action sequence for the eval episode, roll out the LL
    # predictor from the encoded init obs, decode each step. If THIS rollout
    # looks coherent, the LL predictor + decoder pipeline is healthy and any
    # bad-looking CEM rollouts come from OOD actions. If THIS rollout still
    # degrades, LL predictor itself drifts at this horizon.
    try:
        import h5py
        pix_path_sanity = Path(home) / f"{cfg.eval.dataset_name}.h5"
        emb_path_sanity = Path(home) / f"{cfg.eval.dataset_name}_emb.h5"
        with h5py.File(emb_path_sanity, "r") as fe, h5py.File(pix_path_sanity, "r") as fp:
            ep_col_emb = "episode_idx" if "episode_idx" in fe else "ep_idx"
            ep_ids = fe[ep_col_emb][:]; steps = fe["step_idx"][:]
            ep_i, s0 = int(eval_eps[0]), int(eval_starts[0])
            # absolute rows for this episode
            mask = ep_ids == ep_i
            ep_rows = np.nonzero(mask)[0]
            step_to_row = {int(steps[r]): int(r) for r in ep_rows}
            T_HL = int(cfg.unified.T_HL); K_macro = int(cfg.unified.K_macro)
            n_ll_tokens = T_HL * K_macro
            # gather init pre-emb at s0 and the next n_ll_tokens action rows
            init_row = step_to_row[s0]
            act_rows = [step_to_row[s0 + t] for t in range(n_ll_tokens)]
            pre_emb_init = torch.from_numpy(fe["emb"][init_row]).to(device).float()
            actions_raw = torch.from_numpy(
                fe["action"][np.array(act_rows, dtype=np.int64)]
            ).to(device).float().unsqueeze(0)        # (1, n_ll_tokens, 10)
            # ground truth pixels for side-by-side comparison
            gt_rows = [step_to_row[s0 + t] for t in range(n_ll_tokens + 1)]
            gt_pix = fp["pixels"][gt_rows[0]:gt_rows[-1] + 1]
            if gt_pix.dtype != np.uint8:
                gt_pix = np.clip(gt_pix, 0, 255).astype(np.uint8)

        # encode init to post-projector space (matches predictor + decoder_v4)
        with torch.inference_mode():
            post_init = hjepa.ll.projector(pre_emb_init.unsqueeze(0))   # (1, ll_dim)
        # roll out under expert actions (single sample dim S=1)
        init = post_init.unsqueeze(1)                                    # (1, 1, ll_dim)
        action_seq = actions_raw.unsqueeze(1)                            # (1, 1, n_ll_tokens, 10)
        with torch.inference_mode():
            traj = hjepa.ll_rollout_from_emb(init, action_seq)
        traj = traj.squeeze(0).squeeze(0).float()                        # (n+1, ll_dim) post-projector
        with torch.inference_mode():
            pix = ll_decoder(traj.to(next(ll_decoder.parameters()).dtype))
        pred_frames = np.stack([to_uint8_chw(p) for p in pix])           # (n+1, H, W, 3)
        # repeat each LL-token frame action_block times so duration matches env steps
        pred_expanded = np.repeat(pred_frames, int(cfg.action_block), axis=0)
        gt_expanded = gt_pix                                             # already env-step resolution
        # crop to same length
        L = min(len(pred_expanded), len(gt_expanded))
        side = np.concatenate([gt_expanded[:L], pred_expanded[:L]], axis=2)  # GT left, imagined right
        out = viz_dir / "viz.mp4"
        viz_dir.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out, side, fps=10)
        print(f"[viz] wrote expert-action sanity {out}  "
              f"(ep={ep_i} start={s0} {L} frames, GT|IMAGINED side-by-side)")
    except Exception as e:
        print(f"[viz] expert-action rollout failed: {e}")

    # ---- dump the actual expert demo pixels (per chosen episode) ----
    try:
        import h5py
        pix_path = Path(home) / f"{cfg.eval.dataset_name}.h5"
        ep_off = dataset.get_col_data(col_name)
        step_idx_arr = dataset.get_col_data("step_idx")
        with h5py.File(pix_path, "r") as fp:
            for i, (ep_i, s0) in enumerate(zip(eval_eps, eval_starts)):
                # find absolute row indices for this episode's [s0, s0+goal_offset_steps]
                mask = (ep_off == ep_i)
                ep_row0 = int(np.nonzero(mask)[0].min())
                s_end = int(s0) + int(cfg.eval.goal_offset_steps)
                rows = list(range(ep_row0 + int(s0), ep_row0 + s_end))
                frames = fp["pixels"][rows[0]:rows[-1] + 1]
                if frames.dtype != np.uint8:
                    frames = np.clip(frames, 0, 255).astype(np.uint8)
                tag = "" if cfg.eval.num_eval == 1 else f"_{i}"
                out = viz_dir / f"expert{tag}.mp4"
                iio.imwrite(out, frames, fps=10)
                print(f"[viz] saved {out} (ep={ep_i} steps {s0}..{s_end-1}, "
                      f"{len(frames)} frames)")
    except Exception as e:
        print(f"[viz] could not dump expert demo: {e}")

    world.set_policy(policy)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_starts,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_eps,
            callables=callables,
            video_path=viz_dir,
        )
    print(f"metrics: {metrics}")
    policy.finalize()


if __name__ == "__main__":
    run()
