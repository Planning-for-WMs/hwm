"""Unified single-stage CEM planner.

Idea: instead of (a) sampling macro actions then (b) optimizing LL actions to
each subgoal, we sample LL actions directly and let MAE map them to macros.
This keeps macros on-manifold by construction.

Per CEM iter (per env step replan):
  - Sample LL action tokens A of shape (B, S, T_HL * K_macro, 2*action_block).
  - HL path: chunk into (T_HL, K_macro), apply MAE per chunk -> T_HL macros.
    Roll HLP from HLE(LL_emb(obs)) for T_HL steps -> final HL state s_T.
    Long-term cost = ||s_T - HLE(LL_emb(goal))||^2.
  - LL path: take first K_macro tokens, roll LL predictor from LL_emb(obs).
    Short-term cost = ||HLE(LL_final) - s_1||^2 where s_1 is the first subgoal
    from the same HL rollout (so the LL planner anchors to the chosen macro).
  - cost = long + short_weight * short.

Execution: replan every action_block env steps (one LL token).
Warm-start: shift LL action mean left by one token, pad tail with last token.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.transforms import v2 as transforms

from hjepa import HierarchicalJEPA


def img_transform(cfg):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_dataset(cfg, name):
    return swm.data.HDF5Dataset(
        name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=Path(cfg.cache_dir or swm.data.utils.get_cache_dir()),
    )


class UnifiedPolicy(swm.policy.BasePolicy):
    def __init__(self, hjepa: HierarchicalJEPA, cfg, action_block: int,
                 eval_budget: int, process=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "hierarchical"
        self.h = hjepa
        self.cfg = cfg
        self.action_block = action_block
        self.eval_budget = eval_budget
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None
        self._warm_mean: torch.Tensor | None = None
        self._total_steps = 0
        # rolling HL history buffers (populated on first replan, shifted thereafter)
        self._hl_states: torch.Tensor | None = None    # (num_envs, hist, hl_dim)
        self._hl_macros: torch.Tensor | None = None    # (num_envs, hist, hl_dim)
        self._pending_macro: torch.Tensor | None = None  # macro executed this round

    def set_env(self, env):
        self.env = env
        self._replan_every = int(self.cfg.get("replan_every", 1))
        T_HL = int(self.cfg["T_HL"])
        K_macro = int(self.cfg["K_macro"])
        max_exec = min(self._replan_every, T_HL * K_macro)
        self._action_buffer = deque(maxlen=max_exec * self.action_block)
        self._warm_mean = None
        self._total_steps = 0
        # reset HL history at episode boundary
        self._hl_states = None
        self._hl_macros = None
        self._pending_macro = None
        # private CEM RNG, decoupled from global stream so eval_budget changes
        # don't shift the random sequence consumed by CEM.
        device = next(self.h.parameters()).device
        self._cem_gen = torch.Generator(device=device).manual_seed(
            int(self.cfg.get("seed", 0))
        )

    @staticmethod
    def _shift_mean_copy_last(mean, shift=1):
        """Shift left by `shift` LL tokens; pad tail by copying the last token."""
        H = mean.size(1)
        if shift <= 0:
            return mean
        if shift >= H:
            return None
        out = torch.empty_like(mean)
        out[:, : H - shift] = mean[:, shift:]
        out[:, H - shift:] = mean[:, -1:].expand(-1, shift, -1)
        return out

    @torch.inference_mode()
    def _cem(self, ll_curr, s_hl_goal_p, warm_mean=None, ll_goal=None, final=False):
        """CEM. `s_hl_goal_p` is the goal in **projected** HL space (the space
        `predict_hl` lives in). All HL costs are computed there."""
        B = ll_curr.size(0)
        device = ll_curr.device
        T_HL = int(self.cfg["T_HL"])
        K_macro = int(self.cfg["K_macro"])
        S = int(self.cfg["num_samples"])
        n_steps = int(self.cfg["n_steps"])
        topk = int(self.cfg["topk"])
        var_scale = float(self.cfg["var_scale"])
        var_ema = float(self.cfg.get("var_ema", 0.9))
        short_w = float(self.cfg.get("short_weight", 1.0))
        action_dim = 2 * self.action_block
        H_plan = T_HL * K_macro                          # planning horizon (LL tokens)
        hist = self.h.hlp.history                        # HL history buffer size

        s_hl_curr_p = self.h.encode_hl_proj(ll_curr)     # projected HL state

        mean = warm_mean.clone() if warm_mean is not None else torch.zeros(
            B, H_plan, action_dim, device=device)
        var = var_scale * torch.ones(B, H_plan, action_dim, device=device)
        init = ll_curr.unsqueeze(1)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, H_plan, action_dim, device=device, generator=self._cem_gen,
            )
            # MAE per chunk: (B, S, T_HL, K_macro, A) -> (B, S, T_HL, macro_dim)
            blk = cands.view(B, S, T_HL, K_macro, action_dim)
            macros = self.h.encode_macro(
                blk.reshape(B * S * T_HL, K_macro, action_dim)
            ).view(B, S, T_HL, -1)
            # HL rollout: state + macro buffers maintained by the policy across
            # replans. Shape (B, hist, hl_dim) -> tile to (B*S, hist, hl_dim).
            states_init = self._hl_states.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hist, -1
            )
            macros_init = self._hl_macros.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hist, -1
            )
            m_flat = macros.reshape(B * S, T_HL, -1)
            states = states_init
            macros_buf = macros_init
            s_first = None
            for t in range(T_HL):
                macros_buf = torch.cat([macros_buf[:, 1:], m_flat[:, t : t + 1]], dim=1)
                s_next = self.h.predict_hl(states, macros_buf)
                if t == 0:
                    s_first = s_next
                states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            s_final = s_next.view(B, S, -1)
            s_first = s_first.view(B, S, -1)
            long_cost = ((s_final - s_hl_goal_p.unsqueeze(1)) ** 2).sum(dim=-1)
            # LL rollout of first macro's worth of tokens; short cost in proj HL space
            first_k = cands[:, :, :K_macro]
            ll_traj = self.h.ll_rollout_from_emb(init, first_k)
            ll_final = ll_traj[..., -1, :].float()
            short_cost = ((self.h.encode_hl_proj(ll_final) - s_first) ** 2).sum(dim=-1)
            if final:
                cost = ((ll_final - ll_goal.unsqueeze(1)) ** 2).sum(dim=-1)
                long_cost = cost
                short_cost = torch.zeros_like(cost)
            else:
                cost = long_cost + short_w * short_cost
            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, H_plan, action_dim)
            )
            mean = elites.mean(dim=1)
            var = var * var_ema
        # elite-mean costs for logging (final iteration topk)
        elite_long = torch.gather(long_cost, 1, idx).mean().item()
        elite_short = torch.gather(short_cost, 1, idx).mean().item()
        print(f"[replan step={self._total_steps}] long={elite_long:.4f} "
              f"short={elite_short:.4f} total={elite_long + short_w * elite_short:.4f}")
        return mean

    def get_action(self, info_dict, **kwargs):
        assert "pixels" in info_dict and "goal" in info_dict
        info_dict = self._prepare_info(info_dict)
        device = next(self.h.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if len(self._action_buffer) == 0:
            with torch.inference_mode():
                ll_curr = self.h.encode_ll(info_dict["pixels"])[:, -1]
                ll_goal = self.h.encode_ll(info_dict["goal"])[:, -1]
                s_hl_curr_p = self.h.encode_hl_proj(ll_curr)
                s_hl_goal = self.h.encode_hl_proj(ll_goal)
                real_dist = ((s_hl_curr_p - s_hl_goal) ** 2).sum(dim=-1).mean().item()

            # ---- maintain HL history buffer ----
            hist = self.h.hlp.history
            hl_dim = s_hl_curr_p.size(-1)
            macro_dim = self.h.mae.head[-1].out_features
            B_envs = s_hl_curr_p.size(0)
            if self._hl_states is None:
                # episode start: no real history yet; pad with current state and zero macros
                self._hl_states = s_hl_curr_p.unsqueeze(1).expand(-1, hist, -1).clone()
                self._hl_macros = torch.zeros(B_envs, hist, macro_dim, device=device)
            else:
                # shift left, append (new_state, just-executed-macro)
                self._hl_states = torch.cat(
                    [self._hl_states[:, 1:], s_hl_curr_p.unsqueeze(1)], dim=1
                )
                self._hl_macros = torch.cat(
                    [self._hl_macros[:, 1:], self._pending_macro.unsqueeze(1)], dim=1
                )
            print(f"[step={self._total_steps}] real ||HLE_proj(obs)-HLE_proj(goal)||²={real_dist:.4f}")
            warm = None
            if self._warm_mean is not None and self.cfg.get("warm_start", True):
                warm = self._shift_mean_copy_last(self._warm_mean, shift=self._replan_every)
            # final replan: switch to flat LL cost when either (a) within one
            # replan period of the budget, or (b) HL distance to goal drops
            # below `final_dist_threshold` (cost landscape gets flat near goal,
            # so we anchor to the actual LL goal once we're close).
            near_goal = real_dist < float(self.cfg.get("final_dist_threshold", 0.0))
            final_planning = near_goal or (
                self._total_steps + self._replan_every * self.action_block
                >= self.eval_budget
            )
            if final_planning:
                why = "near-goal" if near_goal else "budget"
                print(f"[step={self._total_steps}] FINAL replan ({why}) -> flat LL cost")
            mean = self._cem(
                ll_curr, s_hl_goal, warm_mean=warm,
                ll_goal=ll_goal, final=final_planning,
            )
            self._warm_mean = mean
            # cache MAE(first K_macro tokens of plan) as the macro that will be
            # executed before the next replan — pushed into history next round.
            K_macro = int(self.cfg["K_macro"])
            first_chunk_acts = mean[:, :K_macro]   # (B, K_macro, 2*action_block)
            with torch.inference_mode():
                self._pending_macro = self.h.encode_macro(first_chunk_acts)  # (B, hl_dim)
            # execute the first `replan_every` LL tokens, capped at planned horizon
            H = mean.size(1)
            n_exec = min(self._replan_every, H)
            chunk = mean[:, :n_exec].reshape(
                self.env.num_envs, n_exec * self.action_block, -1
            )
            self._action_buffer.clear()
            self._action_buffer.extend(chunk.transpose(0, 1).cpu())

        action = self._action_buffer.popleft().reshape(
            *self.env.action_space.shape
        ).numpy()
        self._total_steps += 1
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_unified")
def run(cfg: DictConfig):
    # Decouple env's max_episode_steps from eval_budget so changing eval_budget
    # doesn't perturb env construction / wrapper RNG consumption.
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

    hjepa = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    hjepa = hjepa.to("cuda").eval()
    hjepa.requires_grad_(False)
    hjepa.ll.predictor.to(torch.bfloat16)

    unified_cfg = OmegaConf.to_container(cfg.unified)
    unified_cfg["seed"] = int(cfg.seed)
    policy = UnifiedPolicy(
        hjepa=hjepa,
        cfg=unified_cfg,
        action_block=int(cfg.action_block),
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
    eval_eps = dataset.get_row_data(sel)[col_name]
    eval_starts = dataset.get_row_data(sel)["step_idx"]

    world.set_policy(policy)

    t0 = time.time()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_starts.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_eps.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=Path(cfg.policy_ckpt).parent,
        )
    dt = time.time() - t0
    print(metrics)
    print(f"[timing] total: {dt:.2f}s")
    out = Path(cfg.policy_ckpt).parent / cfg.output.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n==== CONFIG ====\n" + OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ====\nmetrics: {metrics}\nevaluation_time: {dt}s\n")


if __name__ == "__main__":
    run()
