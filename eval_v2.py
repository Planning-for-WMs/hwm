"""eval_v2.py — 3-stage hierarchical CEM:

  Stage 1 — HL CEM in macro space
      Sample T_HL macros directly. Roll HLP forward through them to get a
      sequence of subgoals s_1, ..., s_T_HL (in projected HL space). Cost =
      L2(s_T_HL, s_goal). Output: best macros, best subgoals.

  Stage 2 — LL CEM with subgoal-following cost
      Sample LL actions of total length T_HL*K_macro tokens (i.e. T_HL
      segments of K_macro tokens each). Roll the LL predictor forward.
      Cost = sum_{i=1..T_HL-1} L2(HL(ll_emb_{i*K_macro}), s_i)
             + L2(HL(ll_emb_{T_HL*K_macro}), s_goal)        # last segment vs GOAL, not subgoal
      Output: best LL action plan.

  Stage 3 — Terminal refinement
      Warm-start LL CEM from stage 2's mean with small variance, few iters.
      Cost = L2(HL(ll_emb_final), s_goal) only.
      Just nudges the trajectory to actually land on the goal.

Execute the first `replan_every` LL tokens, then re-plan from the new obs.

Run:
  python eval_v2.py policy_ckpt=$STABLEWM_HOME/hjepa_v8/hjepa_v8_epoch_87_object.ckpt
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from collections import deque
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.transforms import v2 as transforms

from eval_unified import img_transform, get_dataset


class HV2Policy(swm.policy.BasePolicy):
    """Hierarchical 3-stage CEM. See module docstring."""

    def __init__(self, hjepa, cfg: dict, action_block: int, eval_budget: int,
                 process=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "hierarchical"
        self.h = hjepa
        self.cfg = cfg
        self.action_block = action_block
        self.eval_budget = eval_budget
        self.process = process or {}
        self.transform = transform or {}
        # Open-loop: plan once at episode start, fill buffer with the full
        # T_HL * K_macro * action_block env-step actions, pop until empty.
        self._action_buffer: deque[torch.Tensor] | None = None
        self._total_steps = 0

    def set_env(self, env):
        self.env = env
        T_HL = int(self.cfg["hl_cem"]["T_HL"])
        K_macro = int(self.cfg["ll_cem"]["K_macro"])
        full_plan_env_steps = T_HL * K_macro * self.action_block
        self._action_buffer = deque(maxlen=full_plan_env_steps)
        self._total_steps = 0
        device = next(self.h.parameters()).device
        self._cem_gen = torch.Generator(device=device).manual_seed(
            int(self.cfg.get("seed", 0))
        )

    # ---- helpers ----

    def _init_hl_buffers(self, s_hl_curr_p):
        """Episode-start defaults: state buffer = current state repeated `hist`
        times; macro buffer = zeros. Open-loop, so these never get updated
        between replans (there are no replans)."""
        hist = self.h.hlp.history
        macro_dim = self.h.mae.head[-1].out_features
        B = s_hl_curr_p.size(0)
        device = s_hl_curr_p.device
        states = s_hl_curr_p.unsqueeze(1).expand(-1, hist, -1).clone()
        macros = torch.zeros(B, hist, macro_dim, device=device)
        return states, macros

    @torch.inference_mode()
    def _rollout_subgoals(self, hl_states, hl_macros, macros):
        """Apply HLP autoregressively over `macros` (B, T_HL, macro_dim)
        starting from (hl_states, hl_macros). Returns (B, T_HL, hl_dim)."""
        states = hl_states
        macros_buf = hl_macros
        out = []
        for t in range(macros.size(1)):
            macros_buf = torch.cat([macros_buf[:, 1:], macros[:, t : t + 1]], dim=1)
            s_next = self.h.predict_hl(states, macros_buf)
            states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            out.append(s_next)
        return torch.stack(out, dim=1)

    # ---- Stage 1: HL CEM in macro space ----

    @torch.inference_mode()
    def _hl_cem(self, hl_states, hl_macros, s_hl_goal_p):
        c = self.cfg["hl_cem"]
        T_HL = int(c["T_HL"])
        S = int(c["num_samples"])
        topk = int(c["topk"])
        n_steps = int(c["n_steps"])
        var_scale = float(c["var_scale"])
        var_ema = float(c.get("var_ema", 0.95))
        macro_dim = self.h.mae.head[-1].out_features
        B = hl_states.size(0)
        device = hl_states.device

        mean = torch.zeros(B, T_HL, macro_dim, device=device)
        var = var_scale * torch.ones(B, T_HL, macro_dim, device=device)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, T_HL, macro_dim, device=device, generator=self._cem_gen,
            )                                                              # (B, S, T_HL, macro_dim)
            # HL rollout over each sample (from the same episode-start buffer)
            states = hl_states.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hl_states.size(1), -1)
            macros_buf = hl_macros.unsqueeze(1).expand(B, S, -1, -1).reshape(
                B * S, hl_macros.size(1), -1)
            mc = cands.reshape(B * S, T_HL, macro_dim)
            for t in range(T_HL):
                macros_buf = torch.cat([macros_buf[:, 1:], mc[:, t : t + 1]], dim=1)
                s_next = self.h.predict_hl(states, macros_buf)
                states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            final = s_next.view(B, S, -1)
            cost = ((final - s_hl_goal_p.unsqueeze(1)) ** 2).sum(dim=-1)   # (B, S)

            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, T_HL, macro_dim))
            mean = elites.mean(dim=1)
            var = var * var_ema
        # log
        elite_cost = torch.gather(cost, 1, idx).mean().item()
        print(f"  [HL CEM] elite_cost={elite_cost:.3f}")
        return mean                                                         # (B, T_HL, macro_dim)

    # ---- Stage 2: LL CEM with subgoal-following cost ----

    @torch.inference_mode()
    def _ll_subgoal_cem(self, ll_curr, subgoals, s_hl_goal_p):
        c = self.cfg["ll_cem"]
        K_macro = int(c["K_macro"])
        S = int(c["num_samples"])
        topk = int(c["topk"])
        n_steps = int(c["n_steps"])
        var_scale = float(c["var_scale"])
        var_ema = float(c.get("var_ema", 0.95))
        T_HL = subgoals.size(1)
        action_dim_raw = 2 * self.action_block
        H = T_HL * K_macro                                                  # total LL tokens
        B = ll_curr.size(0)
        device = ll_curr.device

        mean = torch.zeros(B, H, action_dim_raw, device=device)
        var = var_scale * torch.ones(B, H, action_dim_raw, device=device)
        init = ll_curr.unsqueeze(1)                                         # (B, 1, ll_dim)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, H, action_dim_raw, device=device, generator=self._cem_gen,
            )                                                              # (B, S, H, A)
            # LL rollout under these actions
            traj = self.h.ll_rollout_from_emb(init, cands)                 # (B, S, H+1, ll_dim)
            # checkpoint LL embs at segment boundaries: indices 1..T_HL of length K_macro each
            seg_idxs = [K_macro * (i + 1) for i in range(T_HL)]
            seg_embs = traj[:, :, seg_idxs, :].float()                     # (B, S, T_HL, ll_dim)
            seg_hl = self.h.encode_hl_proj(
                seg_embs.reshape(-1, seg_embs.size(-1))
            ).view(B, S, T_HL, -1)                                          # (B, S, T_HL, hl_dim)
            # cost: sum L2 to each subgoal, except last → goal
            target = subgoals.clone()                                       # (B, T_HL, hl_dim)
            target[:, -1] = s_hl_goal_p                                     # replace last subgoal with goal
            diff = seg_hl - target.unsqueeze(1)                             # (B, S, T_HL, hl_dim)
            cost = (diff ** 2).sum(dim=-1).sum(dim=-1)                      # sum over segments → (B, S)

            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, H, action_dim_raw))
            mean = elites.mean(dim=1)
            var = var * var_ema
        elite_cost = torch.gather(cost, 1, idx).mean().item()
        print(f"  [LL subgoal-CEM] elite_cost={elite_cost:.3f}")
        return mean                                                         # (B, H, A)

    # ---- Stage 3: Terminal refinement ----

    @torch.inference_mode()
    def _refine_cem(self, ll_curr, init_actions, s_hl_goal_p, ll_goal=None):
        c = self.cfg["refine_cem"]
        S = int(c["num_samples"])
        topk = int(c["topk"])
        n_steps = int(c["n_steps"])
        var_scale = float(c["var_scale"])
        var_ema = float(c.get("var_ema", 0.95))
        use_ll_cost = bool(c.get("ll_cost", False))
        B, H, action_dim_raw = init_actions.shape
        device = ll_curr.device

        mean = init_actions.clone()
        var = var_scale * torch.ones(B, H, action_dim_raw, device=device)
        init = ll_curr.unsqueeze(1)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, H, action_dim_raw, device=device, generator=self._cem_gen,
            )
            traj = self.h.ll_rollout_from_emb(init, cands)
            ll_final = traj[:, :, -1, :].float()                            # (B, S, ll_dim)
            if use_ll_cost and ll_goal is not None:
                cost = ((ll_final - ll_goal.unsqueeze(1)) ** 2).sum(dim=-1)
            else:
                hl_final = self.h.encode_hl_proj(
                    ll_final.reshape(-1, ll_final.size(-1))
                ).view(B, S, -1)
                cost = ((hl_final - s_hl_goal_p.unsqueeze(1)) ** 2).sum(dim=-1)
            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, H, action_dim_raw))
            mean = elites.mean(dim=1)
            var = var * var_ema
        elite_cost = torch.gather(cost, 1, idx).mean().item()
        print(f"  [refine CEM] elite_cost={elite_cost:.3f}  "
              f"(space={'LL' if use_ll_cost and ll_goal is not None else 'HL'})")
        return mean

    # ---- main entry point ----

    def get_action(self, info_dict, **kwargs):
        assert "pixels" in info_dict and "goal" in info_dict
        info_dict = self._prepare_info(info_dict)
        device = next(self.h.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if len(self._action_buffer) == 0:
            # Open-loop: plan once at episode start, fill buffer with the full
            # T_HL * K_macro * action_block env-step actions.
            with torch.inference_mode():
                ll_curr    = self.h.encode_ll(info_dict["pixels"])[:, -1]
                ll_goal    = self.h.encode_ll(info_dict["goal"])[:, -1]
                s_hl_curr_p = self.h.encode_hl_proj(ll_curr)
                s_hl_goal_p = self.h.encode_hl_proj(ll_goal)
                real_dist  = ((s_hl_curr_p - s_hl_goal_p) ** 2).sum(dim=-1).mean().item()
            mode = self.cfg.get("mode", "hierarchical")
            print(f"[step={self._total_steps}] mode={mode}  "
                  f"start real_dist={real_dist:.3f}")

            if mode == "hierarchical":
                hl_states, hl_macros = self._init_hl_buffers(s_hl_curr_p)
                best_macros = self._hl_cem(hl_states, hl_macros, s_hl_goal_p)
                subgoals = self._rollout_subgoals(hl_states, hl_macros, best_macros)
                best_acts = self._ll_subgoal_cem(ll_curr, subgoals, s_hl_goal_p)
                best_acts = self._refine_cem(ll_curr, best_acts, s_hl_goal_p,
                                             ll_goal=ll_goal)
            elif mode == "flat":
                # Single flat LL CEM with goal-anchored terminal cost. No HL CEM,
                # no subgoals, no refinement stage. The total horizon matches the
                # hierarchical plan: T_HL * K_macro LL tokens.
                T_HL = int(self.cfg["hl_cem"]["T_HL"])
                K_macro = int(self.cfg["ll_cem"]["K_macro"])
                H = T_HL * K_macro
                B = ll_curr.size(0); action_dim_raw = 2 * self.action_block
                cold_init = torch.zeros(B, H, action_dim_raw, device=device)
                best_acts = self._refine_cem(ll_curr, cold_init, s_hl_goal_p,
                                             ll_goal=ll_goal)
            else:
                raise ValueError(f"unknown policy mode: {mode}")

            # Predicted terminal distances (in both spaces) BEFORE execution.
            with torch.inference_mode():
                init = ll_curr.unsqueeze(1)                                   # (B, 1, ll_dim)
                actions = best_acts.unsqueeze(1)                              # (B, 1, H, A)
                traj = self.h.ll_rollout_from_emb(init, actions)              # (B, 1, H+1, ll_dim)
                ll_final = traj[:, 0, -1, :].float()                          # (B, ll_dim)
                hl_final = self.h.encode_hl_proj(ll_final)
                pred_hl_dist = ((hl_final - s_hl_goal_p) ** 2).sum(dim=-1).mean().item()
                pred_ll_dist = ((ll_final - ll_goal) ** 2).sum(dim=-1).mean().item()
            print(f"  [{mode}] predicted terminal: "
                  f"HL_dist={pred_hl_dist:.3f}  LL_dist={pred_ll_dist:.3f}")

            # Enqueue the full plan (no replanning)
            H_plan = best_acts.size(1)
            chunk = best_acts.reshape(
                self.env.num_envs, H_plan * self.action_block, -1)
            self._action_buffer.clear()
            self._action_buffer.extend(chunk.transpose(0, 1).cpu())

        action = self._action_buffer.popleft().reshape(
            *self.env.action_space.shape).numpy()
        self._total_steps += 1
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_v2")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
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
    hjepa = hjepa.to(device).eval()
    hjepa.requires_grad_(False)
    hjepa.ll.predictor.to(torch.bfloat16)

    pcfg = OmegaConf.to_container(cfg.policy, resolve=True)
    pcfg["seed"] = int(cfg.seed)
    policy = HV2Policy(
        hjepa=hjepa, cfg=pcfg, action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        process=process, transform=transform,
    )

    # episode selection (same as eval_unified)
    def get_episodes_length(ds, eps):
        ep = ds.get_col_data(col_name); st = ds.get_col_data("step_idx")
        return np.array([np.max(st[ep == e]) + 1 for e in eps])
    ep_len = get_episodes_length(dataset, ep_indices)
    max_start = ep_len - cfg.eval.goal_offset_steps - 1
    start_dict = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_per_row = np.array([start_dict[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_per_row)[0]
    if cfg.eval.get("episode_id") is not None:
        # specific-episode mode: pick the first valid start_step for the given episode_id
        ep_target = int(cfg.eval.episode_id)
        ep_rows = np.nonzero(dataset.get_col_data(col_name) == ep_target)[0]
        ep_steps = dataset.get_col_data("step_idx")[ep_rows]
        max_s = start_dict[ep_target]
        valid_starts = ep_rows[ep_steps <= max_s]
        if len(valid_starts) == 0:
            raise ValueError(f"episode {ep_target} has no valid start_step "
                             f"(max_start={max_s})")
        # optional fixed start_step override
        s_override = cfg.eval.get("start_step")
        if s_override is not None:
            mask = dataset.get_col_data("step_idx")[valid_starts] == int(s_override)
            if not mask.any():
                raise ValueError(f"start_step {s_override} not valid for ep {ep_target}")
            sel = np.array([valid_starts[mask.argmax()]])
        else:
            sel = np.array([valid_starts[0]])
        eval_eps = [ep_target]
        eval_starts = [int(dataset.get_row_data(sel)["step_idx"][0])]
    else:
        g = np.random.default_rng(cfg.seed)
        sel = np.sort(valid[g.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
        eval_eps = dataset.get_row_data(sel)[col_name].tolist()
        eval_starts = dataset.get_row_data(sel)["step_idx"].tolist()
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    ckpt_stem = Path(cfg.policy_ckpt).stem.replace("_object", "")
    out_dir = Path(home) / "eval_v2" / ckpt_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_v2] writing to {out_dir}")

    world.set_policy(policy)
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


if __name__ == "__main__":
    run()
