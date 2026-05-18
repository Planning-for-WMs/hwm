"""eval_v2_receding_mpc.py — same 3-stage hierarchical CEM as eval_v2.py, but
**closed-loop** with receding-horizon replanning every `replan_every` LL tokens
(default 5 = 25 env-step actions).

Each replan:
  Stage 1 — HL CEM in macro space (cold init, no warm-start).
  Stage 2 — LL CEM, T_HL segments of K_macro LL tokens; subgoal-following cost
            with the LAST segment compared to the actual goal (not the last subgoal).
  Stage 3 — Terminal refinement (warm-start from stage 2 mean within this replan only).

Between replans the HL history buffer (states + macros) is maintained and
shifted, mirroring eval_unified.py's closed-loop behaviour. CEM means are NOT
warm-started across replans — each replan starts cold.

Run:
  python eval_v2_receding_mpc.py policy_ckpt=$STABLEWM_HOME/hjepa_v12/hjepa_v12_epoch_4_object.ckpt eval.num_eval=5
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


class HV2RecedingPolicy(swm.policy.BasePolicy):
    """3-stage CEM with receding-horizon MPC: plan T_HL*K_macro LL tokens
    ahead, execute the first `replan_every` LL tokens, replan from new obs.
    Maintains an HL state+macro history buffer across replans."""

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
        self._action_buffer: deque[torch.Tensor] | None = None
        self._hl_states: torch.Tensor | None = None        # (B, hist, hl_dim)
        self._hl_macros: torch.Tensor | None = None        # (B, hist, macro_dim)
        self._pending_macro: torch.Tensor | None = None    # macro executed this round
        self._total_steps = 0

    def set_env(self, env):
        self.env = env
        self._replan_every = int(self.cfg["replan_every"])           # in LL tokens
        # buffer holds the env-step actions for the next `replan_every` LL tokens
        self._action_buffer = deque(maxlen=self._replan_every * self.action_block)
        self._hl_states = None
        self._hl_macros = None
        self._pending_macro = None
        self._total_steps = 0
        device = next(self.h.parameters()).device
        self._cem_gen = torch.Generator(device=device).manual_seed(
            int(self.cfg.get("seed", 0))
        )

    # ---- helpers ----

    @torch.inference_mode()
    def _rollout_subgoals(self, hl_states, hl_macros, macros):
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
            )
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
            cost = ((final - s_hl_goal_p.unsqueeze(1)) ** 2).sum(dim=-1)

            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, T_HL, macro_dim))
            mean = elites.mean(dim=1)
            var = var * var_ema
        elite_cost = torch.gather(cost, 1, idx).mean().item()
        print(f"  [HL CEM] elite_cost={elite_cost:.3f}")
        return mean                                                 # (B, T_HL, macro_dim)

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
        init = ll_curr.unsqueeze(1)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, H, action_dim_raw, device=device, generator=self._cem_gen,
            )
            traj = self.h.ll_rollout_from_emb(init, cands)
            seg_idxs = [K_macro * (i + 1) for i in range(T_HL)]
            seg_embs = traj[:, :, seg_idxs, :].float()
            seg_hl = self.h.encode_hl_proj(
                seg_embs.reshape(-1, seg_embs.size(-1))
            ).view(B, S, T_HL, -1)
            target = subgoals.clone()
            target[:, -1] = s_hl_goal_p
            diff = seg_hl - target.unsqueeze(1)
            cost = (diff ** 2).sum(dim=-1).sum(dim=-1)

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
            ll_final = traj[:, :, -1, :].float()
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
            # Closed-loop: re-plan all 3 stages from the new observation each
            # time the action buffer empties.
            with torch.inference_mode():
                ll_curr    = self.h.encode_ll(info_dict["pixels"])[:, -1]
                ll_goal    = self.h.encode_ll(info_dict["goal"])[:, -1]
                s_hl_curr_p = self.h.encode_hl_proj(ll_curr)
                s_hl_goal_p = self.h.encode_hl_proj(ll_goal)
                real_dist  = ((s_hl_curr_p - s_hl_goal_p) ** 2).sum(dim=-1).mean().item()

            # ---- maintain HL history buffer across replans ----
            hist = self.h.hlp.history
            macro_dim = self.h.mae.head[-1].out_features
            B_envs = s_hl_curr_p.size(0)
            if self._hl_states is None:
                # episode start: no real history yet — pad with current state
                self._hl_states = s_hl_curr_p.unsqueeze(1).expand(-1, hist, -1).clone()
                self._hl_macros = torch.zeros(B_envs, hist, macro_dim, device=device)
            else:
                # shift left, append (new s_curr, just-executed macro)
                self._hl_states = torch.cat(
                    [self._hl_states[:, 1:], s_hl_curr_p.unsqueeze(1)], dim=1)
                self._hl_macros = torch.cat(
                    [self._hl_macros[:, 1:], self._pending_macro.unsqueeze(1)], dim=1)

            mode = self.cfg.get("mode", "hierarchical")
            print(f"[step={self._total_steps}] mode={mode}  real_dist={real_dist:.3f}")

            if mode == "hierarchical":
                best_macros = self._hl_cem(self._hl_states, self._hl_macros, s_hl_goal_p)
                subgoals = self._rollout_subgoals(self._hl_states, self._hl_macros, best_macros)
                best_acts = self._ll_subgoal_cem(ll_curr, subgoals, s_hl_goal_p)
                best_acts = self._refine_cem(ll_curr, best_acts, s_hl_goal_p,
                                             ll_goal=ll_goal)
            elif mode == "flat":
                T_HL = int(self.cfg["hl_cem"]["T_HL"])
                K_macro = int(self.cfg["ll_cem"]["K_macro"])
                H = T_HL * K_macro
                B = ll_curr.size(0); action_dim_raw = 2 * self.action_block
                cold_init = torch.zeros(B, H, action_dim_raw, device=device)
                best_acts = self._refine_cem(ll_curr, cold_init, s_hl_goal_p,
                                             ll_goal=ll_goal)
            else:
                raise ValueError(f"unknown policy mode: {mode}")

            # Predicted terminal distances (in both spaces).
            with torch.inference_mode():
                init = ll_curr.unsqueeze(1)
                actions = best_acts.unsqueeze(1)
                traj = self.h.ll_rollout_from_emb(init, actions)
                ll_final = traj[:, 0, -1, :].float()
                hl_final = self.h.encode_hl_proj(ll_final)
                pred_hl_dist = ((hl_final - s_hl_goal_p) ** 2).sum(dim=-1).mean().item()
                pred_ll_dist = ((ll_final - ll_goal) ** 2).sum(dim=-1).mean().item()
            print(f"  [{mode}] predicted terminal: "
                  f"HL_dist={pred_hl_dist:.3f}  LL_dist={pred_ll_dist:.3f}")

            # Cache the macro that will be executed before the next replan
            # (needed for the history-buffer push at the next replan).
            K_macro = int(self.cfg["ll_cem"]["K_macro"])
            with torch.inference_mode():
                self._pending_macro = self.h.encode_macro(best_acts[:, :K_macro])

            # Execute the first `replan_every` LL tokens, then replan.
            H_plan = best_acts.size(1)
            n_exec = min(self._replan_every, H_plan)
            chunk = best_acts[:, :n_exec].reshape(
                self.env.num_envs, n_exec * self.action_block, -1)
            self._action_buffer.clear()
            self._action_buffer.extend(chunk.transpose(0, 1).cpu())

        action = self._action_buffer.popleft().reshape(
            *self.env.action_space.shape).numpy()
        self._total_steps += 1
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_v2_receding")
def run(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    # Receding-horizon MPC: episode budget controls total env-step duration,
    # NOT the plan horizon. Plan covers T_HL*K_macro LL tokens; we only execute
    # the first `policy.replan_every` LL tokens between replans.
    if "replan_every" not in cfg.policy:
        cfg.policy.replan_every = 5  # default: 5 LL tokens = 25 env-step actions
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
    policy = HV2RecedingPolicy(
        hjepa=hjepa, cfg=pcfg, action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        process=process, transform=transform,
    )

    # episode selection (same as eval_v2)
    def get_episodes_length(ds, eps):
        ep = ds.get_col_data(col_name); st = ds.get_col_data("step_idx")
        return np.array([np.max(st[ep == e]) + 1 for e in eps])
    ep_len = get_episodes_length(dataset, ep_indices)
    max_start = ep_len - cfg.eval.goal_offset_steps - 1
    start_dict = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_per_row = np.array([start_dict[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_per_row)[0]
    if cfg.eval.get("episode_id") is not None:
        ep_target = int(cfg.eval.episode_id)
        ep_rows = np.nonzero(dataset.get_col_data(col_name) == ep_target)[0]
        ep_steps = dataset.get_col_data("step_idx")[ep_rows]
        max_s = start_dict[ep_target]
        valid_starts = ep_rows[ep_steps <= max_s]
        if len(valid_starts) == 0:
            raise ValueError(f"episode {ep_target} has no valid start_step "
                             f"(max_start={max_s})")
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
    out_dir = Path(home) / "eval_v2_receding" / ckpt_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_v2_receding_mpc] writing to {out_dir}")

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
