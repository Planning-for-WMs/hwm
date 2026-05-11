"""Hierarchical eval: HL CEM for subgoals + LL CEM to reach the first subgoal.

Mirrors eval.py but plugs a HierarchicalPolicy into world.evaluate_from_dataset.
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


# ---- Hierarchical policy ----
class HierarchicalPolicy(swm.policy.BasePolicy):
    """Truly open-loop hierarchical planner.

    Per episode:
      1. Encode initial obs + goal pixels ONCE.
      2. HL CEM plans T_HL macro actions -> T_HL subgoals (last replaced by HLE(LL_goal)).
      3. For each subgoal in sequence, run an inline LL CEM that rolls out the
         LL predictor from a *latent* init (no pixel encoding); cost is at the
         final predicted LL emb passed through HLE. The optimal mean's predicted
         final LL emb becomes the latent init for the *next* subgoal's LL CEM.
      4. Concatenate all planned actions and execute mechanically. Pixels from
         the env are NEVER read again during the episode.
    """

    def __init__(self, hjepa: HierarchicalJEPA, ll_cfg, hl_cfg, action_block: int,
                 process=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "hierarchical"
        self.h = hjepa
        self.ll_cfg = ll_cfg
        self.hl_cfg = hl_cfg
        self.action_block = action_block
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None

    def set_env(self, env):
        self.env = env
        # buffer holds the entire open-loop plan: T_HL * horizon LL actions,
        # each expanded to action_block env-step actions.
        plan_len = (
            int(self.hl_cfg["T_HL"]) * int(self.ll_cfg["horizon"]) * int(self.action_block)
        )
        self._action_buffer = deque(maxlen=plan_len)

    @torch.inference_mode()
    def _hl_cem(self, s_hl_curr, s_hl_goal):
        """Inline CEM over macro action sequences. Returns optimal macro trajectory:
        macro_actions (B, T_HL, macro_a_dim) and HL state trajectory (B, T_HL+1, hl_dim)."""
        B = s_hl_curr.size(0)
        device = s_hl_curr.device
        T_HL = self.hl_cfg["T_HL"]
        S = self.hl_cfg["num_samples"]
        topk = self.hl_cfg["topk"]
        n_steps = self.hl_cfg["n_steps"]
        var_ema = float(self.hl_cfg.get("var_ema", 0.9))
        macro_dim = self.h.mae.head[-1].out_features

        mean = torch.zeros(B, T_HL, macro_dim, device=device)
        var = self.hl_cfg["var_scale"] * torch.ones(B, T_HL, macro_dim, device=device)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, T_HL, macro_dim, device=device
            )
            # roll out HL predictor for each sample
            s = s_hl_curr.unsqueeze(1).expand(-1, S, -1).reshape(B * S, -1)
            mc = cands.reshape(B * S, T_HL, macro_dim)
            for t in range(T_HL):
                s = self.h.predict_hl(s, mc[:, t])
            s = s.view(B, S, -1)
            cost = ((s - s_hl_goal.unsqueeze(1)) ** 2).sum(dim=-1)  # (B, S)
            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, T_HL, macro_dim)
            )
            mean = elites.mean(dim=1)
            var = var * var_ema

        traj = self.h.rollout_hl(s_hl_curr, mean)
        return mean, traj

    @torch.inference_mode()
    def _ll_cem(self, init_ll_emb, subgoal_hl):
        """Inline open-loop LL CEM. Plans `horizon` LL actions to reach the
        subgoal *in HL space*, starting from a latent LL embedding (no pixels).
        Returns (B, horizon, action_dim) optimal actions and (B, ll_dim)
        predicted final LL emb (rolled out with the optimal mean)."""
        B = init_ll_emb.size(0)
        device = init_ll_emb.device
        S = int(self.ll_cfg["num_samples"])
        n_steps = int(self.ll_cfg["n_steps"])
        topk = int(self.ll_cfg["topk"])
        horizon = int(self.ll_cfg["horizon"])
        var_scale = float(self.ll_cfg["var_scale"])
        var_ema = float(self.ll_cfg.get("var_ema", 0.9))
        action_dim = 2 * self.action_block  # PushT: 2 * action_block

        mean = torch.zeros(B, horizon, action_dim, device=device)
        var = var_scale * torch.ones(B, horizon, action_dim, device=device)
        init = init_ll_emb.unsqueeze(1)  # (B, 1, ll_dim) -- 1-frame history

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, horizon, action_dim, device=device
            )
            traj = self.h.ll_rollout_from_emb(init, cands)               # (B, S, ?, ll_dim)
            final_ll = traj[..., -1, :]
            final_hl = self.h.encode_hl(final_ll)
            cost = ((final_hl - subgoal_hl.unsqueeze(1)) ** 2).sum(dim=-1)
            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, horizon, action_dim)
            )
            mean = elites.mean(dim=1)
            var = var * var_ema

        traj = self.h.ll_rollout_from_emb(init, mean.unsqueeze(1))       # (B, 1, ?, ll_dim)
        return mean, traj[:, 0, -1, :]

    def get_action(self, info_dict, **kwargs):
        assert "pixels" in info_dict and "goal" in info_dict
        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            device = next(self.h.parameters()).device
            for k in list(info_dict.keys()):
                if torch.is_tensor(info_dict[k]):
                    info_dict[k] = info_dict[k].to(device)

            # ----- one-shot encoding (real obs read here is the LAST one) -----
            with torch.inference_mode():
                ll_curr = self.h.encode_ll(info_dict["pixels"])[:, -1]   # (B, ll_dim)
                ll_goal = self.h.encode_ll(info_dict["goal"])[:, -1]
                s_hl_curr = self.h.encode_hl(ll_curr)
                s_hl_goal = self.h.encode_hl(ll_goal)

            # ----- Stage 1: HL CEM (once per episode) -----
            T_HL = int(self.hl_cfg["T_HL"])
            _, hl_traj = self._hl_cem(s_hl_curr, s_hl_goal)              # (B, T_HL+1, D)
            sg_list = [hl_traj[:, t] for t in range(1, T_HL + 1)]
            sg_list[-1] = s_hl_goal                                       # final = real goal

            # ----- Stage 2: chained LL CEM in latent space -----
            current_ll = ll_curr
            all_actions = []
            for subgoal in sg_list:
                actions, current_ll = self._ll_cem(current_ll, subgoal)
                all_actions.append(actions)
            full_plan = torch.cat(all_actions, dim=1)                    # (B, T_HL*H, A_dim)

            plan = full_plan.reshape(
                self.env.num_envs,
                T_HL * int(self.ll_cfg["horizon"]) * self.action_block,
                -1,
            )
            self._action_buffer.extend(plan.transpose(0, 1).cpu())

        action = self._action_buffer.popleft().reshape(*self.env.action_space.shape).numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action


# ---- main ----
@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_hl")
def run(cfg: DictConfig):
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
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

    # ---- load HL model ----
    hjepa = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    hjepa = hjepa.to("cuda").eval()
    hjepa.requires_grad_(False)

    policy = HierarchicalPolicy(
        hjepa=hjepa,
        ll_cfg=OmegaConf.to_container(cfg.ll_cem),
        hl_cfg=OmegaConf.to_container(cfg.hl_cem),
        action_block=int(cfg.action_block),
        process=process, transform=transform,
    )

    # ---- episode/start sampling (mirrors eval.py) ----
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
    out = Path(cfg.policy_ckpt).parent / cfg.output.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write("\n==== CONFIG ====\n" + OmegaConf.to_yaml(cfg))
        f.write(f"\n==== RESULTS ====\nmetrics: {metrics}\nevaluation_time: {dt}s\n")


if __name__ == "__main__":
    run()
