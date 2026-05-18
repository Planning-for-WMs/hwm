"""Hierarchical open-loop eval: HL CEM for T_HL subgoals, then a chained LL
CEM per subgoal in latent space. The env's pixels are encoded exactly once
per episode."""
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
                 eval_budget: int, final_trans_steps: int = 0,
                 process=None, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.type = "hierarchical"
        self.h = hjepa
        self.ll_cfg = ll_cfg
        self.hl_cfg = hl_cfg
        self.action_block = action_block
        self.eval_budget = eval_budget
        self.final_trans_steps = final_trans_steps     # last N env steps: flat LL toward goal
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None

    def set_env(self, env):
        self.env = env
        H_env_LL = int(self.ll_cfg["horizon"]) * int(self.action_block)       # one LL CEM's worth
        H_env_HL = H_env_LL                                                    # one macro
        full_plan = int(self.hl_cfg["T_HL"]) * H_env_LL
        self._ll_replan_every = int(self.ll_cfg.get("replan_every", full_plan))
        self._hl_replan_every = int(self.hl_cfg.get("replan_every", full_plan))
        # closed-loop iff LL replans before the full chain finishes.
        self._closed_loop = self._ll_replan_every < full_plan
        self._H_env_LL = H_env_LL
        self._H_env_HL = H_env_HL
        buf_len = H_env_LL if self._closed_loop else full_plan
        self._action_buffer = deque(maxlen=buf_len)

        # closed-loop state
        self._total_steps = 0
        self._steps_since_ll = 10**9          # force LL replan on first call
        self._steps_since_hl = 10**9          # force HL replan on first call
        self._cached_subgoal: torch.Tensor | None = None
        self._ll_warm_mean: torch.Tensor | None = None
        self._hl_warm_mean: torch.Tensor | None = None
        self._was_near_goal = False

    @staticmethod
    def _shift_mean(mean, shift):
        """Shift the rolling action distribution left by `shift` positions,
        padding the tail with zeros. Returns None if shift exhausts the horizon."""
        if shift <= 0:
            return mean
        H = mean.size(1)
        if shift >= H:
            return None
        out = torch.zeros_like(mean)
        out[:, : H - shift] = mean[:, shift:]
        return out

    @staticmethod
    def _action_smoothness_cost(actions, action_block, alpha=0.5, eps=1e-8):
        """Mirror of kevinghst's ActionChangeObjective. Penalizes abrupt changes
        between consecutive 2-D env-step actions. The LL CEM action tensor is
        (B, S, T, 2*action_block); we reshape to per-env-step (B, S, T*ab, 2)
        and compute squared diffs of L2 magnitudes and atan2 angles.

        Returns (B, S) cost.
        """
        B, S, T, AD = actions.shape
        a = actions.reshape(B, S, T * action_block, 2)
        # magnitude diffs
        mag = a.norm(dim=-1) + eps                           # (B, S, N)
        mag_diff = (mag[..., 1:] - mag[..., :-1]).pow(2).mean(dim=-1)
        # angle diffs (wrap to [-pi, pi] via atan2(sin, cos))
        ang = torch.atan2(a[..., 1], a[..., 0])              # (B, S, N)
        d = ang[..., 1:] - ang[..., :-1]
        ang_diff = torch.atan2(torch.sin(d), torch.cos(d)).pow(2).mean(dim=-1)
        return alpha * ang_diff + (1.0 - alpha) * mag_diff

    @torch.inference_mode()
    def _hl_cem(self, s_hl_curr, s_hl_goal, warm_mean=None):
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

        mean = warm_mean.clone() if warm_mean is not None else torch.zeros(B, T_HL, macro_dim, device=device)
        var = self.hl_cfg["var_scale"] * torch.ones(B, T_HL, macro_dim, device=device)

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, T_HL, macro_dim, device=device
            )
            # roll out HL predictor (history-h interleaved tokens; default at start)
            H = self.h.hlp.history
            s_curr = self.h.hl_projector(s_hl_curr)
            states = s_curr.unsqueeze(1).expand(-1, S, -1).unsqueeze(2).expand(
                -1, -1, H, -1).reshape(B * S, H, -1)
            macros_buf = torch.zeros(B * S, H, macro_dim, device=device)
            mc = cands.reshape(B * S, T_HL, macro_dim)
            for t in range(T_HL):
                macros_buf = torch.cat([macros_buf[:, 1:], mc[:, t : t + 1]], dim=1)
                s_next = self.h.predict_hl(states, macros_buf)
                states = torch.cat([states[:, 1:], s_next.unsqueeze(1)], dim=1)
            s = s_next.view(B, S, -1)
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
    def _ll_cem(self, init_ll_emb, subgoal_hl, warm_mean=None):
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
        smooth_w = float(self.ll_cfg.get("smoothness_weight", 0.0))
        smooth_alpha = float(self.ll_cfg.get("smoothness_alpha", 0.5))
        action_dim = 2 * self.action_block  # PushT: 2 * action_block

        mean = warm_mean.clone() if warm_mean is not None else torch.zeros(B, horizon, action_dim, device=device)
        var = var_scale * torch.ones(B, horizon, action_dim, device=device)
        init = init_ll_emb.unsqueeze(1)  # (B, 1, ll_dim) -- 1-frame history

        for _ in range(n_steps):
            cands = mean.unsqueeze(1) + var.unsqueeze(1).sqrt() * torch.randn(
                B, S, horizon, action_dim, device=device
            )
            traj = self.h.ll_rollout_from_emb(init, cands)               # (B, S, ?, ll_dim)
            final_ll = traj[..., -1, :].float()
            final_hl = self.h.encode_hl(final_ll)
            cost = ((final_hl - subgoal_hl.unsqueeze(1)) ** 2).sum(dim=-1)
            if smooth_w > 0:
                cost = cost + smooth_w * self._action_smoothness_cost(
                    cands, self.action_block, alpha=smooth_alpha
                )
            _, idx = torch.topk(cost, k=topk, dim=1, largest=False)
            elites = torch.gather(
                cands, 1, idx[:, :, None, None].expand(-1, -1, horizon, action_dim)
            )
            mean = elites.mean(dim=1)
            var = var * var_ema

        traj = self.h.ll_rollout_from_emb(init, mean.unsqueeze(1))       # (B, 1, ?, ll_dim)
        return mean, traj[:, 0, -1, :].float()

    def get_action(self, info_dict, **kwargs):
        assert "pixels" in info_dict and "goal" in info_dict
        info_dict = self._prepare_info(info_dict)
        device = next(self.h.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if self._closed_loop:
            self._closed_loop_step(info_dict)
        else:
            self._open_loop_step(info_dict)

        action = self._action_buffer.popleft().reshape(*self.env.action_space.shape).numpy()
        self._total_steps += 1
        self._steps_since_hl += 1
        self._steps_since_ll += 1
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)
        return action

    # ----- open-loop: encode once at episode start, chain T_HL LL CEMs -----
    def _open_loop_step(self, info_dict):
        if len(self._action_buffer) > 0:
            return
        with torch.inference_mode():
            ll_curr = self.h.encode_ll(info_dict["pixels"])[:, -1]
            ll_goal = self.h.encode_ll(info_dict["goal"])[:, -1]
            s_hl_curr = self.h.encode_hl(ll_curr)
            s_hl_goal = self.h.encode_hl(ll_goal)
        T_HL = int(self.hl_cfg["T_HL"])
        _, hl_traj = self._hl_cem(s_hl_curr, s_hl_goal)
        sg_list = [hl_traj[:, t] for t in range(1, T_HL + 1)]
        sg_list[-1] = s_hl_goal
        current_ll = ll_curr
        all_actions = []
        for subgoal in sg_list:
            actions, current_ll = self._ll_cem(current_ll, subgoal)
            all_actions.append(actions)
        full_plan = torch.cat(all_actions, dim=1)
        plan = full_plan.reshape(
            self.env.num_envs, T_HL * int(self.ll_cfg["horizon"]) * self.action_block, -1
        )
        self._action_buffer.extend(plan.transpose(0, 1).cpu())

    # ----- closed-loop: HL and LL replan at separate rates with warm-start -----
    def _closed_loop_step(self, info_dict):
        # encode current real obs + goal
        with torch.inference_mode():
            ll_curr = self.h.encode_ll(info_dict["pixels"])[:, -1]
            ll_goal = self.h.encode_ll(info_dict["goal"])[:, -1]
            s_hl_curr = self.h.encode_hl(ll_curr)
            s_hl_goal = self.h.encode_hl(ll_goal)

        # final-segment fallback: drop HL, target goal directly
        near_goal = self._total_steps >= self.eval_budget - self.final_trans_steps

        # ----- HL replan? -----
        hl_replan = (not near_goal) and (
            self._cached_subgoal is None or self._steps_since_hl >= self._hl_replan_every
        )
        if hl_replan:
            warm = None
            if self._hl_warm_mean is not None and self.hl_cfg.get("warm_start", True):
                shift_hl = self._steps_since_hl // self._H_env_HL
                warm = self._shift_mean(self._hl_warm_mean, shift_hl)
            macros, hl_traj = self._hl_cem(s_hl_curr, s_hl_goal, warm_mean=warm)
            self._hl_warm_mean = macros
            T_HL = int(self.hl_cfg["T_HL"])
            sg_list = [hl_traj[:, t] for t in range(1, T_HL + 1)]
            sg_list[-1] = s_hl_goal
            self._cached_subgoal = sg_list[0]
            self._steps_since_hl = 0
            # subgoal changed -> LL warm-start is stale
            self._ll_warm_mean = None

        ll_target = s_hl_goal if near_goal else self._cached_subgoal
        # near_goal flip: invalidate LL warm-start (target switched)
        if near_goal != self._was_near_goal:
            self._ll_warm_mean = None
            self._was_near_goal = near_goal

        # ----- LL replan? -----
        ll_replan = len(self._action_buffer) == 0 or self._steps_since_ll >= self._ll_replan_every
        if ll_replan:
            warm = None
            if self._ll_warm_mean is not None and self.ll_cfg.get("warm_start", True):
                shift_ll = self._steps_since_ll // self.action_block
                warm = self._shift_mean(self._ll_warm_mean, shift_ll)
            actions, _ = self._ll_cem(ll_curr, ll_target, warm_mean=warm)
            self._ll_warm_mean = actions
            plan = actions.reshape(
                self.env.num_envs, int(self.ll_cfg["horizon"]) * self.action_block, -1
            )
            self._action_buffer.clear()
            self._action_buffer.extend(plan.transpose(0, 1).cpu())
            self._steps_since_ll = 0


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
    # Planning bottleneck is the LL predictor (transformer) called O(11k) times
    # per episode at tiny shapes (~300x3x192). Keep its weights in bf16 to skip
    # the per-call autocast fp32<->bf16 copies that dominated the profile, then
    # torch.compile it to reduce Python/dispatch overhead (CPU was outpacing GPU).
    hjepa.ll.predictor.to(torch.bfloat16)

    policy = HierarchicalPolicy(
        hjepa=hjepa,
        ll_cfg=OmegaConf.to_container(cfg.ll_cem),
        hl_cfg=OmegaConf.to_container(cfg.hl_cem),
        action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        final_trans_steps=int(cfg.get("final_trans_steps", 0)),
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
    # Force the efficient SDPA backend: flash attention's per-kernel setup
    # dominates at T=3, the efficient kernel is ~30% faster on these shapes.
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
