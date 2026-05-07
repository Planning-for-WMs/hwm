"""Time a single CEM solve() with synthetic info_dict to find the bottleneck."""
import time
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf

torch.cuda.synchronize()

device = "cuda"
B, S, T_obs = 10, 300, 1
H_img, W_img = 224, 224

# load model
model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True

# fake info_dict (matches what world.step gives the policy)
n_envs = 50
info = {
    "pixels": torch.rand(n_envs, T_obs, 3, H_img, W_img, device=device),
    "goal":   torch.rand(n_envs, T_obs, 3, H_img, W_img, device=device),
    "action": torch.zeros(n_envs, T_obs, 2, device=device),
    "proprio": torch.zeros(n_envs, T_obs, 4, device=device),
    "state":   torch.zeros(n_envs, T_obs, 7, device=device),
}

# build solver via hydra-like instantiation
cfg = OmegaConf.create({
    "_target_": "stable_worldmodel.solver.CEMSolver",
    "model": None, "batch_size": B,
    "num_samples": S, "var_scale": 1.0,
    "n_steps": 30, "topk": 30, "device": device, "seed": 42,
})
import hydra
solver = hydra.utils.instantiate(cfg, model=model)
plan_cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
solver.configure(action_space=type("A", (), {"shape": (n_envs, 2), "low": -1, "high": 1})(),
                 n_envs=n_envs, config=plan_cfg)
# fake action_space
import gymnasium as gym
solver._action_space = gym.spaces.Box(-1, 1, (n_envs, 2))
solver._action_dim = 2

# patch action_dim
print("action_dim:", solver.action_dim)

# warmup
torch.cuda.synchronize(); t0 = time.time()
solver.solve(info)
torch.cuda.synchronize(); print(f"warmup solve: {time.time()-t0:.2f}s")

# timed run with breakdown via patched get_cost
import jepa
orig_get_cost = jepa.JEPA.get_cost
times = {"encode": 0.0, "rollout": 0.0, "criterion": 0.0, "n_calls": 0}

def patched_get_cost(self, info_dict, action_candidates):
    torch.cuda.synchronize(); t = time.time()
    device = next(self.parameters()).device
    for k in list(info_dict.keys()):
        if torch.is_tensor(info_dict[k]):
            info_dict[k] = info_dict[k].to(device)
    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]
    for k in list(info_dict.keys()):
        if k.startswith("goal_"):
            goal[k[len("goal_"):]] = goal.pop(k)
    goal.pop("action", None)
    goal = self.encode(goal)
    info_dict["goal_emb"] = goal["emb"].unsqueeze(1)
    torch.cuda.synchronize(); times["encode"] += time.time() - t

    t = time.time()
    info_dict = self.rollout(info_dict, action_candidates)
    torch.cuda.synchronize(); times["rollout"] += time.time() - t

    t = time.time()
    cost = self.criterion(info_dict)
    torch.cuda.synchronize(); times["criterion"] += time.time() - t
    times["n_calls"] += 1
    return cost

jepa.JEPA.get_cost = patched_get_cost

torch.cuda.synchronize(); t0 = time.time()
solver.solve(info)
torch.cuda.synchronize(); total = time.time() - t0

print(f"\n== one solve() with B={B}, n_envs={n_envs}, n_steps=30 ==")
print(f"total:     {total:.2f}s")
print(f"get_cost called {times['n_calls']} times (= n_steps * n_batches = 30 * {n_envs//B})")
print(f"  encode:    {times['encode']:.2f}s")
print(f"  rollout:   {times['rollout']:.2f}s")
print(f"  criterion: {times['criterion']:.2f}s")
print(f"  other:     {total - sum(times[k] for k in ('encode','rollout','criterion')):.2f}s")
