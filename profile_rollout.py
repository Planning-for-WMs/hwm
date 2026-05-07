"""Drill into rollout: what's slow inside the autoregressive loop?"""
import time
import torch
import stable_worldmodel as swm
from einops import rearrange

device = "cuda"
model = swm.policy.AutoCostModel("pusht/lewm").to(device).eval()
model.requires_grad_(False)
model.interpolate_pos_encoding = True

B, S, T_obs, T_horizon = 10, 300, 1, 5
info = {
    "pixels": torch.rand(B, S, T_obs, 3, 224, 224, device=device),
    "goal":   torch.rand(B, S, T_obs, 3, 224, 224, device=device),
    "action": torch.zeros(B, S, T_obs, 2, device=device),
}
candidates = torch.zeros(B, S, T_horizon, 10, device=device)

# warm
for _ in range(2):
    info_copy = dict(info)
    out = model.get_cost(info_copy, candidates)
torch.cuda.synchronize()

# breakdown
t_total = t_actenc = t_pred = t_cat = t_rearr = 0.0
n_calls = 30

for _ in range(n_calls):
    info_copy = {k: v for k, v in info.items()}
    torch.cuda.synchronize(); t0 = time.time()

    # mimic rollout internals
    H = info_copy["pixels"].size(2)
    Bm, Sm, Tm = candidates.shape[:3]
    act_0, act_future = torch.split(candidates, [H, Tm - H], dim=2)
    info_copy["action"] = act_0
    n_steps = Tm - H

    _init = {k: v[:, 0] for k, v in info_copy.items() if torch.is_tensor(v)}
    _init = model.encode(_init)
    emb = _init["emb"].unsqueeze(1).expand(Bm, Sm, -1, -1)

    torch.cuda.synchronize(); ta = time.time()
    emb = rearrange(emb, "b s ... -> (b s) ...").clone()
    act = rearrange(act_0, "b s ... -> (b s) ...")
    act_future = rearrange(act_future, "b s ... -> (b s) ...")
    torch.cuda.synchronize(); t_rearr += time.time() - ta

    HS = 3
    for t in range(n_steps):
        torch.cuda.synchronize(); ta = time.time()
        act_emb = model.action_encoder(act)
        torch.cuda.synchronize(); t_actenc += time.time() - ta

        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        torch.cuda.synchronize(); ta = time.time()
        pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
        torch.cuda.synchronize(); t_pred += time.time() - ta

        torch.cuda.synchronize(); ta = time.time()
        emb = torch.cat([emb, pred_emb], dim=1)
        next_act = act_future[:, t:t+1, :]
        act = torch.cat([act, next_act], dim=1)
        torch.cuda.synchronize(); t_cat += time.time() - ta

    torch.cuda.synchronize(); ta = time.time()
    act_emb = model.action_encoder(act)
    t_actenc += time.time() - ta
    emb_trunc = emb[:, -HS:]; act_trunc = act_emb[:, -HS:]
    pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
    emb = torch.cat([emb, pred_emb], dim=1)
    torch.cuda.synchronize(); t_total += time.time() - t0

print(f"{n_calls} rollouts:")
print(f"  total:        {t_total:.2f}s ({t_total/n_calls*1000:.1f} ms/call)")
print(f"  action_encoder: {t_actenc:.2f}s")
print(f"  predict:        {t_pred:.2f}s")
print(f"  torch.cat:      {t_cat:.2f}s")
print(f"  rearrange:      {t_rearr:.2f}s")
