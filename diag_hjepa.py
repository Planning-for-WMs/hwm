"""Diagnose whether the HL hierarchy is learning.

Tests, in order:
  (1) Param counts.
  (2) HL state statistics (post-HLE and post-projector — the loss space).
  (3) MAE output statistics: real vs zero vs random action windows.
  (4) HLP action sensitivity:
        var(pred | vary only s) vs var(pred | vary only m).
        Linear fit pred ≈ A·states + b: fraction explained.
  (5) Prediction error vs baselines:
        HLP(real states, real macros)        — what the model actually predicts
        HLP(real states, shuffled macros)    — does it use the macro?
        HLP(real states, zero macros)        — what happens with no macro?
        HLP(repeat last state, real macros)  — does it use history?
        identity baseline (use s_curr_proj as prediction)
  (6) Time-collapse check: within-traj vs across-traj HL deltas.
  (7) HLE/projector explained variance spectrum.

Args: policy_ckpt=$STABLEWM_HOME/hjepa_v4/hjepa_v4_epoch_13_object.ckpt
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def stats(x, name):
    x = x.float()
    flat = x.reshape(-1)
    print(f"  {name}: shape={tuple(x.shape)} "
          f"mean={flat.mean().item():+.4f} std={flat.std().item():.4f} "
          f"min={flat.min().item():+.4f} max={flat.max().item():+.4f} "
          f"||x||₂_avg={x.flatten(0, -2).norm(dim=-1).mean().item():.3f}")


def rank_me(z, eps=1e-7):
    z = z.float() - z.float().mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    return float(torch.exp(H).item())


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_unified")
def run(cfg: DictConfig):
    device = "cuda"
    ckpt_path = cfg.policy_ckpt
    print(f"\n=== loading {ckpt_path} ===")
    hjepa = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hjepa = hjepa.to(device).eval()
    hjepa.requires_grad_(False)

    # ----- (1) params + arch info -----
    print("\n=== (1) parameter counts ===")
    pairs = [("ll", hjepa.ll), ("hle", hjepa.hle), ("hl_projector", hjepa.hl_projector),
             ("hlp", hjepa.hlp), ("mae", hjepa.mae), ("hl_pred_proj", hjepa.hl_pred_proj)]
    for name, m in pairs:
        print(f"  {name:>14s}: {count_params(m):>10,d} params")
    K = int(hjepa.k)                                  # this is k_max from training
    history = int(hjepa.hlp.history)
    hl_dim = hjepa.hle.net[-1].out_features
    ll_dim = hjepa.hle.net[0].in_features
    print(f"  history={history}, k_max={K}, ll_dim={ll_dim}, hl_dim={hl_dim}")

    # ----- load batch of pre-encoded data -----
    ds_name = cfg.eval.dataset_name + "_emb"
    print(f"\n=== loading dataset {ds_name} ===")
    ds = swm.data.HDF5Dataset(
        ds_name, keys_to_cache=["emb", "action"],
        cache_dir=Path(swm.data.utils.get_cache_dir()),
    )
    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_ids = ds.get_col_data(col)
    step = ds.get_col_data("step_idx")

    K_test = K                                        # use max stride for the probe
    F = 5                                              # frameskip
    BATCH = 1024
    rng = np.random.default_rng(0)
    # require room for (history+1) HL anchors spaced K_test*F raw frames apart
    max_start = {}
    for e in np.unique(ep_ids):
        ms = step[ep_ids == e].max() - history * K_test * F
        max_start[e] = ms
    valid_rows = np.nonzero(np.array(
        [step[i] <= max_start[ep_ids[i]] for i in range(len(step))]))[0]
    sel = np.sort(rng.choice(valid_rows, size=min(BATCH, len(valid_rows)), replace=False))

    # gather indices: history+1 anchor rows + history*K_test action rows
    anchor_idx = np.zeros((BATCH, history + 1), dtype=np.int64)
    action_idx = np.zeros((BATCH, history * K_test * F), dtype=np.int64)
    for i, r in enumerate(sel):
        e = ep_ids[r]; s0 = step[r]
        ep_rows = np.nonzero(ep_ids == e)[0]
        step_to_row = {step[j]: j for j in ep_rows}
        for h in range(history + 1):
            anchor_idx[i, h] = step_to_row[s0 + h * K_test * F]
        for t in range(history * K_test * F):
            action_idx[i, t] = step_to_row[s0 + t]

    def fetch_sorted(idxs, key):
        idxs = np.asarray(idxs)
        uniq, inv = np.unique(idxs, return_inverse=True)
        return ds.get_row_data(uniq)[key][inv]

    anchor_emb = fetch_sorted(anchor_idx.reshape(-1), "emb").reshape(BATCH, history + 1, -1)
    raw_actions = fetch_sorted(action_idx.reshape(-1), "action").reshape(
        BATCH, history * K_test, F * 2)              # pack F=5 env actions per LL token
    anchor_emb = torch.from_numpy(anchor_emb).to(device).float()
    raw_actions = torch.from_numpy(raw_actions).to(device).float()

    # post-projector embeddings (frozen LL projector), then HLE, then HL projector
    flat = anchor_emb.reshape(-1, ll_dim)
    post_emb = hjepa.ll.projector(flat)
    hl_states = hjepa.hle(post_emb).view(BATCH, history + 1, -1)         # pre-projector HL
    hl_proj = hjepa.hl_projector(hl_states.reshape(-1, hl_dim)).view(
        BATCH, history + 1, -1)                                          # loss-space HL

    # macros per slot
    macros = []
    for h in range(history):
        slot = raw_actions[:, h * K_test : (h + 1) * K_test]              # (B, K, 10)
        macros.append(hjepa.encode_macro(slot))
    macros = torch.stack(macros, dim=1)                                  # (B, history, hl_dim)

    states_hist = hl_proj[:, :history]
    s_target = hl_proj[:, -1]

    # ----- (2) HL state stats -----
    print("\n=== (2) HL state stats (loss space = post hl_projector) ===")
    stats(hl_proj[:, 0], "hl_proj[ t-2K ]")
    stats(hl_proj[:, history // 2], "hl_proj[ mid    ]")
    stats(hl_proj[:, -1], "hl_proj[ target ]")
    print(f"  rankme(hl_proj at t-2K) = {rank_me(hl_proj[:, 0]):.2f} / hl_dim={hl_dim}")
    print(f"  rankme(hl_states pre-proj at t-2K) = {rank_me(hl_states[:, 0]):.2f} / hl_dim={hl_dim}")

    # ----- (3) MAE statistics -----
    print("\n=== (3) MAE output statistics on real action windows (one slot) ===")
    slot = raw_actions[:, :K_test]
    m_real = hjepa.encode_macro(slot)
    m_zero = hjepa.encode_macro(torch.zeros_like(slot))
    m_rand = hjepa.encode_macro(torch.randn_like(slot) * slot.std())
    stats(m_real, "MAE(real)")
    print(f"  rankme(MAE on real) = {rank_me(m_real):.2f} / hl_dim={hl_dim}")
    print(f"  ||MAE(real)-MAE(zero)||/||MAE(real)|| = "
          f"{((m_real - m_zero).norm(dim=-1) / m_real.norm(dim=-1).clamp_min(1e-6)).mean().item():.3f}")
    print(f"  ||MAE(real)-MAE(rand)||/||MAE(real)|| = "
          f"{((m_real - m_rand).norm(dim=-1) / m_real.norm(dim=-1).clamp_min(1e-6)).mean().item():.3f}")
    print(f"  -> >0.3 = MAE has some action sensitivity")

    # ----- (4) HLP sensitivity -----
    print("\n=== (4) HLP sensitivity ===")
    pred_real = hjepa.predict_hl(states_hist, macros)
    # vary only macros (states fixed to a single sample tiled)
    s_fixed = states_hist[:1].expand_as(states_hist)
    pred_varyM = hjepa.predict_hl(s_fixed, macros)
    # vary only states (macros fixed)
    m_fixed = macros[:1].expand_as(macros)
    pred_varyS = hjepa.predict_hl(states_hist, m_fixed)
    var_real = pred_real.var(dim=0).mean().item()
    var_varyS = pred_varyS.var(dim=0).mean().item()
    var_varyM = pred_varyM.var(dim=0).mean().item()
    print(f"  mean per-dim var(pred) — vary both : {var_real:.5f}")
    print(f"  mean per-dim var(pred) — vary only s: {var_varyS:.5f}  (high = pred uses state)")
    print(f"  mean per-dim var(pred) — vary only m: {var_varyM:.5f}  (high = pred uses macro)")
    # linear fit pred ≈ A * flatten(states) + b
    X = states_hist.reshape(BATCH, -1)
    X = torch.cat([X, torch.ones(BATCH, 1, device=device)], dim=-1)
    sol = torch.linalg.lstsq(X, pred_real).solution
    resid = pred_real - X @ sol
    frac_linear_in_s = 1.0 - (resid.var(dim=0).mean() / pred_real.var(dim=0).mean().clamp_min(1e-8)).item()
    print(f"  fraction of pred variance linear in states: {frac_linear_in_s:.3f}  "
          f"(1.0 = HLP is affine in s)")

    # ----- (5) Prediction error vs baselines -----
    print("\n=== (5) Prediction error vs baselines (lower is better) ===")
    perm = torch.randperm(macros.size(0), device=device)
    macros_shuf = macros[perm]
    pred_shuf = hjepa.predict_hl(states_hist, macros_shuf)
    pred_zero = hjepa.predict_hl(states_hist, torch.zeros_like(macros))
    # history-collapsed: repeat last state in all 3 slots
    s_last_repeat = states_hist[:, -1:].expand_as(states_hist)
    pred_no_hist = hjepa.predict_hl(s_last_repeat, macros)
    err = lambda p: (p - s_target).pow(2).sum(dim=-1).sqrt().mean().item()
    err_real = err(pred_real)
    err_shuf = err(pred_shuf)
    err_zero = err(pred_zero)
    err_nohist = err(pred_no_hist)
    err_identity = (states_hist[:, -1] - s_target).pow(2).sum(dim=-1).sqrt().mean().item()
    err_zero_baseline = s_target.pow(2).sum(dim=-1).sqrt().mean().item()
    print(f"  HLP(real states, real macros)     : {err_real:.4f}")
    print(f"  HLP(real states, shuffled macros) : {err_shuf:.4f}   "
          f"(if ≈ real, macro is ignored)")
    print(f"  HLP(real states, zero macros)     : {err_zero:.4f}")
    print(f"  HLP(s_curr repeated 3x, real m)   : {err_nohist:.4f}   "
          f"(if ≈ real, history is ignored)")
    print(f"  identity (use s_curr as pred)     : {err_identity:.4f}")
    print(f"  zero-vector baseline              : {err_zero_baseline:.4f}")
    print(f"  -> Healthy: real < shuffled AND real < identity")

    # ----- (6) Time collapse -----
    print("\n=== (6) Time collapse check ===")
    within = (s_target - states_hist[:, -1]).norm(dim=-1)
    across = (states_hist[:, -1] - states_hist[perm, -1]).norm(dim=-1)
    cos = torch.nn.functional.cosine_similarity(states_hist[:, -1], s_target, dim=-1)
    print(f"  ||s_curr_proj - s_target_proj||  within-traj: {within.mean().item():.3f}")
    print(f"  ||s_curr_proj - s_curr_proj'||   across-traj: {across.mean().item():.3f}")
    print(f"  ratio within/across = {within.mean().item()/across.mean().item():.3f}  "
          f"(low → time-collapsed encoder)")
    print(f"  cos(s_curr_proj, s_target_proj) = {cos.mean().item():.3f}")

    # ----- (7) Spectrum -----
    print("\n=== (7) HL projected state explained variance (top 10 PCs) ===")
    z = hl_proj[:, 0] - hl_proj[:, 0].mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    evr = (s ** 2) / (s ** 2).sum()
    for i in range(10):
        print(f"  PC{i+1:>2d}: {evr[i].item():.4f}")
    print(f"  cumsum[1..10] = {evr[:10].sum().item():.4f}  "
          f"cumsum[1..30] = {evr[:30].sum().item():.4f}")


if __name__ == "__main__":
    run()
