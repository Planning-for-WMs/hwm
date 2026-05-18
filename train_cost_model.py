"""train_cost_model.py — supervised cost head: given (HL_target, LL_current),
predict ||LL_target − LL_current||²₂.

  - HL_target  is a projected HL state (post hl_projector), at trajectory index i.
  - LL_current is a post-projector LL embedding, at trajectory index j (j ≠ i,
    both from the SAME episode).
  - Target     is the squared L2 distance in LL space between the LL state at
    index i (the "true" LL underlying HL_target) and the LL state at index j.

At eval time the cost head can be queried as a fine-grained "how far is this
LL state from achieving this HL subgoal." It supplements / replaces the raw
squared HL distance that loses orientation info on PushT.

Build:
  - Pre-encode every row of pusht_expert_train_emb.h5 through the frozen LL
    projector and the frozen HLE+hl_projector pipeline. Store in RAM.
  - Sample `pairs_per_episode` random (i, j) pairs per trajectory; i and j must
    differ.
  - MLP cost head ≈ 100k params: Linear(hl+ll → hidden) → [LN GELU Linear]×(depth-2)
      → Linear(hidden → 1).

Run:
  python train_cost_model.py                            # uses config/train/cost_head.yaml
  python train_cost_model.py trainer.epochs=50         # override via Hydra CLI
"""
import os
import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


class CostHead(nn.Module):
    """MLP that maps (HL_target, LL_current) -> scalar cost."""

    def __init__(self, hl_dim: int = 96, ll_dim: int = 192,
                 hidden_dim: int = 192, depth: int = 3):
        super().__init__()
        assert depth >= 2
        in_dim = hl_dim + ll_dim
        layers = [nn.Linear(in_dim, hidden_dim),
                  nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim), nn.GELU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, hl_target: torch.Tensor, ll_current: torch.Tensor) -> torch.Tensor:
        x = torch.cat([hl_target, ll_current], dim=-1)
        return self.net(x).squeeze(-1)


def load_encoded(hl_emb_h5: str):
    """Read the pre-encoded H5 produced by preprocess_hl.py.
    Returns (LL_post, HL_post, ep_ids) as fp32 CPU tensors / numpy."""
    with h5py.File(hl_emb_h5, "r") as f:
        if "ll_post" not in f or "hl_post" not in f:
            raise KeyError(
                f"{hl_emb_h5} missing 'll_post' or 'hl_post' — run "
                f"preprocess_hl.py first to materialize the HL embeddings.")
        LL = torch.from_numpy(f["ll_post"][:].astype(np.float32, copy=False))
        HL = torch.from_numpy(f["hl_post"][:].astype(np.float32, copy=False))
        ep_col = "episode_idx" if "episode_idx" in f else "ep_idx"
        ep_ids = f[ep_col][:]
    return LL, HL, ep_ids


class CostPairDataset(Dataset):
    """Yields (HL_target, LL_current, target_l2_squared) triples.

    mode="train": resamples (i, j) at every __getitem__ call from the assigned
        episode subset, so each epoch sees fresh pairs. `__len__` returns
        `pairs_per_episode * #episodes` purely to control "samples per epoch."
    mode="val":   fixes pairs at construction time for stable comparison
        across epochs."""

    def __init__(self, LL: torch.Tensor, HL: torch.Tensor,
                 ep_ids: np.ndarray, episode_subset: np.ndarray,
                 pairs_per_episode: int = 30, mode: str = "train",
                 seed: int = 0):
        self.LL = LL
        self.HL = HL
        self.mode = mode
        # group rows by episode (keep only episodes with >=2 rows AND in subset)
        rng = np.random.default_rng(seed)
        idxs_by_ep = []
        for ep in tqdm(episode_subset, desc=f"build {mode} pairs"):
            idxs = np.nonzero(ep_ids == ep)[0]
            if len(idxs) >= 2:
                idxs_by_ep.append(idxs)
        self.idxs_by_ep = idxs_by_ep                          # list of arrays
        self.pairs_per_episode = pairs_per_episode

        if mode == "val":
            pairs = []
            for idxs in idxs_by_ep:
                for _ in range(pairs_per_episode):
                    pair = rng.choice(idxs, size=2, replace=False)
                    pairs.append(pair)
            self.pairs = np.asarray(pairs, dtype=np.int64)
            self._length = len(self.pairs)
            print(f"[CostPairDataset val] {self._length:,} FIXED pairs across "
                  f"{len(idxs_by_ep):,} episodes")
        elif mode == "train":
            self.pairs = None
            self._length = pairs_per_episode * len(idxs_by_ep)
            print(f"[CostPairDataset train] {self._length:,} samples/epoch "
                  f"(resampled each call) across {len(idxs_by_ep):,} episodes")
        else:
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")

    def __len__(self):
        return self._length

    def __getitem__(self, k):
        if self.pairs is not None:                            # val
            i, j = self.pairs[k]
        else:                                                  # train: resample
            ep_idx = np.random.randint(len(self.idxs_by_ep))
            idxs = self.idxs_by_ep[ep_idx]
            choice = np.random.choice(idxs, size=2, replace=False)
            i, j = int(choice[0]), int(choice[1])
        hl_target = self.HL[i]                                # (hl_dim,)
        ll_current = self.LL[j]                               # (ll_dim,)
        target = ((self.LL[i] - self.LL[j]) ** 2).sum()       # scalar
        return hl_target, ll_current, target


def _resolve_hjepa_ckpt(path_or_dir):
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        cands = list(p.glob("*_epoch_*_object.ckpt"))
        if not cands:
            raise FileNotFoundError(f"no *_epoch_*_object.ckpt under {p}")
        return max(cands, key=lambda x: int(x.stem.split("_epoch_")[1].split("_")[0]))
    raise FileNotFoundError(f"{p} not found")


@hydra.main(version_base=None, config_path="./config/train", config_name="cost_head")
def run(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_cost] writing to {out_dir}")
    with open(out_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # ---- wandb (optional) ----
    wb = None
    if cfg.wandb.enabled:
        import wandb as wb
        wb.init(**cfg.wandb.config, config=OmegaConf.to_container(cfg, resolve=True))

    # ---- load pre-encoded (LL_post, HL_post, ep_ids) from H5 ----
    print(f"loading pre-encoded HL emb h5 ← {cfg.hl_emb_h5}")
    LL, HL, ep_ids = load_encoded(cfg.hl_emb_h5)
    print(f"  LL_post shape = {tuple(LL.shape)}  HL_post shape = {tuple(HL.shape)}  "
          f"episodes = {len(np.unique(ep_ids)):,}")
    print(f"  RAM used ≈ {(LL.numel() + HL.numel()) * 4 / 1e9:.2f} GB")

    # ---- per-episode train/val split (no leakage between train and val) ----
    unique_eps = np.unique(ep_ids)
    rng = np.random.default_rng(int(cfg.seed))
    perm = rng.permutation(len(unique_eps))
    n_val_eps = int(float(cfg.val_frac) * len(unique_eps))
    val_eps = unique_eps[perm[:n_val_eps]]
    train_eps = unique_eps[perm[n_val_eps:]]
    print(f"  episodes: train = {len(train_eps):,}  val = {len(val_eps):,}")

    train_set = CostPairDataset(
        LL, HL, ep_ids, episode_subset=train_eps,
        pairs_per_episode=int(cfg.pairs_per_episode),
        mode="train", seed=int(cfg.seed))
    val_set = CostPairDataset(
        LL, HL, ep_ids, episode_subset=val_eps,
        pairs_per_episode=int(cfg.pairs_per_episode),
        mode="val", seed=int(cfg.seed) + 1)

    nw = int(cfg.trainer.num_workers)
    pm = bool(cfg.trainer.pin_memory)
    bs = int(cfg.loader.batch_size)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm,
                              persistent_workers=nw > 0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pm,
                            persistent_workers=nw > 0)

    # ---- model + optimizer ----
    hl_dim = HL.shape[1]; ll_dim = LL.shape[1]
    model = CostHead(hl_dim=hl_dim, ll_dim=ll_dim,
                     hidden_dim=int(cfg.model.hidden_dim),
                     depth=int(cfg.model.depth)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cost head] {n_params:,} params  (hl_dim={hl_dim}, ll_dim={ll_dim}, "
          f"hidden={cfg.model.hidden_dim}, depth={cfg.model.depth})")

    opt_kw = OmegaConf.to_container(cfg.optimizer, resolve=True)
    opt_type = opt_kw.pop("type", "AdamW")
    opt = getattr(torch.optim, opt_type)(model.parameters(), **opt_kw)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.trainer.epochs))

    # baseline: mean target on the val pairs (sanity for the MSE scale)
    val_pairs = val_set.pairs[:5000]
    mean_target = float(((LL[val_pairs[:, 0]] - LL[val_pairs[:, 1]]) ** 2)
                        .sum(dim=-1).mean().item())
    print(f"  mean target on first 5000 val pairs ≈ {mean_target:.3f}")

    log_path = out_dir / "train.log"
    with log_path.open("w") as logf:
        logf.write(f"# {n_params} params  hl={hl_dim} ll={ll_dim} "
                   f"hidden={cfg.model.hidden_dim} depth={cfg.model.depth}\n")
        logf.write(f"# mean target ≈ {mean_target:.3f}\n")
        logf.write("epoch,train_mse,val_mse,lr,wall_s\n")

        for ep in range(int(cfg.trainer.epochs)):
            model.train()
            t0 = time.time(); tot, nb = 0.0, 0
            for hl, ll, tgt in tqdm(train_loader, desc=f"epoch {ep+1}/{cfg.trainer.epochs}",
                                    leave=False):
                hl = hl.to(device, non_blocking=True).float()
                ll = ll.to(device, non_blocking=True).float()
                tgt = tgt.to(device, non_blocking=True).float()
                pred = model(hl, ll)
                loss = nn.functional.mse_loss(pred, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.item()); nb += 1
            train_mse = tot / max(nb, 1)
            sched.step()

            model.eval()
            vtot, vnb = 0.0, 0
            with torch.no_grad():
                for hl, ll, tgt in val_loader:
                    hl = hl.to(device, non_blocking=True).float()
                    ll = ll.to(device, non_blocking=True).float()
                    tgt = tgt.to(device, non_blocking=True).float()
                    pred = model(hl, ll)
                    vtot += float(nn.functional.mse_loss(pred, tgt).item()); vnb += 1
            val_mse = vtot / max(vnb, 1)
            lr = sched.get_last_lr()[0]
            wall = time.time() - t0
            print(f"ep {ep+1:>3d}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}  "
                  f"lr={lr:.2e}  {wall:.1f}s")
            logf.write(f"{ep+1},{train_mse:.4f},{val_mse:.4f},{lr:.2e},{wall:.1f}\n")
            logf.flush()
            if wb is not None:
                wb.log({"train/mse": train_mse, "val/mse": val_mse,
                        "lr": lr, "epoch": ep + 1})

            ck = {"model": model.state_dict(),
                  "config": OmegaConf.to_container(cfg, resolve=True),
                  "hl_dim": hl_dim, "ll_dim": ll_dim,
                  "hidden_dim": int(cfg.model.hidden_dim),
                  "depth": int(cfg.model.depth)}
            torch.save(ck, out_dir / f"{cfg.output_model_name}_epoch_{ep+1}.ckpt")
        torch.save(ck, out_dir / f"{cfg.output_model_name}.ckpt")
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    run()
