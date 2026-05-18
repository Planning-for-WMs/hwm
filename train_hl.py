"""Train hierarchical JEPA on top of a frozen LeWM checkpoint."""
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from hjepa import HierarchicalJEPA
from module import (HLEncoder, HLPredictor,
                    MacroActionEncoder, MLPHead, SIGReg)
from utils import (get_column_normalizer, ModelObjectCallBack, IntervalRankMe,
                   PCALatentViz, LatentViz3D)


def hjepa_forward(self, batch, stage, cfg):
    """HL prediction loss + SIGReg on HL embeddings.  LL is frozen.

    Reads the pre-encoded LL CLS from `batch['emb']` (pre-projector) and
    applies the frozen LL projector once to obtain post-projector embeddings,
    matching the representation the LL predictor was trained on.
    """
    history = cfg.wm.history
    lambd = cfg.loss.sigreg.weight

    pre_emb = batch["emb"]                            # (B, history*K+1, ll_dim) pre-projector
    actions_all = torch.nan_to_num(batch["action"], 0.0)

    # K is now fixed by the sub-dataset this batch was drawn from
    # (HomogeneousKBatchSampler guarantees uniform window size per batch).
    # Recover K from the window: T = history*K + 1.
    T = pre_emb.size(1)
    K = (T - 1) // history
    assert history * K + 1 == T, (
        f"window size {T} not consistent with history={history} * K + 1; "
        f"batches must be homogeneous in K")

    # encode all anchor LL frames in one pass: indices [0, K, 2K, ..., history*K]
    idx = torch.arange(0, history + 1, device=pre_emb.device) * K
    anchor_emb = pre_emb[:, idx]                      # (B, history+1, ll_dim) pre-proj
    B = anchor_emb.size(0)
    flat = rearrange(anchor_emb, "b t d -> (b t) d")
    with torch.no_grad():
        post_emb = self.model.ll.projector(flat)
    hl_states = self.model.encode_hl(post_emb)        # (B*(history+1), hl_dim)
    hl_states_proj = self.model.hl_projector(hl_states).view(B, history + 1, -1)

    # macros: K actions per slot, history slots
    macros = []
    for h in range(history):
        slot = actions_all[:, h * K : (h + 1) * K]    # (B, K, A_raw)
        macros.append(self.model.encode_macro(slot))
    macros = torch.stack(macros, dim=1)               # (B, history, hl_dim)

    states_history = hl_states_proj[:, :history]      # (B, history, hl_dim)
    s_target = hl_states_proj[:, -1]                  # (B, hl_dim)
    s_pred = self.model.predict_hl(states_history, macros)

    pred_loss = (s_pred - s_target).pow(2).mean()
    sig_loss = self.sigreg(hl_states_proj.transpose(0, 1))
    # SIGReg on MAE outputs — keeps macros Gaussian-marginal so all
    # macro_action_dim dims are used and CEM sampling stays well-calibrated.
    lambd_mae = cfg.loss.sigreg_mae.weight
    mae_sig_loss = self.sigreg_mae(macros.transpose(0, 1))
    loss = pred_loss + lambd * sig_loss + lambd_mae * mae_sig_loss

    out = {"emb": hl_states_proj, "pred_loss": pred_loss,
           "sigreg_loss": sig_loss, "sigreg_mae_loss": mae_sig_loss,
           "loss": loss, "k_sample": float(K),
           # flattened queues for RankMe / latent-viz callbacks
           "hl_emb": hl_states_proj.reshape(-1, hl_states_proj.shape[-1]).detach(),
           # macros: (B, history, macro_action_dim) -> flat per-slot samples;
           # since the dataset is expert trajectories these come from the
           # expert action distribution.
           "mae_emb": macros.reshape(-1, macros.shape[-1]).detach()}

    # ---- Macro-use diagnostic (per epoch on val) ----
    # On val batches, predict the same target with (a) real macros, (b) zero
    # macros, (c) shuffled macros across the batch. The gaps tell us whether
    # HLP routes any signal through the macro pathway.
    #   macro_use_gap        = err_zero - err_real   (positive  => macro helps in absolute terms)
    #   macro_specificity_gap = err_shuf - err_real  (positive  => the right macro matters, not just any non-zero)
    if stage in ("val", "validate"):
        with torch.no_grad():
            zero_macros = torch.zeros_like(macros)
            perm = torch.randperm(macros.size(0), device=macros.device)
            shuf_macros = macros[perm]
            s_pred_zero = self.model.predict_hl(states_history, zero_macros)
            s_pred_shuf = self.model.predict_hl(states_history, shuf_macros)
            err_real = (s_pred       - s_target).pow(2).mean()
            err_zero = (s_pred_zero  - s_target).pow(2).mean()
            err_shuf = (s_pred_shuf  - s_target).pow(2).mean()
        diag = {
            "val/macro_err_real": err_real,
            "val/macro_err_zero": err_zero,
            "val/macro_err_shuf": err_shuf,
            "val/macro_use_gap": (err_zero - err_real),
            "val/macro_specificity_gap": (err_shuf - err_real),
        }
        self.log_dict({k: v.detach() for k, v in diag.items()},
                      on_step=False, on_epoch=True, sync_dist=True)

    self.log_dict({f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k},
                  on_step=True, sync_dist=True)
    return out


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_hl")
def run(cfg):
    # ------------ data ------------
    # Build one HDF5Dataset per K in [k_min, k_max]. Each has its own
    # num_steps = history*K + 1, so it admits **every** trajectory long
    # enough to fit that K. The full training set = ConcatDataset of all
    # sub-datasets. Batches are emitted homogeneous-in-K by the
    # `HomogeneousKBatchSampler` so tensor shapes stay uniform per step.
    from torch.utils.data import BatchSampler, ConcatDataset, DataLoader

    history = cfg.wm.history
    k_min, k_max = cfg.wm.k_min, cfg.wm.k_max
    base_ds_cfg = OmegaConf.to_container(cfg.data.dataset)

    # Fit normalizers on the smallest-K dataset (has the most clips).
    with open_dict(cfg):
        norm_cfg = dict(base_ds_cfg); norm_cfg["num_steps"] = history * k_min + 1
        norm_ds = swm.data.HDF5Dataset(**norm_cfg, transform=None)
        transforms = []
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "emb":
                continue
            transforms.append(get_column_normalizer(norm_ds, col, col))
    shared_transform = (spt.data.transforms.Compose(*transforms)
                        if transforms else None)

    rnd = torch.Generator().manual_seed(cfg.seed)
    train_subs, val_subs = [], []
    train_lens, val_lens = [], []
    print("[adaptive-K] sub-datasets:")
    for k in range(k_min, k_max + 1):
        ds_cfg = dict(base_ds_cfg); ds_cfg["num_steps"] = history * k + 1
        ds = swm.data.HDF5Dataset(**ds_cfg, transform=shared_transform)
        n = len(ds)
        t, v = spt.data.random_split(
            ds, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd)
        train_subs.append(t); val_subs.append(v)
        train_lens.append(len(t)); val_lens.append(len(v))
        print(f"  K={k}  num_steps={ds_cfg['num_steps']:>3d}  span={ds_cfg['num_steps']*ds_cfg['frameskip']:>3d} raw  "
              f"clips total={n:,}  train={len(t):,}  val={len(v):,}")
    train_set = ConcatDataset(train_subs)
    val_set = ConcatDataset(val_subs)

    def cumoffsets(lens):
        out = []; acc = 0
        for n in lens:
            out.append(acc); acc += n
        return out
    train_offsets = cumoffsets(train_lens)
    val_offsets = cumoffsets(val_lens)

    class HomogeneousKBatchSampler(BatchSampler):
        """Yields batches where every sample lives in the same sub-dataset,
        so the loaded tensors share shape (history*K+1, ...) for each batch."""
        def __init__(self, sub_lens, sub_offs, batch_size, shuffle, generator,
                     drop_last=True):
            self.sub_lens = sub_lens
            self.sub_offs = sub_offs
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.generator = generator
            self.drop_last = drop_last
        def __iter__(self):
            all_batches = []
            for offset, n in zip(self.sub_offs, self.sub_lens):
                perm = (torch.randperm(n, generator=self.generator).tolist()
                        if self.shuffle else list(range(n)))
                for i in range(0, n, self.batch_size):
                    batch = [perm[j] + offset
                             for j in range(i, min(i + self.batch_size, n))]
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    all_batches.append(batch)
            if self.shuffle:
                ord_ = torch.randperm(len(all_batches),
                                      generator=self.generator).tolist()
                all_batches = [all_batches[i] for i in ord_]
            return iter(all_batches)
        def __len__(self):
            return sum(n // self.batch_size if self.drop_last
                       else (n + self.batch_size - 1) // self.batch_size
                       for n in self.sub_lens)

    loader_kwargs = dict(cfg.loader)
    batch_size = loader_kwargs.pop("batch_size")
    train_sampler = HomogeneousKBatchSampler(
        train_lens, train_offsets, batch_size=batch_size,
        shuffle=True, generator=rnd, drop_last=True)
    val_sampler = HomogeneousKBatchSampler(
        val_lens, val_offsets, batch_size=batch_size,
        shuffle=False, generator=rnd, drop_last=False)
    train = DataLoader(train_set, batch_sampler=train_sampler, **loader_kwargs)
    val = DataLoader(val_set, batch_sampler=val_sampler, **loader_kwargs)

    # ------------ frozen LL ------------
    ll_jepa = torch.load(cfg.ll_ckpt, map_location="cpu", weights_only=False)
    ll_jepa.eval()
    for p in ll_jepa.parameters():
        p.requires_grad_(False)

    # ------------ HL modules ------------
    hle = HLEncoder(ll_dim=cfg.wm.ll_dim, hl_dim=cfg.wm.hl_dim, **cfg.hle)
    hl_projector = MLPHead(dim=cfg.wm.hl_dim, hidden_dim=cfg.hl_projector.hidden_dim)
    hl_pred_proj = MLPHead(dim=cfg.wm.hl_dim, hidden_dim=cfg.hl_pred_proj.hidden_dim)
    hlp = HLPredictor(hl_dim=cfg.wm.hl_dim, macro_action_dim=cfg.wm.macro_action_dim,
                      history=cfg.wm.history, **cfg.hlp)
    mae = MacroActionEncoder(action_dim=cfg.wm.action_dim,
                             macro_action_dim=cfg.wm.macro_action_dim,
                             max_k=cfg.wm.k_max, **cfg.mae)
    hjepa = HierarchicalJEPA(ll_jepa, hle, hl_projector, hl_pred_proj, hlp, mae,
                              k=cfg.wm.k_max)

    optimizers = {
        "model_opt": {
            "modules": r"model\.(hle|hl_projector|hl_pred_proj|hlp|mae)",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=hjepa,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        sigreg_mae=SIGReg(**cfg.loss.sigreg_mae.kwargs),
        forward=partial(hjepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ------------ trainer ------------
    run_dir = Path(swm.data.utils.get_cache_dir(), cfg.get("subdir") or "")
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    object_dump = ModelObjectCallBack(dirpath=run_dir, filename=cfg.output_model_name,
                                      epoch_interval=1)
    callbacks = [object_dump]
    if cfg.get("log") is not None:
        callbacks.append(IntervalRankMe(
            name="hl_rankme", target="hl_emb",
            queue_length=cfg.log.rankme_queue, target_shape=cfg.wm.hl_dim,
            every_n_epochs=cfg.log.every_n_epochs,
        ))
        callbacks.append(PCALatentViz(
            name="hl_pca", target="hl_emb",
            queue_length=cfg.log.pca_queue, target_shape=cfg.wm.hl_dim,
            every_n_epochs=cfg.log.every_n_epochs,
        ))
        # MAE (macro) RankMe + 3D scatter — actions come from expert trajectories
        # (the training dataset is pusht_expert_train), so the queued macros are
        # MAE outputs on expert action chunks.
        callbacks.append(IntervalRankMe(
            name="mae_rankme", target="mae_emb",
            queue_length=cfg.log.rankme_queue,
            target_shape=cfg.wm.macro_action_dim,
            every_n_epochs=cfg.log.every_n_epochs,
        ))
        callbacks.append(PCALatentViz(
            name="mae_pca", target="mae_emb",
            queue_length=cfg.log.pca_queue,
            target_shape=cfg.wm.macro_action_dim,
            every_n_epochs=cfg.log.every_n_epochs,
        ))
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1,
                         logger=logger, enable_checkpointing=True)
    manager = spt.Manager(trainer=trainer, module=module, data=data_module,
                          ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt")
    manager()


if __name__ == "__main__":
    run()
