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
from module import HLEncoder, HLPredictor, MacroActionEncoder, SIGReg
from utils import get_column_normalizer, ModelObjectCallBack, IntervalRankMe, PCALatentViz


def hjepa_forward(self, batch, stage, cfg):
    """HL prediction loss + SIGReg on HL embeddings.  LL is frozen.

    Reads the pre-encoded LL CLS from `batch['emb']` (pre-projector) and
    applies the frozen LL projector once to obtain post-projector embeddings,
    matching the representation the LL predictor was trained on.
    """
    K = cfg.wm.k
    lambd = cfg.loss.sigreg.weight

    pre_emb = batch["emb"]                            # (B, K+1, ll_dim) pre-projector
    actions = torch.nan_to_num(batch["action"][:, :K], 0.0)  # (B, K, A)

    B, T = pre_emb.shape[:2]
    flat = rearrange(pre_emb, "b t d -> (b t) d")
    with torch.no_grad():
        post_emb = self.model.ll.projector(flat)      # frozen LL projector
    hl_states = self.model.encode_hl(post_emb).view(B, T, -1)  # (B, K+1, hl_dim)

    macro = self.model.encode_macro(actions)          # (B, macro_a_dim)
    s_pred = self.model.predict_hl(hl_states[:, 0], macro)
    s_target = hl_states[:, -1]                       # match LeWM (no stop-grad)

    pred_loss = (s_pred - s_target).pow(2).mean()
    sig_loss = self.sigreg(hl_states.transpose(0, 1))
    loss = pred_loss + lambd * sig_loss

    out = {"emb": hl_states, "pred_loss": pred_loss,
           "sigreg_loss": sig_loss, "loss": loss,
           "hl_emb": hl_states.reshape(-1, hl_states.shape[-1]).detach()}  # for queue/RankMe/PCA
    self.log_dict({f"{stage}/{k}": v.detach() for k, v in out.items() if "loss" in k},
                  on_step=True, sync_dist=True)
    return out


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_hl")
def run(cfg):
    # ------------ data ------------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = []  # no pixel preprocessor: we read pre-encoded `emb` directly
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or col == "emb":
                continue
            transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms) if transforms else None

    rnd = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True,
                                        drop_last=True, generator=rnd)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    # ------------ frozen LL ------------
    ll_jepa = torch.load(cfg.ll_ckpt, map_location="cpu", weights_only=False)
    ll_jepa.eval()
    for p in ll_jepa.parameters():
        p.requires_grad_(False)

    # ------------ HL modules ------------
    hle = HLEncoder(ll_dim=cfg.wm.ll_dim, hl_dim=cfg.wm.hl_dim, **cfg.hle)
    hlp = HLPredictor(hl_dim=cfg.wm.hl_dim, macro_action_dim=cfg.wm.macro_action_dim, **cfg.hlp)
    mae = MacroActionEncoder(action_dim=cfg.wm.action_dim, num_actions=cfg.wm.k,
                             macro_action_dim=cfg.wm.macro_action_dim, **cfg.mae)
    hjepa = HierarchicalJEPA(ll_jepa, hle, hlp, mae, k=cfg.wm.k)

    optimizers = {
        "model_opt": {
            "modules": r"model\.(hle|hlp|mae)",  # spt expects a regex; LL is frozen anyway
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=hjepa,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
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
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, num_sanity_val_steps=1,
                         logger=logger, enable_checkpointing=True)
    manager = spt.Manager(trainer=trainer, module=module, data=data_module,
                          ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt")
    manager()


if __name__ == "__main__":
    run()
