"""Train an HL -> pixels decoder by composing a new HL->LL decoder with the
frozen pixel decoder (decoder_v2) trained by train_decoder.py.

Pipeline (per sample):
    pre_emb (192, from emb.h5)
        --(frozen hjepa.ll.projector)--> post_emb (192)
        --(frozen hjepa.encode_hl)----->  HL state (96)
        --(NEW HLDecoder, MLP)--------->  predicted pre_emb (192)
        --(frozen decoder_v2)---------->  predicted pixels (3, 224, 224)

Loss: lpips_weight * LPIPS(pred_pixels, target_pixels) + mse_weight * MSE(...)
Only the HLDecoder receives gradients; everything else is frozen but still
participates in the autograd graph so gradients can flow back.

Run (single GPU):
    python train_hl_decoder.py
Run (2 GPUs):
    torchrun --nproc_per_node=2 train_hl_decoder.py
"""
import os
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 — registers blosc filter for h5py
import h5py
import hydra
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_decoder import (
    ConvDecoder,
    ChunkAwarePixelDataset,
    EmbPixelDataset,
)


class HLDecoder(nn.Module):
    """96-d HL state -> 192-d pre-projector LL CLS (input space of decoder_v2)."""

    def __init__(self, hl_dim=96, ll_dim=192, hidden_dim=512, depth=4):
        super().__init__()
        assert depth >= 2
        layers = [nn.Linear(hl_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim),
                       nn.GELU()]
        layers += [nn.Linear(hidden_dim, ll_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _freeze(mod):
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod.eval()


@hydra.main(version_base=None, config_path="./config/train", config_name="hl_decoder")
def run(cfg):
    # ---------- DDP init ----------
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[local_rank {local_rank}] init_process_group...", flush=True)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[rank {rank}/{world_size}] process group ready", flush=True)
    else:
        rank, world_size = 0, 1
        device = torch.device(cfg.device)
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)
    out_dir = Path(cfg.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    # ---------- wandb (rank 0 only) ----------
    wb = None
    if cfg.wandb.enabled and is_main:
        import wandb as wb
        wb.init(**cfg.wandb.config, config=OmegaConf.to_container(cfg, resolve=True))

    # ---------- data ----------
    with h5py.File(cfg.emb_h5, "r") as f:
        n_total = f["emb"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    n_train = int(cfg.train_split * n_total)
    train_idx_all, val_idx = perm[:n_train], perm[n_train:]
    train_idx = train_idx_all[rank::world_size]

    train_set = ChunkAwarePixelDataset(cfg.emb_h5, cfg.pix_h5, train_idx,
                                       chunk_size=100, seed=cfg.seed + rank)
    val_set = EmbPixelDataset(cfg.emb_h5, cfg.pix_h5, indices=val_idx)
    loader_kw = dict(cfg.loader)
    train_loader = DataLoader(train_set, shuffle=False, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kw)

    # ---------- frozen components ----------
    compute_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else torch.float32

    # HierarchicalJEPA: pickled whole-module checkpoint.
    # If a directory is given, pick the highest-epoch hjepa_*_epoch_N_object.ckpt.
    hjepa_path = Path(cfg.hjepa_ckpt)
    if hjepa_path.is_dir():
        cands = list(hjepa_path.glob("*_epoch_*_object.ckpt"))
        if not cands:
            raise FileNotFoundError(f"no *_epoch_*_object.ckpt under {hjepa_path}")
        hjepa_path = max(cands, key=lambda p: int(p.stem.split("_epoch_")[1].split("_")[0]))
        if is_main:
            print(f"hjepa_ckpt resolved to {hjepa_path}", flush=True)
    hjepa = torch.load(hjepa_path, map_location=device, weights_only=False)
    hjepa.to(device=device, dtype=compute_dtype)
    _freeze(hjepa)

    # Pixel decoder: rebuild ConvDecoder and load state_dict
    pixel_decoder = ConvDecoder(**cfg.pixel_decoder).to(device=device, dtype=compute_dtype)
    pixel_decoder = pixel_decoder.to(memory_format=torch.channels_last)
    sd = torch.load(cfg.pixel_decoder_weights, map_location=device)
    # accept either a raw state_dict or a checkpoint dict that contains one
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    pixel_decoder.load_state_dict(sd, strict=True)
    _freeze(pixel_decoder)

    # ---------- trainable HL decoder ----------
    hl_decoder = HLDecoder(**cfg.model).to(device=device, dtype=compute_dtype)

    # LPIPS (frozen)
    lpips_fn = lpips.LPIPS(net=cfg.loss.lpips_net).to(device=device, dtype=compute_dtype).eval()
    _freeze(lpips_fn)

    # warm-start the HL decoder if requested
    if cfg.get("resume_from", None):
        sd = torch.load(cfg.resume_from, map_location=device)
        hl_decoder.load_state_dict(sd, strict=True)
        if is_main:
            print(f"loaded HLDecoder weights from {cfg.resume_from}", flush=True)

    # DDP wrap (only the trainable module needs it)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        hl_decoder = DDP(hl_decoder, device_ids=[local_rank],
                         gradient_as_bucket_view=True,
                         broadcast_buffers=False)

    compile_mode = cfg.get("compile", None)
    if compile_mode:
        if is_main:
            print(f"torch.compile mode={compile_mode}", flush=True)
        hl_decoder = torch.compile(hl_decoder, mode=compile_mode)
        pixel_decoder = torch.compile(pixel_decoder, mode=compile_mode)
        lpips_fn = torch.compile(lpips_fn, mode=compile_mode)

    opt = torch.optim.AdamW(hl_decoder.parameters(), **cfg.optimizer)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.trainer.epochs)

    if is_main:
        n_params = sum(p.numel() for p in hl_decoder.parameters())
        print(f"HLDecoder params: {n_params/1e6:.2f}M | "
              f"train(per-rank) {len(train_set)} val {len(val_set)} world_size={world_size}",
              flush=True)

    # ---------- pick a fixed trajectory for video logging ----------
    with h5py.File(cfg.emb_h5, "r") as f:
        ep_off = f["ep_offset"][:]; ep_len = f["ep_len"][:]
    rng = np.random.default_rng(cfg.seed)
    ep_idx = int(rng.integers(len(ep_len)))
    t_len = min(int(ep_len[ep_idx]), cfg.log.traj_max_len)
    t_start = int(ep_off[ep_idx])
    with h5py.File(cfg.emb_h5, "r") as f:
        traj_emb = torch.from_numpy(f["emb"][t_start:t_start + t_len].astype(np.float32))
    with h5py.File(cfg.pix_h5, "r") as f:
        traj_tgt = torch.from_numpy(f["pixels"][t_start:t_start + t_len])

    # ---------- helpers ----------
    hl_space = cfg.get("hl_space", "pre")
    if hl_space not in ("pre", "post"):
        raise ValueError(f"hl_space must be 'pre' or 'post', got {hl_space!r}")
    if is_main:
        print(f"[hl_space={hl_space}] HL state = "
              f"{'encode_hl_proj' if hl_space == 'post' else 'encode_hl'}(post_emb)",
              flush=True)

    def forward_chain(pre_emb, target_img):
        """Run the full HL -> pixels pipeline. Returns (pred_pixels, loss, mse, lpips)."""
        # frozen stages
        with torch.no_grad():
            post_emb = hjepa.ll.projector(pre_emb)
            if hl_space == "post":
                hl_state = hjepa.encode_hl_proj(post_emb)
            else:
                hl_state = hjepa.encode_hl(post_emb)
        # trainable: HL -> predicted pre-projector LL CLS
        pred_pre_emb = hl_decoder(hl_state)
        # frozen pixel decoder, gradients flow back through it to hl_decoder
        pred_pix = pixel_decoder(pred_pre_emb)
        mse = F.mse_loss(pred_pix, target_img)
        if cfg.loss.lpips_size and cfg.loss.lpips_size != pred_pix.shape[-1]:
            lp_p = F.interpolate(pred_pix, size=cfg.loss.lpips_size,
                                 mode="bilinear", align_corners=False)
            lp_t = F.interpolate(target_img, size=cfg.loss.lpips_size,
                                 mode="bilinear", align_corners=False)
        else:
            lp_p, lp_t = pred_pix, target_img
        lp = lpips_fn(lp_p, lp_t).mean()
        loss = cfg.loss.mse_weight * mse + cfg.loss.lpips_weight * lp
        return pred_pix, loss, mse, lp

    # ---------- training loop ----------
    if ddp:
        import torch.distributed as dist
        dist.barrier()
        if is_main:
            print("all ranks ready, starting epoch 1", flush=True)
    global_step = 0
    local_steps = torch.tensor([len(train_set) // cfg.loader.batch_size],
                               device=device, dtype=torch.long)
    if ddp:
        dist.all_reduce(local_steps, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local_steps.item())
    if is_main:
        print(f"steps_per_epoch: {steps_per_epoch}", flush=True)

    for epoch in range(cfg.trainer.epochs):
        train_set.set_epoch(epoch)
        hl_decoder.train()
        t0 = time.time()
        tot, n_batches = 0.0, 0
        pbar = tqdm(total=steps_per_epoch,
                    desc=f"epoch {epoch+1}/{cfg.trainer.epochs}",
                    leave=False, dynamic_ncols=True, disable=not is_main)
        loader_iter = iter(train_loader)
        done_flag = torch.zeros(1, device=device, dtype=torch.int32)
        epoch_step = 0
        while epoch_step < steps_per_epoch:
            try:
                emb, img = next(loader_iter)
                done_flag.zero_()
            except StopIteration:
                done_flag.fill_(1)
                emb = img = None
            if ddp:
                dist.all_reduce(done_flag, op=dist.ReduceOp.MAX)
            if done_flag.item():
                break
            epoch_step += 1
            pbar.update(1)

            emb = emb.to(device, dtype=compute_dtype, non_blocking=True)
            img = img.to(device, dtype=compute_dtype, non_blocking=True,
                         memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            _, loss, mse, lp = forward_chain(emb, img)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hl_decoder.parameters(), cfg.trainer.grad_clip)
            opt.step()
            n_batches += 1
            global_step += 1
            if global_step % cfg.log.every_n_steps == 0:
                lv = loss.item(); mv = mse.item(); lpv = lp.item()
                tot += lv * cfg.log.every_n_steps
                pbar.set_postfix(loss=f"{lv:.3f}", mse=f"{mv:.3f}", lpips=f"{lpv:.3f}")
                if wb is not None:
                    wb.log({"train/loss": lv, "train/mse": mv, "train/lpips": lpv,
                            "train/lr": sched.get_last_lr()[0], "epoch": epoch,
                            "global_step": global_step})
        pbar.close()
        sched.step()

        # ---------- val + video + checkpoint (rank 0 only) ----------
        if is_main:
            hl_decoder.eval()
            v_mse = v_lp = 0.0; v_n = 0
            with torch.no_grad():
                for emb, img in tqdm(val_loader, desc="val", leave=False,
                                     dynamic_ncols=True):
                    emb = emb.to(device, dtype=compute_dtype, non_blocking=True)
                    img = img.to(device, dtype=compute_dtype, non_blocking=True,
                                 memory_format=torch.channels_last)
                    _, _, mse, lp = forward_chain(emb, img)
                    v_mse += mse.item() * emb.size(0)
                    v_lp += lp.item() * emb.size(0)
                    v_n += emb.size(0)
            v_mse /= max(v_n, 1); v_lp /= max(v_n, 1)

            # full pipeline decode of the fixed trajectory for the mp4
            if wb is not None:
                hl_mod = hl_decoder.module if hasattr(hl_decoder, "module") else hl_decoder
                hl_mod = hl_mod._orig_mod if hasattr(hl_mod, "_orig_mod") else hl_mod
                pix_mod = pixel_decoder._orig_mod if hasattr(pixel_decoder, "_orig_mod") else pixel_decoder
                with torch.no_grad():
                    preds = []
                    for i in range(0, traj_emb.size(0), cfg.loader.batch_size):
                        chunk = traj_emb[i:i + cfg.loader.batch_size].to(
                            device, dtype=compute_dtype, non_blocking=True)
                        post = hjepa.ll.projector(chunk)
                        if hl_space == "post":
                            hl = hjepa.encode_hl_proj(post)
                        else:
                            hl = hjepa.encode_hl(post)
                        pred_pre = hl_mod(hl)
                        p = pix_mod(pred_pre).float().clamp(-1, 1).add(1).mul(127.5).byte().cpu()
                        preds.append(p)
                    pred_frames = torch.cat(preds, dim=0)
                tgt_frames = traj_tgt.permute(0, 3, 1, 2)
                video = torch.cat([tgt_frames, pred_frames], dim=-1).numpy()
                wb.log({"val/mse": v_mse, "val/lpips": v_lp, "epoch": epoch + 1,
                        "val/traj": wb.Video(video, fps=cfg.log.traj_fps, format="mp4"),
                        "global_step": global_step})

            print(f"epoch {epoch+1}/{cfg.trainer.epochs}  "
                  f"train_loss {tot/max(n_batches,1):.4f}  "
                  f"val_mse {v_mse:.4f}  val_lpips {v_lp:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s",
                  flush=True)

            if (epoch + 1) % cfg.trainer.save_every == 0 or (epoch + 1) == cfg.trainer.epochs:
                hl_mod = hl_decoder.module if hasattr(hl_decoder, "module") else hl_decoder
                hl_mod = hl_mod._orig_mod if hasattr(hl_mod, "_orig_mod") else hl_mod
                sd = hl_mod.state_dict()
                torch.save(sd, out_dir / f"{cfg.output_model_name}_epoch_{epoch+1}_weights.ckpt")
                torch.save(sd, out_dir / f"{cfg.output_model_name}_weights.ckpt")

        if ddp:
            dist.barrier()


if __name__ == "__main__":
    run()
