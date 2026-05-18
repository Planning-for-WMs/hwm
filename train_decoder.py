"""Train an emb->pixels decoder with LPIPS loss.

Reads pre-encoded CLS embeddings from `pusht_expert_train_emb.h5` (192-d) and
target frames from `pusht_expert_train.h5` (224x224x3 uint8). Decoder
architecture mirrors the convolutional upsampler used in DINO-WM / JEPA-WMs:
CLS -> Linear -> spatial 7x7 -> 5x (Upsample + Conv + GN + GELU) -> 3x224x224.
Trained with LPIPS (VGG) + an optional MSE term in [-1, 1] image space.

Run:
    export STABLEWM_HOME=/home/.stable-wm
    python train_decoder.py                       # uses config/train/decoder.yaml
    python train_decoder.py trainer.epochs=50 loader.batch_size=256
"""
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
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm


def _img_to_tensor(img_np):
    """uint8 HWC -> float CHW in [-1, 1]."""
    t = torch.from_numpy(img_np)
    return t.permute(2, 0, 1).float().div_(127.5).sub_(1.0)


class EmbPixelDataset(Dataset):
    """Random-access dataset. Preloads all embeddings to RAM (~1.8 GB for
    PushT-expert; 2.3M x 192 fp32). Pixels stay on disk and are read lazily
    per worker. Used for the val split."""

    def __init__(self, emb_path, pix_path, indices=None):
        self.emb_path = str(emb_path)
        self.pix_path = str(pix_path)
        with h5py.File(self.emb_path, "r") as f:
            self.emb = f["emb"][:]  # all in RAM
            self.emb_dim = self.emb.shape[1]
        self.indices = (np.arange(self.emb.shape[0]) if indices is None
                        else np.asarray(indices))
        self._pix = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self._pix is None:
            self._pix = h5py.File(self.pix_path, "r")["pixels"]
        gi = int(self.indices[i])
        emb = torch.from_numpy(self.emb[gi].astype(np.float32, copy=False))
        return emb, _img_to_tensor(self._pix[gi])


class ChunkAwarePixelDataset(torch.utils.data.IterableDataset):
    """Train-time iterable dataset that reads the pixels H5 chunk-by-chunk.

    The pixels H5 uses chunk size (100, 224, 224, 3) ≈ 15 MB. Random sample
    access decompresses the whole chunk per sample (~100× I/O amplification).
    This iterable groups sample indices by chunk-id, decompresses each chunk
    once, then yields its samples in shuffled order. Across workers, chunks
    are partitioned so each is decompressed by exactly one worker.

    Embeddings are preloaded to RAM (shared via fork). The dataset yields
    samples; shuffle inside each chunk gives stochasticity without re-reading.
    """

    def __init__(self, emb_path, pix_path, indices, chunk_size=100, seed=0):
        super().__init__()
        self.emb_path = str(emb_path); self.pix_path = str(pix_path)
        self.chunk_size = int(chunk_size)
        with h5py.File(self.emb_path, "r") as f:
            self.emb = f["emb"][:]
            self.emb_dim = self.emb.shape[1]
        indices = np.asarray(indices, dtype=np.int64)
        # group indices by chunk id
        cids = indices // self.chunk_size
        order = np.argsort(cids, kind="stable")
        indices = indices[order]; cids = cids[order]
        # list of (chunk_id, [local offsets within chunk])
        uniq, starts = np.unique(cids, return_index=True)
        ends = np.r_[starts[1:], len(indices)]
        self._chunks = [(int(uniq[k]),
                         (indices[starts[k]:ends[k]] - uniq[k] * self.chunk_size).astype(np.int64))
                        for k in range(len(uniq))]
        self._epoch = 0
        self.base_seed = seed
        # length = total samples (used by tqdm via __len__ approximation)
        self._len = int(len(indices))

    def __len__(self):
        return self._len

    def set_epoch(self, e):
        self._epoch = int(e)

    def __iter__(self):
        winfo = torch.utils.data.get_worker_info()
        wid = 0 if winfo is None else winfo.id
        nw = 1 if winfo is None else winfo.num_workers
        rng = np.random.default_rng(self.base_seed + 1000 * self._epoch + wid)

        # shuffle chunk order globally (same on every worker so partition is consistent)
        gshuf = np.random.default_rng(self.base_seed + 1000 * self._epoch).permutation(
            len(self._chunks))
        my_chunks = [self._chunks[gshuf[k]] for k in range(wid, len(self._chunks), nw)]

        pix = h5py.File(self.pix_path, "r")["pixels"]
        cs = self.chunk_size
        for cid, locals_ in my_chunks:
            # one decompress for the whole chunk
            block = pix[cid * cs: cid * cs + cs]  # (<=cs, 224, 224, 3) uint8
            local_shuf = rng.permutation(len(locals_))
            for li in local_shuf:
                lo = int(locals_[li])
                gi = cid * cs + lo
                emb = torch.from_numpy(self.emb[gi].astype(np.float32, copy=False))
                yield emb, _img_to_tensor(block[lo])


class ConvDecoder(nn.Module):
    """emb_dim -> base x init_hw x init_hw -> 5x upsample -> 3x224x224 in [-1, 1]."""

    def __init__(self, emb_dim=192, base=512, init_hw=7):
        super().__init__()
        self.init_hw = init_hw
        self.base = base
        self.fc = nn.Linear(emb_dim, base * init_hw * init_hw)

        chans = [base, base // 2, base // 4, base // 8, base // 16, base // 16]
        blocks = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.GroupNorm(min(32, c_out), c_out),
                nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(min(32, c_out), c_out),
                nn.GELU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)

    def forward(self, emb):
        x = self.fc(emb).view(-1, self.base, self.init_hw, self.init_hw)
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.to_rgb(x))


@hydra.main(version_base=None, config_path="./config/train", config_name="decoder")
def run(cfg):
    # ---------- DDP init ----------
    import os
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if ddp:
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[local_rank {local_rank}] init_process_group...", flush=True)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank(); world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        print(f"[rank {rank}/{world_size}] process group ready on cuda:{local_rank}",
              flush=True)
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
    # Chunk-aware train reader (avoids ~100x blosc decompression amplification
    # from random access on the 100-frame pixel chunks). Embeddings are loaded
    # to RAM once per rank.
    with h5py.File(cfg.emb_h5, "r") as f:
        n_total = f["emb"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n_total)
    n_train = int(cfg.train_split * n_total)
    train_idx_all, val_idx = perm[:n_train], perm[n_train:]
    # Shard train indices across ranks; val stays on rank 0.
    train_idx = train_idx_all[rank::world_size]

    train_set = ChunkAwarePixelDataset(cfg.emb_h5, cfg.pix_h5, train_idx,
                                       chunk_size=100, seed=cfg.seed + rank)
    val_set = EmbPixelDataset(cfg.emb_h5, cfg.pix_h5, indices=val_idx)
    loader_kw = dict(cfg.loader)
    # IterableDataset: shuffle is handled internally; loader must not shuffle.
    train_loader = DataLoader(train_set, shuffle=False, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kw)

    # ---------- frozen LL projector (only if training in post-projector space) ----------
    ll_projector = None
    space = cfg.get("space", "pre")
    if space == "post":
        ll_jepa = torch.load(cfg.ll_ckpt, map_location="cpu", weights_only=False)
        ll_projector = ll_jepa.projector.to(device=device, dtype=torch.float32).eval()
        for _p in ll_projector.parameters():
            _p.requires_grad_(False)
        if is_main:
            print(f"[space=post] loaded frozen LL projector from {cfg.ll_ckpt}",
                  flush=True)
    elif space != "pre":
        raise ValueError(f"space must be 'pre' or 'post', got {space!r}")

    def to_decoder_space(e):
        """Convert raw emb (pre-projector) to the space the decoder consumes."""
        if ll_projector is None:
            return e
        # BN in projector wants fp32 batched input; recast to compute dtype after
        with torch.no_grad():
            e_post = ll_projector(e.float())
        return e_post.to(e.dtype)

    # ---------- model / loss / optim ----------
    # Cast model + loss to bf16 explicitly. autocast leaves GroupNorm /
    # ScalingLayer in fp32, which on L4 makes the whole graph memory-bound at
    # fp32 bandwidth. bf16 throughout avoids that. No GradScaler needed for bf16.
    compute_dtype = torch.bfloat16 if (cfg.amp and device.type == "cuda") else torch.float32
    decoder = ConvDecoder(**cfg.model).to(device=device, dtype=compute_dtype)
    decoder = decoder.to(memory_format=torch.channels_last)
    lpips_fn = lpips.LPIPS(net=cfg.loss.lpips_net).to(device=device, dtype=compute_dtype).eval()
    for p_ in lpips_fn.parameters():
        p_.requires_grad_(False)

    # Warm-start from a previous state_dict if requested
    resume_path = cfg.get("resume_from", None)
    if resume_path:
        sd = torch.load(resume_path, map_location=device)
        missing, unexpected = decoder.load_state_dict(sd, strict=True)
        if is_main:
            print(f"loaded weights from {resume_path}", flush=True)

    # DDP wrap before compile (compile sees the wrapped module fine)
    if ddp:
        print(f"[rank {rank}] before DDP wrap (NCCL handshake)", flush=True)
        from torch.nn.parallel import DistributedDataParallel as DDP
        decoder = DDP(decoder, device_ids=[local_rank],
                      gradient_as_bucket_view=True,
                      broadcast_buffers=False)
        print(f"[rank {rank}] DDP wrap done", flush=True)

    compile_mode = cfg.get("compile", None)
    if compile_mode:
        if is_main:
            print(f"torch.compile mode={compile_mode} (first step will be slow)")
        decoder = torch.compile(decoder, mode=compile_mode)
        lpips_fn = torch.compile(lpips_fn, mode=compile_mode)

    opt = torch.optim.AdamW(decoder.parameters(), **cfg.optimizer)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.trainer.epochs)

    n_params = sum(p.numel() for p in decoder.parameters())
    if is_main:
        print(f"decoder params: {n_params/1e6:.2f}M  | "
              f"train(per-rank) {len(train_set)} val {len(val_set)}  world_size={world_size}")

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
        traj_tgt = torch.from_numpy(f["pixels"][t_start:t_start + t_len])  # T H W C uint8
    if is_main:
        print(f"logging trajectory ep={ep_idx} len={t_len} start={t_start}")

    # ---------- training loop ----------
    if ddp:
        import torch.distributed as dist
        print(f"[rank {rank}] reached training loop, world_size={world_size}", flush=True)
        dist.barrier()
        if is_main:
            print("all ranks ready, starting epoch 1", flush=True)
    global_step = 0
    # Force every rank to run the same number of steps per epoch.
    # The chunk-aware shards have *near*-equal sample counts but the per-rank
    # `len(train_set) // batch_size` can differ by 1, which would deadlock DDP
    # at the boundary (one rank does an extra backward() and waits forever).
    local_steps = torch.tensor([len(train_set) // cfg.loader.batch_size],
                               device=device, dtype=torch.long)
    if ddp:
        dist.all_reduce(local_steps, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local_steps.item())
    if is_main:
        print(f"steps_per_epoch (sync'd across ranks): {steps_per_epoch}", flush=True)
    for epoch in range(cfg.trainer.epochs):
        train_set.set_epoch(epoch)
        decoder.train()
        t0 = time.time()
        tot, n_batches = 0.0, 0
        pbar = tqdm(total=steps_per_epoch,
                    desc=f"epoch {epoch+1}/{cfg.trainer.epochs}",
                    leave=False, dynamic_ncols=True, disable=not is_main)
        loader_iter = iter(train_loader)
        epoch_step = 0
        done_flag = torch.zeros(1, device=device, dtype=torch.int32)
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
            emb = to_decoder_space(emb)
            opt.zero_grad(set_to_none=True)
            pred = decoder(emb)
            mse = F.mse_loss(pred, img)
            if cfg.loss.lpips_size and cfg.loss.lpips_size != pred.shape[-1]:
                lp_p = F.interpolate(pred, size=cfg.loss.lpips_size,
                                     mode="bilinear", align_corners=False)
                lp_t = F.interpolate(img, size=cfg.loss.lpips_size,
                                     mode="bilinear", align_corners=False)
            else:
                lp_p, lp_t = pred, img
            lp = lpips_fn(lp_p, lp_t).mean()
            loss = cfg.loss.mse_weight * mse + cfg.loss.lpips_weight * lp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.trainer.grad_clip)
            opt.step()
            n_batches += 1
            global_step += 1
            # avoid per-step cuda sync; only sync when we actually log
            if global_step % cfg.log.every_n_steps == 0:
                lv = loss.item(); mv = mse.item(); lpv = lp.item()
                tot += lv * cfg.log.every_n_steps
                pbar.set_postfix(loss=f"{lv:.3f}", mse=f"{mv:.3f}", lpips=f"{lpv:.3f}")
                if wb is not None:
                    # let wandb auto-increment its step counter; passing step=N
                    # would be silently dropped if N <= the resumed run's last step.
                    wb.log({"train/loss": lv, "train/mse": mv, "train/lpips": lpv,
                            "train/lr": sched.get_last_lr()[0], "epoch": epoch,
                            "global_step": global_step})
        pbar.close()
        sched.step()

        # validation + checkpoint + video logging: rank 0 only
        if is_main:
            decoder.eval()
            v_mse = v_lp = 0.0; v_n = 0
            with torch.no_grad():
                for emb, img in tqdm(val_loader, desc="val", leave=False,
                                     dynamic_ncols=True):
                    emb = emb.to(device, dtype=compute_dtype, non_blocking=True)
                    img = img.to(device, dtype=compute_dtype, non_blocking=True,
                                 memory_format=torch.channels_last)
                    emb = to_decoder_space(emb)
                    pred = decoder(emb)
                    v_mse += F.mse_loss(pred, img).item() * emb.size(0)
                    if cfg.loss.lpips_size and cfg.loss.lpips_size != pred.shape[-1]:
                        lp_p = F.interpolate(pred, size=cfg.loss.lpips_size,
                                             mode="bilinear", align_corners=False)
                        lp_t = F.interpolate(img, size=cfg.loss.lpips_size,
                                             mode="bilinear", align_corners=False)
                    else:
                        lp_p, lp_t = pred, img
                    v_lp += lpips_fn(lp_p, lp_t).mean().item() * emb.size(0)
                    v_n += emb.size(0)
            v_mse /= max(v_n, 1); v_lp /= max(v_n, 1)

            # decode the fixed trajectory for a single mp4 log
            traj_video = None
            if wb is not None:
                # unwrap DDP/compile to call the bare decoder on variable shapes
                dec_mod = decoder.module if hasattr(decoder, "module") else decoder
                dec_mod = dec_mod._orig_mod if hasattr(dec_mod, "_orig_mod") else dec_mod
                with torch.no_grad():
                    preds = []
                    for i in range(0, traj_emb.size(0), cfg.loader.batch_size):
                        chunk = traj_emb[i:i + cfg.loader.batch_size].to(
                            device, dtype=compute_dtype, non_blocking=True)
                        chunk = to_decoder_space(chunk)
                        p_ = dec_mod(chunk).float().clamp(-1, 1).add(1).mul(127.5).byte().cpu()
                        preds.append(p_)
                    pred_frames = torch.cat(preds, dim=0)
                tgt_frames = traj_tgt.permute(0, 3, 1, 2)
                video = torch.cat([tgt_frames, pred_frames], dim=-1).numpy()
                traj_video = wb.Video(video, fps=cfg.log.traj_fps, format="mp4")
                wb.log({"val/mse": v_mse, "val/lpips": v_lp, "epoch": epoch + 1,
                        "val/traj": traj_video, "global_step": global_step})

            print(f"epoch {epoch+1}/{cfg.trainer.epochs}  train_loss {tot/max(n_batches,1):.4f}  "
                  f"val_mse {v_mse:.4f}  val_lpips {v_lp:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s",
                  flush=True)

            if (epoch + 1) % cfg.trainer.save_every == 0 or (epoch + 1) == cfg.trainer.epochs:
                dec_mod = decoder.module if hasattr(decoder, "module") else decoder
                dec_mod = dec_mod._orig_mod if hasattr(dec_mod, "_orig_mod") else dec_mod
                # state_dict only: pickling the whole module can hang for DDP/compile
                # wrappers. To reload: instantiate ConvDecoder(**cfg.model) then load_state_dict.
                sd = dec_mod.state_dict()
                torch.save(sd, out_dir / f"{cfg.output_model_name}_epoch_{epoch+1}_weights.ckpt")
                torch.save(sd, out_dir / f"{cfg.output_model_name}_weights.ckpt")

        if ddp:
            import torch.distributed as dist
            dist.barrier()


if __name__ == "__main__":
    run()
