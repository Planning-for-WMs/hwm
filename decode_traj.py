"""Decode a random trajectory from `pusht_expert_train_emb.h5` using the
trained ConvDecoder, save as MP4 alongside the ground-truth pixels.

Usage:
  python decode_traj.py [--ckpt PATH] [--episode N] [--out PATH] [--fps 10] [--max-len 200]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import imageio.v3 as iio
import numpy as np
import torch

from train_decoder import ConvDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser(
        os.path.expandvars("$STABLEWM_HOME/decoder_v2/decoder_v2_epoch_1_object.ckpt")))
    ap.add_argument("--emb-h5", default=os.path.expandvars("$STABLEWM_HOME/pusht_expert_train_emb.h5"))
    ap.add_argument("--pix-h5", default=os.path.expandvars("$STABLEWM_HOME/pusht_expert_train.h5"))
    ap.add_argument("--episode", type=int, default=-1, help="-1 = random")
    ap.add_argument("--out", default=os.path.expandvars("$STABLEWM_HOME/decoder_v2/decoded_traj.mp4"))
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    print(f"loading decoder: {args.ckpt}")
    obj = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        # state_dict ckpt: strip any wrapper prefix (e.g. 'module.', '_orig_mod.')
        sd = obj.get("state_dict", obj)
        sd = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
        dec = ConvDecoder(emb_dim=192, base=256, init_hw=7).to(device).eval()
        missing, unexpected = dec.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        dec = obj.eval()
    dec_dtype = next(dec.parameters()).dtype

    print(f"opening emb h5: {args.emb_h5}")
    with h5py.File(args.emb_h5, "r") as fe:
        ep_idx_col = "episode_idx" if "episode_idx" in fe else "ep_idx"
        ep_ids = fe[ep_idx_col][:]
        step_idx = fe["step_idx"][:]
        unique_eps = np.unique(ep_ids)

        ep = args.episode if args.episode >= 0 else int(rng.choice(unique_eps))
        mask = ep_ids == ep
        rows = np.nonzero(mask)[0]
        order = np.argsort(step_idx[rows])
        rows = rows[order]
        if len(rows) > args.max_len:
            rows = rows[: args.max_len]
        print(f"episode {ep}: {len(rows)} frames")
        emb = fe["emb"][rows]                     # (T, 192)
    emb = torch.from_numpy(emb).to(device=device, dtype=dec_dtype)

    # decode in batches to be safe
    BATCH = 64
    frames = []
    with torch.no_grad():
        for i in range(0, emb.size(0), BATCH):
            out = dec(emb[i : i + BATCH])         # (b, 3, 224, 224) in [-1, 1]
            out = (out.float().clamp(-1, 1) + 1) * 0.5    # [0, 1]
            out = (out * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            frames.append(out)
    decoded = np.concatenate(frames, axis=0)      # (T, H, W, 3)

    # load ground-truth pixels for side-by-side
    try:
        with h5py.File(args.pix_h5, "r") as fp:
            gt = fp["pixels"][rows]
            if gt.dtype != np.uint8:
                gt = (gt * 255).astype(np.uint8) if gt.max() <= 1.5 else gt.astype(np.uint8)
        if gt.shape[1:] != decoded.shape[1:]:
            print(f"resizing gt {gt.shape} to match decoded {decoded.shape}")
            from PIL import Image
            gt = np.stack([
                np.array(Image.fromarray(g).resize((decoded.shape[2], decoded.shape[1])))
                for g in gt
            ])
        combined = np.concatenate([gt, decoded], axis=2)   # side-by-side W
        print(f"writing side-by-side mp4 (gt | decoded): {args.out}")
    except Exception as e:
        print(f"could not load gt pixels ({e}); writing decoded-only mp4")
        combined = decoded
        print(f"writing decoded-only mp4: {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, combined, fps=args.fps, codec="libx264")
    print(f"done: {combined.shape[0]} frames @ {args.fps} fps")


if __name__ == "__main__":
    main()
