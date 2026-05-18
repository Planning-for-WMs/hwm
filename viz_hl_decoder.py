"""Render a side-by-side mp4 of three reconstructions of a random episode:
    [ground truth] | [decoder_v2 only] | [HLDecoder + decoder_v2]

Run:
    export STABLEWM_HOME=/home/.stable-wm
    python viz_hl_decoder.py                                # uses defaults below
    python viz_hl_decoder.py --episode 1234 --out out.mp4
"""
import argparse
import os
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import imageio
import numpy as np
import torch

from train_decoder import ConvDecoder
from train_hl_decoder import HLDecoder


def _newest(p: Path, pattern: str):
    cands = list(p.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"no {pattern} under {p}")
    return max(cands, key=lambda x: int(x.stem.split("_epoch_")[1].split("_")[0]))


def newest_hjepa(home):
    return _newest(Path(home) / "hjepa_v6", "*_epoch_*_object.ckpt")


def newest_weights(home, subdir):
    """Pick the highest-epoch *_weights.ckpt under home/subdir."""
    return _newest(Path(home) / subdir, "*_epoch_*_weights.ckpt")


def main():
    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-h5", default=f"{home}/pusht_expert_train_emb.h5")
    ap.add_argument("--pix-h5", default=f"{home}/pusht_expert_train.h5")
    ap.add_argument("--hjepa-ckpt", default=str(newest_hjepa(home)))
    ap.add_argument("--pixel-decoder-weights",
                    default=str(newest_weights(home, "decoder_v4")))
    ap.add_argument("--hl-decoder-weights",
                    default=str(newest_weights(home, "hl_decoder_v2")))
    ap.add_argument("--episode", type=int, default=-1, help="episode idx; -1 = random")
    ap.add_argument("--seed", type=int, default=3072)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=str(Path(home) / "hl_decoder" / "viz.mp4"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    # decoder_v2 shape (must match its training config)
    ap.add_argument("--pix-emb-dim", type=int, default=192)
    ap.add_argument("--pix-base", type=int, default=256)
    ap.add_argument("--pix-init-hw", type=int, default=7)
    # HLDecoder shape (must match its training config)
    ap.add_argument("--hl-hl-dim", type=int, default=96)
    ap.add_argument("--hl-ll-dim", type=int, default=192)
    ap.add_argument("--hl-hidden-dim", type=int, default=768)
    ap.add_argument("--hl-depth", type=int, default=5)
    ap.add_argument("--hl-space", choices=["pre", "post"], default="post",
                    help="which HL state feeds the decoder; must match training "
                         "(v6+ uses 'post'; v5 and earlier used 'pre')")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ---------- load frozen models ----------
    print(f"loading hjepa from {args.hjepa_ckpt}")
    hjepa = torch.load(args.hjepa_ckpt, map_location=device, weights_only=False)
    hjepa.to(device=device, dtype=dtype).eval()

    pix_dec = ConvDecoder(emb_dim=args.pix_emb_dim, base=args.pix_base,
                          init_hw=args.pix_init_hw).to(device=device, dtype=dtype)
    pix_dec = pix_dec.to(memory_format=torch.channels_last)
    sd = torch.load(args.pixel_decoder_weights, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    pix_dec.load_state_dict(sd, strict=True)
    pix_dec.eval()

    hl_dec = HLDecoder(hl_dim=args.hl_hl_dim, ll_dim=args.hl_ll_dim,
                       hidden_dim=args.hl_hidden_dim,
                       depth=args.hl_depth).to(device=device, dtype=dtype)
    hl_dec.load_state_dict(torch.load(args.hl_decoder_weights, map_location=device),
                           strict=True)
    hl_dec.eval()

    # ---------- pick episode ----------
    with h5py.File(args.emb_h5, "r") as f:
        ep_off = f["ep_offset"][:]; ep_len = f["ep_len"][:]
    if args.episode < 0:
        rng = np.random.default_rng(args.seed)
        ep_idx = int(rng.integers(len(ep_len)))
    else:
        ep_idx = args.episode
    t_len = min(int(ep_len[ep_idx]), args.max_len)
    t_start = int(ep_off[ep_idx])
    print(f"episode {ep_idx}: len={t_len} start={t_start}")

    with h5py.File(args.emb_h5, "r") as f:
        traj_emb = torch.from_numpy(
            f["emb"][t_start:t_start + t_len].astype(np.float32))
    with h5py.File(args.pix_h5, "r") as f:
        traj_tgt = torch.from_numpy(f["pixels"][t_start:t_start + t_len])  # T H W C u8

    # ---------- decode trajectory through both pipelines ----------
    pix_v2_frames, pix_hl_frames = [], []
    with torch.no_grad():
        for i in range(0, traj_emb.size(0), args.batch_size):
            chunk = traj_emb[i:i + args.batch_size].to(device=device, dtype=dtype,
                                                       non_blocking=True)
            # decoder_v2 only: pre_emb -> pixels
            pix_v2 = pix_dec(chunk).float().clamp(-1, 1).add(1).mul(127.5).byte().cpu()
            pix_v2_frames.append(pix_v2)
            # HLDecoder pipeline: pre -> post (proj) -> HL -> predicted pre -> pixels
            post = hjepa.ll.projector(chunk)
            hl = (hjepa.encode_hl_proj(post) if args.hl_space == "post"
                  else hjepa.encode_hl(post))
            pred_pre = hl_dec(hl)
            pix_hl = pix_dec(pred_pre).float().clamp(-1, 1).add(1).mul(127.5).byte().cpu()
            pix_hl_frames.append(pix_hl)
    pix_v2 = torch.cat(pix_v2_frames, dim=0)  # T C H W u8
    pix_hl = torch.cat(pix_hl_frames, dim=0)
    pix_gt = traj_tgt.permute(0, 3, 1, 2)     # T C H W u8

    # ---------- compose side-by-side and write mp4 ----------
    # concat along W: [gt | decoder_v2 only | HL pipeline]
    side = torch.cat([pix_gt, pix_v2, pix_hl], dim=-1)  # T C H (3W)
    side = side.permute(0, 2, 3, 1).contiguous().numpy()  # T H (3W) C
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for f in side:
        writer.append_data(f)
    writer.close()
    print(f"wrote {out}  ({side.shape[0]} frames @ {args.fps} fps, "
          f"layout: GT | decoder_v2 | HLDecoder+decoder_v2)")


if __name__ == "__main__":
    main()
