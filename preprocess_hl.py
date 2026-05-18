"""preprocess_hl.py — encode pre-projector LL CLS to (LL_post, HL_post) once,
write to a new H5 file. Used so cost-head / probe-head training doesn't have
to re-run the encoder on every launch.

Inputs:
  --hjepa_ckpt   path to a hjepa *_object.ckpt (or a directory; we pick the
                 highest-epoch ckpt in it)
  --input        pre-encoded H5 (default: $STABLEWM_HOME/pusht_expert_train_emb.h5)
                 must contain "emb" (N, ll_dim) and per-row metadata
                 (episode_idx, step_idx, ep_offset, ep_len, action, proprio, state).
  --output       output H5 path (default: <input_stem>_hl.h5)

Output H5 contents:
  ll_post        (N, ll_dim)  fp32  — post-projector LL embedding
                 = ll.projector(emb)
  hl_post        (N, hl_dim)  fp32  — post-hl_projector HL state
                 = hl_projector(hle(ll.projector(emb)))
  episode_idx   (N,) int64    — copied from input
  step_idx      (N,) int64    — copied from input
  ep_offset     (M,) int64    — copied if present
  ep_len        (M,) int64    — copied if present
  action        (N, A)        — copied if present (forward to downstream consumers)

Run:
  python preprocess_hl.py
  python preprocess_hl.py --hjepa_ckpt $STABLEWM_HOME/hjepa_v12/hjepa_v12_epoch_30_object.ckpt
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm


def resolve_hjepa_ckpt(p):
    p = Path(p)
    if p.is_file():
        return p
    if p.is_dir():
        cands = list(p.glob("*_epoch_*_object.ckpt"))
        if not cands:
            raise FileNotFoundError(f"no *_epoch_*_object.ckpt under {p}")
        return max(cands, key=lambda x: int(x.stem.split("_epoch_")[1].split("_")[0]))
    raise FileNotFoundError(p)


def main():
    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hjepa_ckpt", default=f"{home}/hjepa_v12")
    ap.add_argument("--input", default=f"{home}/pusht_expert_train_emb.h5")
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch_size", type=int, default=8192)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else \
        in_path.with_name(in_path.stem + "_hl.h5")
    print(f"input  : {in_path}")
    print(f"output : {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hjepa_path = resolve_hjepa_ckpt(args.hjepa_ckpt)
    print(f"hjepa  : {hjepa_path}")
    hjepa = torch.load(hjepa_path, map_location=device, weights_only=False)
    hjepa = hjepa.to(device).eval()
    hjepa.requires_grad_(False)

    with h5py.File(in_path, "r") as fin:
        N, ll_dim = fin["emb"].shape
        hl_dim = hjepa.hl_projector.net[-1].out_features
        keys_to_copy = [k for k in fin.keys() if k != "emb"]
        print(f"N rows = {N:,}  ll_dim = {ll_dim}  hl_dim = {hl_dim}")
        print(f"forwarding metadata keys: {keys_to_copy}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()

        with h5py.File(out_path, "w") as fout:
            # encoded targets
            ll_ds = fout.create_dataset(
                "ll_post", shape=(N, ll_dim), dtype="float32",
                chunks=(min(4096, N), ll_dim))
            hl_ds = fout.create_dataset(
                "hl_post", shape=(N, hl_dim), dtype="float32",
                chunks=(min(4096, N), hl_dim))

            # copy metadata column-by-column
            for k in keys_to_copy:
                fout.create_dataset(k, data=fin[k][:], compression=None)

            # encode in batches
            with torch.inference_mode():
                for i in tqdm(range(0, N, args.batch_size), desc="encode"):
                    j = min(i + args.batch_size, N)
                    batch = torch.from_numpy(fin["emb"][i:j]).to(device).float()
                    post_ll = hjepa.ll.projector(batch)            # (b, ll_dim)
                    hle_out = hjepa.hle(post_ll)                   # (b, hl_dim)
                    post_hl = hjepa.hl_projector(hle_out)          # (b, hl_dim)
                    ll_ds[i:j] = post_ll.detach().cpu().float().numpy()
                    hl_ds[i:j] = post_hl.detach().cpu().float().numpy()

    print(f"[done] wrote {out_path}")
    print(f"  ll_post: ({N}, {ll_dim})  hl_post: ({N}, {hl_dim})")
    size_gb = (N * (ll_dim + hl_dim) * 4) / 1e9
    print(f"  approx data size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
