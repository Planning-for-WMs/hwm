"""Encode the pixel dataset to ViT-Tiny CLS embeddings (fp32) and save to a new H5.

Default I/O paths live under $STABLEWM_HOME (defaults to ~/.stable-wm/).
Original dataset is left untouched; output goes to <input_stem>_emb.h5.
"""
import argparse
import time
from pathlib import Path

import h5py
import torch
import stable_pretraining as spt
import stable_worldmodel as swm


def main():
    home = Path(swm.data.utils.get_cache_dir())

    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(home / "pusht_expert_train.h5"))
    p.add_argument("--output", default=str(home / "pusht_expert_train_emb.h5"))
    p.add_argument("--weights", default=str(home / "hf_pusht" / "weights.pt"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)

    encoder = spt.backbone.utils.vit_hf(
        "tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False
    )
    sd = torch.load(args.weights, map_location="cpu", weights_only=False)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    encoder.eval().to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    fin = h5py.File(args.input, "r")
    pixels = fin["pixels"]
    N = pixels.shape[0]
    D = encoder.config.hidden_size

    fout = h5py.File(args.output, "w")
    for k in fin.keys():
        if k == "pixels":
            continue
        fin.copy(k, fout)
    emb_ds = fout.create_dataset(
        "emb", shape=(N, D), dtype="float32", chunks=(min(8192, N), D)
    )

    bs = args.batch_size
    use_amp = device.type == "cuda"
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, N, bs):
            j = min(i + bs, N)
            chunk = pixels[i:j]
            x = torch.from_numpy(chunk).to(device, non_blocking=True).float()
            x = x.permute(0, 3, 1, 2).div_(255.0).sub_(mean).div_(std)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = encoder(x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0].float()
            emb_ds[i:j] = cls.cpu().numpy()
            if (i // bs) % 50 == 0:
                rate = j / max(time.time() - t0, 1e-6)
                eta = (N - j) / max(rate, 1e-6)
                print(
                    f"{j}/{N} ({100*j/N:.1f}%)  {rate:.0f} fps  eta {eta:.0f}s",
                    flush=True,
                )

    fin.close()
    fout.close()
    print(f"done in {time.time()-t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
