
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
# system deps for box2d-py
sudo apt-get install -y swig

uv venv --python=3.10
source .venv/bin/activate

# gym 0.21 / 0.17 need legacy setuptools to build
printf "setuptools<66\nwheel<0.40\n" > /tmp/build-constraints.txt

uv pip install --build-constraints /tmp/build-constraints.txt 'stable-worldmodel[train,env]'

# the resolver may pin datasets to an ancient version; bump it
uv pip install -U datasets
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (the `stable-worldmodel` package default is `~/.stable_worldmodel/`; the README originally referenced `~/.stable-wm/`). Override with:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

### Heldout collection (optional)

`collect_heldout.py` records a PushT dataset using `swm.envs.pusht.WeakPolicy`
(noisy random actions clipped near the block) for evaluating against unseen
init/goal pairs. Output goes to `$STABLEWM_HOME/<name>.h5` in the same schema
as the expert dataset, so eval can use it directly. `--dist-constraint 5`
gives action-magnitude / agent-speed medians within 5% of the expert dataset
(higher values produce a faster-moving agent and shift the eval distribution).

```bash
export STABLEWM_HOME=/home/.stable-wm
python collect_heldout.py --episodes 200 --num-envs 16 --dist-constraint 5
python eval.py --config-name=pusht.yaml policy=pusht/lewm \
    eval.dataset_name=pusht_weak_heldout
```

### Pre-encoding pixels (optional)

`preprocess.py` runs the dataset through the LeWM ViT encoder once and stores
only the CLS embeddings (192-dim, fp32) plus the original metadata
(`action`, `proprio`, `state`, `episode_idx`, `step_idx`, `ep_len`,
`ep_offset`). For pusht-expert-train this shrinks 44 GB → ~1.8 GB.

```bash
export STABLEWM_HOME=/home/.stable-wm   # or wherever the .h5 lives
python preprocess.py \
  --input  $STABLEWM_HOME/pusht_expert_train.h5 \
  --output $STABLEWM_HOME/pusht_expert_train_emb.h5 \
  --weights $STABLEWM_HOME/hf_pusht/weights.pt
```

Original `.h5` is left untouched.

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

To plan more episodes in parallel, raise `solver.batch_size` (defaults to 1 in `config/eval/solver/cem.yaml`):
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm solver.batch_size=16
```
Note: `JEPA.get_cost` was patched to unsqueeze a sample dim on `goal_emb` so it
broadcasts against `pred_emb (B, S, T, D)`; without this, eval fails with a
`expand_as` rank error for any `batch_size`.

`JEPA.predict` is wrapped in `torch.autocast(bfloat16)` (CUDA only) — the
predictor (a depth-6 transformer) is the planning bottleneck (5 sequential
forwards × 30 CEM iters per solve). On L4 this gives ~2× planning speedup;
output is cast back to the input dtype so downstream ops are unchanged.
## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
kw = lambda k: {kk: vv for kk, vv in cfg[k].items() if not kk.startswith("_")}  # drop hydra _target_/_partial_

encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**kw("predictor")),
    action_encoder=Embedder(**kw("action_encoder")),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Hierarchical world model (HL JEPA)

`hjepa.py` adds a hierarchy on top of a frozen LeWM checkpoint:

- `HLEncoder` (MLP): post-projector LL CLS (192) → HL state (96)
- `HL Projector` / `HL Pred-proj`: asymmetric MLP heads `96 → 384 → GELU → 96` (LeWM-style); projector applied post-HLE, pred-proj post-HLP. Loss and SIGReg operate in projected space.
- `HLPredictor`: **causal transformer with AdaLN-zero macro conditioning** (mirrors LL `ARPredictor`). Tokens are 3 projected HL state frames; per-token macros modulate each block's LayerNorm via `Linear(hl_dim, 6·hl_dim)` shift/scale/gate. Gates init at 0 → predictor starts as identity in state, macro pathway is earned by gradient. Depth 6, heads 4, dim_head 24, mlp_dim 192. Head reads the last token's output → next HL state.
- `MacroActionEncoder`: **shallow causal transformer** consuming the frozen LL action encoder's 192-d tokens via a `Linear(192, 96)` input projection. CLS appended at end with learned pos-embed (supports variable K up to `k_max`); depth 2, heads 4, dim 96, mlp_dim 192; CLS → MLP head → macro of dim 96.
- **Variable training stride**: per training batch, `K` is sampled uniformly from `[k_min, k_max]` (defaults 5 and 10). This trains a single HLP that generalizes across HL temporal scales.

**Distributed**: trainer is configured for **2-GPU DDP** by default (`trainer.devices: 2, trainer.strategy: ddp`). Lightning's `DistributedSampler` is enabled, so the per-batch stride `K` is **seeded from `global_step`** in `hjepa_forward` to keep both ranks computing the same prediction problem (different ranks see different data but the *same K*). Drop to 1 GPU with `python train_hl.py trainer.devices=1 trainer.strategy=auto`.

Training reads the pre-encoded `pusht_expert_train_emb.h5` directly (no LL
encoder forward at train time) and applies the frozen LL projector once to
obtain post-projector embeddings — the same representation the LL predictor
operates on. Target is **not** detached (matches LeWM; collapse is prevented
by SIGReg).

Loss (in projected space): `||hl_pred_proj(HLP(states_proj_history, macros_history)) - hl_projector(HLE(LL_emb_{t+history·K}))||²` + `λ · SIGReg(projected HL states)`.
Training windows are `history · k_max + 1` LL tokens long; per-batch sampled stride `K ∈ [k_min, k_max]` slices the window into `history + 1` anchor frames at indices `[0, K, 2K, …, history·K]`.

**Train:**
```bash
export STABLEWM_HOME=/home/.stable-wm
python train_hl.py                                          # uses config/train/lewm_hl.yaml
# checkpoint -> $STABLEWM_HOME/hjepa/hjepa_epoch_<N>_object.ckpt
```

**Eval (two-stage MPC):** Stage 1 — HL CEM over macro-action sequences finds an
HL subgoal. Stage 2 — existing LL CEM (via `SubgoalLLModel` adapter) plans
LL actions to reach the subgoal. Cost in Stage 2 is
`||HLE(LL_predicted_emb) - subgoal||²`.

```bash
python eval_hl.py policy_ckpt=$STABLEWM_HOME/hjepa/hjepa_epoch_50_object.ckpt \
                  eval.dataset_name=pusht_weak_heldout
```

Hyperparameters live in `config/train/lewm_hl.yaml` (HL dims, MAE size, K) and
`config/eval/pusht_hl.yaml` (HL CEM `T_HL`, `num_samples`, `n_steps`, plus the
existing LL solver block).

**Eval (unified single-stage CEM)** — `eval_unified.py`. Samples LL action
tokens directly and feeds each block of `K_macro` tokens through MAE to obtain
on-manifold macros (no extra macro sampling). Cost = long-term HL terminal cost
(after `T_HL` macros via HLP) + `short_weight` × short-term cost (LL predictor
rollout of the first macro's tokens, mapped to HL space and compared to the
first HL subgoal). Closed-loop: replans every `action_block` env steps and
executes only the first LL token. Warm-start shifts the LL action mean left by
one token, padding with a copy of the last token.

```bash
python eval_unified.py policy_ckpt=$STABLEWM_HOME/hjepa/hjepa_epoch_50_object.ckpt
```

Knobs in `config/eval/pusht_unified.yaml`: `unified.T_HL`, `unified.K_macro`
(must match `mae.num_actions`), `unified.num_samples`, `unified.n_steps`,
`unified.short_weight`, `unified.var_scale`, `unified.var_ema`.

**v6 architecture (bounded macro + interleaved HLP, no AdaLN)** —
`config/train/lewm_hl_v6.yaml`. Two changes vs. v5 to reduce planner
exploitation of the HLP: (1) `mae.bounded: true` adds tanh to the MAE head so
macros lie in `[-1, 1]^96`; (2) `hlp.type: interleaved` swaps `HLPredictor`
(AdaLN-zero) for `HLPredictorInterleaved`, which builds a `2H`-token causal
stream `[s_0, m_0, …, s_{H-1}, m_{H-1}]` with type embeddings and reads the
last macro position. Macros enter only via softmax-attended K/V (HWM's PushT
convention); they can no longer multiplicatively rescale state activations.
Single-step teacher-forcing loss (γ_tf=1, γ_roll=0; HWM's PushT setting).
```bash
export STABLEWM_HOME=/home/.stable-wm
python train_hl.py --config-name=lewm_hl_v6
# 1-GPU: python train_hl.py --config-name=lewm_hl_v6 trainer.devices=1 trainer.strategy=auto
# ckpts -> $STABLEWM_HOME/hjepa_v6/hjepa_v6_epoch_<N>_object.ckpt
```

**Experimental copy** — `eval_unified_exp.py` + `config/eval/pusht_unified_exp.yaml`
mirror the unified eval for tuning without touching the canonical files. Points
at `hjepa_v5/hjepa_v5_epoch_81_object.ckpt` (the v5 ckpt has the `hl_projector`
that `eval_unified.encode_hl_proj` requires). On the seed=42 single-episode case
the canonical config fails (HL drifts 188→201). Setting
`unified.final_dist_threshold` to a very large value (e.g. `10000`) forces the
flat LL goal-anchored cost from step 0; the LL predictor closes the loop and
the episode succeeds (100%, distance 188 → 0.4 → 0.12).

```bash
python eval_unified_exp.py        # uses pusht_unified_exp.yaml
```

**Sweep across checkpoints** — `eval_sweep.py` reuses `pusht_unified.yaml` and runs
the unified eval on every Nth epoch checkpoint in `${STABLEWM_HOME}/${subdir}_v2/`.
Saves `pusht_unified_sweep.{csv,png}` alongside the checkpoints.

```bash
python eval_sweep.py sweep_stride=3
```

**Eval (flat closed-loop, non-hierarchical)** — `eval.py` with
`config/eval/pusht_cl.yaml`. Same code path as the open-loop flat eval; the
config exposes a top-level `replan_every` (LL tokens between replans, aliased
into `plan_config.receding_horizon`) so you can sweep open- vs closed-loop
with a single knob. Use this as the apples-to-apples baseline for the
hierarchical evals.

```bash
python eval.py --config-name=pusht_cl.yaml policy=pusht/lewm replan_every=5
```


Caveats:
- LL is fully frozen during HL training (`HierarchicalJEPA` calls `requires_grad_(False)` on the LL JEPA and forces `eval()` even in train mode).
- HL CEM samples macro actions from a fitted Gaussian; the macro distribution at eval can drift from MAE outputs on real action sequences seen during training. If HL planning underperforms, consider initializing the HL CEM mean/var from MAE statistics on the train set.

### `eval_hl.py` planning speedups

`eval_hl.py` is dominated (~93% of wall time per episode) by the LL CEM, which
calls the LL predictor (depth-6 transformer) thousands of times per episode at
tiny shapes (BS≈300, T=3). Two cheap, semantics-preserving changes net ~1.5×
end-to-end on an L4 (40.6s → ~27s/episode at the default config):
- LL predictor weights cast to bf16 at load time (eliminates the per-call
  `autocast` fp32↔bf16 copy kernels — ~10% of CUDA time in profile).
- `EFFICIENT_ATTENTION` SDPA backend forced for the eval loop (flash-attn's
  per-kernel setup dominates at T=3; on these shapes the efficient kernel is
  ~30% faster).
The autoregressive rollout in `hjepa.ll_rollout_from_emb` was updated to feed
bf16 tensors directly to the predictor and cast back only for the fp32 HLE/cost.
GPU is already saturated at BS=300 (throughput is flat from BS=300 to 30k), so
the only "free" wins are the bf16 + efficient-attn changes above; the remaining
time is real GPU compute in the predictor.

Parallel episodes via `eval.num_eval` give an additional throughput win.
`eval_hl_parallel.py` sweeps the value (`+num_evals=N`); on L4:
`num_eval=1 → 27.3s/ep, =4 → 22.4 (1.22×), =8 → 23.3, =16 → 22.4, =32 → 19.6 (1.40×), =64 → 24.0`.
The sweet spot is **`num_eval=32`**. Combined with the bf16 + efficient-attn
changes, total speedup is ~2.1× per episode (40.6 → 19.6 s). Past 32 the
per-episode time degrades (CEM BS = 32×300 = 9600 is past the L4 throughput
plateau).

## Pixel decoder (LPIPS)

`train_decoder.py` trains a convolutional decoder that maps the 192-d LeWM CLS
embeddings back to 224×224 RGB frames. It reads `pusht_expert_train_emb.h5`
(pre-encoded embeddings) for inputs and `pusht_expert_train.h5` for the target
pixels, paired by global step index. Architecture follows the DINO-WM /
JEPA-WMs convention: `Linear → 7×7×512 → 5× (Upsample + Conv + GN + GELU) →
tanh → 3×224×224`. Loss is `mse_weight * MSE + lpips_weight * LPIPS(VGG)` in
[-1, 1] image space.

```bash
uv pip install lpips moviepy imageio-ffmpeg hdf5plugin
export STABLEWM_HOME=/home/.stable-wm
python train_decoder.py                                  # uses config/train/decoder.yaml
# checkpoints -> $STABLEWM_HOME/decoder/decoder_epoch_<N>_object.ckpt
```

**`space` flag** (`pre` | `post`, default `post`). The stored `f["emb"]` is
**pre-projector** (raw ViT CLS), while the LL predictor outputs in **post-projector**
space (the LL training loss target = `ll.projector(encoder(x))`). If you want to
decode rollout frames cleanly, set `space=post` so the decoder is trained on
`ll.projector(emb)`. The default config uses `space=post` and writes to
`$STABLEWM_HOME/decoder_v4/`.

Config: `config/train/decoder.yaml`. Override any field via Hydra CLI, e.g.:
```bash
python train_decoder.py trainer.epochs=50 loader.batch_size=256 \
    loss.lpips_weight=1.0 loss.mse_weight=0.5 wandb.enabled=False
```

WandB logging is on by default (matches the `train_hl.py` config: entity
`wms_at_iitb`, project `hwm`, run id/name `decoder`, `resume=allow`). Logs
per-step `train/{loss,mse,lpips,lr}`, per-epoch `val/{mse,lpips}`, and a single
`val/traj` mp4 of one randomly chosen episode — fixed across epochs for visual
comparison (ep picked at startup via `seed`, capped at `log.traj_max_len`
frames, encoded at `log.traj_fps`). Requires `moviepy` + `imageio-ffmpeg`.

## HL → pixels decoder

`train_hl_decoder.py` trains a small MLP that maps the **96-d HL state** back to
the **192-d LL CLS** that the frozen pixel decoder consumes. The `hl_space` flag
selects which HL state is the input: `pre` (= `encode_hl(post_emb)`, the HLE
output) or `post` (= `encode_hl_proj(post_emb)`, post-`hl_projector`). Use
`hl_space=post` paired with a `space=post` pixel decoder to decode
**predicted subgoals at planning time** (which live in the post-projector / loss
space). Default config uses `hl_space=post`, writes to `$STABLEWM_HOME/hl_decoder_v2/`.
The two frozen modules are chained so the LPIPS+MSE loss is computed in pixel
space and gradients flow back to the HL decoder only.

```
pre_emb  ─(frozen ll.projector)─►  post_emb  ─(frozen encode_hl)─►  HL state (96)
                                                                          │
                                                                          ▼
                                                              [trainable HLDecoder]
                                                                          │
                                                                          ▼
target pix ◄── LPIPS+MSE ── (frozen decoder_v2) ◄── predicted pre_emb (192)
```

Required paths in `config/train/hl_decoder.yaml`:
- `hjepa_ckpt` — the pickled `HierarchicalJEPA` saved by `train_hl.py` (default
  `$STABLEWM_HOME/hjepa/hjepa_object.ckpt`).
- `pixel_decoder_weights` — state_dict from `train_decoder.py` (default
  `$STABLEWM_HOME/decoder_v2/decoder_v2_weights.ckpt`).
- `pixel_decoder.{emb_dim,base,init_hw}` must match what decoder_v2 was trained
  with (default `192/256/7`).

```bash
export STABLEWM_HOME=/home/.stable-wm
# single GPU
python train_hl_decoder.py
# 2 GPUs
torchrun --nproc_per_node=2 train_hl_decoder.py
```

Checkpoints land in `$STABLEWM_HOME/hl_decoder/hl_decoder_epoch_<N>_weights.ckpt`
(state_dict only — reload by instantiating `HLDecoder(**cfg.model)` and calling
`load_state_dict`).

### Visualizing the chain

`viz_hl_decoder.py` writes a side-by-side mp4 of one episode:
`[ground truth] | [decoder_v2 only] | [HLDecoder + decoder_v2]`.

```bash
export STABLEWM_HOME=/home/.stable-wm
python viz_hl_decoder.py                       # random episode -> $STABLEWM_HOME/hl_decoder/viz.mp4
python viz_hl_decoder.py --episode 1234 --out /tmp/ep1234.mp4
```

Picks the highest-epoch `hjepa_v5_epoch_*_object.ckpt` automatically. The HL/
decoder_v2 architecture flags must match what was used at training time
(defaults in the script already match the current configs).

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
