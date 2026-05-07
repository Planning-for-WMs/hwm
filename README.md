
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

- `HLEncoder` (MLP): LL CLS emb (192) → HL state (96)
- `HLPredictor` (MLP): `(s_hl, macro_a)` → next HL state at *K* LL-frames ahead (default `K=5`)
- `MacroActionEncoder`: 2-layer transformer (4 heads, dim 128) over K LL action tokens; CLS → MLP → macro action (16-d)

Loss: `||HLP(s_hl_t, MAE(a_t..t+K-1)) - HLE(LL_emb_{t+K})||²` + `λ * SIGReg(HL embeddings)`.

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

Caveats:
- LL is fully frozen during HL training (`HierarchicalJEPA` calls `requires_grad_(False)` on the LL JEPA and forces `eval()` even in train mode).
- HL CEM samples macro actions from a fitted Gaussian; the macro distribution at eval can drift from MAE outputs on real action sequences seen during training. If HL planning underperforms, consider initializing the HL CEM mean/var from MAE statistics on the train set.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
