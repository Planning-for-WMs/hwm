"""eval_verify.py — visualize how close the model's macro actions are to the expert's.

Runs the same single-episode planning eval as `eval_unified.py`. After the episode:
  - Encodes the **expert** action sequence (from the H5 dataset, same episode +
    start_step as the eval) into a trajectory of macro actions via MAE.
  - Encodes the **agent**'s executed actions (whatever the policy actually did)
    into a trajectory of macro actions via MAE.
  - Saves an interactive plotly HTML 3D viewer of both trajectories.

You can orbit/zoom in your browser. Both trajectories share the same MAE so
proximity in 3D = "model selects expert-like macros at planning time."

Color scheme:
  - Expert trajectory line  : blue
  - Expert start / end      : green (circle) / red (square)
  - Agent trajectory line   : orange
  - Agent  start / end      : magenta (circle) / black (square)

Args (Hydra, same as eval_unified):
  python eval_verify.py policy_ckpt=$STABLEWM_HOME/hjepa_v8/hjepa_v8_epoch_25_object.ckpt
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import h5py
import hydra
import numpy as np
import plotly.graph_objects as go
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval_unified import UnifiedPolicy, img_transform, get_dataset


class VerifyPolicy(UnifiedPolicy):
    """UnifiedPolicy + records every executed env-step action (raw 2-d)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._executed_actions: list[np.ndarray] = []

    def set_env(self, env):
        super().set_env(env)
        self._executed_actions = []

    def get_action(self, info_dict, **kwargs):
        action = super().get_action(info_dict, **kwargs)
        # `action` may be un-normalized via self.process["action"] inside the
        # parent; record what was just returned (raw env-step action, 2-d for
        # PushT). We make a copy to decouple from the env's internal buffer.
        self._executed_actions.append(np.array(action, dtype=np.float32).reshape(-1))
        return action


def chunk_actions_to_ll_tokens(raw_actions, action_block):
    """raw_actions: (N, 2)  ->  (N // action_block, action_block * 2)
    Each LL token packs `action_block` env-step actions."""
    n_full = (len(raw_actions) // action_block) * action_block
    arr = np.asarray(raw_actions[:n_full]).reshape(-1, action_block * 2)
    return arr


def encode_macros(mae, ll_tokens, K_macro, device):
    """Encode (num_ll_tokens, action_dim_raw) into a trajectory of macros.
    Slides a non-overlapping window of K_macro LL tokens; each window -> macro.
    Returns (num_macros, macro_action_dim) numpy array."""
    n_macros = len(ll_tokens) // K_macro
    if n_macros == 0:
        return np.zeros((0, mae.head[-1].out_features
                         if not isinstance(mae.head[-1], torch.nn.Tanh)
                         else mae.head[-2].out_features), dtype=np.float32)
    chunks = ll_tokens[:n_macros * K_macro].reshape(n_macros, K_macro, -1)
    t = torch.from_numpy(chunks).to(device).float()
    with torch.inference_mode():
        m = mae(t)
    return m.float().cpu().numpy()


def make_figure(expert_xyz, agent_xyz, title="MAE macro trajectories"):
    """Build a plotly 3D figure with two trajectories and start/end markers."""
    fig = go.Figure()

    def _add(xyz, line_color, start_color, end_color, name):
        if xyz.shape[0] == 0:
            return
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="lines+markers",
            line=dict(color=line_color, width=4),
            marker=dict(size=3, color=line_color),
            name=f"{name} trajectory",
        ))
        fig.add_trace(go.Scatter3d(
            x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]],
            mode="markers",
            marker=dict(size=10, color=start_color, symbol="circle",
                        line=dict(width=2, color="black")),
            name=f"{name} start",
        ))
        fig.add_trace(go.Scatter3d(
            x=[xyz[-1, 0]], y=[xyz[-1, 1]], z=[xyz[-1, 2]],
            mode="markers",
            marker=dict(size=10, color=end_color, symbol="square",
                        line=dict(width=2, color="black")),
            name=f"{name} end",
        ))

    _add(expert_xyz, line_color="royalblue",
         start_color="lime", end_color="red", name="expert")
    _add(agent_xyz, line_color="orange",
         start_color="magenta", end_color="black", name="agent")

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="macro dim 0",
            yaxis_title="macro dim 1",
            zaxis_title="macro dim 2",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_unified")
def run(cfg: DictConfig):
    # Single-episode visualization only: force num_envs and num_eval to 1.
    # (pusht_unified.yaml interpolates world.num_envs = eval.num_eval; override
    # both explicitly in case the user passed eval.num_eval on the CLI.)
    OmegaConf.set_struct(cfg, False)
    cfg.eval.num_eval = 1
    cfg.world.num_envs = 1
    cfg.world.max_episode_steps = int(cfg.get("max_episode_steps_fixed", 1000))
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        proc = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        proc.fit(col_data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc

    device = torch.device("cuda")
    hjepa = torch.load(cfg.policy_ckpt, map_location="cpu", weights_only=False)
    hjepa = hjepa.to(device).eval()
    hjepa.requires_grad_(False)
    hjepa.ll.predictor.to(torch.bfloat16)

    unified_cfg = OmegaConf.to_container(cfg.unified)
    unified_cfg["seed"] = int(cfg.seed)
    policy = VerifyPolicy(
        hjepa=hjepa, cfg=unified_cfg, action_block=int(cfg.action_block),
        eval_budget=int(cfg.eval.eval_budget),
        process=process, transform=transform,
    )

    # episode selection (same as eval_unified, but force num_eval=1)
    def get_episodes_length(ds, eps):
        ep = ds.get_col_data(col_name); st = ds.get_col_data("step_idx")
        return np.array([np.max(st[ep == e]) + 1 for e in eps])
    ep_len = get_episodes_length(dataset, ep_indices)
    max_start = ep_len - cfg.eval.goal_offset_steps - 1
    start_dict = {ep: max_start[i] for i, ep in enumerate(ep_indices)}
    max_per_row = np.array([start_dict[e] for e in dataset.get_col_data(col_name)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_per_row)[0]
    g = np.random.default_rng(cfg.seed)
    sel = np.sort(valid[g.choice(len(valid) - 1, size=1, replace=False)])
    eval_eps = dataset.get_row_data(sel)[col_name].tolist()
    eval_starts = dataset.get_row_data(sel)["step_idx"].tolist()
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    home = os.environ.get("STABLEWM_HOME", "/home/.stable-wm")
    ckpt_stem = Path(cfg.policy_ckpt).stem.replace("_object", "")
    out_dir = Path(home) / "eval_verify" / ckpt_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- run the single-episode eval ---
    world.set_policy(policy)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_starts,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_eps,
            callables=callables,
            video_path=out_dir,
        )
    print(f"metrics: {metrics}")

    # --- agent macros (from executed env-step actions) ---
    K_macro = int(cfg.unified.K_macro)
    action_block = int(cfg.action_block)
    agent_raw = np.stack(policy._executed_actions)                # (N_env, 2)
    print(f"[verify] agent executed {len(agent_raw)} env-step actions")
    agent_ll = chunk_actions_to_ll_tokens(agent_raw, action_block)
    agent_macros = encode_macros(hjepa.mae, agent_ll, K_macro, device)
    print(f"[verify] agent macros: {agent_macros.shape}")

    # --- expert macros (from H5 dataset, same episode + start_step) ---
    pix_path = Path(home) / f"{cfg.eval.dataset_name}.h5"
    ep_i, s0 = int(eval_eps[0]), int(eval_starts[0])
    n_env_steps = int(cfg.eval.eval_budget)                       # match agent horizon
    with h5py.File(pix_path, "r") as fp:
        ep_off = fp[col_name][:]
        row0 = int(np.nonzero(ep_off == ep_i)[0].min())
        expert_actions = fp["action"][row0 + s0 : row0 + s0 + n_env_steps]
    expert_actions = np.asarray(expert_actions, dtype=np.float32)
    print(f"[verify] expert episode {ep_i} start_step {s0}: {expert_actions.shape}")
    if expert_actions.shape[-1] != 2:
        # some datasets store packed action tokens (10-d); chunk-detect and unpack
        if expert_actions.ndim == 2 and expert_actions.shape[-1] == action_block * 2:
            # already packed per LL token; flatten back to env-step actions
            expert_actions = expert_actions.reshape(-1, 2)
    expert_ll = chunk_actions_to_ll_tokens(expert_actions, action_block)
    expert_macros = encode_macros(hjepa.mae, expert_ll, K_macro, device)
    print(f"[verify] expert macros: {expert_macros.shape}")

    # --- build interactive HTML ---
    title = (f"MAE macros — ep={ep_i} start={s0}  K_macro={K_macro}  "
             f"expert={len(expert_macros)} | agent={len(agent_macros)}")
    fig = make_figure(expert_macros, agent_macros, title=title)
    out_html = out_dir / "macros_3d.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"[verify] wrote interactive viewer -> {out_html}")
    # also save as PNG snapshot for quick previewing (optional, requires kaleido)
    try:
        fig.write_image(str(out_dir / "macros_3d.png"), width=1000, height=800)
        print(f"[verify] wrote static snapshot -> {out_dir / 'macros_3d.png'}")
    except Exception:
        pass


if __name__ == "__main__":
    run()
