"""Collect a heldout PushT dataset using the swm WeakPolicy.

Output goes to $STABLEWM_HOME/<name>.h5 (same format as pusht_expert_train.h5),
so eval can use it via `eval.dataset_name=<name>`.
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pusht_weak_heldout")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--max-episode-steps", type=int, default=100)
    p.add_argument("--dist-constraint", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        history_size=1,
        frame_skip=1,
    )
    policy = WeakPolicy(dist_constraint=args.dist_constraint, seed=args.seed)
    world.set_policy(policy)
    world.record_dataset(
        dataset_name=args.name, episodes=args.episodes, seed=args.seed
    )


if __name__ == "__main__":
    main()
