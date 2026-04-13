"""
Run the best evolved GP controller on BipedalWalker-v3.

Usage:
    uv run deap_demo/run_best.py
    uv run deap_demo/run_best.py --episodes 5 --render          # visual
    uv run deap_demo/run_best.py --no-render --episodes 10      # headless benchmark
"""

import argparse
import os
import pickle

import numpy as np
import gymnasium as gym
from deap import gp

# ── locate saved model ────────────────────────────────────────
_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_DIR, "best_model_v2", "best_individual.pkl")

# ── must import bipedal_gp so creator / pset are registered ──
# (pickle needs the same class definitions that were used to save)
import importlib.util, sys

_spec = importlib.util.spec_from_file_location(
    "bipedal_gp", os.path.join(_DIR, "bipedal_gp.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["bipedal_gp"] = _mod
_spec.loader.exec_module(_mod)

pset = _mod.pset
N_ACTIONS = _mod.N_ACTIONS
ACTION_NAMES = _mod.ACTION_NAMES
MAX_STEPS = _mod.MAX_STEPS


def load_best():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}.\n"
            "Run bipedal_gp.py first to evolve and save a model."
        )
    with open(MODEL_PATH, "rb") as f:
        ind = pickle.load(f)
    return ind


def run(individual, render: bool, n_episodes: int, seed: int = 0):
    fns = [gp.compile(individual[i], pset) for i in range(N_ACTIONS)]

    render_mode = "human" if render else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)

    episode_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(MAX_STEPS):
            try:
                action = np.clip([float(fn(*obs)) for fn in fns], -1.0, 1.0).astype(
                    np.float32
                )
            except Exception:
                action = np.zeros(N_ACTIONS, dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break

        episode_rewards.append(total)
        print(f"  Episode {ep + 1:>2} | reward = {total:8.2f}")

    env.close()
    mean_r = float(np.mean(episode_rewards))
    print(f"\n  Mean reward over {n_episodes} episodes: {mean_r:.2f}")
    return episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Run the best GP controller.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ind = load_best()
    fitness = ind.fitness.values[0]
    print(f"\nLoaded best individual  (training reward = {fitness:.2f})")
    print(f"Running {args.episodes} episode(s)  |  render={args.render}\n")

    print("  Symbolic policy:")
    for i in range(N_ACTIONS):
        print(f"    action[{i}] ({ACTION_NAMES[i]:8s}) = {ind[i]}")
    print()

    run(ind, render=args.render, n_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
