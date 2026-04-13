"""
GP Controller v4 — BipedalWalker-v3  (cgp_python / CGP++ port)
================================================================
Uses the cgp_python library (Python port of CGP++ by Kalkreuth)
instead of DEAP or a hand-rolled CGP.

Individual: one CGP graph, 24 inputs → 4 outputs (one per action)
Fitness   : mean episode reward (maximising, negated internally)
EA        : (1+λ) with neutral genetic drift  (default)
         or (μ+λ) with optional crossover

Run:
    uv run deap_demo/bipedal_cgp_plusplus.py
    uv run deap_demo/bipedal_cgp_plusplus.py --algorithm mupluslambda --render
"""

import argparse
import json
import math
import os
import pickle
import sys

import numpy as np
import gymnasium as gym

# ── locate cgp_python package ─────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cgp_python import (
    Parameters, build,
    ONE_PLUS_LAMBDA, MU_PLUS_LAMBDA,
    PROBABILISTIC_POINT_MUTATION, SINGLE_ACTIVE_GENE_MUTATION,
    MathematicalFunctions,
)
from cgp_python.representation.individual import Individual
from cgp_python.evaluator.evaluator       import Evaluator

BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model_v4")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_OBS     = 24
N_ACTIONS = 4
N_EPISODES = 5
MAX_STEPS  = 1600
RANDOM_SEED = 42

ACTION_NAMES = ["hip_L", "knee_L", "hip_R", "knee_R"]


# ─────────────────────────────────────────────
# FITNESS CALLBACK
# ─────────────────────────────────────────────
def make_eval_fn(n_episodes: int = N_EPISODES, render: bool = False):
    """
    Returns a fitness callback compatible with cgp_python.build().
    Fitness = mean episode reward (negated because cgp_python minimises).
    """
    render_mode = "human" if render else None

    def eval_fn(ind: Individual, evaluator: Evaluator) -> float:
        rewards = []
        env = gym.make("BipedalWalker-v3", render_mode=render_mode)
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=RANDOM_SEED + ep)
            total  = 0.0
            for _ in range(MAX_STEPS):
                out = evaluator.evaluate(ind, list(obs), iterative=True)
                action = np.clip(out, -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
        env.close()
        mean_reward = float(np.mean(rewards))
        # Negate: cgp_python minimises, we want to maximise reward
        return -mean_reward

    return eval_fn


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────
def save_champion(champion: Individual, fitness_reward: float) -> None:
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    with open(os.path.join(BEST_MODEL_DIR, "champion.pkl"), "wb") as f:
        pickle.dump({
            "genome":         champion.genome,
            "fitness_reward": fitness_reward,
            "active_nodes":   champion.active_nodes,
            "expressions":    champion.expressions,
        }, f)

    meta = {
        "fitness_reward":  fitness_reward,
        "n_active_nodes":  champion.num_active_nodes(),
        "expressions":     champion.expressions,
    }
    with open(os.path.join(BEST_MODEL_DIR, "champion.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved → {BEST_MODEL_DIR}/")
    print(f"    champion.pkl   (genome + metadata)")
    print(f"    champion.json  (expressions + reward)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",       default="onepluslambda",
                        choices=["onepluslambda", "mupluslambda"])
    parser.add_argument("--max-generations", type=int, default=500)
    parser.add_argument("--lambda",          type=int, default=4,  dest="lam")
    parser.add_argument("--mu",              type=int, default=1)
    parser.add_argument("--nodes",           type=int, default=80)
    parser.add_argument("--levels-back",     type=int, default=20)
    parser.add_argument("--mutation-rate",   type=float, default=0.04)
    parser.add_argument("--mutation-type",   default="probabilistic",
                        choices=["probabilistic", "single_active"])
    parser.add_argument("--episodes",        type=int, default=N_EPISODES)
    parser.add_argument("--render",          action="store_true")
    args = parser.parse_args()

    mut_type = (SINGLE_ACTIVE_GENE_MUTATION
                if args.mutation_type == "single_active"
                else PROBABILISTIC_POINT_MUTATION)

    algo_flag = ONE_PLUS_LAMBDA if args.algorithm == "onepluslambda" else MU_PLUS_LAMBDA

    params = Parameters(
        num_inputs         = N_OBS,
        num_outputs        = N_ACTIONS,
        num_function_nodes = args.nodes,
        max_arity          = 2,
        levels_back        = args.levels_back,
        mu                 = args.mu,
        lambda_            = args.lam,
        max_generations    = args.max_generations,
        mutation_rate      = args.mutation_rate,
        mutation_operators = [mut_type],
        neutral_genetic_drift = True,
        minimizing_fitness = True,    # we negate reward → minimise
        ideal_fitness      = -300.0,  # episode reward ≥ 300 → solved
        report_interval    = 10,
        algorithm          = algo_flag,
        global_seed        = RANDOM_SEED,
        evaluate_expression = True,
    )

    functions = MathematicalFunctions(extended=True)

    print(__doc__)
    print(f"  Algorithm : {args.algorithm}  (1+{args.lam})")
    print(f"  Nodes     : {args.nodes}  levels_back={args.levels_back}")
    print(f"  Mutation  : {args.mutation_type}  rate={args.mutation_rate}")
    print(f"  Episodes/eval: {args.episodes}  MaxSteps={MAX_STEPS}\n")

    eval_fn = make_eval_fn(n_episodes=args.episodes)
    algo    = build(params, eval_fn, functions=functions, seed=RANDOM_SEED)

    evals, best_loss = algo.evolve()

    best_reward = -best_loss
    champion    = algo.champion()

    print(f"\n{'='*62}")
    print(f"  EVOLUTION COMPLETE")
    print(f"  Fitness evaluations : {evals}")
    print(f"  Best reward         : {best_reward:.2f}")
    print(f"  Active nodes        : {champion.num_active_nodes()}")
    print(f"{'='*62}")

    if champion.expressions:
        print("\n  Symbolic expressions (one per action output):")
        for i, expr in enumerate(champion.expressions):
            name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"out_{i}"
            print(f"    action[{i}] ({name:8s}) = {expr}")

    save_champion(champion, best_reward)

    if args.render:
        print("\nRunning champion with rendering...")
        render_fn = make_eval_fn(n_episodes=1, render=True)
        from cgp_python import build as _build, Evaluator as _Eval, Species, Fitness, Mutation
        # Re-use the existing evaluator from the algorithm
        _algo = algo
        evaluator = _algo.evaluator
        reward = -render_fn(champion, evaluator)
        print(f"Episode reward: {reward:.2f}")


if __name__ == "__main__":
    main()
