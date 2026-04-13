"""
GP Controller v3 — BipedalWalker-v3  (Cartesian Genetic Programming)
======================================================================
CGP represents programs as a fixed-size directed acyclic graph over a
node grid. Key advantages over tree GP:

  - No bloat: genome is a fixed-length integer array
  - Neutral mutations: inactive genes mutate freely without fitness cost,
    providing a reservoir of pre-adapted sub-expressions
  - Implicit reuse: multiple outputs can share interior nodes

Architecture:
  Genome   = N_ROWS × N_COLS interior nodes + N_ACTIONS output pointers
  Each node = [function_gene, input_0, input_1]
  Outputs  = 4 pointers into input/interior node values

Fitness = mean episode reward — same objective as v1/v2.
EA      = (1+λ) evolution strategy with stagnation-triggered restarts.

Run:
    uv run deap_demo/bipedal_gp_v3.py
    uv run deap_demo/bipedal_gp_v3.py --max-generations 1000 --render
"""

import argparse
import json
import math
import os
import pickle

import numpy as np
import gymnasium as gym

BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model_v3")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_OBS     = 24
N_ACTIONS = 4
N_ROWS    = 4
N_COLS    = 20          # 4×20 = 80 interior nodes
LEVELS_BACK = 10        # each node reaches back at most 10 columns

N_NODES       = N_ROWS * N_COLS   # 80
GENES_PER_NODE = 3                 # [func, in0, in1]
GENOME_LEN    = N_NODES * GENES_PER_NODE + N_ACTIONS  # 244

N_GEN         = 500
N_EPISODES    = 5
MAX_STEPS     = 1600
RANDOM_SEED   = 42

MU            = 1     # parents kept
LAMBDA        = 8     # offspring per generation
MUTATION_RATE = 0.04

STAGNATION_LIMIT = 50

ACTION_NAMES = ["hip_L", "knee_L", "hip_R", "knee_R"]


# ─────────────────────────────────────────────
# FUNCTION SET
# All functions have signature f(a, b) — b is ignored for unary ops.
# ─────────────────────────────────────────────
def _safe_div(a, b):  return a / b if abs(b) > 1e-6 else 0.0
def _safe_sqrt(a, _): return math.sqrt(abs(a))
def _safe_log(a, _):  return math.log(abs(a) + 1e-6)
def _safe_exp(a, _):  return math.exp(max(-10.0, min(10.0, a)))
def _tanh(a, _):      return math.tanh(a)
def _sin(a, _):       return math.sin(a)
def _cos(a, _):       return math.cos(a)
def _abs(a, _):       return abs(a)
def _neg(a, _):       return -a
def _c_one(a, _):     return 1.0
def _c_half(a, _):    return 0.5
def _c_neg1(a, _):    return -1.0

FUNCTIONS = [
    lambda a, b: a + b,  # 0  add
    lambda a, b: a - b,  # 1  sub
    lambda a, b: a * b,  # 2  mul
    _safe_div,            # 3  div
    _tanh,                # 4  tanh
    _sin,                 # 5  sin
    _cos,                 # 6  cos
    _abs,                 # 7  abs
    _neg,                 # 8  neg
    _safe_sqrt,           # 9  sqrt
    _safe_log,            # 10 log
    _safe_exp,            # 11 exp
    lambda a, b: max(a, b),  # 12 max
    lambda a, b: min(a, b),  # 13 min
    _c_one,               # 14  1.0
    _c_half,              # 15  0.5
    _c_neg1,              # 16 -1.0
]
N_FUNCS = len(FUNCTIONS)

FUNC_NAMES = [
    "add", "sub", "mul", "div", "tanh", "sin", "cos", "abs", "neg",
    "sqrt", "log", "exp", "max", "min", "1.0", "0.5", "-1.0",
]


# ─────────────────────────────────────────────
# GENOME HELPERS
# ─────────────────────────────────────────────
def _col(node_idx: int) -> int:
    return node_idx // N_ROWS


def _valid_inputs(node_idx: int) -> list[int]:
    """Source indices reachable from interior node node_idx."""
    col = _col(node_idx)
    first_col = max(0, col - LEVELS_BACK)
    interior_start = N_OBS + first_col * N_ROWS
    interior_end   = N_OBS + col * N_ROWS
    return list(range(N_OBS)) + list(range(interior_start, interior_end))


def random_genome(rng: np.random.Generator) -> np.ndarray:
    g = np.zeros(GENOME_LEN, dtype=np.int32)
    for k in range(N_NODES):
        off = k * GENES_PER_NODE
        sources = _valid_inputs(k)
        g[off]     = rng.integers(N_FUNCS)
        g[off + 1] = int(rng.choice(sources))
        g[off + 2] = int(rng.choice(sources))
    all_src = list(range(N_OBS + N_NODES))
    for i in range(N_ACTIONS):
        g[N_NODES * GENES_PER_NODE + i] = int(rng.choice(all_src))
    return g


def active_nodes(g: np.ndarray) -> set[int]:
    """Interior node indices (0-based) that are on the active path."""
    frontier: set[int] = set()
    for i in range(N_ACTIONS):
        src = int(g[N_NODES * GENES_PER_NODE + i])
        if src >= N_OBS:
            frontier.add(src - N_OBS)

    active: set[int] = set()
    while frontier:
        k = frontier.pop()
        if k in active:
            continue
        active.add(k)
        off = k * GENES_PER_NODE
        for j in range(1, GENES_PER_NODE):
            src = int(g[off + j])
            if src >= N_OBS:
                frontier.add(src - N_OBS)
    return active


def compile_genome(g: np.ndarray):
    """Compile genome to a callable f(*obs) → list[float] of N_ACTIONS values."""
    order = sorted(active_nodes(g))

    def f(*obs):
        # vals[0..N_OBS-1] = observation, vals[N_OBS..] = interior nodes
        vals = list(obs) + [0.0] * N_NODES
        for k in order:
            off = k * GENES_PER_NODE
            fn   = FUNCTIONS[int(g[off])]
            in0  = vals[int(g[off + 1])]
            in1  = vals[int(g[off + 2])]
            try:
                r = fn(in0, in1)
                vals[N_OBS + k] = r if math.isfinite(r) else 0.0
            except Exception:
                vals[N_OBS + k] = 0.0
        return [
            vals[int(g[N_NODES * GENES_PER_NODE + i])] for i in range(N_ACTIONS)
        ]

    return f


# ─────────────────────────────────────────────
# MUTATION  — point mutation, one gene at a time
# ─────────────────────────────────────────────
def mutate(g: np.ndarray, rng: np.random.Generator, rate: float = MUTATION_RATE) -> np.ndarray:
    child = g.copy()
    for k in range(N_NODES):
        off = k * GENES_PER_NODE
        if rng.random() < rate:
            child[off] = rng.integers(N_FUNCS)
        sources = _valid_inputs(k)
        for j in range(1, GENES_PER_NODE):
            if rng.random() < rate:
                child[off + j] = int(rng.choice(sources))
    all_src = list(range(N_OBS + N_NODES))
    for i in range(N_ACTIONS):
        if rng.random() < rate:
            child[N_NODES * GENES_PER_NODE + i] = int(rng.choice(all_src))
    return child


# ─────────────────────────────────────────────
# FITNESS
# ─────────────────────────────────────────────
def evaluate(g: np.ndarray, n_episodes: int = N_EPISODES, render: bool = False) -> float:
    f = compile_genome(g)
    render_mode = "human" if render else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=RANDOM_SEED + ep)
        total = 0.0
        for _ in range(MAX_STEPS):
            try:
                actions = f(*obs)
                action = np.clip(actions, -1.0, 1.0).astype(np.float32)
            except Exception:
                action = np.zeros(N_ACTIONS, dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


# ─────────────────────────────────────────────
# (1 + λ)  EVOLUTION STRATEGY
# ─────────────────────────────────────────────
def evolve(rng: np.random.Generator, max_generations: int) -> tuple[np.ndarray, float]:
    parent         = random_genome(rng)
    parent_fitness = evaluate(parent)
    best_genome    = parent.copy()
    best_fitness   = parent_fitness
    stag           = 0

    print(f"  Gen    0 | fitness = {parent_fitness:8.2f}")

    for gen in range(1, max_generations + 1):
        offspring = [mutate(parent, rng) for _ in range(LAMBDA)]
        fits      = [evaluate(g) for g in offspring]

        best_idx = int(np.argmax(fits))
        # neutral selection: accept if offspring is at least as good
        if fits[best_idx] >= parent_fitness:
            parent         = offspring[best_idx]
            parent_fitness = fits[best_idx]

        if parent_fitness > best_fitness + 0.01:
            best_fitness = parent_fitness
            best_genome  = parent.copy()
            stag = 0
        else:
            stag += 1

        if gen % 10 == 0:
            n_active = len(active_nodes(parent))
            print(
                f"  Gen {gen:>4} | fitness = {parent_fitness:8.2f}  "
                f"best = {best_fitness:8.2f}  stag = {stag:>3}  "
                f"active = {n_active}"
            )

        if stag >= STAGNATION_LIMIT:
            print(f"  ↺ stagnant at gen {gen} — restarting from best")
            parent         = best_genome.copy()
            parent_fitness = best_fitness
            stag = 0

        if best_fitness >= 300.0:
            print(f"  ✓ solved at gen {gen}")
            break

    return best_genome, best_fitness


# ─────────────────────────────────────────────
# DESCRIBE
# ─────────────────────────────────────────────
def describe(g: np.ndarray) -> None:
    active = active_nodes(g)
    print(f"  Active nodes : {len(active)} / {N_NODES}")
    for i in range(N_ACTIONS):
        src = int(g[N_NODES * GENES_PER_NODE + i])
        if src < N_OBS:
            label = f"obs[{src}]"
        else:
            k  = src - N_OBS
            fn = FUNC_NAMES[int(g[k * GENES_PER_NODE])]
            label = f"node_{k}[{fn}]"
        print(f"  action[{i}] ({ACTION_NAMES[i]:8s}) ← {label}")


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save(g: np.ndarray, fitness: float) -> None:
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    with open(os.path.join(BEST_MODEL_DIR, "best_genome.pkl"), "wb") as fh:
        pickle.dump({"genome": g, "fitness": fitness}, fh)

    meta = {
        "fitness": fitness,
        "n_active_nodes": len(active_nodes(g)),
        "config": {
            "N_OBS": N_OBS, "N_ACTIONS": N_ACTIONS,
            "N_ROWS": N_ROWS, "N_COLS": N_COLS,
            "LEVELS_BACK": LEVELS_BACK,
        },
    }
    with open(os.path.join(BEST_MODEL_DIR, "best_genome.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  Model saved → {BEST_MODEL_DIR}/")
    print(f"    best_genome.pkl   (reload with pickle + compile_genome)")
    print(f"    best_genome.json  (metadata)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-generations", type=int, default=N_GEN)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    print(__doc__)
    print(f"  n_inputs={N_OBS}  n_outputs={N_ACTIONS}")
    print(f"  Grid: {N_ROWS}×{N_COLS} = {N_NODES} nodes  levels_back={LEVELS_BACK}")
    print(f"  EA: (1+{LAMBDA})  mutation_rate={MUTATION_RATE}")
    print(f"  Max gen={args.max_generations}  Episodes/eval={N_EPISODES}  MaxSteps={MAX_STEPS}\n")

    rng = np.random.default_rng(RANDOM_SEED)
    best_g, best_f = evolve(rng, args.max_generations)

    print(f"\n{'='*62}")
    print(f"  EVOLUTION COMPLETE  |  best fitness = {best_f:.2f}")
    print(f"{'='*62}")
    describe(best_g)

    save(best_g, best_f)

    if args.render:
        print("\nRunning best genome with rendering...")
        r = evaluate(best_g, n_episodes=1, render=True)
        print(f"Episode reward: {r:.2f}")


if __name__ == "__main__":
    main()
