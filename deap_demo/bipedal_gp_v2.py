"""
DEAP GP Controller v2 — BipedalWalker-v3  (A2C-style action gating)
=====================================================================
Architecture: 5 trees per individual
  trees 0-3  (mu)     → raw action proposals
  tree  4    (critic) → TD(0) baseline, gates actions at runtime

Fitness = mean episode reward (unchanged from v1).

The critic never contributes to fitness directly. It modulates
which actions the mu trees commit to each step via a TD advantage:

    v         = critic(*obs)
    v_next    = critic(*obs_next)
    advantage = reward + 0.99*v_next - v        ← TD(0) error
    gate      = sigmoid(advantage)               ← (0,1)
    action    = clip(mu * gate, -1, 1)

Positive surprise → act boldly (gate → 1).
Negative surprise → retreat toward 0 (gate → 0).

Run:
    uv run deap_demo/bipedal_gp_v2.py
"""

import operator
import random
import math
import functools
import pickle
import json
import os
import numpy as np
from deap import base, creator, gp, tools
import gymnasium as gym

BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model_v2")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_OBS     = 24
N_ACTIONS = 4
N_TOTAL   = N_ACTIONS + 1  # 5: 4 mu + 1 critic
IDX_VALUE = -1              # tree index 4

N_ISLANDS = 4
ISLAND_SIZE = 50            # total pop = 200
MIGRATION_FREQ = 10
MIGRATION_SIZE = 5

N_GEN      = 50
N_EPISODES = 5
MAX_STEPS  = 1600

CXPB = 0.65
MUTPB = 0.30
TREE_MAX_H = 6

STAGNATION_LIMIT = 15
IMMIGRANT_RATIO  = 0.15
CRITIC_MUTPB     = 0.10  # critic mutates less — needs to stabilize

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ACTION_NAMES = ["hip_L", "knee_L", "hip_R", "knee_R"]


# ─────────────────────────────────────────────
# SAFE OPERATORS
# ─────────────────────────────────────────────
def safe_div(a, b):
    return a / b if abs(b) > 1e-6 else 0.0


def safe_log(a):
    return math.log(abs(a) + 1e-6)


def safe_sqrt(a):
    return math.sqrt(abs(a))


def safe_exp(a):
    return math.exp(max(-10.0, min(10.0, a)))


def safe_tanh(a):
    return math.tanh(a)


def safe_sin(a):
    return math.sin(a)


def safe_cos(a):
    return math.cos(a)


def safe_sigmoid(a):
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, a))))


def safe_abs(a):
    return abs(a)


def safe_max2(a, b):
    return max(a, b)


def safe_min2(a, b):
    return min(a, b)


def if_pos(cond, a, b):
    """Conditional: if cond > 0 return a else return b."""
    return a if cond > 0.0 else b


def safe_sigmoid_scalar(x):
    """Sigmoid used outside the primitive set (action gating)."""
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))


# ─────────────────────────────────────────────
# PRIMITIVE SET
# ─────────────────────────────────────────────
pset = gp.PrimitiveSet("CTRL", N_OBS)

obs_names = {
    "ARG0": "hull_angle",
    "ARG1": "hull_angvel",
    "ARG2": "vel_x",
    "ARG3": "vel_y",
    "ARG4": "hip_L_angle",
    "ARG5": "hip_L_speed",
    "ARG6": "knee_L_angle",
    "ARG7": "knee_L_speed",
    "ARG8": "leg_L_contact",
    "ARG9": "hip_R_angle",
    "ARG10": "hip_R_speed",
    "ARG11": "knee_R_angle",
    "ARG12": "knee_R_speed",
    "ARG13": "leg_R_contact",
    **{f"ARG{14+i}": f"lidar_{i}" for i in range(10)},
}
pset.renameArguments(**obs_names)

# Binary
pset.addPrimitive(operator.add, 2, name="add")
pset.addPrimitive(operator.sub, 2, name="sub")
pset.addPrimitive(operator.mul, 2, name="mul")
pset.addPrimitive(safe_div, 2, name="div")
pset.addPrimitive(safe_max2, 2, name="max2")
pset.addPrimitive(safe_min2, 2, name="min2")

# Ternary — conditional branching
pset.addPrimitive(if_pos, 3, name="if_pos")

# Unary
pset.addPrimitive(safe_tanh, 1, name="tanh")
pset.addPrimitive(safe_sin, 1, name="sin")
pset.addPrimitive(safe_cos, 1, name="cos")
pset.addPrimitive(safe_sigmoid, 1, name="sigmoid")
pset.addPrimitive(safe_abs, 1, name="abs")
pset.addPrimitive(operator.neg, 1, name="neg")
pset.addPrimitive(safe_sqrt, 1, name="sqrt")
pset.addPrimitive(safe_log, 1, name="log")
pset.addPrimitive(safe_exp, 1, name="exp")

# Two constant pools
pset.addEphemeralConstant("c_s", functools.partial(random.uniform, -1.0, 1.0))
pset.addEphemeralConstant("c_l", functools.partial(random.uniform, -3.0, 3.0))

# ─────────────────────────────────────────────
# INDIVIDUAL & TOOLBOX
# ─────────────────────────────────────────────
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=TREE_MAX_H)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)


def make_individual():
    return creator.Individual([toolbox.tree() for _ in range(N_TOTAL)])


toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ─────────────────────────────────────────────
# FITNESS
# ─────────────────────────────────────────────
def evaluate(individual):
    # ── compile mu trees + critic ──────────────────────────
    try:
        mu_fns   = [gp.compile(individual[i], pset) for i in range(N_ACTIONS)]
        value_fn =  gp.compile(individual[IDX_VALUE], pset)
    except Exception:
        return (-200.0,)

    env = gym.make("BipedalWalker-v3")
    rewards = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=RANDOM_SEED + ep)
        total  = 0.0

        for _ in range(MAX_STEPS):
            # ── evaluate mu and critic at current obs ──────
            try:
                mu = [float(fn(*obs)) for fn in mu_fns]
                v  = float(value_fn(*obs))
                if not all(math.isfinite(x) for x in mu) or not math.isfinite(v):
                    raise ValueError
            except Exception:
                total = -200.0
                break

            # ── probe step with raw mu action ─────────────
            action_raw = np.clip(mu, -1.0, 1.0).astype(np.float32)
            obs_next, reward, terminated, truncated, _ = env.step(action_raw)

            # ── TD(0) advantage ────────────────────────────
            try:
                v_next = float(value_fn(*obs_next))
                if not math.isfinite(v_next):
                    v_next = 0.0
            except Exception:
                v_next = 0.0

            advantage = reward + 0.99 * v_next - v

            # ── gate: positive surprise → act boldly,
            #          negative surprise → retreat toward 0
            gate   = safe_sigmoid_scalar(advantage)
            action = np.clip(
                [mu[i] * gate for i in range(N_ACTIONS)],
                -1.0, 1.0,
            ).astype(np.float32)

            # ── commit gated action ────────────────────────
            obs_next2, reward2, terminated, truncated, _ = env.step(action)
            total += reward2

            obs = obs_next2
            if terminated or truncated:
                break

        rewards.append(total)

    env.close()
    return (float(np.mean(rewards)),)


toolbox.register("evaluate", evaluate)


# ─────────────────────────────────────────────
# GENETIC OPERATORS
# ─────────────────────────────────────────────
def cx_individual(ind1, ind2):
    for i in range(N_TOTAL):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def mut_individual(ind):
    # mu trees — normal rate
    for i in range(N_ACTIONS):
        if random.random() < MUTPB:
            roll = random.random()
            if roll < 0.60:
                (ind[i],) = gp.mutUniform(ind[i], expr=toolbox.expr, pset=pset)
            elif roll < 0.85:
                (ind[i],) = gp.mutNodeReplacement(ind[i], pset=pset)
            else:
                (ind[i],) = gp.mutShrink(ind[i])

    # critic tree — conservative, needs to stabilize
    if random.random() < CRITIC_MUTPB:
        (ind[IDX_VALUE],) = gp.mutUniform(
            ind[IDX_VALUE], expr=toolbox.expr, pset=pset)

    return (ind,)


toolbox.register("mate", cx_individual)
toolbox.register("mutate", mut_individual)
toolbox.register("select", tools.selTournament, tournsize=5)

toolbox.decorate(
    "mate",
    gp.staticLimit(
        key=lambda ind: max(t.height for t in ind), max_value=TREE_MAX_H + 2
    ),
)
toolbox.decorate(
    "mutate",
    gp.staticLimit(
        key=lambda ind: max(t.height for t in ind), max_value=TREE_MAX_H + 2
    ),
)


# ─────────────────────────────────────────────
# ISLAND MODEL
# ─────────────────────────────────────────────
def migrate(islands):
    """Ring migration: best of island i → replaces worst of island i+1."""
    n = len(islands)
    migrants = []
    for isle in islands:
        top = tools.selBest(isle, MIGRATION_SIZE)
        migrants.append([toolbox.clone(ind) for ind in top])
    for i, isle in enumerate(islands):
        worst_idx = sorted(range(len(isle)), key=lambda j: isle[j].fitness.values[0])[
            :MIGRATION_SIZE
        ]
        incoming = migrants[(i - 1) % n]
        for rank, idx in enumerate(worst_idx):
            isle[idx] = incoming[rank]


def inject_immigrants(island, n_immigrants):
    """Replace worst n individuals with brand-new random ones."""
    new_inds = [make_individual() for _ in range(n_immigrants)]
    for ind, fit in zip(new_inds, map(toolbox.evaluate, new_inds)):
        ind.fitness.values = fit
    worst_idx = sorted(range(len(island)), key=lambda j: island[j].fitness.values[0])[
        :n_immigrants
    ]
    for rank, idx in enumerate(worst_idx):
        island[idx] = new_inds[rank]


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def print_individual(ind, label=""):
    print(f"\n{'─'*62}")
    print(f"  {label}  |  reward = {ind.fitness.values[0]:.2f}")
    print(f"{'─'*62}")
    for i in range(N_ACTIONS):
        print(f"  mu [{i}] ({ACTION_NAMES[i]:8s}) = {ind[i]}")
    print(f"  critic              = {ind[IDX_VALUE]}")
    print(f"{'─'*62}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(__doc__)
    print(f"  Islands={N_ISLANDS} × {ISLAND_SIZE} = {N_ISLANDS*ISLAND_SIZE} total")
    print(f"  Gen={N_GEN}  Episodes/eval={N_EPISODES}  MaxSteps={MAX_STEPS}")
    print(
        f"  TreeMaxH={TREE_MAX_H}  MigrateEvery={MIGRATION_FREQ}  "
        f"StagLimit={STAGNATION_LIMIT}\n"
    )

    # init
    islands = [toolbox.population(n=ISLAND_SIZE) for _ in range(N_ISLANDS)]
    for isle in islands:
        for ind, fit in zip(isle, map(toolbox.evaluate, isle)):
            ind.fitness.values = fit

    global_hof = tools.HallOfFame(5)
    for isle in islands:
        global_hof.update(isle)

    stag_counters = [0] * N_ISLANDS
    isle_best = [
        max(isle, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        for isle in islands
    ]

    print(f"Gen   0 | global_max={global_hof[0].fitness.values[0]:7.2f}")
    print_individual(global_hof[0], "Gen 0 best")

    for gen in range(1, N_GEN + 1):

        for idx, isle in enumerate(islands):

            offspring = toolbox.select(isle, len(isle))
            offspring = list(map(toolbox.clone, offspring))

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            # elitism — keep best of previous generation
            elite = tools.selBest(isle, 1)
            offspring[-1] = toolbox.clone(elite[0])

            islands[idx][:] = offspring

            # stagnation check
            cur_best = max(isle, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
            if cur_best <= isle_best[idx] + 0.01:
                stag_counters[idx] += 1
            else:
                stag_counters[idx] = 0
                isle_best[idx] = cur_best

            if stag_counters[idx] >= STAGNATION_LIMIT:
                n_imm = max(1, int(ISLAND_SIZE * IMMIGRANT_RATIO))
                inject_immigrants(isle, n_imm)
                stag_counters[idx] = 0
                print(f"  ↺ island[{idx}] stagnant → +{n_imm} immigrants  (gen {gen})")

        # ring migration
        if gen % MIGRATION_FREQ == 0:
            migrate(islands)
            print(f"  ↔ migration  (gen {gen})")

        global_hof.update([ind for isle in islands for ind in isle])

        all_fits = [ind.fitness.values[0] for isle in islands for ind in isle]
        print(
            f"Gen {gen:>3} | max={max(all_fits):7.2f}  "
            f"avg={np.mean(all_fits):7.2f}  "
            f"stag={stag_counters}"
        )

        if gen % 10 == 0:
            print_individual(global_hof[0], f"Gen {gen} best")

    # final
    print("\n" + "=" * 62)
    print("  EVOLUTION COMPLETE — TOP 5 INDIVIDUALS")
    print("=" * 62)
    for rank, ind in enumerate(global_hof, 1):
        print_individual(ind, f"Rank #{rank}")

    # ── save best model ───────────────────────────────────
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    best = global_hof[0]

    with open(os.path.join(BEST_MODEL_DIR, "best_individual.pkl"), "wb") as f:
        pickle.dump(best, f)

    meta = {
        "fitness": best.fitness.values[0],
        "mu_trees": {ACTION_NAMES[i]: str(best[i]) for i in range(N_ACTIONS)},
        "critic_tree": str(best[IDX_VALUE]),
    }
    with open(os.path.join(BEST_MODEL_DIR, "best_individual.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved → {BEST_MODEL_DIR}/")
    print(f"    best_individual.pkl   (reload with pickle)")
    print(f"    best_individual.json  (human-readable expressions)")

    return islands, global_hof


if __name__ == "__main__":
    islands, hof = main()
