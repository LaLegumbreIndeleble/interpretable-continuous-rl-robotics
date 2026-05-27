"""
DEAP GP Controller — BipedalWalker-v3  (D4PG architecture, 15-tree)
====================================================================
Architecture: 4 + N_ATOMS trees per individual
  trees 0-3    (actor)  → deterministic action means, tanh output
  trees 4-14   (critic) → N_ATOMS=11 logits for categorical Q distribution

vs GP DDPG (5-tree):
  - Critic outputs a categorical distribution over N_ATOMS return bins
    (C51-style) instead of a single scalar Q
  - Gaussian noise epsilon=0.3 replaces OU (simpler, stateless — D4PG
    authors found both perform equally)
  - Expected Q = dot(softmax(logits), atom_supports) collapses the
    distribution to a scalar for action gating

    mu         = tanh(actor_i(*obs))
    a_gauss    = clip(mu + EPSILON * N(0,1), -1, 1)    ← explore
    logits     = [critic_k(*obs, *a_gauss) for k in 0..N_ATOMS-1]
    probs      = softmax(logits)
    E[Q]       = dot(probs, supports)                   ← expected Q
    gate       = sigmoid(E[Q])
    action     = clip(a_gauss * gate, -1, 1)            ← commit

Atom supports: N_ATOMS=11 atoms over [Vmin=-200, Vmax=300]
  → DELTA_Z = 50 per atom (tuned to BipedalWalker's return range)

pset_actor : 24 inputs (obs)
pset_critic: 28 inputs (24 obs + 4 actions)  — same as GP DDPG

Run:
    uv run bipedal_gp_a2c/bipedal_gp_d4pg.py
"""

import operator
import random
import math
import functools
import pickle
import json
import os
import time
import numpy as np
from deap import base, creator, gp, tools
import gymnasium as gym

BEST_MODEL_DIR = os.path.join(os.path.dirname(__file__), "best_model_d4pg")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_OBS     = 24
N_ACTIONS = 4

# Distributional critic
N_ATOMS = 11
Vmin    = -200.0
Vmax    =  300.0
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
SUPPORTS = [Vmin + i * DELTA_Z for i in range(N_ATOMS)]

IDX_CRITIC_START = N_ACTIONS              # first critic tree index (4)
IDX_CRITIC_END   = N_ACTIONS + N_ATOMS    # one past last critic tree (15)
N_TOTAL          = N_ACTIONS + N_ATOMS    # 15 total

# Gaussian exploration (D4PG replaces OU with plain Gaussian)
EPSILON = 0.3

N_ISLANDS      = 4
ISLAND_SIZE    = 50
MIGRATION_FREQ = 10
MIGRATION_SIZE = 5

N_GEN      = 2000
N_EPISODES = 5
MAX_STEPS  = 1600

CXPB = 0.65
MUTPB = 0.30
TREE_MAX_H = 6

STAGNATION_LIMIT = 15
IMMIGRANT_RATIO  = 0.15
CRITIC_MUTPB     = 0.10   # per-atom tree mutation rate

TIMEOUT_PENALTY    = 50.0  # max penalty when truncated and fully stuck
PROGRESS_WINDOW    = 200   # recent steps used to measure movement
PROGRESS_THRESHOLD = 0.5   # reward/step above this → considered still moving, no penalty

TIME_LIMIT_HOURS = 20.0

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
    return a if cond > 0.0 else b

def safe_softplus(a):
    return a if a > 20.0 else math.log(1.0 + math.exp(a))

def safe_sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, x))))


# ─────────────────────────────────────────────
# DISTRIBUTIONAL HELPERS
# ─────────────────────────────────────────────
def softmax_list(xs):
    """Numerically stable softmax over a Python list."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def expected_q(logits):
    """E[Q] = sum(softmax(logits) * supports)."""
    probs = softmax_list(logits)
    return sum(p * z for p, z in zip(probs, SUPPORTS))


# ─────────────────────────────────────────────
# PRIMITIVE SETS  (actor: 24 inputs, critic: 28)
# ─────────────────────────────────────────────
_PRIMITIVES = [
    (operator.add,  2, "add"),
    (operator.sub,  2, "sub"),
    (operator.mul,  2, "mul"),
    (safe_div,      2, "div"),
    (safe_max2,     2, "max2"),
    (safe_min2,     2, "min2"),
    (if_pos,        3, "if_pos"),
    (safe_tanh,     1, "tanh"),
    (safe_sin,      1, "sin"),
    (safe_cos,      1, "cos"),
    (safe_sigmoid,  1, "sigmoid"),
    (safe_abs,      1, "abs"),
    (operator.neg,  1, "neg"),
    (safe_sqrt,     1, "sqrt"),
    (safe_log,      1, "log"),
    (safe_exp,      1, "exp"),
    (safe_softplus, 1, "softplus"),
]

def _build_pset(name, n_inputs):
    ps = gp.PrimitiveSet(name, n_inputs)
    for fn, arity, pname in _PRIMITIVES:
        ps.addPrimitive(fn, arity, name=pname)
    ps.addEphemeralConstant(f"c_s_{name}", functools.partial(random.uniform, -1.0, 1.0))
    ps.addEphemeralConstant(f"c_l_{name}", functools.partial(random.uniform, -3.0, 3.0))
    return ps

_OBS_NAMES = {
    "ARG0":  "hull_angle",    "ARG1":  "hull_angvel",
    "ARG2":  "vel_x",         "ARG3":  "vel_y",
    "ARG4":  "hip_L_angle",   "ARG5":  "hip_L_speed",
    "ARG6":  "knee_L_angle",  "ARG7":  "knee_L_speed",
    "ARG8":  "leg_L_contact",
    "ARG9":  "hip_R_angle",   "ARG10": "hip_R_speed",
    "ARG11": "knee_R_angle",  "ARG12": "knee_R_speed",
    "ARG13": "leg_R_contact",
    **{f"ARG{14+i}": f"lidar_{i}" for i in range(10)},
}

pset_actor = _build_pset("ACTOR", N_OBS)
pset_actor.renameArguments(**_OBS_NAMES)

pset_critic = _build_pset("CRITIC", N_OBS + N_ACTIONS)
pset_critic.renameArguments(**{
    **_OBS_NAMES,
    "ARG24": "action_hip_L",
    "ARG25": "action_knee_L",
    "ARG26": "action_hip_R",
    "ARG27": "action_knee_R",
})


# ─────────────────────────────────────────────
# INDIVIDUAL & TOOLBOX
# ─────────────────────────────────────────────
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_actor",  gp.genHalfAndHalf, pset=pset_actor,  min_=1, max_=TREE_MAX_H)
toolbox.register("expr_critic", gp.genHalfAndHalf, pset=pset_critic, min_=1, max_=TREE_MAX_H)


def make_individual():
    actor_trees  = [gp.PrimitiveTree(toolbox.expr_actor())  for _ in range(N_ACTIONS)]
    critic_trees = [gp.PrimitiveTree(toolbox.expr_critic()) for _ in range(N_ATOMS)]
    return creator.Individual(actor_trees + critic_trees)


toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ─────────────────────────────────────────────
# FITNESS
# ─────────────────────────────────────────────
def evaluate(individual):
    try:
        actor_fns  = [gp.compile(individual[i],                   pset_actor)  for i in range(N_ACTIONS)]
        critic_fns = [gp.compile(individual[IDX_CRITIC_START + k], pset_critic) for k in range(N_ATOMS)]
    except Exception:
        return (-200.0,)

    env = gym.make("BipedalWalker-v3")
    rewards = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=RANDOM_SEED + ep)
        total          = 0.0
        timed_out      = False
        recent_rewards = []

        for _ in range(MAX_STEPS):
            # ── deterministic actor ────────────────────────────
            try:
                mu_raw = [float(fn(*obs)) for fn in actor_fns]
                if not all(math.isfinite(x) for x in mu_raw):
                    raise ValueError
            except Exception:
                total = -200.0
                break

            mu = np.array([math.tanh(m) for m in mu_raw], dtype=np.float32)

            # ── Gaussian exploration (D4PG: no OU, plain N(0,1)) ──
            a_gauss = np.clip(
                mu + EPSILON * np.random.normal(size=N_ACTIONS).astype(np.float32),
                -1.0, 1.0,
            ).astype(np.float32)

            # ── distributional Q: N_ATOMS logits → expected Q ──
            try:
                logits = [float(fn(*obs, *a_gauss)) for fn in critic_fns]
                if not all(math.isfinite(x) for x in logits):
                    raise ValueError
                eq = expected_q(logits)
            except Exception:
                eq = 0.0

            # ── gate by expected Q ─────────────────────────────
            gate   = safe_sigmoid_scalar(eq)
            action = np.clip(a_gauss * gate, -1.0, 1.0).astype(np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            recent_rewards.append(reward)
            if terminated or truncated:
                timed_out = truncated and not terminated
                break
        else:
            timed_out = True  # MAX_STEPS exhausted without env signal

        if timed_out:
            window = recent_rewards[-PROGRESS_WINDOW:]
            progress_rate = float(np.mean(window)) if window else 0.0
            penalty_scale = max(0.0, 1.0 - max(0.0, progress_rate) / PROGRESS_THRESHOLD)
            total -= TIMEOUT_PENALTY * penalty_scale

        rewards.append(total)

    env.close()
    return (float(np.mean(rewards)),)


toolbox.register("evaluate", evaluate)


# ─────────────────────────────────────────────
# GENETIC OPERATORS
# ─────────────────────────────────────────────
def cx_individual(ind1, ind2):
    for i in range(N_ACTIONS):  # actor trees
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    for k in range(N_ATOMS):    # critic atom trees (same pset, safe to cross)
        if random.random() < 0.5:
            j = IDX_CRITIC_START + k
            ind1[j], ind2[j] = gp.cxOnePoint(ind1[j], ind2[j])
    return ind1, ind2


def _mutate_tree(tree, expr, pset):
    roll = random.random()
    if roll < 0.60:
        return gp.mutUniform(tree, expr=expr, pset=pset)[0]
    elif roll < 0.85:
        return gp.mutNodeReplacement(tree, pset=pset)[0]
    else:
        return gp.mutShrink(tree)[0]


def mut_individual(ind):
    for i in range(N_ACTIONS):
        if random.random() < MUTPB:
            ind[i] = _mutate_tree(ind[i], toolbox.expr_actor, pset_actor)
    for k in range(N_ATOMS):
        if random.random() < CRITIC_MUTPB:
            j = IDX_CRITIC_START + k
            ind[j] = _mutate_tree(ind[j], toolbox.expr_critic, pset_critic)
    return (ind,)


toolbox.register("mate",   cx_individual)
toolbox.register("mutate", mut_individual)
toolbox.register("select", tools.selTournament, tournsize=5)

toolbox.decorate("mate",   gp.staticLimit(
    key=lambda ind: max(t.height for t in ind), max_value=TREE_MAX_H + 2))
toolbox.decorate("mutate", gp.staticLimit(
    key=lambda ind: max(t.height for t in ind), max_value=TREE_MAX_H + 2))


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
        worst_idx = sorted(range(len(isle)), key=lambda j: isle[j].fitness.values[0])[:MIGRATION_SIZE]
        incoming  = migrants[(i - 1) % n]
        for rank, idx in enumerate(worst_idx):
            isle[idx] = incoming[rank]


def inject_immigrants(island, n_immigrants):
    """Replace worst n individuals with brand-new random ones."""
    new_inds = [make_individual() for _ in range(n_immigrants)]
    for ind, fit in zip(new_inds, map(toolbox.evaluate, new_inds)):
        ind.fitness.values = fit
    worst_idx = sorted(range(len(island)), key=lambda j: island[j].fitness.values[0])[:n_immigrants]
    for rank, idx in enumerate(worst_idx):
        island[idx] = new_inds[rank]


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def print_individual(ind, label=""):
    print(f"\n{'─'*70}")
    print(f"  {label}  |  fitness = {ind.fitness.values[0]:.4f}")
    print(f"{'─'*70}")
    for i in range(N_ACTIONS):
        print(f"  actor  [{i}] ({ACTION_NAMES[i]:8s}) = {ind[i]}")
    for k in range(N_ATOMS):
        support_val = Vmin + k * DELTA_Z
        print(f"  critic [{k:2d}] (atom {support_val:+.0f}) = {ind[IDX_CRITIC_START + k]}")
    print(f"{'─'*70}")


# ─────────────────────────────────────────────
# DETERMINISTIC EVAL  (no noise, no gate — mirrors test_net)
# ─────────────────────────────────────────────
def eval_deterministic(individual, n_episodes=10, render=False):
    """Evaluate using actor output only (no Gaussian noise, no gate)."""
    try:
        actor_fns = [gp.compile(individual[i], pset_actor) for i in range(N_ACTIONS)]
    except Exception:
        return -200.0

    render_mode = "human" if render else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    ep_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=RANDOM_SEED + ep)
        total = 0.0
        while True:
            try:
                mu_raw = [float(fn(*obs)) for fn in actor_fns]
                action = np.clip([math.tanh(m) for m in mu_raw], -1.0, 1.0)
            except Exception:
                action = np.zeros(N_ACTIONS)
            obs, reward, terminated, truncated, _ = env.step(
                np.array(action, dtype=np.float32))
            total += reward
            if terminated or truncated:
                break
        ep_rewards.append(total)

    env.close()
    mean_r = float(np.mean(ep_rewards))
    print(f"  Deterministic eval ({n_episodes} eps): mean={mean_r:.2f}")
    return mean_r


# ─────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────
def _save_best(best, training_log):
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    with open(os.path.join(BEST_MODEL_DIR, "best_individual.pkl"), "wb") as f:
        pickle.dump(best, f)
    meta = {
        "fitness":      best.fitness.values[0],
        "actor_trees":  {ACTION_NAMES[i]: str(best[i]) for i in range(N_ACTIONS)},
        "critic_trees": {
            f"atom_{k}_support_{Vmin + k * DELTA_Z:+.0f}": str(best[IDX_CRITIC_START + k])
            for k in range(N_ATOMS)
        },
        "supports": SUPPORTS,
    }
    with open(os.path.join(BEST_MODEL_DIR, "best_individual.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(BEST_MODEL_DIR, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(__doc__)
    print(f"  Islands={N_ISLANDS} × {ISLAND_SIZE} = {N_ISLANDS*ISLAND_SIZE} total")
    print(f"  Gen={N_GEN}  Episodes/eval={N_EPISODES}  MaxSteps={MAX_STEPS}")
    print(f"  N_ATOMS={N_ATOMS}  Vmin={Vmin}  Vmax={Vmax}  DELTA_Z={DELTA_Z:.1f}")
    print(f"  EPSILON={EPSILON}  Trees/individual={N_TOTAL}")
    print(
        f"  TreeMaxH={TREE_MAX_H}  MigrateEvery={MIGRATION_FREQ}  "
        f"StagLimit={STAGNATION_LIMIT}  TimeLimit={TIME_LIMIT_HOURS}h\n"
    )

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    start_time   = time.time()
    training_log = []

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

    elapsed_h = (time.time() - start_time) / 3600.0
    print(f"Gen   0 | global_max={global_hof[0].fitness.values[0]:7.2f}  t={elapsed_h:.2f}h")
    print_individual(global_hof[0], "Gen 0 best")

    for gen in range(1, N_GEN + 1):
        elapsed_h = (time.time() - start_time) / 3600.0
        if elapsed_h >= TIME_LIMIT_HOURS:
            print(f"\n  Time limit ({TIME_LIMIT_HOURS}h) reached at gen {gen}. Stopping.")
            break

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

            elite = tools.selBest(isle, 1)
            offspring[-1] = toolbox.clone(elite[0])
            islands[idx][:] = offspring

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

        if gen % MIGRATION_FREQ == 0:
            migrate(islands)
            print(f"  ↔ migration  (gen {gen})")

        global_hof.update([ind for isle in islands for ind in isle])

        all_fits  = [ind.fitness.values[0] for isle in islands for ind in isle]
        max_fit   = max(all_fits)
        avg_fit   = float(np.mean(all_fits))
        elapsed_h = (time.time() - start_time) / 3600.0

        print(
            f"Gen {gen:>4} | max={max_fit:7.2f}  "
            f"avg={avg_fit:7.2f}  "
            f"stag={stag_counters}  t={elapsed_h:.2f}h"
        )

        if gen % 5 == 0:
            print_individual(global_hof[0], f"Gen {gen} best")
            training_log.append({
                "gen": gen,
                "max": round(max_fit, 4),
                "avg": round(avg_fit, 4),
                "elapsed_h": round(elapsed_h, 4),
            })
            _save_best(global_hof[0], training_log)

    print("\n" + "=" * 70)
    print("  EVOLUTION COMPLETE — TOP 5 INDIVIDUALS")
    print("=" * 70)
    for rank, ind in enumerate(global_hof, 1):
        print_individual(ind, f"Rank #{rank}")

    training_log.append({
        "gen": "final",
        "max": round(global_hof[0].fitness.values[0], 4),
        "elapsed_h": round((time.time() - start_time) / 3600.0, 4),
    })
    _save_best(global_hof[0], training_log)

    print(f"\n  Model saved → {BEST_MODEL_DIR}/")
    print(f"    best_individual.pkl")
    print(f"    best_individual.json  (actor + {N_ATOMS} critic atom trees + supports)")
    print(f"    training_log.json")

    print("\n── Deterministic Eval of Best Individual ──")
    eval_deterministic(global_hof[0], n_episodes=5)

    return islands, global_hof


if __name__ == "__main__":
    islands, hof = main()
