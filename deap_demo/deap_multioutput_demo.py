"""
DEAP Multi-Output GP — Minimal Proof of Concept
================================================
Proves that DEAP can evolve a VECTOR of symbolic expressions
(one independent tree per output dimension) that are compiled
and evaluated on-the-fly each generation.

Problem (no Gym, no LLM):
  Inputs  : x0, x1   (2 variables)
  Outputs : 4 target functions the GP must rediscover

  y0 = sin(x0) + x1              (trigonometric + linear)
  y1 = x0 * x1 - x0              (bilinear interaction)
  y2 = x0**2 + cos(x1)           (quadratic + trigonometric)
  y3 = x1 / (x0 + 1) - 0.5      (rational)

Each individual = list of 4 PrimitiveTrees  (one per output)
Fitness        = negative MSE summed across all 4 outputs (maximize → 0)

Setup & Run with uv:
    uv init deap_demo
    cd deap_demo
    uv add deap numpy
    uv run deap_multioutput_demo.py
"""

import operator
import random
import numpy as np
from deap import base, creator, gp, tools, algorithms

# ──────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ──────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# 1.  DATASET  — 2 inputs, 4 known target outputs
# ──────────────────────────────────────────────
N_INPUTS = 2
N_OUTPUTS = 4
N_SAMPLES = 40

X = np.random.uniform(-2, 2, (N_SAMPLES, N_INPUTS))
x0, x1 = X[:, 0], X[:, 1]

# Ground-truth functions (what we want the GP to find)
Y_TRUE = np.stack(
    [
        np.sin(x0) + x1,  # y0
        x0 * x1 - x0,  # y1
        x0**2 + np.cos(x1),  # y2
        x1 / (x0 + 1.0) - 0.5,  # y3
    ],
    axis=1,
)  # shape (N_SAMPLES, N_OUTPUTS)

TARGET_NAMES = [
    "sin(x0) + x1",
    "x0*x1 - x0",
    "x0**2 + cos(x1)",
    "x1/(x0+1) - 0.5",
]


# ──────────────────────────────────────────────
# 2.  PRIMITIVE SET  (shared by all 4 trees)
# ──────────────────────────────────────────────
def protected_div(a, b):
    return a / b if abs(b) > 1e-6 else 1.0


pset = gp.PrimitiveSet("MAIN", N_INPUTS)
pset.renameArguments(ARG0="x0", ARG1="x1")

# Binary operators
pset.addPrimitive(operator.add, 2, name="add")
pset.addPrimitive(operator.sub, 2, name="sub")
pset.addPrimitive(operator.mul, 2, name="mul")
pset.addPrimitive(protected_div, 2, name="div")

# Unary operators
pset.addPrimitive(np.sin, 1, name="sin")
pset.addPrimitive(np.cos, 1, name="cos")
pset.addPrimitive(np.abs, 1, name="abs")

# Ephemeral random constants  [-2, 2]
pset.addEphemeralConstant("c", lambda: round(random.uniform(-2, 2), 2))

# ──────────────────────────────────────────────
# 3.  INDIVIDUAL = list of N_OUTPUTS trees
#     FITNESS    = single scalar (sum of MSE, minimise)
# ──────────────────────────────────────────────
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Register a single-tree generator
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)


# An individual is a list of N_OUTPUTS independent trees
def make_individual():
    return creator.Individual([toolbox.tree() for _ in range(N_OUTPUTS)])


toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ──────────────────────────────────────────────
# 4.  EVALUATION
#     Compile each tree → callable → evaluate on X → MSE vs Y_TRUE
# ──────────────────────────────────────────────
def evaluate(individual):
    total_mse = 0.0
    for i, tree in enumerate(individual):
        try:
            fn = gp.compile(tree, pset)  # ← on-the-fly compilation
            y_pred = np.array([fn(x0[j], x1[j]) for j in range(N_SAMPLES)], dtype=float)
            # Guard against NaN/Inf from dangerous operations
            if not np.all(np.isfinite(y_pred)):
                return (1e6,)
            mse = np.mean((y_pred - Y_TRUE[:, i]) ** 2)
        except Exception:
            return (1e6,)
        total_mse += mse
    return (total_mse,)  # must return a tuple


toolbox.register("evaluate", evaluate)


# ──────────────────────────────────────────────
# 5.  GENETIC OPERATORS
#     Applied independently to each tree inside the individual
# ──────────────────────────────────────────────
def cx_individual(ind1, ind2):
    """Crossover: for each output dimension, swap subtrees with 50% prob."""
    for i in range(N_OUTPUTS):
        if random.random() < 0.5:
            ind1[i], ind2[i] = gp.cxOnePoint(ind1[i], ind2[i])
    return ind1, ind2


def mut_individual(ind):
    """Mutation: randomly mutate one or more trees in the individual."""
    for i in range(N_OUTPUTS):
        if random.random() < 0.3:  # 30% chance per output
            (ind[i],) = gp.mutUniform(ind[i], expr=toolbox.expr, pset=pset)
    return (ind,)


toolbox.register("mate", cx_individual)
toolbox.register("mutate", mut_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

# Bloat control — limit max tree height per output
MAX_HEIGHT = 5
toolbox.decorate(
    "mate",
    gp.staticLimit(key=lambda ind: max(t.height for t in ind), max_value=MAX_HEIGHT),
)
toolbox.decorate(
    "mutate",
    gp.staticLimit(key=lambda ind: max(t.height for t in ind), max_value=MAX_HEIGHT),
)

# ──────────────────────────────────────────────
# 6.  EVOLUTION LOOP
# ──────────────────────────────────────────────
POP_SIZE = 200
N_GEN = 40
CXPB = 0.7  # crossover probability
MUTPB = 0.2  # mutation probability


def print_best(ind, gen):
    print(f"\n── Generation {gen:>3}  |  total MSE = {ind.fitness.values[0]:.4f} ──")
    for i, tree in enumerate(ind):
        fn = gp.compile(tree, pset)
        y_p = np.array([fn(x0[j], x1[j]) for j in range(N_SAMPLES)])
        mse = np.mean((y_p - Y_TRUE[:, i]) ** 2)
        print(f"  output[{i}]  MSE={mse:.4f}  expr= {tree}")
        print(f"           target= {TARGET_NAMES[i]}")


def main():
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(
        1, similar=lambda a, b: str(a) == str(b)
    )  # compare by expression string

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    print("=" * 60)
    print("  DEAP Multi-Output GP  |  2 inputs → 4 symbolic outputs")
    print("=" * 60)
    print(f"Population={POP_SIZE}  Generations={N_GEN}  MaxTreeHeight={MAX_HEIGHT}")
    print(f"Samples={N_SAMPLES}  Operators: add sub mul div sin cos abs")
    print("\nTarget functions:")
    for i, t in enumerate(TARGET_NAMES):
        print(f"  y{i} = {t}")
    print()

    # Print generation-0 best
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    print_best(hof[0], gen=0)

    # Run evolution
    for gen in range(1, N_GEN + 1):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate only the individuals that need it
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        # Print stats every 10 generations
        record = stats.compile(pop)
        if gen % 10 == 0 or gen == N_GEN:
            print(
                f"\nGen {gen:>3}  min_MSE={record['min']:.4f}  "
                f"avg_MSE={record['avg']:.4f}"
            )
            print_best(hof[0], gen=gen)

    # ── FINAL RESULTS ─────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL BEST INDIVIDUAL")
    print("=" * 60)
    best = hof[0]
    print(f"Total MSE = {best.fitness.values[0]:.6f}\n")

    for i, tree in enumerate(best):
        fn = gp.compile(tree, pset)
        y_p = np.array([fn(x0[j], x1[j]) for j in range(N_SAMPLES)])
        mse = np.mean((y_p - Y_TRUE[:, i]) ** 2)
        r2 = 1 - np.var(y_p - Y_TRUE[:, i]) / (np.var(Y_TRUE[:, i]) + 1e-12)

        print(f"  ┌─ output[{i}] ─────────────────────────────")
        print(f"  │  Evolved : {tree}")
        print(f"  │  Target  : {TARGET_NAMES[i]}")
        print(f"  │  MSE     : {mse:.6f}")
        print(f"  │  R²      : {r2:.4f}")
        print(f"  └─────────────────────────────────────────")

    print("\nKey takeaway:")
    print("  Each output is an INDEPENDENT tree compiled and evaluated on-the-fly.")
    print("  Individual = [tree_0, tree_1, tree_2, tree_3]")
    print("  At each generation: compile(tree_i, pset) → fn(x0, x1) → float")
    print("  Crossover and mutation act on each tree independently.")
    return pop, hof


if __name__ == "__main__":
    pop, hof = main()
