"""
CGP++ Python Port — Parameters
===============================
Direct translation of Parameters.h / Parameters.cpp.
Mirrors every field and default value from the C++ implementation.
"""

from dataclasses import dataclass, field
from typing import List

# Gene-type constants (mirror typedef in Parameters.h)
PROBABILISTIC_POINT_MUTATION = 0
SINGLE_ACTIVE_GENE_MUTATION  = 1
INVERSION_MUTATION           = 2
DUPLICATION_MUTATION         = 3

BLOCK_CROSSOVER    = 0
DISCRETE_CROSSOVER = 1

ONE_PLUS_LAMBDA = 0
MU_PLUS_LAMBDA  = 1

SYMBOLIC_REGRESSION = 0
LOGIC_SYNTHESIS     = 1

FITNESS_EVALUATIONS_TO_TERMINATION = 0
BEST_FITNESS_OF_RUN                = 1


@dataclass
class Parameters:
    # ── graph topology ────────────────────────────────────────
    num_function_nodes: int = 100
    num_inputs:         int = 2
    num_outputs:        int = 1
    num_functions:      int = 4
    max_arity:          int = 2
    levels_back:        int = 100          # connectivity reach

    # ── EA ────────────────────────────────────────────────────
    mu:     int = 1
    lambda_: int = 4                       # C++ uses "lambda" (keyword in Python)
    population_size: int = 5              # set automatically by algorithm

    max_generations:        int  = 1_000_000
    max_fitness_evaluations: int = 100_000_000

    neutral_genetic_drift: bool = True    # OnePlusLambda NGD

    # ── mutation ──────────────────────────────────────────────
    mutation_rate:      float = 0.01
    mutation_type:      int   = PROBABILISTIC_POINT_MUTATION
    mutation_operators: List[int] = field(default_factory=lambda: [PROBABILISTIC_POINT_MUTATION])

    inversion_rate:        float = 0.1
    duplication_rate:      float = 0.1
    max_inversion_depth:   int   = 5
    max_duplication_depth: int   = 5

    # ── crossover ─────────────────────────────────────────────
    crossover_rate: float = 0.0
    crossover_type: int   = DISCRETE_CROSSOVER

    # ── algorithm / problem ───────────────────────────────────
    algorithm: int = ONE_PLUS_LAMBDA
    problem:   int = SYMBOLIC_REGRESSION

    # ── fitness ───────────────────────────────────────────────
    ideal_fitness:     float = 0.0
    minimizing_fitness: bool = True        # True = minimise (e.g. MSE)

    # ── reporting ─────────────────────────────────────────────
    report_interval:      int  = 100
    report_during_job:    bool = True
    report_after_job:     bool = True
    report_simple:        bool = False
    simple_report_type:   int  = 0
    print_configuration:  bool = False

    # ── checkpointing ─────────────────────────────────────────
    checkpointing:      bool = False
    checkpoint_modulo:  int  = 50

    # ── misc ──────────────────────────────────────────────────
    evaluate_expression:  bool = True
    global_seed:          int  = 42
    generate_random_seed: bool = False
    num_jobs:             int  = 1
    num_eval_threads:     int  = 1

    # ── derived (set by set_genome_size) ──────────────────────
    genome_size: int = 0

    def set_genome_size(self) -> None:
        self.genome_size = self.num_function_nodes * (self.max_arity + 1) + self.num_outputs

    def get_lambda(self) -> int:
        return self.lambda_
