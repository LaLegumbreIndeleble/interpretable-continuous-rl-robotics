"""
CGP++ Python Port — (μ+λ) Evolutionary Algorithm
==================================================
Mirrors MuPlusLambda.h (Beyer & Schwefel 2002).

  - Population = μ parents + λ offspring
  - Parents selected uniformly at random from the top-μ pool
  - Optional crossover before mutation
  - Population sorted each generation; top μ become next parents
"""

from __future__ import annotations
import math
from typing import Callable, Optional, Tuple
import numpy as np

from ..parameters import Parameters
from ..representation.individual import Individual
from ..representation.species    import Species
from ..evaluator.evaluator       import Evaluator
from ..mutation.mutation         import Mutation
from ..population.static_population import StaticPopulation
from ..fitness.fitness           import Fitness
from ..variation.crossover.discrete_crossover import DiscreteCrossover
from ..variation.crossover.block_crossover    import BlockCrossover
from ..parameters import DISCRETE_CROSSOVER, BLOCK_CROSSOVER


class MuPlusLambda:
    """
    (μ+λ) ES with optional crossover.

    Args:
        params          : configuration
        species         : genome layout
        evaluator       : CGP graph evaluator
        mutation        : mutation operator
        fitness         : fitness comparator
        eval_fn         : user callback — signature:
                              eval_fn(individual, evaluator) -> float
        rng             : numpy random generator
        crossover_rate  : probability of applying crossover before mutation
                          (0.0 = mutation only, mirrors C++ behaviour)
        report_interval : print progress every N generations (0 = silent)
    """

    def __init__(
        self,
        params: Parameters,
        species: Species,
        evaluator: Evaluator,
        mutation: Mutation,
        fitness: Fitness,
        eval_fn: Callable[[Individual, Evaluator], float],
        rng: np.random.Generator,
        report_interval: int = 100,
    ) -> None:
        self.params          = params
        self.species         = species
        self.evaluator       = evaluator
        self.mutation        = mutation
        self.fitness         = fitness
        self.eval_fn         = eval_fn
        self.rng             = rng
        self.report_interval = report_interval

        self.mu              = params.mu
        self.lambda_         = params.get_lambda()
        self.crossover_rate  = params.crossover_rate
        self.max_generations = params.max_generations

        pop_size = self.mu + self.lambda_
        self.population = StaticPopulation(pop_size, species, fitness, rng)

        # Build crossover operator
        if params.crossover_type == BLOCK_CROSSOVER:
            self._crossover = BlockCrossover(params, rng)
        else:
            self._crossover = DiscreteCrossover(params, rng)

    # ── internal helpers ──────────────────────────────────────

    def _evaluate_offspring(self) -> None:
        for i in range(self.mu, self.mu + self.lambda_):
            ind = self.population.get_individual(i)
            if not ind.evaluated:
                self.evaluator.decode_path(ind)
                ind.fitness   = self.eval_fn(ind, self.evaluator)
                ind.evaluated = True

    def _select_parent_idx(self) -> int:
        return int(self.rng.integers(0, self.mu))

    def _breed(self) -> None:
        for i in range(self.lambda_):
            idx1 = self._select_parent_idx()
            idx2 = self._select_parent_idx()
            p1   = self.population.get_individual(idx1)
            p2   = self.population.get_individual(idx2)
            o1   = p1.clone()
            o2   = p2.clone()

            if self.crossover_rate > 0 and self.rng.random() < self.crossover_rate:
                self._crossover.variate(o1, o2)

            self.mutation.mutate(o1)
            o1.evaluated = False
            self.population.set_individual(o1, self.mu + i)

    # ── initialise the parent pool ────────────────────────────

    def _init_parents(self) -> None:
        for i in range(self.mu):
            ind = self.population.get_individual(i)
            self.evaluator.decode_path(ind)
            ind.fitness   = self.eval_fn(ind, self.evaluator)
            ind.evaluated = True

    # ── main loop ─────────────────────────────────────────────

    def evolve(self) -> Tuple[int, float]:
        """
        Run evolution.

        Returns:
            (fitness_evaluations, best_fitness)
        """
        best_fitness  = self.fitness.worst_value()
        fitness_evals = 0

        # Evaluate initial parents
        self._init_parents()
        self.population.sort()
        best_fitness = self.population.get_individual(0).fitness

        for gen in range(self.max_generations + 1):
            # Breed and evaluate offspring
            self._breed()
            self._evaluate_offspring()
            fitness_evals += self.lambda_

            # Sort full population; top-μ become new parents
            self.population.sort()
            best_fitness = self.population.get_individual(0).fitness

            if self.report_interval > 0 and gen % self.report_interval == 0:
                active = self.population.get_individual(0).num_active_nodes()
                print(f"  Gen {gen:>6} | best = {best_fitness:.6g}  active = {active}")

            if self.fitness.is_ideal(best_fitness):
                print(f"  Ideal fitness reached at gen {gen}.")
                break

        return fitness_evals, best_fitness

    def champion(self) -> Individual:
        return self.population.best()
