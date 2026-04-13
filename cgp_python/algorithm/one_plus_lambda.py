"""
CGP++ Python Port — (1+λ) Evolutionary Algorithm
==================================================
Mirrors OnePlusLambda.h.

  - Population = 1 parent + λ offspring
  - Neutral genetic drift (NGD): accept offspring with equal fitness
    as new parent (proposed by Miller, 2019)
  - Fitness evaluation is delegated to a user-supplied callable
"""

from __future__ import annotations
import math
from typing import Callable, List, Optional, Tuple
import numpy as np

from ..parameters import Parameters
from ..representation.individual import Individual
from ..representation.species    import Species
from ..evaluator.evaluator       import Evaluator
from ..mutation.mutation         import Mutation
from ..population.static_population import StaticPopulation
from ..fitness.fitness           import Fitness


class OnePlusLambda:
    """
    (1+λ) ES with optional neutral genetic drift.

    Args:
        params          : configuration
        species         : genome layout
        evaluator       : CGP graph evaluator
        mutation        : mutation operator
        fitness         : fitness comparator
        eval_fn         : user callback — signature:
                              eval_fn(individual, evaluator) -> float
        rng             : numpy random generator
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

        self.lambda_         = params.get_lambda()
        self.ngd             = params.neutral_genetic_drift
        self.max_generations = params.max_generations

        pop_size = 1 + self.lambda_
        self.population = StaticPopulation(pop_size, species, fitness, rng)

    # ── internal helpers ──────────────────────────────────────

    def _evaluate_all(self) -> None:
        for i in range(self.population.size()):
            ind = self.population.get_individual(i)
            if not ind.evaluated:
                self.evaluator.decode_path(ind)
                ind.fitness   = self.eval_fn(ind, self.evaluator)
                ind.evaluated = True

    def _select_parent(self, parent_idx: Optional[int], best_fitness: float) -> Tuple[int, float]:
        """Return (new_parent_index, new_best_fitness)."""
        pop = self.population

        if parent_idx is None:
            # First selection: just pick the best individual
            best_idx = 0
            best_f   = pop.get_individual(0).fitness
            for i in range(1, pop.size()):
                f = pop.get_individual(i).fitness
                if self.fitness.is_better(f, best_f):
                    best_idx = i
                    best_f   = f
            return best_idx, best_f

        better  = []
        equal   = []
        for i in range(pop.size()):
            f = pop.get_individual(i).fitness
            if self.fitness.is_better(f, best_fitness):
                better.append(i)
            elif f == best_fitness:
                equal.append(i)

        if better:
            idx = int(self.rng.choice(better))
            return idx, pop.get_individual(idx).fitness
        if self.ngd and equal:
            idx = int(self.rng.choice(equal))
            return idx, best_fitness
        return parent_idx, best_fitness

    def _breed(self, parent_idx: int) -> None:
        parent = self.population.get_individual(parent_idx)
        # Place parent at slot 0
        if parent_idx != 0:
            self.population.set_individual(parent, 0)
        # Create λ mutated offspring
        for i in range(1, 1 + self.lambda_):
            offspring = parent.clone()
            self.mutation.mutate(offspring)
            offspring.evaluated = False
            self.population.set_individual(offspring, i)

    # ── main loop ─────────────────────────────────────────────

    def evolve(self) -> Tuple[int, float]:
        """
        Run evolution.

        Returns:
            (fitness_evaluations, best_fitness)
        """
        best_fitness   = self.fitness.worst_value()
        parent_idx: Optional[int] = None
        fitness_evals  = 0

        for gen in range(self.max_generations + 1):
            self._evaluate_all()
            fitness_evals += self.lambda_

            parent_idx, best_fitness = self._select_parent(parent_idx, best_fitness)

            if self.report_interval > 0 and gen % self.report_interval == 0:
                active = self.population.get_individual(parent_idx).num_active_nodes()
                print(f"  Gen {gen:>6} | best = {best_fitness:.6g}  active = {active}")

            if self.fitness.is_ideal(best_fitness):
                print(f"  Ideal fitness reached at gen {gen}.")
                break

            self._breed(parent_idx)
            parent_idx = 0

        return fitness_evals, best_fitness

    def champion(self) -> Individual:
        """Return the best individual after evolve()."""
        return self.population.best()
