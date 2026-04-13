"""
CGP++ Python Port — StaticPopulation
======================================
Mirrors StaticPopulation.h.

Fixed-size list of Individual objects with sort-by-fitness support.
"""

from __future__ import annotations
from typing import List
import numpy as np

from ..representation.individual import Individual
from ..representation.species    import Species
from ..fitness.fitness            import Fitness


class StaticPopulation:
    """Fixed-size population of CGP individuals."""

    def __init__(
        self,
        size: int,
        species: Species,
        fitness: Fitness,
        rng: np.random.Generator,
    ) -> None:
        self._fitness   = fitness
        self._individuals: List[Individual] = [
            Individual(species, rng) for _ in range(size)
        ]

    def size(self) -> int:
        return len(self._individuals)

    def get_individual(self, index: int) -> Individual:
        return self._individuals[index]

    def set_individual(self, individual: Individual, index: int) -> None:
        self._individuals[index] = individual

    def sort(self) -> None:
        """Sort in-place so that the best individual is at index 0."""
        self._individuals.sort(key=lambda ind: self._fitness.sort_key(ind.fitness))

    def best(self) -> Individual:
        return min(
            self._individuals,
            key=lambda ind: self._fitness.sort_key(ind.fitness),
        )
