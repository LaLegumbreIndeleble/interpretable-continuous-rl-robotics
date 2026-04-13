"""
CGP++ Python Port — Probabilistic Point Mutation
=================================================
Mirrors ProbabilisticPoint.h.

Selects floor(mutation_rate × genome_size) random positions and replaces
each gene with a uniformly drawn value within [min_gene, max_gene].
"""

from __future__ import annotations
import numpy as np

from ...representation.species import Species
from ...representation.individual import Individual
from ...parameters import Parameters


class ProbabilisticPoint:
    """Standard probabilistic point mutation for CGP."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
    ) -> None:
        self.params        = params
        self.rng           = rng
        self.species       = species
        self.mutation_rate = params.mutation_rate
        self.genome_size   = species.genome_size
        self.name          = "Probabilistic Point"

    def variate(self, individual: Individual) -> None:
        genome        = individual.genome
        num_mutations = max(1, int(self.mutation_rate * self.genome_size))

        for _ in range(num_mutations):
            pos = int(self.rng.integers(0, self.genome_size))
            lo  = self.species.min_gene(pos)
            hi  = self.species.max_gene(pos)
            if lo <= hi:
                genome[pos] = int(self.rng.integers(lo, hi + 1))
