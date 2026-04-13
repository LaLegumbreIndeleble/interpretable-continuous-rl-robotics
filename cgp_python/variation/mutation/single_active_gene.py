"""
CGP++ Python Port — Single Active Gene Mutation
================================================
Mirrors SingleActiveGene.h.

Mutates exactly one gene drawn uniformly at random from the set of
genes belonging to active nodes (function gene OR any connection gene).
"""

from __future__ import annotations
import numpy as np

from ...representation.species import Species
from ...representation.individual import Individual
from ...parameters import Parameters


class SingleActiveGene:
    """Mutates exactly one active gene per call."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
    ) -> None:
        self.params    = params
        self.rng       = rng
        self.species   = species
        self.max_arity = params.max_arity
        self.name      = "Single Active Gene"

    def variate(self, individual: Individual) -> None:
        active_nodes = individual.active_nodes
        if not active_nodes:
            return

        genome         = individual.genome
        n_active       = len(active_nodes)
        rand_node_idx  = int(self.rng.integers(0, n_active))
        rand_node_num  = active_nodes[rand_node_idx]
        rand_gene_idx  = int(self.rng.integers(0, self.max_arity + 1))  # 0=func, 1..=conn

        mutation_pos   = self.species.position_from_node_number(rand_node_num) + rand_gene_idx
        lo = self.species.min_gene(mutation_pos)
        hi = self.species.max_gene(mutation_pos)
        if lo <= hi:
            genome[mutation_pos] = int(self.rng.integers(lo, hi + 1))
