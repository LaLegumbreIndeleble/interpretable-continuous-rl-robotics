"""
CGP++ Python Port — Inversion Mutation
========================================
Mirrors Inversion.h (Kalkreuth 2022).

Reverses the order of function genes across a stochastically chosen
subsequence of active nodes.
"""

from __future__ import annotations
import math
import numpy as np

from .phenotypic import Phenotypic
from ...representation.individual import Individual
from ...representation.species import Species
from ...parameters import Parameters


class Inversion(Phenotypic):
    """Phenotypic inversion mutation for CGP."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
    ) -> None:
        super().__init__(
            params, rng, species,
            rate      = params.inversion_rate,
            max_depth = params.max_inversion_depth,
        )
        self.name = "Inversion"

    def variate(self, individual: Individual) -> None:
        active_nodes = individual.active_nodes
        num_active   = len(active_nodes)
        if num_active <= 1:
            return

        genome = individual.genome
        depth  = self.stochastic_depth(self.max_depth, num_active)
        start  = self.start_index(num_active, depth)
        end    = start + depth
        middle = round(depth / 2.0)

        # Pairwise swap function genes between [start, end]
        for i in range(middle):
            left_node  = active_nodes[start + i]
            right_node = active_nodes[end - i]
            left_pos   = self.species.position_from_node_number(left_node)
            right_pos  = self.species.position_from_node_number(right_node)
            genome[left_pos], genome[right_pos] = int(genome[right_pos]), int(genome[left_pos])
