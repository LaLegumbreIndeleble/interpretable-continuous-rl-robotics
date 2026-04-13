"""
CGP++ Python Port — Duplication Mutation
=========================================
Mirrors Duplication.h (Kalkreuth 2022).

Copies the function gene of a randomly chosen active node to a
subsequent sequence of active nodes of stochastically determined length.
"""

from __future__ import annotations
import numpy as np

from .phenotypic import Phenotypic
from ...representation.individual import Individual
from ...representation.species import Species
from ...parameters import Parameters


class Duplication(Phenotypic):
    """Phenotypic duplication mutation for CGP."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
    ) -> None:
        super().__init__(
            params, rng, species,
            rate      = params.duplication_rate,
            max_depth = params.max_duplication_depth,
        )
        self.name = "Duplication"

    def variate(self, individual: Individual) -> None:
        active_nodes   = individual.active_nodes
        num_active     = len(active_nodes)
        if num_active <= 1:
            return

        genome = individual.genome
        depth  = self.stochastic_depth(self.max_depth, num_active)
        start  = self.start_index(num_active, depth)
        end    = start + depth

        # Source: function gene of active_nodes[start]
        src_node = active_nodes[start]
        src_pos  = self.species.position_from_node_number(src_node)
        fn_gene  = int(genome[src_pos])

        # Copy to active_nodes[start+1 .. start+depth]
        for i in range(start + 1, end + 1):
            node = active_nodes[i]
            pos  = self.species.position_from_node_number(node)
            genome[pos] = fn_gene
