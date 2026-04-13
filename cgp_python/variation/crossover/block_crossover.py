"""
CGP++ Python Port — Block Crossover
=====================================
Mirrors BlockCrossover.h.

Swaps a contiguous block of function genes between two parents.
The block start and length are chosen uniformly at random within
the active-node sequences.
"""

from __future__ import annotations
import numpy as np

from ...representation.individual import Individual
from ...parameters import Parameters


class BlockCrossover:
    """Block-based crossover on active function genes."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
    ) -> None:
        self.params     = params
        self.rng        = rng
        self.num_inputs = params.num_inputs
        self.max_arity  = params.max_arity
        self.name       = "Block Crossover"

    def variate(self, p1: Individual, p2: Individual) -> None:
        if not p1.active_nodes or not p2.active_nodes:
            return

        g1  = p1.genome
        g2  = p2.genome
        an1 = p1.active_nodes
        an2 = p2.active_nodes
        min_len = min(len(an1), len(an2))

        if min_len < 2:
            return

        # Pick a random block within [0, min_len)
        start  = int(self.rng.integers(0, min_len - 1))
        length = int(self.rng.integers(1, min_len - start + 1))
        end    = min(start + length, min_len)

        for x in range(start, end):
            node1 = an1[x]
            node2 = an2[x]
            idx1  = (node1 - self.num_inputs) * (1 + self.max_arity)
            idx2  = (node2 - self.num_inputs) * (1 + self.max_arity)
            g1[idx1], g2[idx2] = int(g2[idx2]), int(g1[idx1])
