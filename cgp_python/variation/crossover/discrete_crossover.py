"""
CGP++ Python Port — Discrete Crossover
========================================
Mirrors DiscreteCrossover.h (Kalkreuth 2021/2022).

Phenotypic discrete recombination: swaps function genes between
corresponding active nodes of two parents, with boundary extension
when the parents have different numbers of active nodes.
"""

from __future__ import annotations
import numpy as np

from ...representation.individual import Individual
from ...parameters import Parameters


class DiscreteCrossover:
    """Discrete (uniform) recombination on active function genes."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
    ) -> None:
        self.params    = params
        self.rng       = rng
        self.num_inputs = params.num_inputs
        self.max_arity  = params.max_arity
        self.name       = "Discrete Crossover"

    def variate(self, p1: Individual, p2: Individual) -> None:
        if not p1.active_nodes or not p2.active_nodes:
            return

        g1 = p1.genome
        g2 = p2.genome
        an1 = p1.active_nodes
        an2 = p2.active_nodes
        len1, len2 = len(an1), len(an2)
        min_len = min(len1, len2)
        max_len = max(len1, len2)
        boundary_extension = True

        for x in range(min_len):
            if self.rng.random() < 0.5:
                # boundary extension: on last shared index, pick from longer parent
                if boundary_extension and x == (min_len - 1) and len1 != len2:
                    r = int(self.rng.integers(0, max_len - x))
                    if len1 < len2:
                        swap1 = an1[x]
                        swap2 = an2[x + r]
                    else:
                        swap1 = an1[x + r]
                        swap2 = an2[x]
                else:
                    swap1 = an1[x]
                    swap2 = an2[x]

                idx1 = (swap1 - self.num_inputs) * (1 + self.max_arity)
                idx2 = (swap2 - self.num_inputs) * (1 + self.max_arity)
                g1[idx1], g2[idx2] = int(g2[idx2]), int(g1[idx1])
