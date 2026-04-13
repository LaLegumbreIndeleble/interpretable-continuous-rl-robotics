"""
CGP++ Python Port — Phenotypic (base for Duplication / Inversion)
==================================================================
Mirrors Phenotypic.h: stochastic depth and start-index helpers shared
by the two phenotypic mutation operators.
"""

from __future__ import annotations
import numpy as np

from ...representation.species import Species
from ...parameters import Parameters


class Phenotypic:
    """Abstract base for phenotypic mutation operators."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
        rate: float = 0.1,
        max_depth: int = 5,
    ) -> None:
        self.params    = params
        self.rng       = rng
        self.species   = species
        self.rate      = rate
        self.max_depth = max_depth
        self.name      = "Phenotypic"

    def stochastic_depth(self, max_depth: int, num_active_nodes: int) -> int:
        """Return a random depth in [1, min(max_depth, num_active_nodes-1)]."""
        cap = min(max_depth, num_active_nodes - 1)
        cap = max(cap, 1)
        return int(self.rng.integers(1, cap + 1))

    def start_index(self, num_active_nodes: int, depth: int) -> int:
        """Return a valid start index such that start+depth < num_active_nodes."""
        start_max = num_active_nodes - depth
        if start_max <= 0:
            return 0
        return int(self.rng.integers(0, start_max))
