"""
CGP++ Python Port — Individual
================================
Mirrors Individual.h: genome array + fitness + active-node list +
symbolic expressions.
"""

from __future__ import annotations
import copy
from typing import List, Optional
import numpy as np

from .species import Species


class Individual:
    """CGP individual with integer genome, fitness, and active-node metadata."""

    def __init__(self, species: Species, rng: np.random.Generator) -> None:
        self.species     = species
        self.rng         = rng
        self.genome_size = species.genome_size
        self.genome: np.ndarray = species.random_genome(rng)
        self.fitness: float     = float("inf")    # ∞ = worst for minimisation
        self.evaluated: bool    = False
        self.active_nodes: List[int]  = []
        self.expressions:  List[str]  = []

    # ── cloning ───────────────────────────────────────────────

    def clone(self) -> "Individual":
        child = Individual.__new__(Individual)
        child.species      = self.species
        child.rng          = self.rng
        child.genome_size  = self.genome_size
        child.genome       = self.genome.copy()
        child.fitness      = self.fitness
        child.evaluated    = self.evaluated
        child.active_nodes = list(self.active_nodes)
        child.expressions  = list(self.expressions)
        return child

    # ── reset ────────────────────────────────────────────────

    def reset(self) -> None:
        self.genome        = self.species.random_genome(self.rng)
        self.active_nodes  = []
        self.evaluated     = False
        self.fitness       = float("inf")

    # ── active nodes ─────────────────────────────────────────

    def clear_active_nodes(self) -> None:
        self.active_nodes.clear()

    def add_active_node(self, node_num: int) -> None:
        self.active_nodes.append(node_num)

    def num_active_nodes(self) -> int:
        return len(self.active_nodes)

    # ── expressions ──────────────────────────────────────────

    def clear_expressions(self) -> None:
        self.expressions.clear()

    def add_expression(self, expr: str) -> None:
        self.expressions.append(expr)

    # ── genome string ────────────────────────────────────────

    def to_string(self, delimiter: str = " ") -> str:
        return delimiter.join(str(g) for g in self.genome)

    def __repr__(self) -> str:
        return (f"Individual(fitness={self.fitness:.6g}, "
                f"active={len(self.active_nodes)}, "
                f"genome=[{self.to_string()}])")
