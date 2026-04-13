"""
CGP++ Python Port — Fitness
============================
Mirrors Fitness.h: comparators for minimising and maximising fitness,
plus worst/ideal sentinel values.
"""

from __future__ import annotations
import math


class Fitness:
    """
    Fitness comparison logic.

    Args:
        minimizing: True  → lower fitness is better (e.g. MSE)
                    False → higher fitness is better (e.g. reward)
        ideal_fitness: target value at which evolution may terminate early.
    """

    def __init__(self, minimizing: bool = True, ideal_fitness: float = 0.0) -> None:
        self.minimizing    = minimizing
        self.ideal_fitness = ideal_fitness

    def worst_value(self) -> float:
        return math.inf if self.minimizing else -math.inf

    def is_better(self, a: float, b: float) -> bool:
        """Return True if fitness `a` is strictly better than `b`."""
        if self.minimizing:
            return a < b
        return a > b

    def is_ideal(self, fitness: float) -> bool:
        if self.minimizing:
            return fitness <= self.ideal_fitness
        return fitness >= self.ideal_fitness

    def sort_key(self, fitness: float) -> float:
        """Key for sorting a population: best first."""
        return fitness if self.minimizing else -fitness
