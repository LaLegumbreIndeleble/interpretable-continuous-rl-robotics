"""
CGP++ Python Port — Mutation Pipeline
========================================
Mirrors MutationPipeline.h.

Chains multiple mutation operators.  Each operator's variate() is called
in registration order whenever mutate() is invoked.
"""

from __future__ import annotations
from typing import List

from ..representation.individual import Individual


class MutationPipeline:
    """Pipeline that applies a sequence of mutation operators."""

    def __init__(self) -> None:
        self._operators: List = []

    def add(self, operator) -> "MutationPipeline":
        """Register a mutation operator (any object with a variate() method)."""
        self._operators.append(operator)
        return self

    def mutate(self, individual: Individual) -> None:
        for op in self._operators:
            op.variate(individual)
