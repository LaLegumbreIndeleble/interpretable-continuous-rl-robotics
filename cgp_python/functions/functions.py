"""
CGP++ Python Port — Functions (abstract base)
==============================================
Mirrors Functions.h: abstract interface for a function set.
Subclass this to define custom primitive sets.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Functions(ABC):
    """Abstract base for CGP function sets."""

    @abstractmethod
    def call_function(self, inputs: List[float], function: int) -> float:
        """Evaluate function `function` on `inputs`."""

    @abstractmethod
    def function_name(self, function: int) -> str:
        """Return a human-readable name for function index `function`."""

    @abstractmethod
    def input_name(self, input_idx: int) -> str:
        """Return a human-readable name for input index `input_idx`."""

    @abstractmethod
    def arity_of(self, function: int) -> int:
        """Return the arity of function `function`."""

    @abstractmethod
    def num_functions(self) -> int:
        """Return the total number of available functions."""
