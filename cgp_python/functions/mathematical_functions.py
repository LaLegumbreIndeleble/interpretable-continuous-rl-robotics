"""
CGP++ Python Port — Mathematical Function Set
==============================================
Mirrors MathematicalFunctions.h: ADD, SUB, MUL, DIV (protected).
Extended with trig, exp/log, abs, neg, and constants for RL control.
"""

from __future__ import annotations
import math
from typing import List

from .functions import Functions


# ── safe wrappers ─────────────────────────────────────────────────────────────

def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-6 else 1.0   # mirrors C++ (returns 1 on /0)

def _safe_log(a: float, _: float) -> float:
    return math.log(abs(a) + 1e-6)

def _safe_sqrt(a: float, _: float) -> float:
    return math.sqrt(abs(a))

def _safe_exp(a: float, _: float) -> float:
    return math.exp(max(-10.0, min(10.0, a)))

def _tanh(a: float, _: float) -> float:
    return math.tanh(a)

def _sin(a: float, _: float) -> float:
    return math.sin(a)

def _cos(a: float, _: float) -> float:
    return math.cos(a)

def _abs(a: float, _: float) -> float:
    return abs(a)

def _neg(a: float, _: float) -> float:
    return -a

def _max2(a: float, b: float) -> float:
    return max(a, b)

def _min2(a: float, b: float) -> float:
    return min(a, b)

def _const_zero(a: float, _: float) -> float:
    return 0.0

def _const_one(a: float, _: float) -> float:
    return 1.0

def _const_half(a: float, _: float) -> float:
    return 0.5

def _const_neg1(a: float, _: float) -> float:
    return -1.0


# ── base mathematical set (mirrors C++ MathematicalFunctions) ─────────────────

_BASE = [
    (lambda a, b: a + b,  "ADD", 2),   # 0
    (lambda a, b: a - b,  "SUB", 2),   # 1
    (lambda a, b: a * b,  "MUL", 2),   # 2
    (_safe_div,            "DIV", 2),   # 3
]

# ── extended set (adds trig, exp/log, unary ops, constants) ───────────────────

_EXTENDED = _BASE + [
    (_tanh,      "TANH",  1),   # 4
    (_sin,       "SIN",   1),   # 5
    (_cos,       "COS",   1),   # 6
    (_abs,       "ABS",   1),   # 7
    (_neg,       "NEG",   1),   # 8
    (_safe_sqrt, "SQRT",  1),   # 9
    (_safe_log,  "LOG",   1),   # 10
    (_safe_exp,  "EXP",   1),   # 11
    (_max2,      "MAX",   2),   # 12
    (_min2,      "MIN",   2),   # 13
    (_const_zero, "0.0",  0),   # 14  (0-arity: ignores both inputs)
    (_const_one,  "1.0",  0),   # 15
    (_const_half, "0.5",  0),   # 16
    (_const_neg1, "-1.0", 0),   # 17
]


class MathematicalFunctions(Functions):
    """
    Mathematical function set.

    Args:
        extended: if True (default) use the full 18-function set;
                  if False use only the 4-function base (ADD/SUB/MUL/DIV)
                  matching the original C++ MathematicalFunctions.
    """

    def __init__(self, extended: bool = True) -> None:
        self._table = _EXTENDED if extended else _BASE

    def call_function(self, inputs: List[float], function: int) -> float:
        fn, _, _ = self._table[function]
        a = inputs[0] if len(inputs) > 0 else 0.0
        b = inputs[1] if len(inputs) > 1 else 0.0
        try:
            result = fn(a, b)
            return result if math.isfinite(result) else 0.0
        except Exception:
            return 0.0

    def function_name(self, function: int) -> str:
        return self._table[function][1]

    def input_name(self, input_idx: int) -> str:
        return f"x{input_idx}"

    def arity_of(self, function: int) -> int:
        return self._table[function][2]

    def num_functions(self) -> int:
        return len(self._table)
