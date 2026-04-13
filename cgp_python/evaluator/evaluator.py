"""
CGP++ Python Port — Evaluator
==============================
Mirrors Evaluator.h: iterative and recursive evaluation of CGP individuals,
active-node detection (decode_path / visit_node), and symbolic expression
generation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from ..representation.species import Species
from ..representation.individual import Individual
from ..functions.functions import Functions
from ..parameters import Parameters


class Evaluator:
    """
    Decodes and evaluates CGP individuals.

    Provides:
      - evaluate_iterative  : fast forward-pass over pre-decoded active nodes
      - evaluate_recursive  : recursive evaluation + expression building
      - decode_path         : discover and sort active nodes
    """

    def __init__(
        self,
        params: Parameters,
        functions: Functions,
        species: Species,
    ) -> None:
        self.params     = params
        self.functions  = functions
        self.species    = species
        self.num_inputs  = params.num_inputs
        self.num_outputs = params.num_outputs
        self.genome_size = species.genome_size
        self.max_arity   = params.max_arity
        self.evaluate_expression = params.evaluate_expression

        # Working maps (cleared before each evaluation)
        self._node_value_map: Dict[int, float] = {}
        self._expression_map: Dict[int, str]   = {}
        self._visited: Dict[int, bool]         = {}

    # ── internal helpers ──────────────────────────────────────

    def _gene_at(self, genome: "np.ndarray", position: int) -> int:
        return int(genome[position])

    def _clear_maps(self) -> None:
        self._node_value_map.clear()
        self._expression_map.clear()

    # ── active-node detection ─────────────────────────────────

    def _visit_node(
        self,
        genome: "np.ndarray",
        active_nodes: List[int],
        node_num: int,
    ) -> None:
        """Recursive DFS to collect active nodes (mirrors visit_node)."""
        if node_num in self._visited:
            return
        if node_num < self.num_inputs:
            return

        active_nodes.append(node_num)
        self._visited[node_num] = True

        pos = self.species.position_from_node_number(node_num)
        for i in range(1, self.max_arity + 1):
            conn = self._gene_at(genome, pos + i)
            self._visit_node(genome, active_nodes, conn)

    def decode_path(self, individual: Individual) -> None:
        """
        Discover active nodes by tracing from outputs backwards.
        Sorts the resulting list and stores it on the individual.
        Mirrors Evaluator::decode_path.
        """
        genome = individual.genome
        active_nodes: List[int] = []
        self._visited = {}

        for i in range(self.num_outputs):
            output_pos = self.genome_size - 1 - i
            output_val = self._gene_at(genome, output_pos)
            self._visit_node(genome, active_nodes, output_val)

        active_nodes.sort()
        individual.active_nodes = active_nodes

    # ── iterative evaluation ──────────────────────────────────

    def evaluate_iterative(
        self,
        individual: Individual,
        inputs: List[float],
        outputs: List[float],
    ) -> None:
        """
        Fast iterative evaluation over pre-decoded active_nodes.
        Mirrors Evaluator::evaluate_iterative.
        Caller must call decode_path first (or use evaluate_recursive).
        """
        genome = individual.genome
        self._clear_maps()

        for node_num in individual.active_nodes:
            pos      = self.species.position_from_node_number(node_num)
            function = self._gene_at(genome, pos)
            args: List[float] = []
            for i in range(self.max_arity):
                conn = self._gene_at(genome, pos + i + 1)
                if conn < self.num_inputs:
                    args.append(inputs[conn])
                else:
                    args.append(self._node_value_map.get(conn, 0.0))
            result = self.functions.call_function(args, function)
            self._node_value_map[node_num] = result

        outputs.clear()
        for i in range(self.num_outputs):
            output_pos = self.genome_size - 1 - i
            output_val = self._gene_at(genome, output_pos)
            if output_val < self.num_inputs:
                outputs.append(inputs[output_val])
            else:
                outputs.append(self._node_value_map.get(output_val, 0.0))

    # ── recursive evaluation ──────────────────────────────────

    def _evaluate_node(
        self,
        genome: "np.ndarray",
        active_nodes: List[int],
        inputs: List[float],
        node_num: int,
    ) -> Tuple[float, str]:
        """
        Recursively evaluate node_num.
        Returns (value, expression_string).
        Mirrors Evaluator::evaluate_node.
        """
        # Memoised?
        if node_num in self._node_value_map:
            value = self._node_value_map[node_num]
            expr  = self._expression_map.get(node_num, "")
            return value, expr

        # Input node
        if node_num < self.num_inputs:
            value = inputs[node_num]
            name  = self.functions.input_name(node_num) if self.evaluate_expression else ""
            self._node_value_map[node_num] = value
            self._expression_map[node_num] = name
            return value, name

        # Interior node
        active_nodes.append(node_num)
        pos      = self.species.position_from_node_number(node_num)
        function = self._gene_at(genome, pos)
        num_args = self.functions.arity_of(function)
        fn_name  = self.functions.function_name(function) if self.evaluate_expression else ""

        args: List[float] = []
        sub_exprs: List[str] = []

        for i in range(1, self.max_arity + 1):
            conn = self._gene_at(genome, pos + i)
            val, sub_expr = self._evaluate_node(genome, active_nodes, inputs, conn)
            args.append(val)
            sub_exprs.append(sub_expr)

        result = self.functions.call_function(args, function)
        self._node_value_map[node_num] = result

        expr = ""
        if self.evaluate_expression:
            used = sub_exprs[:num_args]
            expr = f"{fn_name}({' '.join(used)})"
            self._expression_map[node_num] = expr

        return result, expr

    def evaluate_recursive(
        self,
        individual: Individual,
        inputs: List[float],
        outputs: List[float],
    ) -> None:
        """
        Recursive evaluation with simultaneous expression building.
        Populates individual.active_nodes and individual.expressions.
        Mirrors Evaluator::evaluate_recursive.
        """
        genome = individual.genome
        self._clear_maps()
        active_nodes: List[int] = []

        if self.evaluate_expression:
            individual.clear_expressions()

        outputs.clear()
        for i in range(self.num_outputs):
            output_pos = self.genome_size - 1 - i
            output_val = self._gene_at(genome, output_pos)
            value, expr = self._evaluate_node(genome, active_nodes, inputs, output_val)
            outputs.append(value)
            if self.evaluate_expression:
                individual.add_expression(expr)

        individual.active_nodes = sorted(set(active_nodes))

    # ── convenience: evaluate and return outputs ───────────────

    def evaluate(
        self,
        individual: Individual,
        inputs: List[float],
        *,
        iterative: bool = True,
    ) -> List[float]:
        """
        Evaluate individual on inputs.  Returns list of output values.

        Args:
            individual : CGP individual (active_nodes must be populated for
                         iterative mode; call decode_path first).
            inputs     : input values.
            iterative  : use fast iterative mode (default True).
        """
        outputs: List[float] = []
        if iterative:
            if not individual.active_nodes:
                self.decode_path(individual)
            self.evaluate_iterative(individual, inputs, outputs)
        else:
            self.evaluate_recursive(individual, inputs, outputs)
        return outputs
