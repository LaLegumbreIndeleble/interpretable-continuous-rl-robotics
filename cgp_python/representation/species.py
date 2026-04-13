"""
CGP++ Python Port — Species
============================
Mirrors Species.h: genome layout, gene-range queries, and node/position
mapping for an integer-encoded CGP genome.

Genome layout (for num_nodes interior nodes, max_arity connection slots):
  [f0, c0_0, c0_1, ...,  f1, c1_0, c1_1, ...,  fN-1, ...,  out0, out1, ...]
   ^--- node 0 block ---^  ^--- node 1 block ---^           ^--- outputs ---^

  Block size = max_arity + 1  (1 function gene + max_arity connection genes)
  Output genes start at index: num_nodes * (max_arity + 1)

Node numbering:
  input nodes  : 0 .. num_inputs - 1
  interior nodes: num_inputs .. num_inputs + num_nodes - 1
  output nodes (conceptual): num_inputs + num_nodes .. (ignored in genome)
"""

from __future__ import annotations
import numpy as np
from ..parameters import Parameters

CONNECTION_GENE = 0
FUNCTION_GENE   = 1
OUTPUT_GENE     = 2


class Species:
    """Genome layout and gene-range logic (port of Species<G>)."""

    def __init__(self, params: Parameters) -> None:
        self.params = params
        self.num_nodes     = params.num_function_nodes
        self.num_inputs    = params.num_inputs
        self.num_outputs   = params.num_outputs
        self.num_functions = params.num_functions
        self.max_arity     = params.max_arity
        self.levels_back   = params.levels_back
        self.genome_size   = self._calc_genome_size()

    # ── genome size ───────────────────────────────────────────

    def _calc_genome_size(self) -> int:
        return self.num_nodes * (self.max_arity + 1) + self.num_outputs

    # ── gene-type decoding ────────────────────────────────────

    def decode_genotype_at(self, position: int) -> int:
        """Return CONNECTION_GENE, FUNCTION_GENE, or OUTPUT_GENE."""
        if position >= self.num_nodes * (self.max_arity + 1):
            return OUTPUT_GENE
        if position % (self.max_arity + 1) == 0:
            return FUNCTION_GENE
        return CONNECTION_GENE

    # ── node ↔ position mapping ───────────────────────────────

    def node_number_from_position(self, position: int) -> int:
        gene_type = self.decode_genotype_at(position)
        if gene_type == OUTPUT_GENE:
            return self.num_inputs + self.num_nodes + (
                position - self.num_nodes * self.max_arity)
        return self.num_inputs + (position // (self.max_arity + 1))

    def position_from_node_number(self, node_number: int) -> int:
        """Return the position of the *function gene* of an interior node."""
        return (node_number - self.num_inputs) * (self.max_arity + 1)

    # ── gene bounds ───────────────────────────────────────────

    def min_gene(self, position: int) -> int:
        gene_type = self.decode_genotype_at(position)
        if gene_type == OUTPUT_GENE:
            val = self.num_inputs + self.num_nodes - self.levels_back
        elif gene_type == FUNCTION_GENE:
            return 0
        else:
            node_number = self.node_number_from_position(position)
            val = node_number - self.levels_back
        return max(val, 0)

    def max_gene(self, position: int) -> int:
        gene_type = self.decode_genotype_at(position)
        if gene_type == OUTPUT_GENE:
            return self.num_inputs + self.num_nodes - 1
        if gene_type == FUNCTION_GENE:
            return self.num_functions - 1
        return self.node_number_from_position(position) - 1

    # ── random genome initialisation ─────────────────────────

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        """Return a freshly randomised integer genome."""
        g = np.zeros(self.genome_size, dtype=np.int32)
        for i in range(self.genome_size):
            lo = self.min_gene(i)
            hi = self.max_gene(i)
            if lo <= hi:
                g[i] = int(rng.integers(lo, hi + 1))
            else:
                g[i] = lo
        return g
