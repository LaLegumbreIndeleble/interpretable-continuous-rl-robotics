"""
CGP++ Python Port — Mutation
==============================
Mirrors Mutation.h.

Factory that constructs a MutationPipeline from the mutation_operators
list in Parameters, then exposes a single mutate() entry-point.
"""

from __future__ import annotations
import numpy as np

from ..parameters import (
    Parameters,
    PROBABILISTIC_POINT_MUTATION,
    SINGLE_ACTIVE_GENE_MUTATION,
    INVERSION_MUTATION,
    DUPLICATION_MUTATION,
)
from ..representation.species import Species
from ..representation.individual import Individual
from ..variation.mutation.probabilistic_point import ProbabilisticPoint
from ..variation.mutation.single_active_gene  import SingleActiveGene
from ..variation.mutation.inversion           import Inversion
from ..variation.mutation.duplication         import Duplication
from .mutation_pipeline import MutationPipeline


class Mutation:
    """Builds and owns the mutation pipeline."""

    def __init__(
        self,
        params: Parameters,
        rng: np.random.Generator,
        species: Species,
    ) -> None:
        self.pipeline = MutationPipeline()

        _factory = {
            PROBABILISTIC_POINT_MUTATION: lambda: ProbabilisticPoint(params, rng, species),
            SINGLE_ACTIVE_GENE_MUTATION:  lambda: SingleActiveGene(params, rng, species),
            INVERSION_MUTATION:           lambda: Inversion(params, rng, species),
            DUPLICATION_MUTATION:         lambda: Duplication(params, rng, species),
        }

        for op_type in params.mutation_operators:
            if op_type in _factory:
                self.pipeline.add(_factory[op_type]())

        # Fallback: if the list was empty, use probabilistic point
        if not self.pipeline._operators:
            self.pipeline.add(ProbabilisticPoint(params, rng, species))

    def mutate(self, individual: Individual) -> None:
        self.pipeline.mutate(individual)
