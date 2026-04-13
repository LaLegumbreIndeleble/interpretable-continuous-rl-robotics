"""
cgp_python — Python port of CGP++ (github.com/RomanKalkreuth/cgp-plusplus)
===========================================================================
Public API surface.  Import everything you need from here:

    from cgp_python import Parameters, build, OnePlusLambda, MuPlusLambda

Quick-start example (symbolic regression):

    import numpy as np
    from cgp_python import Parameters, build

    params = Parameters(
        num_inputs=2, num_outputs=1, num_function_nodes=50,
        num_functions=4, max_arity=2, levels_back=50,
        lambda_=4, mutation_rate=0.01, max_generations=50_000,
        minimizing_fitness=True, ideal_fitness=0.0,
    )

    def eval_fn(ind, evaluator):
        total_error = 0.0
        for x0, x1, y_true in dataset:
            out = evaluator.evaluate(ind, [x0, x1])
            total_error += (out[0] - y_true) ** 2
        return total_error / len(dataset)

    algo = build(params, eval_fn)
    evals, best = algo.evolve()
    champion = algo.champion()
"""

from .parameters import (
    Parameters,
    PROBABILISTIC_POINT_MUTATION,
    SINGLE_ACTIVE_GENE_MUTATION,
    INVERSION_MUTATION,
    DUPLICATION_MUTATION,
    BLOCK_CROSSOVER,
    DISCRETE_CROSSOVER,
    ONE_PLUS_LAMBDA,
    MU_PLUS_LAMBDA,
)
from .representation.species    import Species
from .representation.individual import Individual
from .functions.functions       import Functions
from .functions.mathematical_functions import MathematicalFunctions
from .evaluator.evaluator       import Evaluator
from .fitness.fitness           import Fitness
from .mutation.mutation         import Mutation
from .mutation.mutation_pipeline import MutationPipeline
from .population.static_population import StaticPopulation
from .algorithm.one_plus_lambda import OnePlusLambda
from .algorithm.mu_plus_lambda  import MuPlusLambda
from .variation.mutation.probabilistic_point import ProbabilisticPoint
from .variation.mutation.single_active_gene  import SingleActiveGene
from .variation.mutation.inversion           import Inversion
from .variation.mutation.duplication         import Duplication
from .variation.crossover.discrete_crossover import DiscreteCrossover
from .variation.crossover.block_crossover    import BlockCrossover

import numpy as np
from typing import Callable, Union


def build(
    params: Parameters,
    eval_fn: Callable,
    *,
    functions: Union[Functions, None] = None,
    seed: Union[int, None] = None,
    extended_functions: bool = True,
) -> Union[OnePlusLambda, MuPlusLambda]:
    """
    Convenience factory — wire up all components and return a ready-to-run
    algorithm object.

    Args:
        params             : configuration (Parameters dataclass)
        eval_fn            : fitness callback:
                                 eval_fn(individual, evaluator) -> float
        functions          : custom function set; if None, uses
                             MathematicalFunctions(extended=extended_functions)
        seed               : random seed (overrides params.global_seed)
        extended_functions : use extended 18-function set (default True);
                             ignored when `functions` is provided

    Returns:
        OnePlusLambda or MuPlusLambda instance (call .evolve() to run)
    """
    rng = np.random.default_rng(seed if seed is not None else params.global_seed)

    if functions is None:
        functions = MathematicalFunctions(extended=extended_functions)

    params.num_functions = functions.num_functions()
    params.set_genome_size()

    species   = Species(params)
    fitness   = Fitness(minimizing=params.minimizing_fitness, ideal_fitness=params.ideal_fitness)
    evaluator = Evaluator(params, functions, species)
    mutation  = Mutation(params, rng, species)

    if params.algorithm == ONE_PLUS_LAMBDA:
        return OnePlusLambda(
            params, species, evaluator, mutation, fitness, eval_fn, rng,
            report_interval=params.report_interval,
        )
    return MuPlusLambda(
        params, species, evaluator, mutation, fitness, eval_fn, rng,
        report_interval=params.report_interval,
    )
