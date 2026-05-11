# DEAP API Catalog

Flat index: link → method/class names. DEAP 1.4.3.

---

## https://deap.readthedocs.io/en/master/py-modindex.html

- `deap.algorithms`
- `deap.base`
- `deap.benchmarks`
- `deap.benchmarks.binary`
- `deap.benchmarks.gp`
- `deap.benchmarks.movingpeaks`
- `deap.benchmarks.tools`
- `deap.cma`
- `deap.creator`
- `deap.gp`
- `deap.tools`

---

## https://deap.readthedocs.io/en/master/api/creator.html

- `creator.create(name, base, **attrs)`
- `creator.class_replacers`

---

## https://deap.readthedocs.io/en/master/api/base.html

- `base.Toolbox`
- `Toolbox.register(alias, function, *args, **kwargs)`
- `Toolbox.unregister(alias)`
- `Toolbox.decorate(alias, *decorators)`
- `Toolbox.clone` (default attribute, deepcopy)
- `Toolbox.map` (default attribute, builtin map)
- `base.Fitness`
- `Fitness.dominates(other, obj=slice(None))`
- `Fitness.valid` (property)
- `Fitness.values` (property)
- `Fitness.weights` (class attribute)
- `Fitness.wvalues` (attribute)

---

## https://deap.readthedocs.io/en/master/api/tools.html

Initialization:
- `tools.initRepeat(container, func, n)`
- `tools.initIterate(container, generator)`
- `tools.initCycle(container, seq_func, n=1)`

Crossover:
- `tools.cxOnePoint(ind1, ind2)`
- `tools.cxTwoPoint(ind1, ind2)`
- `tools.cxUniform(ind1, ind2, indpb)`
- `tools.cxPartialyMatched(ind1, ind2)`
- `tools.cxUniformPartialyMatched(ind1, ind2, indpb)`
- `tools.cxOrdered(ind1, ind2)`
- `tools.cxBlend(ind1, ind2, alpha)`
- `tools.cxESBlend(ind1, ind2, alpha)`
- `tools.cxESTwoPoint(ind1, ind2)`
- `tools.cxSimulatedBinary(ind1, ind2, eta)`
- `tools.cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)`
- `tools.cxMessyOnePoint(ind1, ind2)`

Mutation:
- `tools.mutGaussian(individual, mu, sigma, indpb)`
- `tools.mutShuffleIndexes(individual, indpb)`
- `tools.mutFlipBit(individual, indpb)`
- `tools.mutUniformInt(individual, low, up, indpb)`
- `tools.mutPolynomialBounded(individual, eta, low, up, indpb)`
- `tools.mutESLogNormal(individual, c, indpb)`

Selection:
- `tools.selTournament(individuals, k, tournsize, fit_attr='fitness')`
- `tools.selRoulette(individuals, k, fit_attr='fitness')`
- `tools.selNSGA2(individuals, k, nd='standard')`
- `tools.selNSGA3(individuals, k, ref_points, nd='log', best_point=None, worst_point=None, extreme_points=None, return_memory=False)`
- `tools.selNSGA3WithMemory(ref_points, nd='log')`
- `tools.uniform_reference_points(nobj, p=4, scaling=None)`
- `tools.selSPEA2(individuals, k)`
- `tools.selRandom(individuals, k)`
- `tools.selBest(individuals, k, fit_attr='fitness')`
- `tools.selWorst(individuals, k, fit_attr='fitness')`
- `tools.selDoubleTournament(individuals, k, fitness_size, parsimony_size, fitness_first, fit_attr='fitness')`
- `tools.selStochasticUniversalSampling(individuals, k, fit_attr='fitness')`
- `tools.selTournamentDCD(individuals, k)`
- `tools.selLexicase(individuals, k)`
- `tools.selEpsilonLexicase(individuals, k, epsilon)`
- `tools.selAutomaticEpsilonLexicase(individuals, k)`
- `tools.sortNondominated(individuals, k, first_front_only=False)`
- `tools.sortLogNondominated(individuals, k, first_front_only=False)`

Migration:
- `tools.migRing(populations, k, selection, replacement=None, migarray=None)`

Statistics:
- `tools.Statistics(key=None)`
- `Statistics.register(name, function, *args, **kwargs)`
- `Statistics.compile(data)`
- `tools.MultiStatistics(**kwargs)`
- `MultiStatistics.register(name, function, *args, **kwargs)`
- `MultiStatistics.compile(data)`

Logbook:
- `tools.Logbook`
- `Logbook.record(**infos)`
- `Logbook.select(*names)`
- `Logbook.pop(index=0)`
- `Logbook.chapters` (attribute)
- `Logbook.header` (attribute)
- `Logbook.log_header` (attribute)
- `Logbook.stream` (property)

Hall of Fame:
- `tools.HallOfFame(maxsize, similar=operator.eq)`
- `HallOfFame.update(population)`
- `HallOfFame.insert(item)`
- `HallOfFame.remove(index)`
- `HallOfFame.clear()`
- `tools.ParetoFront(similar=operator.eq)`
- `ParetoFront.update(population)`

History:
- `tools.History()`
- `History.update(individuals)`
- `History.decorator` (property)
- `History.getGenealogy(individual, max_depth=None)`
- `History.genealogy_tree` (attribute)
- `History.genealogy_history` (attribute)

Constraints:
- `tools.DeltaPenalty(feasibility, delta, distance=None)`
- `tools.ClosestValidPenalty(feasibility, feasible, alpha, distance=None)`

---

## https://deap.readthedocs.io/en/master/api/algo.html

Complete algorithms:
- `algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)`
- `algorithms.eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)`
- `algorithms.eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True)`
- `algorithms.eaGenerateUpdate(toolbox, ngen, stats=None, halloffame=None, verbose=True)`

Variations:
- `algorithms.varAnd(population, toolbox, cxpb, mutpb)`
- `algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)`

CMA-ES (`deap.cma`, documented on same page):
- `cma.Strategy(centroid, sigma, **kwargs)` — kwargs: `lambda_`, `mu`, `cmatrix`, `weights`, `cs`, `damps`, `ccum`, `ccov1`, `ccovmu`
- `Strategy.computeParams(params)`
- `Strategy.generate(ind_init)`
- `Strategy.update(population)`
- `cma.StrategyOnePlusLambda(parent, sigma, **kwargs)` — kwargs: `lambda_`, `d`, `ptarg`, `cp`, `cc`, `ccov`, `pthresh`
- `StrategyOnePlusLambda.computeParams(params)`
- `StrategyOnePlusLambda.generate(ind_init)`
- `StrategyOnePlusLambda.update(population)`
- `cma.StrategyMultiObjective(population, sigma, **kwargs)` — kwargs: `mu`, `lambda_`, `indicator`, `d`, `ptarg`, `cp`, `cc`, `ccov`, `pthresh`
- `StrategyMultiObjective.generate(ind_init)`
- `StrategyMultiObjective.update(population)`

---

## https://deap.readthedocs.io/en/master/api/gp.html

Classes:
- `gp.PrimitiveTree(content)`
- `PrimitiveTree.from_string(string, pset)` (classmethod)
- `PrimitiveTree.height` (property)
- `PrimitiveTree.root` (property)
- `PrimitiveTree.searchSubtree(begin)`
- `gp.PrimitiveSet(name, arity, prefix='ARG')`
- `PrimitiveSet.addPrimitive(primitive, arity, name=None)`
- `PrimitiveSet.addTerminal(terminal, name=None)`
- `PrimitiveSet.addEphemeralConstant(name, ephemeral)`
- `gp.PrimitiveSetTyped(name, in_types, ret_type, prefix='ARG')`
- `PrimitiveSetTyped.addPrimitive(primitive, in_types, ret_type, name=None)`
- `PrimitiveSetTyped.addTerminal(terminal, ret_type, name=None)`
- `PrimitiveSetTyped.addEphemeralConstant(name, ephemeral, ret_type)`
- `PrimitiveSetTyped.addADF(adfset)`
- `PrimitiveSetTyped.renameArguments(**kwargs)`
- `PrimitiveSetTyped.terminalRatio` (property)
- `gp.Primitive(name, args, ret)`
- `gp.Terminal(terminal, symbolic, ret)`
- `gp.MetaEphemeral(name, func, ret=object, id_=None)`

Compile:
- `gp.compile(expr, pset)`
- `gp.compileADF(expr, psets)`

Generation:
- `gp.genFull(pset, min_, max_, type_=None)`
- `gp.genGrow(pset, min_, max_, type_=None)`
- `gp.genHalfAndHalf(pset, min_, max_, type_=None)`
- `gp.genRamped(pset, min_, max_, type_=None)` (deprecated alias)

GP crossover:
- `gp.cxOnePoint(ind1, ind2)`
- `gp.cxOnePointLeafBiased(ind1, ind2, termpb)`
- `gp.cxSemantic(ind1, ind2, gen_func=genGrow, pset=None, min=2, max=6)`

GP mutation:
- `gp.mutShrink(individual)`
- `gp.mutUniform(individual, expr, pset)`
- `gp.mutNodeReplacement(individual, pset)`
- `gp.mutEphemeral(individual, mode)` — `mode` in `{"one", "all"}`
- `gp.mutInsert(individual, pset)`
- `gp.mutSemantic(individual, gen_func=genGrow, pset=None, ms=None, min=2, max=6)`

Bloat control:
- `gp.staticLimit(key, max_value)` (decorator factory)

Visualization:
- `gp.graph(expr)` → `(nodes, edges, labels)`

---

## https://deap.readthedocs.io/en/master/api/benchmarks.html

Note: page returned 403 on direct fetch; entries below come from source modules and indirect references.

`deap.benchmarks` (single-objective fitness functions, all returning `(value,)`):
- `benchmarks.rand(individual)`
- `benchmarks.plane(individual)`
- `benchmarks.sphere(individual)`
- `benchmarks.cigar(individual)`
- `benchmarks.rosenbrock(individual)`
- `benchmarks.h1(individual)`
- `benchmarks.ackley(individual)`
- `benchmarks.bohachevsky(individual)`
- `benchmarks.griewank(individual)`
- `benchmarks.rastrigin(individual)`
- `benchmarks.rastrigin_scaled(individual)`
- `benchmarks.rastrigin_skew(individual)`
- `benchmarks.schaffer(individual)`
- `benchmarks.schwefel(individual)`
- `benchmarks.himmelblau(individual)`
- `benchmarks.shekel(individual, a, c)`
- Multi-objective (return tuples): `benchmarks.fonseca`, `benchmarks.kursawe`, `benchmarks.schaffer_mo`, `benchmarks.zdt1`...`zdt6`, `benchmarks.dtlz1`...`dtlz7`

`deap.benchmarks.binary`:
- `benchmarks.binary.chuang_f1(individual)`
- `benchmarks.binary.chuang_f2(individual)`
- `benchmarks.binary.chuang_f3(individual)`
- `benchmarks.binary.royal_road1(individual, order)`
- `benchmarks.binary.royal_road2(individual, order)`

`deap.benchmarks.gp` (symbolic regression test problems):
- `benchmarks.gp.kotanchek(data)`
- `benchmarks.gp.salustowicz_1d(data)`
- `benchmarks.gp.salustowicz_2d(data)`
- `benchmarks.gp.unwrapped_ball(data)`
- `benchmarks.gp.rational_polynomial(data)`
- `benchmarks.gp.sin_cos(data)`
- `benchmarks.gp.ripple(data)`
- `benchmarks.gp.rational_polynomial2(data)`

`deap.benchmarks.movingpeaks`:
- `benchmarks.movingpeaks.MovingPeaks(dim, random=random, **kwargs)`
- `MovingPeaks.__call__(individual, count=True)`
- `MovingPeaks.changePeaks()` — re-orders position/height/width/number
- `MovingPeaks.globalMaximum()`
- `MovingPeaks.maximums()`
- `benchmarks.movingpeaks.cone(individual, position, height, width)`
- `benchmarks.movingpeaks.function1(individual, position, height, width)`
- `benchmarks.movingpeaks.SCENARIO_1` (dict)
- `benchmarks.movingpeaks.SCENARIO_2` (dict)
- `benchmarks.movingpeaks.SCENARIO_3` (dict)

`deap.benchmarks.tools`:
- `benchmarks.tools.convergence(first_front, optimal_front)`
- `benchmarks.tools.diversity(first_front, first, last)`
- `benchmarks.tools.hypervolume(front, ref=None)`
- `benchmarks.tools.igd(A, Z)` (inverted generational distance)
- `benchmarks.tools.translate(vector)` (decorator class)
- `benchmarks.tools.rotate(matrix)` (decorator class)
- `benchmarks.tools.scale(factor)` (decorator class)
- `benchmarks.tools.noise(noise)` (decorator class)