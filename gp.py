import itertools
import random
import operator
from deap import algorithms, base, creator, tools, gp
from deap.gp import Primitive
import numpy

import demoBackups
import imageProcessing as improc
import bayesianLearning as bayes
from bayesianLearning import Node, evaluateTree, initSourceTrees, sourceCol
from multiprocessing import Pool


class Generic: pass


class Series(Generic):
    def __init__(self, inputs):
        # inputs should be a list of Parallel or Element instances
        self.inputs = inputs

    def __repr__(self):
        return f"Series({self.inputs})"


class Parallel(Generic):
    def __init__(self, input):
        # Assuming Parallel takes a single input, modify as needed
        self.input = input

    def __repr__(self):
        return f"Parallel({self.input})"


class Element(Generic):
    def __init__(self, char, int1, int2):
        self.char = char
        self.int1 = int1
        self.int2 = int2

    def __repr__(self):
        return f"Element('{self.char}', {self.int1}, {self.int2})"


class Symbol:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.value


class Layer:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


class Accent:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


# Define your primitives (operations and terminals)
def createSeries(*args):
    # args will be a tuple of inputs, can be Parallel or Element instances
    return Series(list(args))


def createParallel(*args):
    return Parallel(list(args))


def createElement(char, int1, int2):
    return Element(char, int1, int2)


def createSimp(n):
    return n


# Trigrams as terminals
symbols = improc.reversedCharMapDbl.keys()
layers = [-1, 0]
accents = [0, 1]

pset = gp.PrimitiveSetTyped("MAIN", [], Parallel)  # arity=0 for a tree-based representation
# pset.addPrimitive(createParallel, [Series] * 4, Parallel)
# pset.addPrimitive(createParallel, [Parallel] * 4, Series)
for arity in range(1, 9):  # From 1 to 4
    pset.addPrimitive(createParallel, [Series] * arity, Parallel)
for arity in range(1, 9):  # From 1 to 4
    pset.addPrimitive(createSeries, [Generic] * arity, Series)

defaultParallel = Parallel([])  # Assuming Parallel can be instantiated like this
defaultSeries = Series([])  # Assuming Series can be instantiated like this

pset.addTerminal(defaultSeries, Series)
pset.addTerminal(defaultParallel, Parallel)

pset.addPrimitive(createElement, [Symbol, Accent, Layer], Element)
pset.addPrimitive(createSimp, [Symbol], Symbol)
pset.addPrimitive(createSimp, [Layer], Layer)
pset.addPrimitive(createSimp, [Accent], Accent)

for c in symbols:
    pset.addTerminal(c, Symbol)
for l in layers:
    pset.addTerminal(l, Layer)
for a in accents:
    pset.addTerminal(a, Accent)

# Define fitness measure
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def traverseTree(individual, index=0, depth=0):
    """
    Recursively traverse the tree from a given index to the bottom.

    :param individual: The GP tree (individual).
    :param index: The current index in the individual list.
    :param depth: The current depth in the tree.
    """
    node = individual[index]
    indent = " " * depth
    print(f"{indent}Node: {node.name}, Depth: {depth}")

    # Check if the node is a primitive (i.e., has children)
    if isinstance(node, gp.Primitive):
        # The next child's index is the current index + 1
        next_index = index + 1
        for _ in range(node.arity):
            # Recursively visit each child and update the next_index
            next_index = traverseTree(individual, next_index, depth + 1)
        return next_index
    else:
        # If it's a terminal node, the next index is just the current index + 1
        return index + 1


def convert2GuideTree(indiv, index=0, depth=0, verbose=False):
    node = indiv[index]
    indent = " " * depth
    if verbose:
        print(f"{indent}Node: {node.name}, Depth: {depth}")
    guideTree = []

    if node.name[:6] == 'create':
        if node.name not in ['createSimp', 'createElement']:
            guideTree.append(node.name[6:].lower())
    elif node.name[0] in ['S', 'P']:
        return [], index + 1
    else:
        try:
            val = int(node.name)
        except ValueError:
            val = node.name
        guideTree.append(val)

    # Check if the node is a primitive (i.e., has children)
    if isinstance(node, gp.Primitive):
        # The next child's index is the current index + 1
        nextIndex = index + 1
        for _ in range(node.arity):
            # Recursively visit each child and update the nextIndex
            insertion, nextIndex = convert2GuideTree(indiv, nextIndex, depth + 1, verbose=verbose)
            if insertion == 0 or insertion:
                # print("inser", insertion)
                if isinstance(insertion, list) and guideTree[0] == 'series' and insertion[0] == 'series':
                    guideTree += insertion[1:]
                else:
                    guideTree.append(insertion)
        if len(guideTree) == 1 and guideTree[0] in ['series', 'parallel']:
            return [], nextIndex
        else:
            if node.name == 'createSimp':
                return guideTree[0], nextIndex
            else:
                return guideTree, nextIndex
    else:
        # If it's a terminal node, the next index is just the current index + 1
        return guideTree[0], index + 1


def convert2PrimTree(tree):
    if tree[0] == 'parallel':
        children = [subchild for child in [convert2PrimTree(child) for child in tree[1:]] for subchild in child]
        return [gp.Primitive("createParallel", args=[Series] * len(tree[1:]), ret=Parallel)] + children
    if tree[0] == 'series':
        children = [subchild for child in [convert2PrimTree(child) for child in tree[1:]] for subchild in child]
        return [gp.Primitive("createSeries", args=[Generic] * len(tree[1:]), ret=Series)] + children
    else:
        return [gp.Primitive("createElement", [Symbol, Accent, Layer], ret=Element),
                                 gp.Primitive("createSimp", [Symbol], ret=Symbol),
                                 gp.Terminal(str(tree[0]), tree[0], Symbol),
                                 gp.Primitive("createSimp", [Accent], ret=Accent),
                                 gp.Terminal(str(tree[1]), tree[1], Accent),
                                 gp.Primitive("createSimp", [Layer], ret=Layer),
                                 gp.Terminal(str(tree[2]), tree[2], Layer)]


sourceNodeCol = initSourceTrees(sourceCol, normalize=True)


def evaluate(individual):
    global sourceNodeCol
    # for node in individual:
    #    print(node.name, end=', ')
    guideTree = convert2GuideTree(individual, verbose=False)[0]
    print('Current Tree', guideTree)
    if not guideTree:
        guideTree = improc.unknownTree
    candidatePrior, candidateLikelihood, candidateTree, videos, layerLst, zeroGraphs = evaluateTree(guideTree,
                                                                                                    sourceNodeCol,
                                                                                                    normalize=True,
                                                                                                    keyLen=8,
                                                                                                    precision=1,
                                                                                                    acceleration=1)

    print('Prior', candidatePrior, '* Likelihood', candidateLikelihood, '=', candidatePrior * candidateLikelihood)
    # Convert individual to a functional representation
    # circuit_function = toolbox.compile(expr=individual)
    # Define your evaluation logic here
    # print("circuit", circuit_function)
    # For example, measure the efficiency of the circuit
    return (candidatePrior * candidateLikelihood,)


def customGenHalfAndHalf(pset, type_=None, min_=0, max_=2, arity_probabilities=None):
    """
    Generate an expression with a custom probability for choosing arity,
    using 'Half and Half' method.
    """
    if type_ is None:
        type_ = pset.ret
    method = random.choice([genFull, genGrow])
    return method(pset, min_, max_, type_, arity_probabilities)


def genFull(pset, min_, max_, type_, arity_probabilities):
    def expr(depth, type_):
        if depth == max_:
            term = random.choice(pset.terminals[type_])
            return term.value if isinstance(term, gp.Terminal) else term
        else:
            prim = random.choice(pset.primitives[type_])
            print("context", pset.context)
            func = pset.context[prim.name]  # Get the function
            arity = choose_arity(prim, arity_probabilities)
            args = [expr(depth + 1, _type) for _type in prim.args[:arity]]
            return func(*args)

    return expr(min_, type_)


def genGrow(pset, min_, max_, type_, arity_probabilities):
    def expr(depth, type_):
        if depth == max_ or (depth >= min_ and random.random() < pset.terminalRatio):
            term = random.choice(pset.terminals[type_])
            return term.value if isinstance(term, gp.Terminal) else term
        else:
            prim = random.choice(pset.primitives[type_])
            func = pset.context[prim.name]  # Get the function
            arity = choose_arity(prim, arity_probabilities)
            args = [expr(depth + 1, _type) for _type in prim.args[:arity]]
            return func(*args)

    return expr(min_, type_)


def choose_arity(prim, arity_probabilities):
    """Choose arity for a primitive based on custom probabilities."""
    if arity_probabilities and len(prim.args) in arity_probabilities:
        arity_options = list(range(1, len(prim.args) + 1))
        probabilities = arity_probabilities[len(prim.args)]
        return random.choices(arity_options, probabilities, k=1)[0]
    else:
        return random.randint(1, len(prim.args))


# Usage example
# Set custom arity probabilities: {arity: [probabilities]}
arity_probabilities = {
    4: [0.2, 0.3, 0.3, 0.2]  # Probabilities for primitives with arity 4
}

toolbox = base.Toolbox()
pool = Pool()
toolbox.register("map", pool.map)
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Evolve the population
population = toolbox.population(n=100)

hof = tools.HallOfFame(5)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)


def customEvoAlg(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True, addIdeal=False, refresh=True):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'max', 'avg', 'min']

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(pop)

    restartCond = 25
    lastFewPops = restartCond

    # Begin the evolution
    for gen in range(1, ngen + 1):
        print("Current Generation", gen)

        # Insert the ideal individual at the start of the second generation
        if gen == 2 and addIdeal:
            ideal = convert2PrimTree(demoBackups.shortSlideShow)
            pop[0] = creator.Individual(ideal)
            pop[0].fitness.values = toolbox.evaluate(pop[0])  # Evaluate its fitness

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, **record)
        if verbose:
            print('-' * 200)
            print("Log Stream")
            print(logbook.stream)

        # Custom action at the end of each generation
        # Example: Print the best individual
        if halloffame:
            print("Best individual so far:", convert2GuideTree(halloffame[0])[0])

        if gen > restartCond and refresh:
            minStats = logbook.select('min')
            print("minimum stats", minStats)
            if abs(numpy.mean(minStats) - minStats[0]) < 0.1:
                pop = toolbox.population(n=100)
                restartCond = lastFewPops + gen
                print("Restarting Everything", restartCond)

    return pop, logbook


# Running the algorithm
# result, log = algorithms.eaSimple(population, toolbox, 0.5, 0.2, 40, stats=stats, halloffame=hof, verbose=True)
pop, log = customEvoAlg(population, toolbox, 0.7, 0.4, 100, stats=stats, halloffame=hof, verbose=True)

for gen in log:
    print(f"Generation: {gen['gen']}")
    print(f"  Max Fitness: {gen['max']}")
    print(f"  Avg Fitness: {gen['avg']}")
    print(f"  Min Fitness: {gen['min']}")

print("Hall of Fame Individuals:")
for individual in hof:
    print(convert2GuideTree(individual)[0])
