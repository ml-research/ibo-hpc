import random
import numpy as np
from ihpo.search_spaces import SearchSpace
from ihpo.consts.dtypes import MetaType
from copy import deepcopy
from deap import base, creator, tools, algorithms

def get_sampling_function(search_space: SearchSpace, key: str):

    sample = search_space.sample()[0]

    def get_sample(*args):
        return sample[key]

    return get_sample

def get_crossover(search_space: SearchSpace):

    cont_params = [idx for idx, val in enumerate(search_space.get_search_space_definition().values()) if val['dtype'] == 'float']
    disc_params = [idx for idx, val in enumerate(search_space.get_search_space_definition().values()) if val['dtype'] != 'float']

    def do_crossover(ind1, ind2, alpha=0.5):

        # Continuous: Blend Crossover
        for i in cont_params:
            d = abs(ind1[i] - ind2[i])
            min_val, max_val = min(ind1[i], ind2[i]), max(ind1[i], ind2[i])
            ind1[i] = random.uniform(min_val - alpha * d, max_val + alpha * d)
            ind2[i] = random.uniform(min_val - alpha * d, max_val + alpha * d)

        # Discrete: One-Point Crossover
        if random.random() < 0.5:  # 50% chance of swapping discrete value
            ind1_copy, ind2_copy = deepcopy(ind1), deepcopy(ind2)
            for idx in disc_params:
                ind1[idx] = ind2_copy[idx]
                ind2[idx] = ind1_copy[idx]

        return ind1, ind2

    return do_crossover

def get_mutate(search_space: SearchSpace):

    cont_params = [(idx, val) for idx, val in enumerate(search_space.get_search_space_definition().values()) if val['dtype'] == 'float']
    disc_params = [(idx, val) for idx, val in enumerate(search_space.get_search_space_definition().values()) if val['dtype'] == 'int']
    cat_params = [(idx, val) for idx, val in enumerate(search_space.get_search_space_definition().values()) if val['dtype'] == 'cat']

    def do_mutation(individual, low, up, eta=20, indpb=0.2):
        # Continuous: Polynomial Mutation
        for i, val in cont_params:
            if random.random() < indpb:
                delta = (val['max'] - val['min']) * (random.random() - 0.5) * 0.1  # Small change
                individual[i] = max(val['min'], min(val['max'], individual[i] + delta))

        # Discrete: Increment/Decrement
        for i, val in disc_params:
            if random.random() < indpb:
                individual[i] += random.choice([-1, 1])
                low, up = min(val['allowed']), max(val['allowed'])
                individual[i] = max(low, min(up, individual[i]))

        # Categorical: Random Reassignment
        for i, val in cat_params:
            if random.random() < indpb:
                allowed_vals = val['allowed']
                rnd_idx = random.randint(0, len(allowed_vals) - 1)  # Random category selection
                individual[i] = allowed_vals[rnd_idx]

        return (individual,)
    
    return do_mutation

def setup_deap(search_space: SearchSpace, fitness_function):
    # DEAP Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness function
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    for key, val in search_space.get_search_space_definition().items():
        if val['type'] == MetaType.REAL:
            toolbox.register(key, get_sampling_function(search_space, key), val['min'], val['max'])

        elif val['type'] == MetaType.DISCRETE:
            toolbox.register(key, get_sampling_function(search_space, key), 0, len(val['allowed']) - 1)

    # Define Individual (Combination of continuous + discrete parameters)
    indiviaual = [getattr(toolbox, var) for var in search_space.get_search_space_definition().keys()]
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    tuple(indiviaual), n=1)

    # Population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic Operators
    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", get_crossover(search_space), alpha=0.5)
    toolbox.register("mutate", get_mutate(search_space), low=None, up=None, eta=1.0, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# Evolutionary Optimization
def optimize_evolutionary(search_space: SearchSpace, fitness_function, best_n=10):
    toolbox = setup_deap(search_space, fitness_function)
    population = toolbox.population(n=50)  # Create initial population
    ngen, cxpb, mutpb = 10, 0.5, 0.2  # Generations, crossover prob, mutation prob

    # Run genetic algorithm
    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
                        stats=None, halloffame=None, verbose=True)

    best_ind = tools.selBest(population, k=best_n)  # Get best individuals

    # transform to dict-fitness tuples
    configs, scores = [], []
    for cfg in best_ind:
        genotype = list(cfg)
        scores.append(cfg.fitness.values[-1])
        cfg_dict = {k: v for k, v in zip(search_space.get_search_space_definition().keys(), genotype)}
        configs.append(cfg_dict)

    return configs, scores

