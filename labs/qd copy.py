# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv

from pcgym.envs import PcgrlEnv
from typing import Tuple
from pcgym.envs.helper import get_string_map

import qdax
from qdax.core.map_elites import MAPElites

# %% n-dimensional function with a strange topology
@partial(np.vectorize, signature="(d)->()")
def griewank_function(pop):  # this is kind of our fitness function (except we a minimizing)
    return 1 + np.sum(pop**2) / 4000 - np.prod(np.cos(pop / np.sqrt(np.arange(1, pop.size + 1))))

"""
@partial(np.vectorize, signature="(d)->(d)", excluded=[0])
def mutate(sigma, pop):  # What are we doing here?
    return pop + np.random.normal(0, sigma, pop.shape)
"""


@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover2(x1, x2):
    n = np.random.randint(0,99)
    if n <= 33:
        return uniform_crossover(x1,x2)
    if n <= 66:
        return one_point_crossover(x1,x2)
    if n <= 99:
        return two_point_crossover(x1,x2)

    return x1 * np.random.rand() + x2 * (1 - np.random.rand())
    
@partial(np.vectorize, signature="(d),(d)->(d)")
def crossover(x1, x2):
    return x1 * np.random.rand() + x2 * (1 - np.random.rand())

def step(pop, cfg):
    loss = griewank_function(pop)
    idxs = np.argsort(loss)[: int(cfg.population * cfg.proportion)]  # select best
    best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))  # cross over
    pop = crossover(best, best[np.random.permutation(best.shape[0])])  # mutate
    return mutate(cfg.sigma, pop), loss  # return new generation and loss

def uniform_crossover(x1, x2):
    mask = np.random.rand(*x1.shape) < 0.5
    return np.where(mask, x1, x2)

def one_point_crossover(x1, x2):
    point = np.random.randint(1, len(x1))
    return np.concatenate([x1[:point], x2[point:]])

def two_point_crossover(x1, x2):
    p1, p2 = sorted(np.random.choice(len(x1), 2, replace=False))
    return np.concatenate([x1[:p1], x2[p1:p2], x1[p2:]])


# %% Setup
def main(cfg):
    """
    pop = np.random.rand(cfg.population, cfg.dimensions)
    for gen in range(cfg.generation):
        break
        pop, fitness = step(pop, cfg)
        fitnesses.append(fitness.min())
        print(f"Generation {gen}: Best fitness = {fitness.min()}")
    """
    # map gym shit
    env, pop = init_pcgym(cfg)
    fitnesses = []

    for gen in range(cfg.generation):
        #print(pop)
        print(pop.shape)
        fitness, behaviour = map(np.array, zip(*list(map(lambda p: eval(env, p,cfg), pop))))
        #print (fitness)
        idxs = np.argsort(-fitness)[: int(cfg.population * cfg.proportion)]
        #best = np.tile(pop[idxs], (int(cfg.population * cfg.proportion), 1))
        best = pop[idxs]
        print(np.sort(fitness)[::-1][:20])
        #print(idxs)
        elite = best.copy()
        pop = crossover3(best, cfg)

        fitnesses.append(fitness.max())
        
        #print(pop)
        pop = mutate(cfg,env,pop)
        #pop = np.append(mutate(cfg, env, pop), best[0:5], axis=0)
        pop = np.concatenate([pop, elite], axis=0)
        print(f"Generation {gen}: Best fitness = {fitness.max()}")
        if gen == 0:
            bestest = pop[idxs.max()]

            env.map = bestest
            snot = env.render()
            plt.figure(figsize=(10,2), dpi=160)
            plt.imshow(snot,interpolation='nearest')
            plt.show()

    bestest = pop[idxs.max()]

    #fitness, behaviour = eval(env,pop[0])
    env.map = bestest
    snot = env.render()
    plt.figure(figsize=(10,2), dpi=160)
    plt.imshow(snot,interpolation='nearest')
    plt.show()
    
    # our_plot_function(griewank_function)
    # plt.plot(fitnesses)
    # plt.yscale("log")
    # plt.xlabel("Generation")
    # plt.ylabel("Best Fitness")

def crossover3(pop, cfg):
    d = pop.shape[1:]
    newpop = np.empty((cfg.population, *d), dtype=pop.dtype)
    for i in range(cfg.population):
        x1 = pop[np.random.randint(len(pop))]
        x2 = pop[np.random.randint(len(pop))]
        n = np.random.randint(0, 100)
        #if n <= 33:
        #    child = uniform_crossover(x1, x2)
        if n <= 66:
            child = one_point_crossover(x1, x2)
        else:
            child = two_point_crossover(x1, x2)
        newpop[i] = child
    
    return newpop
# %% Init population (maps)
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    env.reset()
    pop = np.random.randint(0, env.get_num_tiles(), (cfg.n, *env._rep._map.shape))  # type: ignore
    
    return env, pop

def init_pcgym2(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")

    
    pop = []
    for _ in range(cfg.n):
        env.reset()
        pop.append(env._rep._map.copy())
    
    pop = np.array(pop)
    
    return env, pop

#########################################################################
##  BELOW HERE THERE BE DRAGONS (left over stuff i played around with) ##
#########################################################################

# %% Plotting function that I think we should put in utils.py
# def our_plot_function(fn):
#     x1 = np.linspace(-10, 10, 100)
#     x2 = np.linspace(-10, 10, 100)
#     xs = np.stack(np.meshgrid(x1, x2), axis=-1)
#     ys = fn(xs)
#     plt.imshow(ys, cmap="viridis")
#     plt.colorbar()
#     plt.show()


# env, pop = init(ctx.config)

# import qdax
# from qdax.core.map_elites import MAPElites
# from PIL import Image
# from qdax.core.emitters.mutation_operators import polynomial_mutation

# fitness, behaviour = map(np.array, zip(*list(map(lambda p: eval(env, p), pop))))
# print(behaviour.shape, fitness.shape)
# # print(fitness)
# # print(behaviour)
# # print(list(map(lambda p: mutate(ctx.config, env, p), pop)))


 # %% using gym-pcg to evaluate map quality
def eval(env, p,cfg) -> Tuple[int, np.ndarray]:
    env._rep._map = p
    string_map = get_string_map(env._rep._map, env._prob.get_tile_types())
    stats = env._prob.get_stats(string_map)
    behaviour = None
    if cfg.game == 'smb':
        behaviour = np.array([stats["disjoint-tubes"], stats["empty"]])
    
    return env._prob.get_reward(stats, {k: 0 for k in stats.keys()}), behaviour


def mutate(cfg, env, p) -> np.ndarray:
    mask = np.random.random(p.shape) < cfg.p
    p[mask] = np.random.randint(0, env.get_num_tiles(), p[mask].shape)
    return p


# def archive(state, p):
#     raise NotImplementedError
