# %% qd.py
#   quality diversity exercises
# by: Noah Syrkis

# Imports
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from typing import Tuple
from pcgym import PcgrlEnv
from pcgym.envs.helper import get_int_prob, get_string_map

from tqdm import tqdm


#crossover takes a random point on the map and splits and merges the two
def one_point_crossover(x1, x2):
    point = np.random.randint(1, len(x1))
    return np.concatenate([x1[:point], x2[point:]])

#same as one point crossover but transposed
def one_point_crossover_T(x1, x2):
    t1, t2 = x1.T, x2.T
    f = int(len(t1) * 0.1)
    t = len(t1) - f
    point = np.random.randint(f, t)
    return np.concatenate([t1[:point], t2[point:]]).T

# %% Setup
def main(cfg):
    env, pop = init_pcgym(cfg)
    
    best_elite = Map_Elites(cfg,pop,env)
    env.map = best_elite
    visual = env.render()
    plt.figure(figsize=(10,2), dpi=160)
    plt.imshow(visual,interpolation='nearest')
    plt.show()
    exit()

#initiating the gym, this is gives a slightly "better" starting map than complete random
def init_pcgym(cfg) -> Tuple[PcgrlEnv, np.ndarray]:
    env = PcgrlEnv(prob=cfg.game, rep=cfg.rep, render_mode="rgb_array")
    pop = []
    for _ in range(cfg.population):
        env.reset()
        pop.append(env._rep._map.copy())
    
    pop = np.array(pop)
    
    return env, pop

def variation_operator(Archive,cfg, env):
    keys = list(Archive.keys())
    parent1 = Archive[keys[np.random.randint(0, len(keys))]]["solution"]
    parent2 = Archive[keys[np.random.randint(0, len(keys))]]["solution"]
    n = np.random.rand()
    if n < cfg.mutants:
        env.reset()
        parent2 = env._rep._map.copy()

    #I have 6 different crossover versions the first two takes half of each map split horizontal or vertical respectively,
    #the four others take a quater from one and add to the other
    r = np.random.randint(0,6)
    if r == 0:
        child = one_point_crossover_T(parent1, parent2)
    if r == 1:
        child = one_point_crossover(parent1, parent2)
    if r == 2:
        child = one_point_crossover_T(parent1, parent2)
        child = one_point_crossover(child, parent2)
    if r == 3:
        child = one_point_crossover_T(parent1, parent2)
        child = one_point_crossover(parent1, child)
    if r == 4:
        child = one_point_crossover(parent1, parent2)
        child = one_point_crossover_T(parent1, child)
    if r == 5:
        child = one_point_crossover(parent1, parent2)
        child = one_point_crossover_T(child, parent2)

    return child


def mutate(cfg, env, p) -> np.ndarray:
    mask = np.random.random(p.shape) < cfg.p
    p[mask] = np.random.randint(0, env.get_num_tiles(), p[mask].shape)
    return p

#I tried to normalize the keys in order to reduce the size of the map
def get_key(b, resolution, global_max = 100):
    key = []
    for v in b:
        # normalize to [0,1] using global min/max
        normalized = (v) / (global_max)
        # scale to resolution bins and clamp
        bin_index = int(normalized * resolution)
        if bin_index >= resolution:
            bin_index = resolution - 1
        key.append(bin_index)
    return tuple(key)

#I spent more time crossing over than evaluating, maybe I should have spent more on defining behaviour and fitness
def evaluate(candidate, env):
    env._rep._map = candidate
    
    string_map = get_string_map(env._rep._map, env._prob.get_tile_types())

    stats = env._prob.get_stats(string_map)
    behavior = stats.values()

    fitness = env._prob.get_reward(stats, {k: 0 for k in stats.keys()})
    return fitness, behavior


def Map_Elites(cfg, population, env):
        
    # MAP-Elites hyperparameters
    n_budget = cfg.n  # total number of evaluations
    resolution = cfg.resolution  # number of cells per dimension
    g_max = cfg.gmax

    # MAP-Elites:
    Archive = {}  # empty archive
    #fill in the population
    nbest = -100000
    for candidate in population:
        f, b = evaluate(candidate, env)

        #return candidate
        key = get_key(b, resolution, g_max)  # get the index of the niche/cell
        if key not in Archive or Archive[key]["fitness"] < f:  # add if new behavior or better fitness
            Archive[key] = {"fitness": f, "behavior": b, "solution": candidate}
            print(len(Archive.keys()))
    for i in range(n_budget):  # mutation and/or crossover
        candidate = variation_operator(Archive, cfg, env)
        f, b = evaluate(candidate, env)
        key = get_key(b, resolution, g_max)  # get the index of the niche/cell

        if key not in Archive or Archive[key]["fitness"] < f:  # add if new behavior or better fitness
            Archive[key] = {"fitness": f, "behavior": b, "solution": candidate}
        best = max(Archive.values(), key=lambda v: v["fitness"])
        if best['fitness'] != nbest:
            nbest = best['fitness']
            print(nbest,i,len(Archive.keys()))
        if i% 1000 == 0:
            print(nbest,i,len(Archive.keys()))

    
    print(len(Archive.keys()))
    print(best['fitness'])
    return best["solution"]

# %%
