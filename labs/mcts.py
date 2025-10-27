# imports
from __future__ import annotations
import numpy as np
import aigs
import time
from aigs import State, Env
from dataclasses import dataclass, field
from copy import copy
import math

# %% Setup
env: Env

#this heuristic is simply based on having pieces as close to the center of the board
def heuristic_value(state: State) -> float:
    result = 0
    x = 0
    for l in state.board:
        y = 0
        for il in l:
            if il != 0:
                if state.maxim:
                    result += placement_value(x,y,il)
                else:
                    result += placement_value(x,y,-il)
            y+=1
    return result

def placed_heuristic_value(state: State) -> float:
    heuristic = placement_value(state.col,state.row)
    if state.maxim: return heuristic
    return - heuristic

def placement_value(x: int,y :int, maxim: int = 1):
    return (np.abs(x -3) + np.abs(y - 3))/100 * maxim
    
def minimax(state: State, maxim: bool, depth: int = 0):
    if state.ended: 
        return state.point
    if depth > 5: 
        return heuristic_value(state)

    temp = 0
    for action in np.where(state.legal)[0]:
        value = minimax(env.step(state, action), not maxim, depth + 1)
        if value == 1 and maxim:
            return 1
        if value == -1 and not maxim:
            return -1
        temp = max(temp, value) if maxim else min(temp, value)
    return temp

def alpha_beta(state: State, maxim: bool, alpha: int, beta: int, depth: int = 0):
    if state.ended: 
        return state.point
    if depth > 5:
        if not maxim: return heuristic_value(state)
        return -heuristic_value(state)
    if maxim:
        temp = -100
        for action in np.where(state.legal)[0]:
            value = alpha_beta(env.step(state, action), not maxim, alpha, beta, depth +1)
            temp = max(temp, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return temp
    else:
        temp = 100
        for action in np.where(state.legal)[0]:
            value = alpha_beta(env.step(state, action), not maxim, alpha, beta, depth +1)
            temp = min(temp, value)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return temp

#this is a slightly faster alpha_beta that uses a heuristic based on the placement of the first piece placed,
#this will result in fewer possible heuristic values, which will in general lead to earlier termination
def alpha_beta_first(state: State, maxim: bool, alpha: int, beta: int, depth: int = 0, firstStepHeuristic: float = 0):
    if depth == 0: firstStepHeuristic = placed_heuristic_value(state) #basing the heuristic on the abstract value of the first move
    if state.ended: return state.point
    if depth > 6:
        return firstStepHeuristic
    if maxim:
        temp = -100
        for action in np.where(state.legal)[0]:
            value = alpha_beta_first(env.step(state, action), not maxim, alpha, beta, depth +1,firstStepHeuristic)
            temp = max(temp, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return temp
    else:
        temp = 100
        for action in np.where(state.legal)[0]:
            value = alpha_beta_first(env.step(state, action), not maxim, alpha, beta, depth +1,firstStepHeuristic)
            temp = min(temp, value)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return temp

#this is some lazy code duplication for having a alpha-beta first algorithm for reaching the bottom of the monte-carlo search
def alpha_beta_first2(state: State, maxim: bool, alpha: int, beta: int, depth: int = 5, firstStepHeuristic = None):
    if firstStepHeuristic is None: 
        firstStepHeuristic = placed_heuristic_value(state) #basing the heuristic on the abstract value of the first move
    if state.ended: 
        return state.point, True
    if depth == 0:
        return firstStepHeuristic, False
    ended = False
    if maxim:
        temp = -100
        for action in np.where(state.legal)[0]:
            value, tended = alpha_beta_first2(env.step(state, action), not maxim, alpha, beta, depth -1,firstStepHeuristic)
            temp = max(temp, value)
            alpha = max(alpha, value)
            ended = ended or tended
            if beta <= alpha:
                break
        return temp, ended
    else:
        temp = 100
        for action in np.where(state.legal)[0]:
            value, tended = alpha_beta_first2(env.step(state, action), not maxim, alpha, beta, depth -1,firstStepHeuristic)
            temp = min(temp, value)
            beta = min(beta, value)
            ended = ended or tended
            if beta <= alpha:
                break
        return temp, ended


@dataclass
class Node:
    state: State
    parent: Node
    children: dict
    t: int
    n: int

    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.t = 0
        self.n = 0
        self.children = {}


#monte Carlo tree search
def monte_carlo(state: State, cfg, tree = None) -> int:
    if tree is None:
        root = Node(state, None)
    else:
        root = tree

    #using time as a qualifier
    end_time = time.time() + (cfg.time)
    done = False
    i = 1
    while time.time() < end_time and not done:

        done = expand(root,root)
        i +=1
        
    
    #return the action with the highest score
    best = -10000
    actions = np.where(state.legal)[0]
    np.random.shuffle(actions)
    action = None
    for a in actions:
        if a in root.children and root.children[a].t/root.children[a].n > best:
            action = a
            best = root.children[a].t/root.children[a].n

    return (action, root)


def expand(v: Node, root:Node):
    actions = np.where(v.state.legal)[0]
    best = -10000
    node = None
    for a in actions:
        if a not in v.children:
            v.children[a] = Node(env.step(v.state, a),v)
            rollout(v.children[a], root)
            backup(v.children[a])
            return False
        tmp = v.children[a]
        value = ucb1_value(tmp)
        if value > best:
            best = value
            node = tmp

    if node == None: return True
    return expand(node, root)

def rollout(v: Node, root: Node):
    state = v.state
    while not state.ended:
        state = end_game_policy(state,root)
    if root.state.maxim:
        v.t = state.point * 1
    else:
        v.t = state.point * -1
    v.n = 1

#using ucb as the guide for expansion selection
def ucb1_value(v: Node) -> float:
    return v.t/v.n + 2 * math.sqrt(math.log(v.parent.n)/v.n)

#backpropegating the tree
def backup(node):
    t = node.t
    #the root of the tree will will have None as a parent, so terminate the back propagation
    while node.parent != None:
        node.parent.t += t
        node.parent.n += 1
        node = node.parent

#this is the part exploring random branches
#currently it is using alpha-beta minimax to find the last couple of steps so it doesnt create unnescecarily long branches
def end_game_policy(state, root: Node):
    
    actions = np.where(state.legal)[0]
    np.random.shuffle(actions)
    
    for a in actions:
        points, ended = alpha_beta_first2(env.step(state, a), root.state.maxim ^ state.maxim, -1, 1,depth=2)
        if ended:
            
            return env.step(state, a)

    a = np.random.choice(actions).item()
    state = env.step(state,a)
    
    return state

#this is a custom printer
def printer(state:State, t:float):
    p = 'o'
    if state.maxim: p = 'x' #here it is flipped, because we want to show the one that TOOK the move rather than who's turn it "is" this is due to place in loop
    print(p, ": " +str(t))
    print(state)

#a little timer to get time for each player
def timer(state: State, t:float, to:float, tx:float):
    tmp = (time.time() - t)
    if state.minim: tx += tmp
    else: to += tmp
    return to,tx,tmp

# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    tree = None
    
    for n in range(1): 
        state = env.init()
        tx = 0
        to = 0
        tmp = 0
        
        while not state.ended:
            actions = np.where(state.legal)[0]  # the actions to choose from
            np.random.shuffle(actions) #combat determinism
            

            match getattr(cfg, state.player):
                case "random":
                    t = time.time()
                    a = np.random.choice(actions).item()
                    to,tx,tmp = timer(state,t,to,tx)

                case "human":
                    t = time.time()
                    print(state, end="\n\n")
                    a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))
                    

                case "minimax":
                    t = time.time()
                    values = [minimax(env.step(state, a), not state.maxim) for a in actions]
                    a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                    to,tx,tmp = timer(state,t,to,tx)

                case "alpha_beta":
                    t = time.time()
                    values = [alpha_beta(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                    a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                    to,tx,tmp = timer(state,t,to,tx)

                case "alpha_beta_first":
                    t = time.time()
                    values = [alpha_beta_first(env.step(state, a), not state.maxim, -1, 1) for a in actions]
                    a = actions[np.argmax(values) if state.maxim else np.argmin(values)]
                    to,tx,tmp = timer(state,t,to,tx)
                
                case "monte_carlo":
                    t = time.time()
                    a, tree = monte_carlo(state, cfg, tree)
                    to,tx,tmp = timer(state,t,to,tx)

                case _:
                    raise ValueError(f"Unknown player {state.player}")


            state = env.step(state, a)
            printer(state,tmp)
            
            #this is experimenting with keeping the tree and building on it continuosly between turns
            if cfg.persistent_tree and tree != None and not state.ended:
                while a not in tree.children:
                    expand(tree,tree)
                tree = tree.children[a]

        print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
        print(f"x: " + str(tx))
        print(f"o: " + str(to))
        if tree != None:
            while tree.parent != None:
                tree = tree.parent

# %%
