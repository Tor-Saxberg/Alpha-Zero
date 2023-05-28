import os, sys; sys.path.append('./')
import numpy as np
from random import choice

from connect4_env import Connect4Env
from tictactoe_env import TictactoeEnv
from agent import Agent
from arena import Arena

def test_agents():
    env = Connect4Env()
    agent_1 = Agent(env, epoch='last',  net='cnn')
    arena = Arena(env)
    game = 'connect4'
    net = None
    i = 0
    """
    while True:
        filepath = f'./checkpoints/Connect4_cnn_{i}.hdf5'
        breakpoint()
        if os.path.exists(filepath): i += 1
        else: break
    """
    for epoch in range(10):
        agent_2 = Agent(env, epoch=epoch, net='cnn')
        test_rates = []
        #test_rate = arena.test(agent_2, 30)
        test_rate = arena.compete(agent_1, agent_2, 100)
        test_rates.append(test_rate)
        print(f'Epoch {epoch}, Game {i+1}')
        print("test rate: {}".format(test_rate))

def test():
    env = Connect4Env()
    agent_1 = Agent(env, epoch='last',  net='cnn')
    arena = Arena(env)
    agent_2 = Agent(env, epoch=6, net='cnn')
    test_rates = []
    #test_rate = arena.test(agent_2, 30)
    test_rate = arena.compete(agent_1, agent_2, 50)
    test_rates.append(test_rate)
    print("test rate: {}".format(test_rate))

def test_compete():
    class ones():
        def __init__(self): pass
        def play(self, env):
            print("ones")
            if 0 in env.legal_moves(): return env.step(0)# 2/nd col
            else:
                return env.step(choice(env.legal_moves() )) # random move
    class random():
        def __init__(self): pass
        def play(self, env):
            print("random")
            return env.step(choice(env.legal_moves() )) # random move
    env = Connect4Env()
    arena = Arena(env)
    print(arena.compete(ones(), random(), 30))

if __name__ == "__main__":
    #test()
    #test_compete()
    test_agents()
