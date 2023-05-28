import sys; sys.path.append('../')
import glob
import os

import numpy as np

from connect4_env import Connect4Env
from tictactoe_env import TictactoeEnv
from agent import Agent
from network.cnn import CNN

def test():
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    board = np.block([*env._separate_players() ])
    board_reshape = np.reshape(board, (-1,board.shape[0], board.shape[1]) )
    Q, pi = agent.net.model.predict_on_batch(board_reshape)
    # exclude illegal moves
    #pi[np.setdiff1d(range(env.action_size), env.legal_moves() )] = float('-inf')
    #for node in sorted(parent.children, key=lambda x: x.last()): #incomplete
    #predict = agent.net.model.predict(board.reshape((-1,*board.shape))
    #pi = predict[1][0]
    #Q, pi = self.net.model.predict_on_batch(board_reshape)
    #Q = Q[0][0]; pi = pi[0]
    print("pi")
    print(pi)
    print("Q")
    print(Q)
    return env.step(np.argmax(pi))

def test_load():
    env = Connect4Env()
    net = CNN(env)
    net.load_checkpoint(0)
    net.load_checkpoint('last')

if __name__ == "__main__":
    test()
    #test_load()
