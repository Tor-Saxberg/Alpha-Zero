import sys; sys.path.append('../')
import numpy as np
from connect4_env import Connect4Env
from agent import Agent
from MCTS_minimal import MCTS, Node
from random import sample
from random import choice

def test_predict():
    env = Connect4Env()
    node = MCTS(env, net)
    board = np.block([*env._separate_players() ])
    board_reshape = np.reshape(board, (-1,board.shape[0], board.shape[1]) )
    start = time()
    Q, pi = net.model.predict_on_batch(board_reshape)
    end = time()
    print("Q: {}".format(Q))
    print("pi: {}".format(pi))
    print(f'predict time: {end-start}')


def test_backpropogate():
    """_backpropogate should update pi and Q for each node"""
    env = Connect4Env()
    net = CNN(env)
    node = MCTS(env, net)
    current = node
    for _ in range(1):
        while not current.done:
            if current.Q > 0.8 and current.turn > 20: break;
            if not current.children: current._expand()
            # random first move for stochasticity
            #if not current.parent: current = sample(current.children,1)[0]
            else: current = current._action()
        current._backpropogate() # reversed tree path from leaf to root
        print(current)

def test_expand():
    env = Connect4Env()
    net = CNN(env)
    node = MCTS(env, net)
    current = node
    times = 0
    while not current.done:
        if current.Q > 0.8 and current.turn > 20: break;
        if not current.children:
            start = time()
            current._expand()
            end = time()
            times += end - start
        else: current = current._action()
    print(current)
    print(times)

def test_simulate():
    env = Connect4Env()
    net = CNN(env)
    node = MCTS(env, net)
    start = time()
    node._simulate()
    end = time()
    children = sorted(node.children, key=lambda x: x.last())
    next_state = np.random.choice(children, p=node.pi[node.legal_moves() ])
    node._print_children()
    print(next_state)
    print("mcts_simulate time: {}".format(end - start))

def test_play():
    env = Connect4Env()
    net = 'cnn'
    agent = Agent(env, net=net, epoch=-1)
    initial= MCTS(agent.env, agent.net)
    tstart = time()
    for i in range(1):
        tree = []
        current = initial
        while not agent._check(current):
            start = time()
            current, tree = current.play(sims=4)
            end = time()
            print(current)
            print("mcts_play time: {}".format(end - start))
    tend = time()
    print("mcts_complete time: {}".format(tend - tstart))

if __name__ == "__main__":
    #test_predict()
    #test_Q()
    #test_backpropogate()
    #test_expand()
    #test_simulate()
    test_play()
