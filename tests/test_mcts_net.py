import pytest
import cProfile

import numpy as np
from random import sample
from time import time
from Alpha_Zero.connect4_env import Connect4Env
from Alpha_Zero.MCTS_net import MCTS
#from ..network.cnn import CNN

@pytest.mark.skip(reason="This test is currently skipped")
def test_predict():
    env = Connect4Env()
    node = MCTS(env)
    Q, pi = node._predict()
    print(f"Q: {Q}")
    print(f"pi: {pi}")
    print(f'predict time: {end-start}')

    while not node.done(): node.play(20)
    start = time()
    Q,pi = node.parent._predict()
    end = time()
    print(f"Q: {Q}")
    print(f"pi: {pi}")
    print(f'predict time: {end-start}')

@pytest.mark.skip(reason="This test is currently skipped")
def test_net_predict():
    env = Connect4Env()
    net = CNN(env)
    node = MCTS(env, net=net)
    board = np.block([*node.env._separate_players() ])
    board_reshape = np.reshape(board, (-1,board.shape[0], board.shape[1]) )
    start = time()
    Q, pi = net.model.predict_on_batch(board_reshape)
    end = time()

    while not node.done(): node.play(20)
    board = np.block([*node.env._separate_players() ])
    board_reshape = np.reshape(board, (-1,board.shape[0], board.shape[1]) )
    start = time()
    Q, pi = net.model.predict_on_batch(board_reshape)
    end = time()
    print("Q: {}".format(Q))
    print("pi: {}".format(pi))
    print(f'predict time: {end-start}')

@pytest.mark.skip(reason="This test is currently skipped")
def test_Q():
    env = Connect4Env()
    # net = CNN(env)
    #initial = MCTS(env, net)
    initial = MCTS(env)
    node = initial
    for _ in range(10):
        current = node
        while not current.done():
            if not current.children: current._expand()
            if not current.parent: current = sample(current.children,1)[0]
            else: current = sample(current.children,1)[0]
        current._backpropogate(node) # update Q's
    print(node.Q)

@pytest.mark.skip(reason="This test is currently skipped")
def test_pi():
    env = Connect4Env()
    # net = CNN(env)
    #initial = MCTS(env, net)
    initial = MCTS(env)
    node = initial
    for _ in range(10):
        current = node
        while not current.done():
            if not current.children: current._expand()
            if not current.parent: current = sample(current.children,1)[0]
            else: current = sample(current.children,1)[0]
        current._backpropogate(node) # update Q's
    print(node.pi)

@pytest.mark.skip(reason="This test is currently skipped")
def test_expand():
    env = Connect4Env()
    #net = CNN(env)
    #node = MCTS(env, net)
    node = MCTS(env)
    current = node
    times = 0
    while not current.done():
        #if current.Q > 0.8 and current.turn > 20: break;
        if not current.children:
            start = time()
            current._expand()
            end = time()
            times += end - start
            current.env.render()
        else: current = current._action()
    print(times)

@pytest.mark.skip(reason="This test is currently skipped")
def test_backpropogate():
    """_backpropogate should update pi and Q for each node"""
    env = Connect4Env()
    #net = CNN(env)
    #node = MCTS(env, net)
    initial = MCTS(env)
    games = 10
    time_list = [[] for _ in range(games)]
    for game in range(games): # x games
        current = initial
        # simulate to leaf
        while not current.done():
            if not current.children: current._expand()
            current = current._action()
        # backpropogate
        start = time()
        current._backpropogate(initial) # leaf=current; root=initial
        end = time()
        current.env.render()
        time_list[game].append(end-start)
    # record time
    for game in range(games):
        print(f"game: {game}")
        for i,t in enumerate(time_list[game]):
            if i: print("\t", end='')
            print(f"{round(t,4)}\n")

@pytest.mark.skip(reason="This test is currently skipped")
def test_simulate():
    env = Connect4Env()
    #net = CNN(env)
    #node = MCTS(env, net)
    node = MCTS(env)
    start = time()
    for _ in range(20): node._simulate()
    end = time()
    children = sorted(node.children, key=lambda x: x.last())
    next_state = np.random.choice(children, p=node.pi[node.env.legal_moves() ])
    node._print_children()
    print(next_state)
    print("mcts_simulate time: {}".format(end - start))

#@pytest.mark.skip(reason="This test is currently skipped")
def test_play():
    print('hello world')
    env = Connect4Env()
    #net = 'cnn'
    net = None
    initial= MCTS(env, net)
    current = None
    games = 1
    time_list = [[] for _ in range(games)]
    # start profiler
    profiler = cProfile.Profile()
    profiler.enable()
    for game in range(games):
        print("new game")
        current = initial
        while not current.done():
            start = time()
            current = current.play(sims=40)
            end = time()
            current.env.render()
            time_list[game].append(end-start)
        current.env.render()
    # stop profiler
    profiler.disable()
    profiler.print_stats(sort='time')
    """
    for game in range(games):
        print(f"game: {game}")
        for i,t in enumerate(time_list[game]):
            if i: print("\t", end='')
            print(f"{round(t,3)}\n")
    """


@pytest.mark.skip(reason="This test is currently skipped")
def test_effective():
    #from random import choice
    env = Connect4Env()
    initial = MCTS(env)
    wins = 0
    games = 0
    states = 0

    def check(node, winner, states, games, wins):
        if node.done():
            games += 1
            print(f"number of games: {games}")
            wins += 1 if winner == 'agent' \
                else 0 if winner == 'bot' \
                else 0.5
            print(winner + ' wins!')
            if wins > 0: print(f"agent win rate: {wins/games}")
            else: print('wins: 0')
            return 1, states, games, wins
        return 0, states, games, wins

    node = initial
    #player = choice([1,2])
    player = 2
    while True:
        if player == 1:
            if not node.children: node._expand()
            node = sample(node.children, 1)[0] # random move
            #node = list(node.children)[0] # random move
            result, states, games, wins = check(node, 'bot', states, games, wins)
            #if result: node = initial
            if result: break;
            else: player = 2
        if player == 2:
            node = node.play(20) # MCTS move
            result, states, games, wins = check(node, 'agent', states, games, wins)
            #if result: node = initial
            if result: break;
            else: player = 1
    #node_list = node._get_tree()
    #for move in node_list[-1::-1]:
    #    move._print_self()
    print(node)
