import pytest
import cProfile

import numpy as np
from random import choice
from time import time
from Alpha_Zero.connect4_env import Connect4Env
from Alpha_Zero.MCTS_net import MCTS
from Alpha_Zero.network.cnn import CNN

#@pytest.mark.skip(reason="This test is currently skipped")
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

#@pytest.mark.skip(reason="This test is currently skipped")
def test_net_predict():
    env = Connect4Env()
    net = CNN(env)
    node = MCTS(env, net=net)
    start = time()
    Q, pi = node._predict()
    end = time()
    print(node.env)
    print(pi)
    print(Q)
    print(f"_predict time: {end-start}")

    # start profiler
    profiler = cProfile.Profile()
    profiler.enable()
    #while not node.done(): 
    for _ in range(5):
        node.play(10)
        print(node.env.turn)
    # stop profiler
    profiler.disable()
    profiler.print_stats(sort='cumtime')

    #Q, pi = net.model.predict_on_batch(board_reshape)
    Q, pi = node._predict()
    print(node.env)
    print(pi)
    print(Q)


#@pytest.mark.skip(reason="This test is currently skipped")
def test_Q():
    env = Connect4Env()
    # net = CNN(env)
    #initial = MCTS(env, net)
    initial = MCTS(env)
    node = initial
    for _ in range(1):
        current = node
        while not current.done():
            if not current.children: current._expand()
            #current = choice(current.children)
            current = current.children[0]
        current._backpropogate(node) # update Q's
    current._print_parents()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_pi():
    env = Connect4Env()
    # net = CNN(env)
    #initial = MCTS(env, net)
    initial = MCTS(env)
    node = initial
    for sim in range(1000): # x simulations
        current = node
        while not current.done():
            if not current.children: current._expand()
            else: current = current._action()
        current._backpropogate(node) # update Q's
    while current.parent: 
        current._print_self()
        #print(current.pi, end='\n')
        current = current.parent

#@pytest.mark.skip(reason="This test is currently skipped")
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

#@pytest.mark.skip(reason="This test is currently skipped")
def test_backpropogate():
    """_backpropogate should update pi and Q for each node"""
    env = Connect4Env()
    #net = CNN(env)
    #node = MCTS(env, net)
    initial = MCTS(env)
    games = 1
    for _ in range(games): # x games
        current = initial
        # simulate to leaf
        while not current.done():
            if not current.children: current._expand()
            current = current.children[0]
        # backpropogate
        current._backpropogate(initial) # leaf=current; root=initial
        current._print_parents()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_simulate():
    env = Connect4Env()
    #net = CNN(env)
    #node = MCTS(env, net)
    node = MCTS(env)
    start = time()
    node._simulate(20)
    end = time()
    children = sorted(node.children, key=lambda x: x.last())
    next_state = np.random.choice(children, p=node.pi[node.env.legal_moves() ])
    node._print_children()
    print(next_state)
    next_state._print_self()
    print("mcts_simulate time: {}".format(end - start))

#@pytest.mark.skip(reason="This test is currently skipped")
def test_play():
    env = Connect4Env()
    #net = 'cnn'
    net = None
    initial= MCTS(env, net)
    current = None
    games = 1
    time_list = [[] for _ in range(games)]
    # start profiler
    #profiler = cProfile.Profile()
    #profiler.enable()

    for game in range(games):
        print("new game")
        current = initial
        while not current.done():
            start = time()
            current = current.play(sims=80)
            print(current.parent)
            end = time()
            time_list[game].append(end-start)
        print(current)
        breakpoint()
        #for node in current._get_tree():
            #node._print_children()

    # stop profiler
    #profiler.disable()
    #profiler.print_stats(sort='time')

    """
    for game in range(games):
        print(f"game: {game}")
        for i,t in enumerate(time_list[game]):
            if i: print("\t", end='')
            print(f"{round(t,3)}\n")
    """


#@pytest.mark.skip(reason="This test is currently skipped")
def test_effective():
    #from random import choice
    env = Connect4Env()
    initial = MCTS(env)
    wins = 0
    games = 0
    states = 0
    node = initial
    player = 2
    #player = choice([1,2])

    def check(node, winner, states, games, wins):
        if node.done():
            if not games % 10:  node._print_siblings(); node.env.render()
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

    # start profiler
    #profiler = cProfile.Profile()
    #profiler.enable()

    # play 100 games
    for _ in range(50):
        node = node.reset()
        # run each game to terminal
        while not node.done():
            if player == 1:
                if not node.children: node._expand()
                node = choice(node.children) # random move
                #node = list(node.children)[0] # random move
                result, states, games, wins = check(node, 'bot', states, games, wins)
                #if result: node = initial
                if result: break
                else: player = 2

            if player == 2:
                node = node.play(20) # MCTS move
                result, states, games, wins = check(node, 'agent', states, games, wins)
                #if result: node = initial
                if result: break
                else: player = 1
            if not games % 10:  node._print_siblings(); node.env.render()
    # stop profiler
    #profiler.disable()
    #profiler.print_stats(sort='cumtime')

    #node_list = node._get_tree()
    #for move in node_list[-1::-1]:
    #    move._print_self()
