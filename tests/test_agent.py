import pytest
import cProfile

import numpy as np
from Alpha_Zero.connect4_env import Connect4Env
from Alpha_Zero.agent import Agent
from Alpha_Zero.MCTS_net import MCTS


#@pytest.mark.skip(reason="This test is currently skipped")
def test_make_examples():
    print("testing separate")
    env = Connect4Env()
    #agent = Agent(env, net='cnn')
    agent = Agent(env)
    initial_state = MCTS(agent.env,agent.net)
    examples = []
    node_list = []
    current = initial_state
    # simulate a game
    while not current.done: current = current.play(sims=1)
    # add all board states to examples
    node_list = current._get_tree() # get the tree
    for node in node_list:
        states = agent._make_examples(
        examples.append(agent._make_examples(node_list)) # augment and add states
    print("testing node, board1, board2")
    node = node_list[0]
    player1_board, player2_board, next_player = node._separate_players()
    print(node.board,  "\n")
    print(player1_board, "\n")
    print(player2_board, "\n")
    print(next_player, "\n")

@pytest.mark.skip(reason="This test is currently skipped")
def test_examples():
    print("testing separate")
    # initialize
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    initial_state = MCTS(env)
    examples = []
    node_list = []
    current = initial_state
    # play a game
    while not current.done():
        current, current_state, node_list = current.play(sims=1)
    # add states to examples


    examples.append(agent._make_examples(node_list))
    node = node_list[0]
    win = 0 if node_list[0].winner is None else 1
    print(node.board)
    print("")
    boards, Qs, policies = agent._make_examples(node_list)
    print(boards[0])
    print("testing cnn reshape")
    boards_list = []; Qs_list = []; policies_list = []
    for example in examples:
        boards, Qs, policies = example
        boards_list.extend(boards)
        Qs_list.extend(Qs)
        policies_list.extend(policies)

    boards = np.reshape(boards_list, (-1,*boards[0].shape) )
    Qs = np.reshape(Qs_list, (-1,1))
    policies = np.reshape(policies_list, (-1,agent.env.action_size))

    print(f"fitting {len(boards)} boards")
    print(boards[0])
    print(Qs[0])
    print(policies[0])

@pytest.mark.skip(reason="This test is currently skipped")
def test_train():
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    agent.train(games=3, sims=1,  virtual=1)
