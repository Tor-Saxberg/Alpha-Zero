import sys; sys.path.append('../')
import numpy as np

from connect4_env import Connect4Env
from agent import Agent
from MCTS_net import create_MCTS_instance

def test():
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    agent.train(games=3, sims=1,  virtual=1)

def test_separate():
    print("testing separate")
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    examples = []
    initial_state = create_MCTS_instance(agent.env,agent.net)
    node_list = []
    next_state = initial_state
    while True:
        next_state, current_state, node_list = next_state.play(sims=7)
        if agent._check(next_state):
            examples.append(agent._make_examples(node_list))
            break
    print("testing node, board1, board2")
    node = node_list[0]
    player1_board, player2_board, next_player = node._separate_players()
    print(node.board,  "\n")
    print(player1_board, "\n")
    print(player2_board, "\n")
    print(next_player, "\n")

def test_examples():
    print("testing separate")
    env = Connect4Env()
    agent = Agent(env, net='cnn')
    examples = []
    initial_state = create_MCTS_instance(agent.env,agent.net)
    node_list = []
    next_state = initial_state
    while True:
        next_state, current_state, node_list = next_state.play(sims=7)
        if agent._check(next_state):
            examples.append(agent._make_examples(node_list))
            break
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

if __name__ == "__main__":
    test()
    #test_separate()
    #test_examples()
