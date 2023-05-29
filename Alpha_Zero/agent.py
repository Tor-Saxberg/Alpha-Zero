import numpy as np
from time import time
from typing import List
from MCTS_net import MCTS
from network.cnn import CNN
#from network.rnn import RNN
#from network.dnn import DNN
#from network.actor_critic.agent import DDPG
#import sys
#from IPython.display import clear_output

class Agent():
    """learns to play any board game"""

    def __init__(self, env, epoch=0, net=None):
        """load or create network"""
        self.env = env
        if net: 
            if net == 'cnn': self.net = CNN(env)
            #elif net == 'rnn': self.net = RNN(env)
            #elif net == 'dnn': self.net = DNN(env)
            #elif net == 'ddpg': self.net = DDPG(env)
            self.net.load_checkpoint(epoch=epoch)
        else: self.net = None

    def train(self, games=10, sims=7, epoch=0, expl=1, virtual=0):
        """train the network via MCTS. virtual=1 skips learning.  """
        examples = []
        initial_state = MCTS(self.env, self.net, expl=expl)
        start = time()
        for i in range(games):
            node_list = []
            current = initial_state
            while True:
                current, node_list = current.play(sims=sims)
                print(current)
                print(f'Epoch {epoch}, Game {i+1}')
                if current.done():
                    examples.append(self.env.generate_examples(node_list)) # this should be a set to avoid duplicates
                    break
        if self.net: self.net.train(examples, virtual, epoch=epoch)
        end = time()
        print(f'agent training time: {end-start}')

    def play(self, env):
        """return best move from env"""
        board = np.block([*env._separate_players() ])
        # do I need to load the pre-trained model first?
        if self.net: pi = self.net.model.predict(board.reshape((-1,*board.shape)), verbose=0 )[1][0]
        else: pi = [1/len(env.legal_moves())] * env.action_size
        # exclude illegal moves
        pi[np.setdiff1d(range(env.action_size), env.legal_moves() )] = float('-inf') # don't make illegal moves

        return env.step(np.argmax(pi))

    def _make_examples(self, node_list: List[MCTS]):
        """augments a list of MCTS nodes for symmetry.
            Returns 
                a list of separated and augmented boards, 
                a list of Qs, 
                a list of pi's"""
        examples = []; Qs = []; Ps = []
        # last player won the game
        if not node_list[0].done(): breakpoint()
        for node in node_list:
            # split boards into board0 and board1
            board_0, board_1, boards_player = node.env._separate_players()
            # get list of augmented boards, etc. baords_0 are the player_0 boards
            boards_0, Qs_0, Ps_0 = node.augment(board_0, node.pi, node.Q)
            boards_1, Qs_1, Ps_1 = node.augment(board_1, node.pi, node.Q)
            # Ps_0[0] != Ps_0[1], but Ps_0 == Ps_1

            # creaate a list of boards, Qs, and Ps
            for i in range(len(boards_0) ):
                # an example is a board_0, board_1, and a boards_player board
                baord_examples.append([np.block([boards_0[i], boards_1[i], boards_player[i]]) ])
                Qs.extend((Qs_0[i], Qs_1[i]) ])
                Ps.extend([np.block([boards_0[i], boards_1[i], boards_player[i]]) ])
            win = -win # 2nd to last player lost
        return examples, Qs, Ps




