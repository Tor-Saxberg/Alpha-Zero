from time import time
from MCTS_net import MCTS
from network.cnn import CNN
from network.rnn import RNN
from network.dnn import DNN
from network.actor_critic.agent import DDPG
import numpy as np
import sys
from IPython.display import clear_output

class Agent():
    """learns to play any board game"""

    def __init__(self, env, epoch=0, net=None):
        """load or create network"""
        self.env = env
        if net:
            if net == 'cnn': self.net = CNN(env)
            elif net == 'rnn': self.net = RNN(env)
            elif net == 'dnn': self.net = DNN(env)
            elif net == 'ddpg': self.net = DDPG(env)
            self.net.load_checkpoint(epoch=epoch)
        else: self.net = None

    def train(self, games=10, sims=7, epoch=0, expl=1, virtual=0):
        """train the network via MCTS. virtual=1 skips learning.  """
        examples = []
        initial_state = MCTS(self.env, self.net, expl=expl)
        start = time()
        for i in range(games):
            tree = []
            current = initial_state
            while True:
                current, tree = current.play(sims=sims)
                print(current)
                print(f'Epoch {epoch}, Game {i+1}')
                if self._check(current):
                    examples.append(self._make_examples(tree))
                    break
        if self.net: self.net.train(examples, virtual, epoch=epoch)
        end = time()
        print(f'agent training time: {end-start}')

    def _check(self, node):
        """print last game state"""
        #if node.done:
            #clear_output(wait=True)
            #print(chr(27) + "[2J")
            #print(node)
            #sys.stdout.flush()
        return node.done


    def play(self, env):
        """return best move from env"""
        board = np.block([*env._separate_players() ])
        # do I need to load the pre-trained model first?
        if self.net: pi = self.net.model.predict(board.reshape((-1,*board.shape)), verbose=0 )[1][0]
        else: pi = [1/len(env.legal_moves())] * env.action_size
        # exclude illegal moves
        pi[np.setdiff1d(range(env.action_size), env.legal_moves() )] = float('-inf') # don't make illegal moves

        return env.step(np.argmax(pi))


    def _make_examples(self, tree):
        """augment data, assign values, and reverse order"""
        boards = []; Qs = []; policies = []
        # last player won the game
        win = 0 if tree[0].winner is None else 1
        for node in tree:
            """split boards into player1_board and player2_board"""
            player1_board, player2_board, next_player = node._separate_players()
            """get list of augmented boards, etc"""
            player1_board, Q, policy = node.augment(player1_board, node.pi, win)
            win = -win # 2nd to last player lost
            player2_board, Q, policy = node.augment(player2_board, node.pi, win)
            """extend list of examples"""
            for i in range(len(player1_board) ):
                boards.extend([np.block([player1_board[i], player2_board[i], next_player]) ])
            Qs.extend(Q)
            policies.extend(policy)
            win = -win # 2nd to last player lost
        return boards, Qs, policies

