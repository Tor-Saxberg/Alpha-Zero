import pdb, traceback, sys, os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category= FutureWarning)

from arena import Arena 
from connect4_env import Connect4Env
from tictactoe_env import TictactoeEnv
from agent import Agent

from time import sleep
from matplotlib import pyplot as plt

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--game', default='tictactoe', help='[tictactoe connect4] (string)')
parser.add_argument('--net', default='rnn', help='[rnn cnn dnn] (string)')
parser.add_argument('--start',default=0, help='starting epoch (int) or last (string)')
parser.add_argument('--epochs',default=20, type=int, help='stop after "end" epochs (int)')
parser.add_argument('--games',default=20, type=int, help='games before each test (int)')
args=parser.parse_args()

def env(game):
    if game == 'tictactoe':
        return TictactoeEnv()
    elif game == 'connect4':
        return Connect4Env()

class Configs():
    def __init__(self, game, net, start, epochs, games):
        self.draws = 4 # when has learning has stopped
        self.tests = 30 # games in each test
        self.threshold = .55 # win rate before updating
        self.epochs = epochs # epochs to train networks
        self.games = games # games to train on

        # epoch to start from
        if start == 'last': self.start = self.last(game,net)
        else: self.start = int(start)
        # MCTS searches each move
        if game == 'tictactoe': self.sims = 4
        elif game == 'connect4': self.sims = 14

    def last(self, game, net): # continue from last epoch
        i = 0
        while True:
            filepath = f'./checkpoints/{game}_{net}_{i+1}.hdf5'
            if os.path.exists(filepath): i += 1
            else: return i+1


def main(args):
    """trains an agent to play {game} with {net}."""
    game = args.game
    net = args.net
    configs = Configs(game, net, args.start, args.epochs, args.games)
    arena = Arena(env(game) )
    #new = Agent(TictactoeEnv(),et='rnn',virtual=1)

    win_rates = []
    test_rates = []

    old = Agent(env(game), net=net, epoch=configs.start-1)
    new = Agent(env(game), net=net, epoch=configs.start-1)

    for epoch in range(configs.start, configs.start + configs.epochs):
        draws = 0
        win_rate = 0
        while win_rate < configs.threshold and draws < configs.draws: 
            draws += 1
            new.train(games=configs.games, sims=configs.sims, epoch=epoch, expl=1)

            win_rate = arena.compete(new, old, configs.tests)
            win_rates.append(win_rate)
            test_rate = arena.test(new, configs.tests)
            test_rates.append(test_rate)

            print("win rate: {}".format(win_rate))
            print("test rate: {}".format(test_rate))
            print("draws: {}".format(draws))
            print("epoch: {}".format(epoch))
            sys.stdout.flush()
            #new.train(games=configs.games, sims=configs.sims, epoch=epoch, expl=1, virtual=True)
        old = Agent(env(game), net=net, epoch=epoch)
        #old = Agent(TictactoeEnv(),et='rnn', epoch=0, virtual=True)

    plt.plot(win_rates, 'g') 
    plt.plot(test_rates, 'r')
    plt.show()

if __name__ == '__main__':
    # you can run '$python3 main.py' with no arguments
    try: main(args)
    except: 
        extype, value, tb = sys.exc_info()
        print(extype, value)
        traceback.print_exc()
        pdb.post_mortem(tb)
