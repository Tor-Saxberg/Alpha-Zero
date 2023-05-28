import pdb, traceback, sys, os, glob, shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category= FutureWarning)
from time import sleep
from matplotlib import pyplot as plt

from arena import Arena
from connect4_env import Connect4Env
from tictactoe_env import TictactoeEnv
from agent import Agent


import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--game', default='connect4', help='[tictactoe connect4] (string)')
parser.add_argument('--net', default='cnn', help='[rnn cnn dnn] (string)')
parser.add_argument('--start',default=0, help='starting epoch (int) or last (string)')
parser.add_argument('--epochs',default=10, type=int, help='stop after "end" epochs (int)')
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
        self.tests = 50 # games in each test
        self.threshold = .55 # win rate before updating
        self.epochs = epochs # epochs to train networks
        self.games = games # games to train on
        self.explore = 0.5
        self.sims = 7

        if start == 'last': self.start = self.last(game,net)
        else: self.start = int(start)

    def last(self, game, net): # continue from last epoch
        i = 0
        while True:
            filepath = f'./checkpoints/{game}_{net}_{i+1}.hdf5'
            if os.path.exists(filepath): i += 1
            else: return i+1


def main(args):
    """trains an agent to play {game} with {net}."""
    # wipe network data
    if args.start == 0:
        print("wiping data")
        for f in glob.glob('./checkpoints/*.hdf5'): os.remove(f)
        for f in glob.glob('./logging/*/*/*.v2'): os.remove(f)
        for f in glob.glob('./logging/*'): shutil.rmtree(f)

    game = args.game
    net = args.net
    configs = Configs(game, net, args.start, args.epochs, args.games)
    arena = Arena(env(game) )
    agent = Agent(env(game), net=net, epoch=args.start)
    #new = Agent(TictactoeEnv(),et='rnn',virtual=1)

    win_rates = []
    test_rates = []

    for epoch in range(configs.start, configs.start + configs.epochs):
        agent.train(games=configs.games, sims=configs.sims, epoch=epoch, expl=configs.explore)
        old = Agent(env(game), epoch=epoch-1)
        test_rate = arena.compete(agent, old, configs.tests)
        test_rates.append(test_rate)
        print("test rate: {}".format(test_rate))
        print("epoch: {}".format(epoch))
        sys.stdout.flush()

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
