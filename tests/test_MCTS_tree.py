from random import choice
import sys; sys.path.append('../')
from MCTS_tree import MCTS


node = MCTS()
wins = 0
games = 0
boards = 0

def check(winner):
    global  node, wins, games, boards
    if node.done:
        games += 1
        wins += 1 if winner == 'agent' \
            else 0 if winner == 'bot' \
            else 0.5
        node.render()
        print(winner + ' wins!')
        if wins > 0: print(f"agent win rate: {wins/games}")
        else: print('wins:  0')
        print(f"number of boards: {boards}")
        print(f"number of games: {games}")
        return 1
    return 0

def play_bot():
    global  node, wins, games, boards
    player = choice([1,2])
    while True:
        if player == 1:
            node = node.step(choice(node.legal_moves() ))
            if check('bot'): node.reset()
            else: player = 2

        if player == 2:
            for _ in range(14):
                boards = node.rollout(node.player_turn(), boards)
            node = node.choose()
            if check('agent'): node.reset()
            else: player = 1

if __name__ == "__main__":
    play_bot()
