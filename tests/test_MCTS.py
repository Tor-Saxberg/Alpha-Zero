import sys; sys.path.append('../')
from random import choice
from MCTS_minimal import MCTS, Node

tree = MCTS()
node = Node()
wins = 0
games = 0

def check(node, winner):
    global wins, games
    if node.done:
        games += 1
        wins += 1 if winner == 'agent' \
            else 0 if winner == 'bot' \
            else 0.5
        node.render()
        print(winner + 'wins!')
        if wins > 0: print('agent win rate: ' + str(wins/games))
        else: print ('wins:  0')
        print('MCTS boards: ' + str(len(tree.children.keys() )))
        node.reset();
        return 1
    return 0

def play_bot():
    global tree, node
    player = choice([0,1])
    while True:
        if player == 0:
            node = node.step(choice(node.legal_moves() ))
            if check(node,'bot'): player = choice([0,1])

        if player == 1:
            for _ in range(20):
                tree.rollout(node, node.player_turn())
            node = tree.choose(node)
            if check(node, 'agent'): player = choice([0,1])


def play_human():
    tree = MCTS()
    node = Connect4Board()

    while True:
        action = input("enter a row")
        node = node.make_move(action)
        node.render()
        if node.terminal: break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(50):
            tree.rollout(node)
        node = tree.choose(node)
        node.render()
        if node.terminal: break

if __name__ == "__main__":
    print("playing bot")
    play_bot()


