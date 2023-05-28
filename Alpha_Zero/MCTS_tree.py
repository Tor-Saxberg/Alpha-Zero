import math
from copy import copy
from connect4_env import Connect4Env


class MCTS(Connect4Env):
    """an MCTS using a tree structure"""
    def __init__(self, parent=None, explore=1):
        super().__init__()
        self.parent = parent
        self.children = set()
        self.N = 0
        self.Q = int()
        self.explore = explore

    def choose(self):
        """return next move"""
        def score(child):
            if child.N == 0: return float("-inf")
            return child.Q / child.N
        return max(self.children, key=score)

    def rollout(self, player, boards):
        """update move preferences"""
        leaf = self._select(); leaf.N += 1 # same as _action() in mcts_net.py
        boards = leaf._propogate(player, boards) # same as _simulate() in mcts_net.py
        return boards

    def _select(self):
        """retrieve leaf node"""
        current = self
        while True:
            if not current.children: return current
            current = current._best_child()

    def _best_child(self):
        """return best child"""
        self._expand()
        def uct(child): return child.Q / (child.N+1) + child.explore * \
            math.sqrt(math.log(child.N+1) / (child.N+1) )
        new = max(self.children, key = uct)
        return new

    def _expand(self):
        """create children nodes"""
        for i in self.legal_moves():
            new = copy(self).step(i)
            new.parent = self
            self.children.add(new)

    def _propogate(self, player, boards):
        """ update weights along path to next terminal"""
        current = self
        while not current.done:
            current = current._best_child()
        reward = 1 if current.winner == player \
            else 0 if current.winner is not None \
            else 0.5
        boards = current._backpropogate(reward, boards)
        return boards

    def _backpropogate(self, reward, boards):
        """update weights up the tree"""
        current = self
        while current.parent is not None:
            current.N += 1
            current.Q += reward
            reward = - reward
            current = current.parent
            boards += 1
        current.N += 1
        current.Q += reward
        return boards

    def __copy__(self):
        """copy Connect4Env variables, parent, and empty children"""
        new = type(self)(self)
        new.board = copy(self.board)
        new.turn = self.turn
        new.done = self.done
        new.winner = self.winner
        new.winning_moves = copy(self.winning_moves)
        new.last_move = self.last_move
        #new.parent = self.parent
        new.children = set()
        return new

    def __repr__(self):
        """display parent list, current board, and child list"""
        print('\nparent:', end=' ')
        if self.parent is None: print('None')
        else:
            string = []
            parent = self
            while parent.parent is not None:
                parent = parent.parent
                string.append('\tturn: {},last: {}, N: {}, Q: {}, ID: {}'.format( \
                    parent.turn, parent.last_move, parent.N, parent.Q, id(parent)) )
            string = reversed(string)
            for parent in string: print(parent,end="\n")

        print('\nself', end=' '); self.render()
        print('\tturn: {},last: {}, N: {}, Q: {}, id: {}'.format( \
            self.turn, self.last_move, self.N, self.Q, id(self) ) )

        print('\nchildren:', end=" ")
        string = ''
        if not self.children: print('None')
        else:
            for child in self.children:
                string += '\n\tturn: {},last: {}, N: {}, Q: {}, id: {}'.format( \
            child.turn, child.last_move, child.N, child.Q, id(child))
        return string+'\n'
