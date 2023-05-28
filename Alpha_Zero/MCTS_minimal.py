from connect4_env import Connect4Env
from collections import defaultdict
import math
from random import choice
from copy import copy
import sys
import numpy as np


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node not in self.children: return node.find_random_child()
        def score(n):
            if self.N[n] == 0: return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward
        return max(self.children[node], key=score)

    def rollout(self, node, player):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(copy(node))
        leaf = path[-1]
        if leaf not in self.children: # if it's new
            self.children[leaf] = leaf.find_all_children()

        while not leaf.is_terminal():
            leaf = leaf.find_random_child()
        reward = leaf.reward(player)

        for board in reversed(path):
            self.N[board] += 1
            self.Q[board] += reward
            reward = 1 - reward

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children: return path
            if node.is_terminal(): return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"
        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] \
                + self.exploration_weight \
                * math.sqrt( math.log(self.N[node]) / self.N[n])
        return max(self.children[node], key=uct)

class Node(Connect4Env):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def find_all_children(self):
        "All possible successors of this board state"
        if self.done: return None
        return {copy(self).step(i) for i in self.legal_moves() }
        #return self.step(i) for i in self.legal_moves() }

    def find_best_child(self, N, Q):
        "Random successor of this board state"""
        if self.done: return None
        return copy(self).step(choice(self.legal_moves() ))

    def find_random_child(self):
        "Random successor of this board state"""
        if self.done: return None
        return self.step(choice(self.legal_moves() ))
        #return self.step(choice(self.legal_moves() ))

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.done

    def reward(self, player):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        if self.winner is player: return 1 # agent won
        elif self.winner is not player: return -1 # agent lost
        else: return 0.5 # draw

