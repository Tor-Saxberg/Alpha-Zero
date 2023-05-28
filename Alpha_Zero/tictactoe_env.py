import numpy as np
import enum
from termcolor import colored
from copy import copy
from pprint import pprint

# noinspection PyArgumentList

class TictactoeEnv:
    def __init__(self, board=None):
        """initialize environment"""
        self.name = 'tictactoe'
        self.board_size = (3,3)
        self.action_size = 9

        if board is None:
            self.board = np.zeros(self.board_size).astype('int')
            self.board[:] = 0 # for empty
        else: self.board = board

        self.turn = 0
        self.done = False
        self.winner = None
        self.winning_moves = np.zeros(self.board_size)
        self.last_move = None

    def reset(self):
        """reset environment state"""
        self.board[:] = 0 # for empty
        self.turn = 0
        self.done = False
        self.winner = None
        self.winning_moves = np.zeros(self.board_size)
        self.last_move = None

    def player_turn(self):
        """return current player: 0 or 1"""
        return self.turn % 2 + 1 # {1,2}, 1 goes first

    def last(self):
        return self.last_move[0]*3 + self.last_move[1]

    def step(self, action):
        """increment player, then play move"""
        action = int(action) 
        if self.done:
            raise ValueError("game already over"); return
        if type(action) != int:
            raise ValueError('bad action'); return
        if action not in self.legal_moves():
            raise ValueError("illegal move"); return
        
        act = [0,0]; act[0] = int(action/3); act[1] = action%3
        self.turn += 1

        if self.board[act[0]][act[1]] == 0:
            self.board[act[0]][act[1]] = self.player_turn()
            self.last_move = act

        self.check(act)
        if self.turn >= 9:
            self.done = True
        if self is None: print("Nonetype TictactoeEnv")
        return self

    def legal_moves(self):
        row,col = np.where(self.board==0)
        return [row*3 + col][0]


    def augment(self, board, pi, win):
        """augment data, and assign values"""
        boards = []; Qs = []; policies = []

        boards.append(board)
        Qs.append(win)
        policies.append(pi)

        """flip horizontally"""
        boards.append(np.flip(board, axis=0))
        Qs.append(win)
        policies.append(np.flip(pi))

        """flip vertically"""
        boards.append(np.flip(board, axis=1))
        Qs.append(win)
        policies.append(np.flip(pi))

        """flip horizontally and vertically"""
        temp = np.flip(board, axis=0)
        boards.append(np.flip(temp, axis=1))
        Qs.append(win)
        policies.append(np.flip(pi))

        return boards, Qs, policies


    def check(self, action):
        if self.vertical_check(action): self.done = True
        elif self.horizontal_check(action): self.done = True
        elif self.diagonal(action): self.done = True
        if self.done == True: self.winner = self.player_turn()

    def vertical_check(self, action):
        """return winner if vertical 3-in-a-row"""
        count = 0
        for i in range(3):
            if self.board[i][action[1]] == self.player_turn():
                count += 1
                self.winning_moves[i][action[1]] = 1
            else: break
        if count == 3: return True
        else: self.winning_moves[:] = 0; return False

    def horizontal_check(self, action):
        """return winner if horizontal 3-in-a-row"""
        count = 0
        for i in range(3):
            if self.board[action[0]][i] == self.player_turn():
                count += 1
                self.winning_moves[action[0]][i] = 1
            else: break
        if count >= 3: return True
        else: self.winning_moves[:] = 0; return False

    def diagonal(self, action):
        """return winnfer if diagonal 4-in-a-row"""
        # check main diagnoal
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == self.player_turn():
            self.winning_moves[0][0] = self.winning_moves[1][1] = self.winning_moves[2][2] = 1
            return True
        #check off diagnoal
        if self.board[0][2] == self.board[1][1] == self.board[2][0] == self.player_turn():
            self.winning_moves[0][2] = self.winning_moves[1][1] = self.winning_moves[2][0] = 1
            return True

    def render(self):
        print('\nturn: ' + str(self.turn) + ', last: ' + str(self.last_move))

        for i in range(3):
            print(f"\t{i} ", end="")
            for j in range(3):
                if self.board[i][j] == 1:
                    if self.last_move == [i,j]: print(colored('| X', 'green'), end=" ")
                    elif self.winning_moves[i][j]: print(colored('| X', 'red'), end=" ")
                    else: print("| " + 'X', end=" ")
                elif self.board[i][j] == 2:
                    if self.last_move == [i,j]: print(colored('| O', 'green'), end=" ")
                    elif self.winning_moves[i][j]: print(colored('| O','red'), end=" ")
                    else: print("| " + 'O', end=" ")
                else: print("|  ", end=" ")
                #print("| " + str(self.board[i][j]), end=" ")
            print("|")

        print("\t   _   _   _ ")
        print("\t   0   1   2 ")
        if self.done:
            print("Game Over!")
            if self.winner == 1:
                print("X is the winner")
            elif self.winner == 2:
                print("O is the winner")
            else:
                print("draw game")

    def __copy__(self):
        """copy board"""
        new = type(self)()
        new.board = copy(self.board)
        new.turn = self.turn
        new.done = self.done
        new.winner = self.winner
        new.winning_moves = copy(self.winning_moves)
        new.last_move = self.last_move
        return new

    def __repr__(self):
        #self.render()
        self.render()
        return f'\nturn: {self.turn},last: {self.last_move}, id: {id(self)}'

    def __hash__(self):
        "Nodes must be hashable"
        return hash(tuple(self.board.flatten() ))

    def __eq__(node1, node2):
        "Nodes must be comparable"
        if node1 is None: return True
        if node1 is not None and node2 is None: return False
        return np.array_equal(node1.board, node2.board)

    def __gt__(node1, node2):
        return node1.turn > node2.turn

