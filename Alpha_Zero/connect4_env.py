import numpy as np
from termcolor import colored
from copy import copy, deepcopy

# noinspection PyArgumentList

class Connect4Env:
    def __init__(self, board=None):
        """initialize environment"""
        self.name = 'Connect4'
        self.board_size = (6,7)
        self.action_size = 7

        if board is None:
            self.board = np.zeros(self.board_size).astype('int')
            self.board[:] = 2 # for empty
        else: self.board = board

        self.turn = 0
        self.done = False
        self.winner = None
        self.winning_moves = np.zeros(self.board_size).astype('int')
        self.last_move = None
        self.legal = [x for x in range(7)]

    def reset(self):
        """reset environment state"""
        self.board[:] = 2 # for empty
        self.turn = 0
        self.done = False
        self.winner = None
        self.winning_moves = np.zeros(self.board_size).astype('int')
        self.last_move = None
        self.legal = [x for x in range(7)]

    def player_turn(self):
        """return current player: 0 or 1"""
        # X goes first at turn 0
        return self.turn % 2

    def last(self):
        return self.last_move[1]

    def step(self, action):
        """increment player, then play move"""
        if self.done: raise ValueError("game already over")
        if self.legal_moves() == []: raise ValueError("no legal moves")
        if action not in self.legal_moves(): raise ValueError("illegal move")
        for i in range(5,-1,-1):
            if self.board[i][action] == 2:
                self.board[i][action] = self.player_turn()
                self.last_move = (i, action) # i = row, action = column
                if i == 0: self.legal.remove(action) # reduce action space when col is filled
                break
        self.check_for_fours(action)
        self.turn += 1
        return self

    def legal_moves(self):
        return self.legal

    def augment(self, board, pi, win):
        """augment data, and assign values
            return boards, Qs, policies"""
        boards = []; Qs = []; policies = []
        boards.append(board)
        Qs.append(win)
        policies.append(pi)
        # flip board and policies for symmetry
        boards.append(np.flip(board, axis=1))
        Qs.append(win)
        policies.append(np.flip(pi))
        return boards, Qs, policies

    def _separate_players(self):
        """split board into player1 board and player2 board.
        return player1_board, player2_board, current_player"""
        board0 = np.array(self.board) == 0
        board1 = np.array(self.board) == 1
        current_player = np.ones(self.board_size) * self.player_turn()
        return board0.astype(int), board1.astype(int), current_player.astype(int)

    def check_for_fours(self, action):
        """update self.done (False->True), self.winner (None->self.player_turn())"""
        if self.is_winner():
            self.done = True
            self.winner = self.player_turn()
            return;
        elif self.turn >= 41: # turn changes after check
            self.done = True;
            return

    def is_winner(self):
        """return winner if 4-in-a-row vertically, horizontally, or diagonally"""
        if self.turn < 6: 
            return False  # can't win before 7th turn
        current_player = self.player_turn()
        row, col = self.last_move
        row = 5 - row # row[0] is the top of the board
        # Directions: vertical, horizontal, main-diagonal, off-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            self.winning_moves[:] = 0  # Reset winning_moves at the start of each direction
            self.winning_moves[-(row-5)][col] = 1

            # Check in the positive direction
            r, c = row + dr, col + dc
            while 0 <= r <= 5 and 0 <= c <= 6:
                if self.board[-(r-5)][c] == current_player:
                    count += 1
                    self.winning_moves[-(r-5)][c] = 1
                else:
                    break
                r, c = r + dr, c + dc

            # Check in the negative direction
            r, c = row - dr, col - dc
            while 0 <= r <= 5 and 0 <= c <= 6:
                if self.board[-(r-5)][c] == current_player:
                    count += 1
                    self.winning_moves[-(r-5)][c] = 1
                else:
                    break
                r, c = r - dr, c - dc

            # If a winning line is found, stop checking and return True
            if count >= 4:
                return True

        # If no winning line is found, reset winning_moves and return False
        self.winning_moves[:] = 0
        return False

    def render(self, indent = 1):
        print('\nturn: ' + str(self.turn) + ', last: ' + str(self.last_move))
        if self.done:
            print("Game Over!")
            if self.winner == None: print("draw game")
            else: print(f"player {self.player_turn()} is the winner")

        for i in range(6):
            print("\t"*indent, end="")
            for j in range(7):
                if self.board[i][j] == 0:
                    if self.last_move == (i,j): print(colored('| O', 'green'), end=" ")
                    elif self.winning_moves[i][j]: print(colored('| O', 'red'), end=" ")
                    else: print("| " + 'O', end=" ")
                elif self.board[i][j] == 1:
                    if self.last_move == (i,j): print(colored('| X', 'green'), end=" ")
                    elif self.winning_moves[i][j]: print(colored('| X','red'), end=" ")
                    else: print("| " + 'X', end=" ")
                else: print("|  ", end=" ")
                #print("| " + str(self.board[i][j]), end=" ")
            print("|")
        print("\t  _   _   _   _   _   _   _ ")
        print("\t  0   1   2   3   4   5   6 ")

    def __copy__(self):
        """copy board"""
        new = type(self)() # create a new MCTS, not a new Connect4
        new.turn = self.turn
        new.done = self.done
        new.winner = self.winner
        new.last_move = self.last_move
        new.legal = copy(self.legal) # copy 1d array
        new.board = deepcopy(self.board) # copy 2d array
        new.winning_moves = np.zeros(self.board_size).astype('int')
        #new.winning_moves = deepcopy(self.winning_moves) 
        return new

    def __repr__(self):
        self.render()
        return f"\nturn: {self.turn}, player: {self.player_turn}, last: {self.last_move}, hash: {self.__hash__}"

    def __eq__(self, other):
        "Nodes must be comparable"
        if not self and not other: return True
        elif not other: return False
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        """Nodes must be hashable"""
        return hash(tuple(self.board.flatten() ))

    def __gt__(self, other):
        return self.turn > other.turn

