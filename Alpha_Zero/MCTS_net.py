""" MCTS class"""
__package__ = 'Alpha_Zero.tests'
#from .connect4_env import Connect4Env
from termcolor import colored
#from sklearn.preprocessing import normalize
import math
import numpy as np
from numpy.random import choice
from copy import copy
#from time import time

class MCTS():
    """an MCTS using a tree structure"""
    def __init__(self, env, net=None, expl=1, copy=None):
        """initialize N, Q, pi, parent, children"""
        self.env = env
        self.net = net
        self.t = 0.5 # pi[child.last()] = child.N**t/sum(children.N)
        self.expl = expl # max(child.Q + expl * pi * sqrt(N) / (child.N+1))
        self.sim_flag = 0

        self.N = 0
        self.U = 0
        if not copy: self.Q, self.pi = self._predict() # don't run _predict() twice
        self.children = []
        self.parent = None

        self.print_N = 0
        self.print_Q = 0
        self.print_U = 0
        self.print_pi = [0] * 7

    def done(self):
        return self.env.done

    def last(self):
        return self.env.last()

    def reset(self):
        current = self
        while current.parent:
            current = current.parent
        return current

    def play(self, sims=1):
        """update policy and return next state"""
        #sims = min(sims, len(self.legal_moves() ))
        #update policies via network simulations
        self._simulate(sims)
        # pick a move based on improved policy
        # next_state = choice(self.children, p=self.pi[self.env.legal_moves() ])
        next_state = max(self.children, key=lambda x: x.U)
        return next_state

    def _get_tree(self):
        """propogate from terminal, update N,Q; return path
           Return the branch as a list of nodes"""
        current = self
        nodes = []
        while current is not None:
            nodes.append(current)
            current = current.parent
        return nodes # reversed tree path

    def _predict(self):
        """get policy from net"""
        if self.net:
            # divide board
            player1_board, player2_board, next_player = self.env._separate_players()
            board = np.block([player1_board, player2_board, next_player])
            board_reshape = np.reshape(board, (-1,board.shape[0], board.shape[1]) )
            # assign Q, pi from net
            Q, pi = self.net.model.predict_on_batch(board_reshape)
            Q = Q[0][0]; pi = pi[0] # why are these 2D arrays?
            pi[pi==0] = 0.5 # never say never
            pi = np.random.dirichlet(pi*self.expl + 1) # add noise
        else:
            Q = 0 # unkown outcome
            pi = np.zeros(7)
            legal_moves = self.env.legal_moves()
            n_moves = len(legal_moves)
            if n_moves: pi[legal_moves] = 1/n_moves
            return Q, np.array(pi)

    def _simulate(self, sims):
        """update pi and Q via simulations
           randomize the first move"""
        for _ in range(sims):
            #if self.N * self.env.turn > 50: break
            current = self
            while not current.done():
                #if current.Q > 0.8 and current.turn > 20: break;
                if not current.children: current._expand()
                # if there's a simulated terminal node
                if any((child.sim_flag and child.done()) for child in current.children):
                    breakpoint()
                    break
                # if branch fully explored:
                if all(child.sim_flag for child in current.children):
                    breakpoint()
                    break
                # random first move for stochasticity
                if not current.parent: current = choice(current.children)
                else: current = current._action()
            current._backpropogate(self) # reversed tree path from leaf to self
        # you should set a sim flag = 1 if all(child.flag... ) or any(child.done...) to prevent duplicates.
            # when using a net, the whole node tree is forgotten anyways
        # one simulation should not affect another. you need to parralelize this operation.
            # simulations are independent, so you don't need to worry about opening a node
                # _simulate: don't lock for _action(), but wait for it to be unlocked.
            # simulations are overlapping, so you need to lock a node while updating
                # _simulate: lock a node before expanding it
                # _backpropogate: lock current, update, lock next, release last

    def _expand(self):
        """create children nodes"""
        for action in self.env.legal_moves():
            #new = MCTS(node=self)
            new = copy(self)
            new.N = 0
            new.parent = self
            new.children = []
            new.env.step(action)
            self.children.append(new) # children are sorted by move
            new.Q, new.pi = new._predict() # must do this after steping

    def _action(self):
        """select best child"""
            #return child.Q
        max_U = max(self.children, key=lambda x: abs(x.U)).U # U > 0 for all or all but 1
        if max_U == 0: return choice(self.children)
        children_U = np.array([child.U/(child.N+1) for child in self.children])
        children_U += max_U # all positive values
        return choice(self.children, p=children_U/sum(children_U)) # add randomness
        #return choice(self.children, p=self.pi[self.env.legal_moves()]) # add randomness

    def _backpropogate(self, root):
        """propogate from leaf; update N,Q; return path
           Arguments:
               self: leaf of current simulation
               root: last non-simulated move
        """
        flag = 0
        current = self
        while current: 
            current._Q()
            if current == root: flag = 1;
            if not flag: current.N += 1 # don't update N of root
            current._pi()
            current.print_pi = copy(current.pi)
            if not flag: current._U()
            #if not flag: current._printing_update()
            if current.parent == root: current._printing_update()
            current = current.parent

    def _Q(self):
        """update Q values: 0 (tie), 1 (win), or -AVG(children.Q)"""
        # current turn won or tied
        if self.done(): 
            self.Q = 1 if self.env.winner else 0; 
        # next turn wins
        elif any(child.Q == 1 and child.done() == True for child in self.children):
            self.Q = -1
        # Q = avg child.Q
        else:
            Qs = np.array([child.Q for child in self.children if child.N > 0])
            if Qs.size > 0:
                self.Q = -np.mean(Qs)

    def _pi(self):
        """update policies after MCTS in last-move order"""
        if self.done() or self.N == 0: return # leaf or non-agent played without simulating
        if self.env.turn > 10: self.t = (self.N + self.env.turn) / (self.N)
        for child in self.children: 
            # at least one child was simulated
            if child.N: self.pi[child.last()] = child.N**self.t/(self.N+1)
        self.pi /= sum(self.pi)

    def _U(self):
        if not self.parent: return
        self.U = self.Q + self.parent.expl * self.parent.pi[self.last()] * math.sqrt(self.N) / (self.N+1)

    def _printing_update(self):
        self.print_N = self.N
        self.print_Q = self.Q
        self.print_U = self.U
        self.print_pi = copy(self.pi)

    def _print_parents(self):
        current = self
        while current.parent:
            current._print_siblings()
            current = current.parent

    def _print_siblings(self):
        """print list of parents"""
        print('\nsiblings:')
        if self.parent is None: print('None')
        else:
            parent = self.parent
            s = ''
            sN = ''
            sQ = ''
            sP = ''
            sU = ''

            for node in parent.children:
                sN += '\t'
                sQ += '\t'
                sP += '\t'
                U = node.Q + node.expl * parent.pi[node.last()] * math.sqrt(node.N) / (node.N+1)
                sU += '\t'

                # highlight self
                if node is self: 
                    sN += colored(f'{node.print_N}','red')
                    sQ += colored(f'{node.print_Q:2.3f}','red')
                    sU += colored(f'{node.print_U:2.3f}','red')
                    sP += colored(f'{parent.print_pi[node.last()]:2.3f}','red')

                # highlight max sibling
                else:
                    # color max N
                    if node == max(parent.children, key=lambda x: x.print_N):
                        sN += colored(f'{node.print_N}','green')
                    else:
                        sN += f'{node.print_N}'
                    # color max Q
                    if node == max(parent.children, key=lambda x: x.print_Q):
                        sQ += colored(f'{node.print_Q:2.3f}','green')
                    else:
                        sQ += f'{node.print_Q:2.3f}'
                    # color max U
                    if node == max(parent.children, key=lambda x: x.print_Q):
                        sU += colored(f'{node.print_U:2.3f}','green')
                    else:
                        sU += f'{node.print_U:2.3f}'
                    # color max pi
                    if node == max(parent.children, key=lambda x: parent.print_pi[x.last()]):
                        sP += colored(f'{parent.print_pi[node.last()]:2.3f}','green')
                    else:
                        sP += f'{parent.print_pi[node.last()]:2.3f}'
            # print parent
            s += f"turn {self.env.turn} \n"
            s += "N" + sN + '\n'
            s += "Q" + sQ + '\n'
            s += "U" + sU + '\n'
            s += "P" + sP + '\n'
            print(s)

    def _print_self(self):
        """print current node info"""
        #print(f"\nself: \nN: {self.N} \nQ: {self.Q} \npi: {self.pi}"); 
        s = ''
        s += f'N: {self.N}, \n'
        s += f'Q: {self.Q:.2f}, \n'
        s += f'U: {self.U:2.3f}\n'
        s += f'policy: ' + '   '.join(f"{x:2.3f}" for x in self.pi)
        print(s)
        self.env.render()

    def _print_children(self):
        """print list of children"""
        print('\nchildren:', end=" ")
        if not self.children: print('None'); return ''
        else:
            s = ''
            for child in self.children:
                s += '\n\t'
                #s += f'\n\tmove: {child.last()} '
                s += f'N: {child.N} \t'
                s += f'Q: {(child.Q):.2f} \t'
                s += f'U: {(child.U):.2f} \t'
                s += f'\tpolicy: '
                s += f'  {["{:.3f}".format(pi) for pi in child.pi]}'
                #s += f' policy: ' + '   '.join(f"{x:2.2f}" for x in self.pi)
            print(s)

    def __repr__(self):
        """print MCTS node representation"""
        self._print_siblings()
        self.env.render()
        #self._print_children()
        return ''

    def __copy__(self):
        """copy node attributes except parent/children, predict Q and pi"""
        #new = MCTS(copy=True) # don't run _predict() twice
        new = MCTS(self.env, copy=True) # don't set pi and Q twice
        new.env = self.env.__copy__()
        # can't use __dict__.update() without effecting env __copy__()
        # in theory, you don't need to copy the env. just use one copy for simulating, and restore it to root
        new.pi = []
        new. Q = 0
        new.net = self.net
        new.t = self.t
        new.expl = self.expl
        new.children = []
        new.parent = None
        return new
