from random import choice

class Arena():
    def __init__(self, env): self.env = env

    def compete(self, player1, player2, games):
        """
        return (player1) win rate for x games"""
        wins_player1 = 0
        wins_player2 = 0

        for _ in range(games):
            turn = choice([1,-1]) #random player order
            #play (games) games
            while True:
                if turn == 1:
                    self.env = player1.play(self.env)
                    if self._check('player1'):
                        print("player 1 wins")
                        wins_player1 += 1
                        break
                    turn = -turn

                if turn == -1:
                    self.env = player2.play(self.env)
                    if self._check('player2'):
                        print("player 2 wins")
                        wins_player2 += 1
                        break
                    turn = -turn
        print("returning")
        return wins_player1/games

    def test(self, player, runs):
        """return player win rate against random agent"""
        class testser():
            def __init__(self): pass
            def play(self, env): return env.step(choice(env.legal_moves() )) # random move
        return self.compete(player, testser(), runs) #

    def _check(self, player):
        """detect gameover, """
        if self.env.done:
            self.env.render()
            self.env.reset()
            #print(player + ' self.wins!')
            return True
        return False
