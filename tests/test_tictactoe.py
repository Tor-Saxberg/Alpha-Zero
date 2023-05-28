import pdb, traceback, sys
from tictactoe_env import TictactoeEnv
from random import choice

def test_legal(env):
    print(env.legal_moves() )

def test_step(env):
    """testing step()"""
    env.step(4)
    env.step(0)
    print(env.board)

def test_render(env):
    """testing render()"""
    env.render()

#testing vertical check
def test_vert(env):
    """testing vertical check"""
    env.reset()
    for i in range(3):
        env.step(i*3)
        env.step(i*3+1)
    env.render()

def test_horiz(env):
    """testing horizontal check"""
    env.reset()
    for i in range(3):
        env.step(i)
        env.step(i+3)
    env.render()
    breakpoint()

def test_diag(env):
    """testing diagonal check"""
    env.reset()
    for i in range(9):
        env.step(i)
    env.render()

def test_rand(env):
    """play randomly"""
    env.reset()
    while env.done is False:
        print(env.legal_moves() )
        print(env)
        #env.render()
        env.step(choice(env.legal_moves() ))
    env.render()



def test():
    print(chr(27) + "[2J")
    env = TictactoeEnv()
    print("\ntesting init")
    #print(env.board)
    print("\ntesting legal moves")
    #test_legal(env)
    print("\ntesting step")
    #test_step(env)
    print("\ntesting render")
    #test_render(env)
    print("\ntesting vertical check")
    #test_vert(env)
    print("\ntesting horizontal check")
    #test_horiz(env)
    print("\ntesting diagonal check")
    #test_diag(env)
    print('\nrandom testing')
    test_rand(env)

if __name__ == '__main__':
    try: test()
    except: 
        extype, value, tb = sys.exc_info()
        print(extype, value)
        traceback.print_exc()
        pdb.post_mortem(tb)
