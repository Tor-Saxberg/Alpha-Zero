import pytest
import cProfile

from Alpha_Zero.connect4_env import Connect4Env
from random import choice

env = Connect4Env()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_legal():
    print(env.legal_moves() )

#@pytest.mark.skip(reason="This test is currently skipped")
def test_render():
    """testing render()"""
    env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_step():
    """testing step()"""
    print("\ntesting step")
    env.reset()
    while not env.done:
        env.step(choice(env.legal_moves()))
        env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_vert():
    """testing vertical check"""
    print("\ntesting vertical")
    env.reset()
    for i in range(3):
        env.step(0)
        env.step(1)
    env.step(0)
    env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_horiz():
    """testing horizontal check"""
    print("\ntesting horizontal")
    env.reset()
    while not env.done:
        env.step(env.legal_moves()[0])
    env.render()
    return

#@pytest.mark.skip(reason="This test is currently skipped")
def test_diag():
    """testing diagonal check"""
    print("\ntesting main diagonal")
    env.reset()
    for row in range(3):
        for col in range(7):
            if not env.done:
                env.step(col)
    env.step(4)
    env.render()

    print("\ntesting off diagonal")
    env.reset()
    for row in range(3):
        for col in range(7):
            env.step(col)
    env.step(2)
    env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_full():
    print("\ntesting full")
    env.reset()
    for row in range(3):
        for col in range(7):
            env.step(col)
            env.render()
    env.render()
    # block diagonal 
    for col in range(5):
        env.step(col+1)
    for col in range(7):
        env.step(col)
    for col in range(7):
        env.step(col)
    env.step(env.legal_moves()[0])
    env.step(env.legal_moves()[0])
    env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_rand():
    """play randomly"""
    print('\nrandom testing')
    env.reset()
    while not env.done:
        env.step(choice(env.legal_moves() ))
        if env.winning_moves.all() != 0 and not env.done:
            env.render()
    env.render()

#@pytest.mark.skip(reason="This test is currently skipped")
def test_separate_players():
    #test_rand()
    baord0,baord1,turn = env._separate_players()
    print("\n")
    print(baord0, end="\n\n")
    print(baord1, end="\n\n")
    print(turn)