this project learns to play connect4 or tic-tac-toe. you can run it simply with '$python3 main.py' for tic-tac-toe. you can also add flags --game [connect4 tictactoe] --net [rnn cnn] --start [number start last(begin/resume epoch). You can display the training archs for each epoch with 'tensorboard --logdir logging'.]

this program works by creating an agent and training it until it can beat an older version of itself. Then, the older agent is updated to the newer version, and the cycle repeats. some parameters determine how many wins in a set of games are needed to update the older agent and how many sequential draws in a game signal that learning has leveled off. 



the files in this project are straightforward: 
the nerual netowrks it can use are in the Network folder. 
main.py imports the other files and runs the program. 
conenct4_env.py and tictactoe_env.py encode the game rules.
arena.py contains the code to compete agents. 
agent.py creates training examples and trains the network
    Agent
        train: simulates games by playing the best moves according to the network, then sends the board states and win/loss results back to the network
        _check: prints last baord state at game end
        _make_examples: splits each board state of a game into player1 and player2 boards and creates symmetric board states for training
        play: uses the agent to play against a human
MCTS_minimal.py is a naive implementation of MCTS.
MCTS_tree uses a tree structure to play full rollouts.
MCTS_net uses the networks in /Network/ to skip the rollout phase and just backpropogate the predicted values.
the /img/ folder is used for the connect4.ipynb file.
the /actor_critic/ folder contains code for the actor_critic network model, but is not working at this time.
