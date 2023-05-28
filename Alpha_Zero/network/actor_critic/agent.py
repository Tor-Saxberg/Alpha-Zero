#import sys
#sys.path.insert(0, './actor_critic')

from .actor import Actor
from .critic import Critic
from .replaybuffer import ReplayBuffer
from .ounoise import OUNoise

import numpy as np
from collections import namedtuple, deque

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self):
        self.state_size = (6,7)
        self.action_size = 7

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size)
        self.actor_target = Actor(self.state_size, self.action_size)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.3
        self.exploration_sigma = 0.5
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.50  # discount factor
        self.tau = 0.5  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        self.last_state = state
        return state

    def train(self,nodes):
         # Save experience / reward
         for i, node in enumerate(reversed(nodes) ):
            if node.done: next_state = node.board
            if node.parent: self.last_state = node.parent.board
            else: self.last_state = node.board
            if not self.last_state: self.last_state = node.board

            #self.memory.add(self.last_state, action, reward, next_state, done)
            self.memory.add(self.last_state, node.last_move[1], node.Q, node.board, node.done)
            self.last_state = node.board

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

            if node.done: self.noise.reset()
                
    def play(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        v,pi = self.actor_local.model.predict(state)
        return list(pi + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        netx_actions = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, netx_actions])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        v_gradients, pi_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, v_gradients, pi_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
