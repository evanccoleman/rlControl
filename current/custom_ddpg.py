"""
Tabular Q-Value DDPG Algorithm

A simplified DDPG-style algorithm using tabular Q-learning instead of neural networks.
This approach discretizes the state and action spaces and uses Q-tables for both
the critic (Q-function) and a tabular policy for the actor.
"""

import numpy as np
from collections import defaultdict
import random


class ReplayBuffer:
    """Experience replay buffer for storing transitions."""

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        if len(self.buffer) < self.capacity: # allocate memory
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class CustomDDPG:
    """Custom DDPG agent by Olivia Buchanan."""

    def __init__(self,
                 policy: str = None,
                 env=None,
                 
                 ):
        """
        Create a CustomDDPG agent.

        Employ neural networks with a MlpPolicy for actor and critic networks.
        """

