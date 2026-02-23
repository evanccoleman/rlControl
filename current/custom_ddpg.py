# custom_ddpg.py

import numpy as np
from collections import defaultdict
import random
import torch
import torch.nn as nn


class ReplayBuffer:
    """
    Experience replay buffer for storing transitions.
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        """
        if len(self.buffer) < self.capacity: # allocate memory
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    Simple Actor network.
    """

    def __init__(self, obs_dim, action_dim, action_high):
        """
        Create Actor network with three linear layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.action_high = torch.FloatTensor(action_high)

    def forward(self, state):
        """
        One forward pass through the network.
        """
        return self.net(state) * self.action_high


class Critic(nn.Module):
    """
    Simple Critic network.
    """
    def __init__(self, obs_dim, action_dim):
        """
        Create Critic network.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        """
        One forward pass through the network.
        """
        return self.net(torch.cat([state, action], dim=-1))


class CustomDDPG:
    """
    Custom DDPG agent.
    """

    def __init__(self,
                 env=None,
                 action_noise=None,
                 seed: int = None,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 actor_lr: float = 0.003,
                 critic_lr: float = 0.003,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 ):
        """
        Create a CustomDDPG agent.

        Employ a MlpPolicy for actor and critic networks.
        """
        # policy and env
        self.policy = policy
        self.env = env

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # other settings
        self.action_noise = action_noise
        self.learning_rate = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.seed = seed

        # get dimensions from env
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high

        # actor and target actor
        self.actor = Actor(obs_dim, action_dim, action_high)
        self.actor_target = Actor(obs_dim, action_dim, action_high)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)

        # critic and target critic
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
    
    def learn(total_timesteps: int = 0, callback = None):
        """
        Train agent.
        """
        k = 1

    def predict(self, state):
        """
        Select an action using the current policy.
        """
        state_tensor = torch.FloatTensor(state)
        action = self.actor(state_tensor)
        if self.action_noise is not None:
            action = action + torch.FloatTensor(self.action_noise())
        return action.detach().numpy()
    
    def _store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _sample_batch():
        """
        Sample a minibatch of experiences.
        """
        raw_minibatch = self.replay_buffer.sample(self.batch_size)


    def _update_actor():
        """
        Update the actor network.
        """
        k = 1

    def _update_critic():
        """
        Update the critic network.
        """
        k = 1

    def _soft_update_target(self, network_type):
        """
        Soft a target network using Polyak averaging.

        Loop through network parameters in pair-wise fashion.
        
        Use torch.no_grad() to prevent parameter manipulation from
        impacting later gradient calculations.
        """
        # select which network pair to update
        if network_type == "actor":
            online_params = self.actor.parameters()
            target_params = self.actor_target.parameters()
        elif network_type == "critic":
            online_params = self.critic.parameters()
            target_params = self.critic_target.parameters()

        # loop through parameters to update weights and biases
        with torch.no_grad():
            for online_p, target_p in zip(online_params, target_params):
                new_target_weight = self.tau * online_p.data + \
                                    (1 - self.tau) * target_p.data
                target_p.data.copy_(new_target_weight)

    def set_logger(self, logger = None):
        """
        Change where logger output goes.
        """
        self._logger = logger
        # might need to import logger.py from stable_baselines3


