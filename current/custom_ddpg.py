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

    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        """
        if len(self.buffer) < self.buffer_size: # allocate memory
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

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
                 action_noise=0.1,
                 seed: int = None,
                 buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 ):
        """
        Create a CustomDDPG agent.

        Learning rate is same for all actor and critic networks.

        Employ a MlpPolicy for actor and critic networks.

        This agent is specifically designed to be compatible with
        the walkers_v5.py program and functional against 
        Stablebaselines3 agents.
        """
        # env
        self.env = env

        # replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # other settings
        self.action_noise = action_noise
        self.learning_rate = learning_rate 
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
                                                lr=learning_rate)

        # critic and target critic
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=learning_rate)
    
    def learn(self,
              total_timesteps: int = 0,
              reset_num_timesteps=False,
              callback = None,
              ):
        """
        Train agent.
        """

        # train for total_timesteps
        i = 1 # start step
        while i < total_timesteps + 1:
            obs, info = self.env.reset() # reset env
            is_episode_over = False # loop control variable

            # one full episode
            while (i < total_timesteps + 1) and (not is_episode_over):

                # agent chooses action
                action, _ = self.predict(obs, deterministic=False)

                # environment applies action
                next_obs, reward, terminated, trunc, info = self.env.step(action)
                i = i + 1

                # store transition
                self._store_transition(obs, action, reward, next_obs, terminated)

                # sample minibatch and learn from past experiences
                if len(self.replay_buffer.buffer) >= self.batch_size:

                    # sampel minibatch
                    batch_of_tensors = self._sample_batch()

                    # update online networks
                    self._update_critic(batch_of_tensors[0], # states
                                        batch_of_tensors[1], # actions
                                        batch_of_tensors[2], # rewards
                                        batch_of_tensors[3], # next_states
                                        batch_of_tensors[4], # dones
                                        )
                    self._update_actor(batch_of_tensors[0])

                    # update target networks
                    self._soft_update_target("actor")
                    self._soft_update_target("critic")

                # move to the next state
                obs = next_obs
                is_episode_over = terminated or trunc

    def predict(self, state, deterministic=False):
        """
        Select an action using the current policy.
        """
        state_tensor = torch.FloatTensor(state)
        action = self.actor(state_tensor)
        if not deterministic:
            action = action + torch.FloatTensor(self.action_noise())
        return action.detach().numpy(), None
    
    def _store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _sample_batch(self):
        """
        Sample a batch of experiences.

        Raw batch is list of tuples (state, action, reward, next_state, done);
        instead, return list of tensors [state tensor, action tensor, etc.]
        """
        batch_as_tuples = self.replay_buffer.sample(self.batch_size)
        grouped_tuples = zip(*batch_as_tuples)
        batch_as_tensors = [torch.Tensor(the_tuple) for the_tuple
                            in grouped_tuples]
        return batch_as_tensors


    def _update_actor(self, state):
        """
        Update the actor network.
        """
        # get the action and qvalue
        action = self.actor(state)
        qvalue = self.critic(state, action)

        # compute loss
        loss = -qvalue.mean()
        
        # remove gradient from previous updates,
        # backpropagate the loss, and then update weights
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

    def _update_critic(self, state, action, reward, next_state, done):
        """
        Update the critic network.
        """
        # get the target part of the TD
        # prevent use of target networks from affecting their gradients
        with torch.no_grad():
            target_action = self.actor_target(next_state)
            target_qvalue = self.critic_target(next_state, target_action)
            td_target = reward + (self.gamma * target_qvalue * (1 - done))

        # get the online critic's Q-value
        online_qvalue = self.critic(state, action)

        # compute loss
        loss = nn.MSELoss()
        td = loss(online_qvalue, td_target)

        # remove gradient from previous updates,
        # backpropagate the loss, and then update weights
        self.critic_optimizer.zero_grad()
        td.backward()
        self.critic_optimizer.step()

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

    def save(self, save_path : str):
        """
        Save the CustomDDPG agent to a zip file.
        """
        torch.save({"actor": self.actor.state_dict(),
                    "actor_target": self.actor_target.state_dict(),
                    "critic": self.critic.state_dict(),
                    "critic_target": self.critic_target.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "action_noise": self.action_noise,
                    "buffer_size": self.replay_buffer.buffer_size,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "tau": self.tau
                    },
                   save_path)

    @classmethod
    def load(cls,
             save_path : str,
             env=None,
             seed=0):
        """
        Load an old CustomDDPG agent from a zip file.
        """
        # get hyperparameters from zip file
        hyperparameters_dict = torch.load(save_path)

        # create agent
        agent = cls(env=env,
                    action_noise=hyperparameters_dict["action_noise"],
                    seed=seed,
                    buffer_size=hyperparameters_dict["buffer_size"],
                    batch_size=hyperparameters_dict["batch_size"],
                    learning_rate=hyperparameters_dict["learning_rate"],
                    gamma=hyperparameters_dict["gamma"],
                    tau=hyperparameters_dict["tau"]
                    )

        # load network weights
        agent.actor.load_state_dict(hyperparameters_dict["actor"])
        agent.actor_target.load_state_dict(hyperparameters_dict["actor_target"])
        agent.critic.load_state_dict(hyperparameters_dict["critic"])
        agent.critic_target.load_state_dict(hyperparameters_dict["critic_target"])
        agent.actor_optimizer.load_state_dict(hyperparameters_dict["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(hyperparameters_dict["critic_optimizer"])

        return agent
