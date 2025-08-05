# ddpg.py

# good ol' numpy and plt
import numpy as np
import matplotlib.pyplot as plt

# gymnasium
import gymnasium as gym

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# stuff for custom OUActionNoise
import copy
import random

class ReplayBuffer():
    """
    Experience Replay Buffer stores transitions of the
    form (state, action, reward, next_state, done)
    into parallel numpy arrays.

    When the buffer reaches max capacity, new transitions
    are stored starting back at the beginning.

    Note: a deque of tuples is a more Pythonic way of storing
    experiences, but parallel numpy arrays is more performance
    and memory efficient.
    """

    def __init__(self,
                 size: int = 0,
                 obs_dim: int = 0,
                 action_dim: int = 0,
                 batch_size: int = 0,
                 ) -> None:
        """
        Initialize the replay buffer.
        """
    
        # buffer as parallel arrays
        self.obs_buff = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buff = np.zeros([size, obs_dim], dtype=np.float32)
        self.action_buff = np.zeros([size, action_dim], dtype=np.float32)
        self.reward_buff = np.zeros([size], dtype=np.float32)
        self.done_buff = np.zeros([size], dtype=np.float32)

        # other buffer attributes
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(self,
              obs: np.ndarray = None,
              action: np.ndarray = None,
              reward: float = 0.,
              next_obs: np.ndarray = None,
              done: bool = False,
              ) -> None:
        """
        Stores a transition in the buffer.
        Also updates the ptr and the buffer's size so far.
        """

        # store transition into the buffer
        self.obs_buff[self.ptr] = obs
        self.next_obs_buff[self.ptr] = next_obs
        self.action_buff[self.ptr] = action
        self.reward_buff[self.ptr] = reward
        self.done_buff[self.ptr] = done
        
        # update the pointer and how full buffer is
        self.ptr = (self.ptr + 1) % self.max_size # ring buffer
        self.size += 1
        self.size = min(self.size, self.max_size) # keep size capped

    def sample_batch(self) -> dict:
        """
        Randomly sample a minibatch of experiences from buffer
        according to the batch_size. 
        Returns a dictionary of {str : np.ndarray}
        """

        # randomly select from filled indices
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        # return the minibatch
        return dict(
                obs=self.obs_buff[idxs],
                next_obs=self.next_obs_buff[idxs],
                action=self.action_buff[idxs],
                reward=self.reward_buff[idxs],
                done=self.done_buff[idxs],
                )

    def __len__(self) -> int:
        """
        Returns an int of how full the buffer is.
        """
        return self.size

class OUActionNoise:
    """
    Ornstein-Uhlenbeck action noise.
    """
    
    def __init__(self,
                 size: int,
                 mu: float = 0.0,
                 theta: float = 0.0,
                 sigma: float = 0.0,
                 ):
        """
        Initialize parameters and noise process.
        """

        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Reset the internal state (= noise) to mean (mu).
        """
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """
        Update internal state and return it as a noise sample.
        """

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
                [random.random() for _ in range(len(x))]
                )
        self.state = x + dx
        return self.state

class Actor(nn.Module):
    """
    Outputs a single action for what to do next.
    Consists of three fully connected layers and three
    non-linearity functions: two ReLU's and tanh.
    """

    def __init__(self,
                in_dim: int,
                out_dim: int,
                init_w: float = 0.003,
                ):
        """
        Initialize actor network.
        """

        # call super constructor so we can assign parameters
        super(Actor, self).__init__()

        # create layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        # create weights and biases from uniform distro
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self,
                state: torch.Tensor,
                ) -> torch.Tensor:
        """
        One forward pass through the actor network.
        """

        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()
        return action

class Critic(nn.Module):
    """
    Evaluates the actor's choice using the state
    and action to estimate Q-values.
    Consists of three fully connected layers and two
    non-linearity functions: two ReLU's.
    """

    def __init__(self,
                 in_dim: int,
                 init_w: float = 0.003,
                 ):
        """
        Initialize critic network.
        """

        # call super constructor so we can assign parameters
        super(Critic, self).__init__()

        # create layers
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        # create weights and biases from uniform distro
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor,
                ) -> torch.Tensor:
        """
        One forward pass through the critic network.
        """
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        return value

class CustomDDPG():
    """
    CustomDDPG agent.
    Utilizes a custom replay buffer and OUActionNoise.
    This agent's environment needs to have actions normalized.
    """

    def __init__(self,
                 env: gym.Env,
                 learning_rate: float = 0,
                 gamma: float = 0,
                 batch_size: int = 0,
                 buffer_size: int = 0,
                 ou_noise_mean: float = 0.0,
                 ou_noise_theta: float = 0.15,
                 ou_noise_sigma: float = 0.2,
                 tau: float = 0.005,
                 initial_random_steps: int = 10000,
                 ):
        """
        Initialize CustomDDPG agent.
        """

        # get dimensions for buffer
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # initialize DDPGAgent's attributes
        self.env = env
        self.buffer = ReplayBuffer(size=buffer_size,
                                   obs_dim=obs_dim,
                                   action_dim=action_dim,
                                   batch_size=batch_size,
                                   )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        # OU action noise
        self.noise = OUActionNoise(size=action_dim,
                                   mu=ou_noise_mean,
                                   sigma=ou_noise_sigma,
                                   theta=ou_noise_theta,
                                   )

        # device: cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # actor, critic, and target networks
        self.actor = Actor(in_dim=obs_dim, out_dim=action_dim).to(self.device)
        self.actor_target = Actor(in_dim=obs_dim, out_dim=action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(in_dim=obs_dim+action_dim).to(self.device)
        self.critic_target = Critic(in_dim=obs_dim+action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # adam optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(),
                                          lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=self.learning_rate)

        # transition to store in buffer
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode: train or test
        self.is_test = False

    def select_action(self,
                      state: np.ndarray,
                      ) -> np.ndarray:
        """
        Returns the action to take in the input state.
        """

        # if initial random action should be conducted
        selected_action = None
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = (
                    self.actor(torch.FloatTensor(state)\
                    .to(self.device))\
                    .detach().cpu().numpy()
                    )

        # add noise for exploration during training
        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        self.transition = [state, selected_action]

        return selected_action

    def step(self,
             action: np.ndarray,
             ) -> tuple:
        """
        Take and execute an action in the env.
        Returns the next_state, reward, and done
        """

        # execute action
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        # store experience tracked by transition attribute in buffer
        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.buffer.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """
        Update the model by gradient descent.
        Returns actor_loss and critic_loss data.
        """

        device = self.device # shorten the next chunk of code

        # take numpy arrays returned by minibatch and convert to tensors  
        samples = self.buffer.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["action"]).to(device)
        reward = torch.FloatTensor(samples["reward"]).reshape(-1, 1).to(device)
        done = torch.FloatTensor(samples["done"]).reshape(-1, 1).to(device)

        # find the action for next_state and value of next_state 
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        masks = 1 - done # there is no "next_state" of terminal states
        curr_return = reward + self.gamma * next_value * masks

        # train critic
        # calculate loss, gradients, optimize weights
        values = self.critic(state, action)
        critic_loss = F.mse_loss(values, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        # calculate loss, gradients, optimize weights
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target update
        self._target_soft_update()

        return actor_loss.data, critic_loss.data

    def learn(self,
              total_timesteps: int,
              log_interval: int,
              progress_bar: bool,
              ):
        """
        Train the agent.
        The log_interval and progress_bar parameters
        don't do anything. Just provide compatibility with
        walkers.py
        """

        self.is_test = False

        # reset env and create some variables
        state, _ = self.env.reset()
        actor_losses = []
        critic_losses = []
        scores = []
        score = 0

        # training loop
        for self.total_step in range(1, total_timesteps + 1):

            # get and execute action
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            # update state and score
            state = next_state
            score += reward

            # reset stuff and add score before next episode
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # start improving model when warm up period is over
            if len(self.buffer) >= self.batch_size and \
                    self.total_step > self.initial_random_steps:
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss.cpu().numpy())
                critic_losses.append(critic_loss.cpu().numpy())

        # training is over
        self.env.close()

    def predict(self,
                observation,
                deterministic=True,
                ) -> np.ndarray:
        """
        Run the agent in testing mode.
        """
        # get the action
        action = self.select_action(state=observation)

        # return the action and an extra thing to make walkers.py
        # implementation happy
        return action, 0

    def save(self,
             path: str,
             ) -> None:
        """
        Save agent to a zip file.
        """
        torch.save({"state_dict": self.state_dict(),
                    "data": self._get_constructor_parameters(),
                    },
                   path,
                   )

    # def load(self,
    #          path: str,
    #          ):

    def _target_soft_update(self):
        """
        Soft update the target networks:
        target_net = tau*local + (1-tau)*target
        """

        for t_param, l_param in \
                zip(self.actor_target.parameters(), self.actor.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in \
                zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 -tau) * t_param.data)

class ActionNormalizer(gym.ActionWrapper):
    """
    Rescale and relocate the actions.
    """

    def action(self,
               action: np.ndarray,
               ) -> np.ndarray:
        """
        Change the range (-1, 1) to (low, high).
        """

        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self,
                       action: np.ndarray,
                       ) -> np.ndarray:
        """
        Change the range (low, high) to (-1, 1).
        """

        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action
