# episodes.py

import random

import torch
import numpy as np
import gymnasium as gym

from agents import turn_off_training_mode, restore_training_mode

def set_seed(seed):
    """
    Sets the seed for Python packages.

    Ensures consistent seeding.
    """

    # for Python's built-in random module
    random.seed(seed)

    # for numpy
    np.random.seed(seed)

    # for PyTorch and PyTorch with GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_episode(agent,
                env: gym.Env,
                discount: bool = False,
                ) -> int:
    """
    Executes a single episode for an agent without LSTM.

    Returns the episode rewards.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0 # track episode returns
    total_discount = 1
    is_episode_over = False # loop control variable

    # take actions and update agent until episode termination
    while not is_episode_over:

        # agent chooses action
        action, _states = agent.predict(obs, deterministic=True)

        # environment applies action
        next_obs, reward, terminated, trunc, info = env.step(action)

        # move to the next state
        obs = next_obs
        is_episode_over = terminated or trunc

        # update episode returns
        episode_rewards += reward * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if discount:
        return episode_rewards
    else:
        return info["episode"]["r"]

def run_episode_lstm(agent,
                     env: gym.Env,
                     discount: bool = False,
                     ) -> int:
    """
    Executes a single episode for an agent with LSTM.

    Returns the episode rewards.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0 # track episode returns
    total_discount = 1
    is_episode_over = False # loop control variable
    lstm_states = None # track hidden state of LSTM stuff
    episode_starts = np.array([True]) # helps reset lstm_states

    # take actions and update agent until episode termination
    while not is_episode_over:

        # agent chooses action
        action, lstm_states = agent.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)

        # environment applies action
        next_obs, reward, terminated, trunc, info = env.step(action)

        # move to the next state
        obs = next_obs
        is_episode_over = terminated or trunc
        episode_starts = np.array([terminated or trunc])

        # update episode returns
        episode_rewards += reward * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if discount:
        return episode_rewards
    else:
        return info["episode"]["r"]

def run_many_episodes(agent,
                      env,
                      num_episodes: int = 0,
                      agent_type: str = None,
                      discount: bool = False,
                      ) -> None:
    """
    Executes episodes in testing mode for an agent.

    Returns the average returns from the testing run.
    """

    # turn off training mode and begin exploitation
    training_settings = (agent.learning_rate, agent.action_noise)
    turn_off_training_mode(agent)

    # track episodic returns
    rewards = []

    # run the episodes
    for i in range(1, num_episodes + 1):
        episode_rewards = 0 # initialize and set this in scope

        # decide whether to run LSTM episode
        if agent_type == "rppo":
            episode_rewards = run_episode_lstm(agent, env,
                                             discount=discount)
        else:
            episode_rewards = run_episode(agent, env,
                                         discount=discount)

        # add episode returns to running list
        rewards.append(episode_rewards)

    # restore training mode
    restore_training_mode(agent, training_settings)

    # calculate performance
    avg_reward = np.mean(rewards)

    return avg_reward
