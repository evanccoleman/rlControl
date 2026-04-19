# evaluate_agents.py

import collections
import numpy as np

def evaluate_agent(agent, env, agent_type, n_episodes=5):
    """
    Evaluate a custom agent by running episodes and averaging rewards.
    """
    if agent_type == "frameddpg":
        return evaluate_framestacking(agent, env, n_episodes)
    else:
        return evaluate_standard(agent, env, n_episodes)

def evaluate_standard(agent, env, n_episodes=5):
    """
    Evaluate an agent that uses raw observations.
    """
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards)

def evaluate_framestacking(agent, env, n_episodes=5):
    """
    Evaluate a framestacking agent that uses stacked observations.
    """
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        # fill deque with duplicated obs
        deque = collections.deque(maxlen=agent.stack_size)
        for _ in range(agent.stack_size):
            deque.append(obs)

        while not done:
            stacked_obs = np.array(deque).flatten()
            action, _ = agent.predict(stacked_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            deque.append(obs)

        rewards.append(episode_reward)

    return np.mean(rewards)
