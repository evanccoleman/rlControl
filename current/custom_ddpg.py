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
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class TabularDDPG:
    """
    Tabular Deep Deterministic Policy Gradient.

    Uses Q-tables instead of neural networks for both actor and critic.
    Suitable for environments with discrete or discretized state/action spaces.
    """

    def __init__(
        self,
        state_bins,
        action_bins,
        action_low,
        action_high,
        gamma=0.99,
        tau=0.005,
        actor_lr=0.1,
        critic_lr=0.1,
        buffer_size=10000,
        batch_size=64,
        exploration_noise=0.1
    ):
        """
        Initialize the Tabular DDPG agent.

        Args:
            state_bins: Number of bins for discretizing each state dimension
            action_bins: Number of bins for discretizing each action dimension
            action_low: Lower bound of action space (array)
            action_high: Upper bound of action space (array)
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            actor_lr: Learning rate for actor (policy) updates
            critic_lr: Learning rate for critic (Q-function) updates
            buffer_size: Size of replay buffer
            batch_size: Batch size for learning
            exploration_noise: Standard deviation of exploration noise
        """
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.action_low = np.array(action_low)
        self.action_high = np.array(action_high)
        self.action_dim = len(action_low)

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise

        # Critic Q-tables: Q(s, a)
        self.q_table = defaultdict(lambda: 0.0)
        self.q_table_target = defaultdict(lambda: 0.0)

        # Actor policy table: maps state -> continuous action
        self.policy_table = defaultdict(self._random_action)
        self.policy_table_target = defaultdict(self._random_action)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Action space discretization
        self.action_values = self._create_action_discretization()

    def _random_action(self):
        """Generate a random action within bounds."""
        return np.random.uniform(self.action_low, self.action_high)

    def _create_action_discretization(self):
        """Create discrete action values for each dimension."""
        action_values = []
        for i in range(self.action_dim):
            values = np.linspace(
                self.action_low[i],
                self.action_high[i],
                self.action_bins
            )
            action_values.append(values)
        return action_values

    def _discretize_state(self, state):
        """Convert continuous state to discrete tuple."""
        if isinstance(state, (int, float)):
            state = [state]
        return tuple(int(s * self.state_bins) % self.state_bins for s in state)

    def _discretize_action(self, action):
        """Convert continuous action to discrete tuple."""
        if isinstance(action, (int, float)):
            action = [action]
        discrete = []
        for i, a in enumerate(action):
            # Find closest discrete action
            idx = np.argmin(np.abs(self.action_values[i] - a))
            discrete.append(idx)
        return tuple(discrete)

    def _continuous_action(self, discrete_action):
        """Convert discrete action tuple to continuous action."""
        continuous = []
        for i, idx in enumerate(discrete_action):
            continuous.append(self.action_values[i][idx])
        return np.array(continuous)

    def select_action(self, state, add_noise=True):
        """
        Select an action using the current policy.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            Continuous action array
        """
        discrete_state = self._discretize_state(state)
        action = self.policy_table[discrete_state].copy()

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, self.action_low, self.action_high)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Perform one update step of the DDPG algorithm.

        Returns:
            Dictionary containing critic_loss and actor_loss (or None if buffer too small)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        critic_losses = []
        actor_improvements = []

        for state, action, reward, next_state, done in batch:
            discrete_state = self._discretize_state(state)
            discrete_action = self._discretize_action(action)
            discrete_next_state = self._discretize_state(next_state)

            # Critic update (Q-learning style)
            # Get target action from target policy
            target_action = self.policy_table_target[discrete_next_state]
            discrete_target_action = self._discretize_action(target_action)

            # Compute target Q-value
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * self.q_table_target[
                    (discrete_next_state, discrete_target_action)
                ]

            # Update critic
            current_q = self.q_table[(discrete_state, discrete_action)]
            critic_loss = (target_q - current_q) ** 2
            critic_losses.append(critic_loss)

            # TD update for Q-table
            self.q_table[(discrete_state, discrete_action)] += self.critic_lr * (
                target_q - current_q
            )

            # Actor update (policy improvement)
            # Find best action by evaluating Q for all discrete actions
            best_action = None
            best_q = float('-inf')

            # Grid search over discrete actions
            for action_indices in self._action_grid():
                q_val = self.q_table[(discrete_state, action_indices)]
                if q_val > best_q:
                    best_q = q_val
                    best_action = action_indices

            if best_action is not None:
                # Soft policy update toward best action
                current_policy = self.policy_table[discrete_state]
                best_continuous = self._continuous_action(best_action)
                self.policy_table[discrete_state] = (
                    (1 - self.actor_lr) * current_policy +
                    self.actor_lr * best_continuous
                )
                actor_improvements.append(best_q - current_q)

        # Soft update target networks
        self._soft_update_targets()

        return {
            'critic_loss': np.mean(critic_losses),
            'actor_improvement': np.mean(actor_improvements) if actor_improvements else 0.0
        }

    def _action_grid(self):
        """Generate all discrete action combinations."""
        if self.action_dim == 1:
            for i in range(self.action_bins):
                yield (i,)
        else:
            # For higher dimensions, use a subset to avoid explosion
            # Sample random action combinations
            for _ in range(min(self.action_bins ** self.action_dim, 100)):
                yield tuple(random.randint(0, self.action_bins - 1)
                           for _ in range(self.action_dim))

    def _soft_update_targets(self):
        """Soft update target tables using Polyak averaging."""
        # Update target Q-table
        for key in self.q_table:
            self.q_table_target[key] = (
                self.tau * self.q_table[key] +
                (1 - self.tau) * self.q_table_target[key]
            )

        # Update target policy table
        for key in self.policy_table:
            self.policy_table_target[key] = (
                self.tau * self.policy_table[key] +
                (1 - self.tau) * self.policy_table_target[key]
            )

    def train(self, env, num_episodes=1000, max_steps=200, verbose=True):
        """
        Train the agent on an environment.

        Args:
            env: Gym-like environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress

        Returns:
            List of episode rewards
        """
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            episode_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, add_noise=True)

                result = env.step(action)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, _ = result

                self.store_transition(state, action, reward, next_state, done)
                self.update()

                episode_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)

            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward (last 100): {avg_reward:.2f}")

        return episode_rewards

    def save(self, filepath):
        """Save the agent's Q-tables and policy tables."""
        data = {
            'q_table': dict(self.q_table),
            'q_table_target': dict(self.q_table_target),
            'policy_table': {k: v.tolist() for k, v in self.policy_table.items()},
            'policy_table_target': {k: v.tolist() for k, v in self.policy_table_target.items()},
        }
        np.save(filepath, data)

    def load(self, filepath):
        """Load the agent's Q-tables and policy tables."""
        data = np.load(filepath, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: 0.0, data['q_table'])
        self.q_table_target = defaultdict(lambda: 0.0, data['q_table_target'])
        self.policy_table = defaultdict(
            self._random_action,
            {k: np.array(v) for k, v in data['policy_table'].items()}
        )
        self.policy_table_target = defaultdict(
            self._random_action,
            {k: np.array(v) for k, v in data['policy_table_target'].items()}
        )


if __name__ == "__main__":
    # Example usage with a simple custom environment
    class SimpleEnv:
        """Simple 1D continuous control environment for testing."""

        def __init__(self):
            self.state = 0.0
            self.goal = 0.5

        def reset(self):
            self.state = np.random.uniform(-1, 1)
            return np.array([self.state])

        def step(self, action):
            action = np.clip(action, -1, 1)[0]
            self.state = np.clip(self.state + action * 0.1, -1, 1)

            distance = abs(self.state - self.goal)
            reward = -distance
            done = distance < 0.05

            return np.array([self.state]), reward, done, False, {}

    # Create and train agent
    env = SimpleEnv()
    agent = TabularDDPG(
        state_bins=20,
        action_bins=10,
        action_low=[-1.0],
        action_high=[1.0],
        gamma=0.99,
        tau=0.01,
        actor_lr=0.1,
        critic_lr=0.1,
        exploration_noise=0.2
    )

    print("Training Tabular DDPG...")
    rewards = agent.train(env, num_episodes=500, max_steps=100)
    print(f"Final average reward: {np.mean(rewards[-50:]):.2f}")
