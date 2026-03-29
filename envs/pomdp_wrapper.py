# pomdp_wrapper.py

"""
POMDP Wrapper for Gymnasium Environments

Updated for compatibility with:
- gymnasium >= 0.29.0
- stable-baselines3 >= 2.0.0
- Python >= 3.9

Original implementation from: https://github.com/LinghengMeng/LSTM-TD3
Paper: "Memory-based Deep Reinforcement Learning for POMDPs" (IROS 2021)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class POMDPWrapper(gym.ObservationWrapper):
    """
    A wrapper that converts fully observable MDP environments into 
    Partially Observable MDPs (POMDPs) by modifying observations.
    
    Supported POMDP types:
        1. remove_velocity: Remove velocity-related observations
        2. flickering: Randomly zero out entire observation with probability flicker_prob
        3. random_noise: Add Gaussian noise N(0, sigma) to each observation element
        4. random_sensor_missing: Zero out individual sensors with probability sensor_miss_prob
        5. remove_velocity_and_flickering: Combination of 1 and 2
        6. remove_velocity_and_random_noise: Combination of 1 and 3
        7. remove_velocity_and_random_sensor_missing: Combination of 1 and 4
        8. flickering_and_random_noise: Combination of 2 and 3
        9. random_noise_and_random_sensor_missing: Combination of 3 and 4
        10. random_sensor_missing_and_random_noise: Combination of 4 and 3 (different order)
    """
    
    # Mapping from old environment names to new ones (prefer v5 when available)
    ENV_NAME_MAPPING = {
        # Classic Control (old -> new)
        "Pendulum-v0": "Pendulum-v1",
        "Acrobot-v1": "Acrobot-v1",
        "MountainCarContinuous-v0": "MountainCarContinuous-v0",
        
        # MuJoCo (old -> new, v2/v3/v4 -> v5 preferred)
        "HalfCheetah-v2": "HalfCheetah-v5",
        "HalfCheetah-v3": "HalfCheetah-v5",
        "HalfCheetah-v4": "HalfCheetah-v5",
        "Ant-v2": "Ant-v5",
        "Ant-v3": "Ant-v5",
        "Ant-v4": "Ant-v5",
        "Walker2d-v2": "Walker2d-v5",
        "Walker2d-v3": "Walker2d-v5",
        "Walker2d-v4": "Walker2d-v5",
        "Hopper-v2": "Hopper-v5",
        "Hopper-v3": "Hopper-v5",
        "Hopper-v4": "Hopper-v5",
        "InvertedPendulum-v2": "InvertedPendulum-v5",
        "InvertedPendulum-v4": "InvertedPendulum-v5",
        "InvertedDoublePendulum-v2": "InvertedDoublePendulum-v5",
        "InvertedDoublePendulum-v4": "InvertedDoublePendulum-v5",
        "Swimmer-v2": "Swimmer-v5",
        "Swimmer-v3": "Swimmer-v5",
        "Swimmer-v4": "Swimmer-v5",
        "Humanoid-v2": "Humanoid-v5",
        "Humanoid-v3": "Humanoid-v5",
        "Humanoid-v4": "Humanoid-v5",
        "HumanoidStandup-v2": "HumanoidStandup-v5",
        "HumanoidStandup-v4": "HumanoidStandup-v5",
        "Reacher-v2": "Reacher-v5",
        "Reacher-v4": "Reacher-v5",
        "Pusher-v2": "Pusher-v5",
        "Pusher-v4": "Pusher-v5",
        
        # PyBulletGym -> MuJoCo v5 equivalents (PyBullet is deprecated)
        "HalfCheetahPyBulletEnv-v0": "HalfCheetah-v5",
        "HalfCheetahBulletEnv-v0": "HalfCheetah-v5",
        "HalfCheetahMuJoCoEnv-v0": "HalfCheetah-v5",
        "AntPyBulletEnv-v0": "Ant-v5",
        "AntBulletEnv-v0": "Ant-v5",
        "AntMuJoCoEnv-v0": "Ant-v5",
        "Walker2DPyBulletEnv-v0": "Walker2d-v5",
        "Walker2DBulletEnv-v0": "Walker2d-v5",
        "Walker2DMuJoCoEnv-v0": "Walker2d-v5",
        "HopperPyBulletEnv-v0": "Hopper-v5",
        "HopperBulletEnv-v0": "Hopper-v5",
        "HopperMuJoCoEnv-v0": "Hopper-v5",
        "InvertedPendulumPyBulletEnv-v0": "InvertedPendulum-v5",
        "InvertedPendulumBulletEnv-v0": "InvertedPendulum-v5",
        "InvertedPendulumMuJoCoEnv-v0": "InvertedPendulum-v5",
        "InvertedDoublePendulumPyBulletEnv-v0": "InvertedDoublePendulum-v5",
        "InvertedDoublePendulumBulletEnv-v0": "InvertedDoublePendulum-v5",
        "InvertedDoublePendulumMuJoCoEnv-v0": "InvertedDoublePendulum-v5",
        "ReacherPyBulletEnv-v0": "Reacher-v5",
        "ReacherBulletEnv-v0": "Reacher-v5",
    }

    def __init__(self, env_name, pomdp_type='remove_velocity',
                 flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
                 **env_kwargs):
        """
        Initialize the POMDP wrapper.
        
        Args:
            env_name: Name of the gymnasium environment (supports both old and new names)
            pomdp_type: Type of partial observability to apply
            flicker_prob: Probability of zeroing entire observation (for flickering)
            random_noise_sigma: Standard deviation of Gaussian noise (for random_noise)
            random_sensor_missing_prob: Probability of zeroing each sensor (for random_sensor_missing)
            **env_kwargs: Additional arguments passed to gymnasium.make()
        """
        # Map old environment names to new ones
        self.original_env_name = env_name
        actual_env_name = self.ENV_NAME_MAPPING.get(env_name, env_name)
        
        if actual_env_name != env_name:
            print(f"Note: Mapping '{env_name}' to '{actual_env_name}' for compatibility")
        
        # Create the underlying environment
        super().__init__(gym.make(actual_env_name, **env_kwargs))
        
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma
        self.random_sensor_missing_prob = random_sensor_missing_prob
        self.remain_obs_idx = None
        
        # Configure observation space based on POMDP type
        if pomdp_type == 'remove_velocity':
            self.remain_obs_idx, self.observation_space = self._remove_velocity(self.original_env_name)
        elif pomdp_type == 'flickering':
            pass  # Keep original observation space
        elif pomdp_type == 'random_noise':
            pass  # Keep original observation space
        elif pomdp_type == 'random_sensor_missing':
            pass  # Keep original observation space
        elif pomdp_type == 'remove_velocity_and_flickering':
            self.remain_obs_idx, self.observation_space = self._remove_velocity(self.original_env_name)
        elif pomdp_type == 'remove_velocity_and_random_noise':
            self.remain_obs_idx, self.observation_space = self._remove_velocity(self.original_env_name)
        elif pomdp_type == 'remove_velocity_and_random_sensor_missing':
            self.remain_obs_idx, self.observation_space = self._remove_velocity(self.original_env_name)
        elif pomdp_type == 'flickering_and_random_noise':
            pass  # Keep original observation space
        elif pomdp_type == 'random_noise_and_random_sensor_missing':
            pass  # Keep original observation space
        elif pomdp_type == 'random_sensor_missing_and_random_noise':
            pass  # Keep original observation space
        else:
            raise ValueError(f"Unknown pomdp_type: {pomdp_type}")

    def observation(self, obs):
        """
        Transform the observation according to the POMDP type.
        
        Args:
            obs: Original observation from the environment
            
        Returns:
            Modified observation based on POMDP type
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()
        
        # Single source of POMDP
        if self.pomdp_type == 'remove_velocity':
            return obs[self.remain_obs_idx].astype(np.float32)
            
        elif self.pomdp_type == 'flickering':
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(obs.shape, dtype=np.float32)
            else:
                return obs.astype(np.float32)
                
        elif self.pomdp_type == 'random_noise':
            noisy_obs = obs + np.random.normal(0, self.random_noise_sigma, obs.shape)
            return noisy_obs.astype(np.float32)
            
        elif self.pomdp_type == 'random_sensor_missing':
            obs_copy = obs.copy()
            mask = np.random.rand(len(obs_copy)) <= self.random_sensor_missing_prob
            obs_copy[mask] = 0
            return obs_copy.astype(np.float32)
            
        # Multiple sources of POMDP
        elif self.pomdp_type == 'remove_velocity_and_flickering':
            new_obs = obs[self.remain_obs_idx]
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(new_obs.shape, dtype=np.float32)
            else:
                return new_obs.astype(np.float32)
                
        elif self.pomdp_type == 'remove_velocity_and_random_noise':
            new_obs = obs[self.remain_obs_idx]
            noisy_obs = new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)
            return noisy_obs.astype(np.float32)
            
        elif self.pomdp_type == 'remove_velocity_and_random_sensor_missing':
            new_obs = obs[self.remain_obs_idx].copy()
            mask = np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob
            new_obs[mask] = 0
            return new_obs.astype(np.float32)
            
        elif self.pomdp_type == 'flickering_and_random_noise':
            if np.random.rand() <= self.flicker_prob:
                new_obs = np.zeros(obs.shape, dtype=np.float32)
            else:
                new_obs = obs.copy()
            noisy_obs = new_obs + np.random.normal(0, self.random_noise_sigma, new_obs.shape)
            return noisy_obs.astype(np.float32)
            
        elif self.pomdp_type == 'random_noise_and_random_sensor_missing':
            new_obs = obs + np.random.normal(0, self.random_noise_sigma, obs.shape)
            mask = np.random.rand(len(new_obs)) <= self.random_sensor_missing_prob
            new_obs[mask] = 0
            return new_obs.astype(np.float32)
            
        elif self.pomdp_type == 'random_sensor_missing_and_random_noise':
            obs_copy = obs.copy()
            mask = np.random.rand(len(obs_copy)) <= self.random_sensor_missing_prob
            obs_copy[mask] = 0
            noisy_obs = obs_copy + np.random.normal(0, self.random_noise_sigma, obs_copy.shape)
            return noisy_obs.astype(np.float32)
            
        else:
            raise ValueError(f"Unknown pomdp_type: {self.pomdp_type}")

    def _remove_velocity(self, env_name):
        """
        Determine which observation indices to keep (removing velocity-related ones).
        
        Args:
            env_name: Name of the environment (original name for lookup)
            
        Returns:
            Tuple of (remain_obs_idx, new_observation_space)
        """
        # Normalize environment name for lookup
        # Handle both old PyBullet names and new MuJoCo names
        env_name_lower = env_name.lower()
        
        # Classic Control
        if env_name in ["Pendulum-v0", "Pendulum-v1"]:
            remain_obs_idx = np.arange(0, 2)  # cos(theta), sin(theta) - remove angular velocity
            
        elif env_name == "Acrobot-v1":
            remain_obs_idx = list(np.arange(0, 4))  # Keep cos/sin of both joints, remove velocities
            
        elif env_name == "MountainCarContinuous-v0":
            remain_obs_idx = [0]  # Keep position, remove velocity
            
        # MuJoCo environments (v2, v3, v4, v5)
        # HalfCheetah: obs = [rootz, rooty, ...joint angles..., rootx_vel, rooty_vel, ...joint vels...]
        elif "halfcheetah" in env_name_lower:
            # For HalfCheetah-v4: 17 total obs, first 8 are positions, last 9 are velocities
            # Original paper used indices 0-7 for position
            remain_obs_idx = np.arange(0, 8)
            
        # Ant
        elif "ant" in env_name_lower:
            # Ant has complex observation structure
            # Original: positions (0-12) + contact forces (27-110), remove velocities (13-26)
            # For Ant-v4: structure may differ slightly, but we preserve the ratio
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
            
        # Walker2d
        elif "walker2d" in env_name_lower or "walker2D" in env_name_lower:
            remain_obs_idx = np.arange(0, 8)
            
        # Hopper
        elif "hopper" in env_name_lower:
            remain_obs_idx = np.arange(0, 5)
            
        # InvertedPendulum
        elif "invertedpendulum" in env_name_lower and "double" not in env_name_lower:
            remain_obs_idx = np.arange(0, 2)
            
        # InvertedDoublePendulum
        elif "inverteddoublependulum" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
            
        # Swimmer
        elif "swimmer" in env_name_lower:
            remain_obs_idx = np.arange(0, 3)
            
        # Humanoid
        elif "humanoidstandup" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
        elif "humanoid" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
            
        # Reacher
        elif "reacher" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))
            
        # Pusher
        elif "pusher" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            
        # Thrower
        elif "thrower" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            
        # Striker
        elif "striker" in env_name_lower:
            remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            
        else:
            raise ValueError(f'POMDP velocity removal for {env_name} is not defined! '
                           f'Please add the appropriate index mapping.')

        # Ensure indices are within bounds of actual observation space
        actual_obs_dim = self.env.observation_space.shape[0]
        remain_obs_idx = np.array(remain_obs_idx)
        remain_obs_idx = remain_obs_idx[remain_obs_idx < actual_obs_dim]
        
        if len(remain_obs_idx) == 0:
            raise ValueError(f"No valid observation indices for {env_name}. "
                           f"Observation space has {actual_obs_dim} dimensions.")

        # Create new observation space
        obs_low = np.full(len(remain_obs_idx), -np.inf, dtype=np.float32)
        obs_high = np.full(len(remain_obs_idx), np.inf, dtype=np.float32)
        observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        return remain_obs_idx, observation_space


def make_pomdp_env(env_name, pomdp_type='remove_velocity', **kwargs):
    """
    Factory function to create a POMDP-wrapped environment.
    
    This is a convenience function for use with stable-baselines3's
    make_vec_env or similar utilities.
    
    Args:
        env_name: Name of the base environment
        pomdp_type: Type of partial observability
        **kwargs: Additional arguments for POMDPWrapper
        
    Returns:
        A callable that creates the wrapped environment
    """
    def _make():
        return POMDPWrapper(env_name, pomdp_type=pomdp_type, **kwargs)
    return _make


if __name__ == '__main__':
    import warnings
    
    print("Testing POMDP Wrapper with modern Gymnasium")
    print("=" * 60)
    
    # Test environments - using MuJoCo v4 (requires mujoco package)
    test_configs = [
        # Test with new-style names
        ("HalfCheetah-v4", "remove_velocity"),
        ("HalfCheetah-v4", "flickering"),
        ("HalfCheetah-v4", "random_noise"),
        ("HalfCheetah-v4", "random_sensor_missing"),
        ("HalfCheetah-v4", "remove_velocity_and_flickering"),
        
        # Test with old-style names (will be mapped)
        ("HalfCheetahPyBulletEnv-v0", "remove_velocity"),
        ("AntPyBulletEnv-v0", "remove_velocity_and_flickering"),
    ]
    
    # Also test classic control (no MuJoCo required)
    classic_configs = [
        ("Pendulum-v1", "remove_velocity"),
        ("Pendulum-v1", "flickering"),
        ("Acrobot-v1", "random_noise"),
    ]
    
    # Test classic control environments first (always available)
    print("\nTesting Classic Control environments:")
    print("-" * 40)
    for env_name, pomdp_type in classic_configs:
        try:
            env = POMDPWrapper(env_name, pomdp_type)
            obs, info = env.reset()
            print(f"✓ {env_name} ({pomdp_type})")
            print(f"  Action space: {env.action_space}")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Sample obs shape: {obs.shape}")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step test passed")
            env.close()
        except Exception as e:
            print(f"✗ {env_name} ({pomdp_type}): {e}")
    
    # Test MuJoCo environments (require mujoco package)
    print("\nTesting MuJoCo environments:")
    print("-" * 40)
    for env_name, pomdp_type in test_configs:
        try:
            env = POMDPWrapper(env_name, pomdp_type)
            obs, info = env.reset()
            print(f"✓ {env_name} ({pomdp_type})")
            print(f"  Action space: {env.action_space}")
            print(f"  Observation space: {env.observation_space}")
            print(f"  Sample obs shape: {obs.shape}")
            
            # Test step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step test passed")
            env.close()
        except Exception as e:
            if "mujoco" in str(e).lower() or "mjcf" in str(e).lower():
                print(f"⊘ {env_name}: MuJoCo not installed (skipping)")
            else:
                print(f"✗ {env_name} ({pomdp_type}): {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
