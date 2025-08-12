# statemaskingwrapper.py

# good ol' numpy
import numpy as np

# gymnasium
import gymnasium as gym
from gymnasium.spaces import Box

class StateMaskingWrapper(gym.ObservationWrapper):
    """
    Wrapper that masks (removes) specific indixes from a
    Box observation space.

    Useful for inducing partial observability in standard
    MDP environments by withholding state information
    from the agent.
    """

    def __init__(self,
                 env: gym.Env,
                 indices_to_mask: list,
                 ):
        """
        Initialize a StateMaskingWrapper.
        """

        super().__init__(env)

        # check if obs space is Box
        if not isinstance(env.observation_space, Box):
            raise TypeError(f"StateMaskingWrapper only supports Box \
                    observation spaces but got \
                    {type(env.observation_space)}")

        # remove duplicate indices and sort biggest to smallest
        self.indices_to_mask = sorted(list(set(indices_to_mask)), reverse=True)
        self.original_obs_space = env.observation_space

        # validate indices
        # assumes the original obs space is a 1D array
        for idx in self.indices_to_mask:
            if not 0 <= idx < self.original_obs_space.shape[0]:
                raise ValueError(f"Index {idx} is out of bounds for \
                        observation space shape \
                        {self.original_obs_space.shape}")

        # create a mask for keeping elements
        self.keep_mask = np.ones(self.original_obs_space.shape,
                                 dtype=bool)
        self.keep_mask[self.indices_to_mask] = False

        # modify the observation space
        new_low = self.original_obs_space.low[self.keep_mask]
        new_high = self.original_obs_space.high[self.keep_mask]
        self.observation_space = Box(low=new_low,
                                     high=new_high,
                                     dtype=self.original_obs_space.dtype,
                                     )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Applies the mask to the observation.
        """
        return obs[self.keep_mask]
