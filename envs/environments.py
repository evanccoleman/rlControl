# environments.py

import gymnasium as gym

from envs.pomdp_wrapper import POMDPWrapper

def create_env(env_type: str = None,
               quiet: bool = False,
               pomdp_type: str = None,
               ):
    """
    Creates a Gymnasium environment.

    Can make the environment partially observable
    using the POMDPWrapper class.
    """

    # make pomdp env
    if pomdp_type != None:
        if quiet:
            env = POMDPWrapper(env_type,
                               pomdp_type=pomdp_type,
                               render_mode=None,
                               )
        else:
            env = POMDPWrapper(env_type,
                               pomdp_type=pomdp_type,
                               render_mode="human",
                               )
    # make mdp env
    else:
        if quiet:
            env = gym.make(env_type,
                           render_mode=None,
                           )
        else:
            env = gym.make(env_type,
                           render_mode="human",
                           )

    # track non-discounted returns automatically
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
