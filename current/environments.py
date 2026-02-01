# environments.py

import gymnasium as gym

from pomdp_wrapper import POMDPWrapper

def create_env(env_type: str,
               quiet: bool,
               pomdp_type: str,
               ):
    """
    Creates a Gymnasium environment.

    Can make the environment partially observable
    using the POMDPWrapper class.
    """

    # decide if environment is POMDP and/or is rendered
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
