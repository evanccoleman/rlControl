# environments.py

import gymnasium as gym

from pomdp_wrapper import POMDPWrapper
from gymnasium.wrappers import FrameStackObservation

def create_env(env_type: str = None,
               quiet: bool = False,
               pomdp_type: str = None,
               framestack: int = 0,
               ):
    """
    Creates a Gymnasium environment.

    Can make the environment partially observable
    using the POMDPWrapper class.

    Can make the environment use FrameStacking with
    FrameStack wrapper from Gymnasium.
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

    # add framestacking
    if framestack > 0:
        print("Framestacking")
        env = FrameStackObservation(env, stack_size=framestack)

    # track non-discounted returns automatically
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env
