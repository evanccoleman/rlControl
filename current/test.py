# test.py
import random
import array

import numpy as np
from datetime import datetime as dt
import sys
from tqdm import tqdm
import os

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from config import read_command
from agents import create_agent, save_agent
from environments import create_env
from episodes import run_many_episodes, set_seed
from write_output import write_output

def main() -> None:
    """
    Runs test.py.
    """

    args = read_command(sys.argv[1:])

    env = create_env(env_type=args.env_type,
                     quiet=args.quiet,
                     pomdp_type=args.pomdp_type,
                     framestack=args.framestack,
                     )

    print(env.observation_space)
    obs, _ = env.reset()
    print(obs.shape)

    agent = create_agent(agent_type="frameddpg",
                         load_agent=None,
                         env=env,
                         params_file=None,
                         seed=42,
                         )

                         

if __name__ == "__main__":

    """
    Note to self: by defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
