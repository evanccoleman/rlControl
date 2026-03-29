# config.py

import argparse
from argparse import Namespace
import sys

def read_command(argv) -> Namespace:
    """
    Reads in command line options that set
    the optimizer.
    """

    # instructions for how to run optimizer.py
    usage_str = """
    USAGE:      python optimizer.py -a {agent} -e {env} <options>
    EXAMPLES:   (1) python optimizer.py -a ppo -env Ant-v5
                    - Runs the optimizer on an ant ppo agent
                      where timesteps per trial is 10000 (default)
                      and the number of trials to optimize for is
                      10000 (default).
    """

    # create argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options agent and env creation
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-e", "--env_type",
                        type=str, default=None,
                        help="Which environment to put agent in (default None).")
    parser.add_argument("-p", "--pomdp_type",
                        type=str, default=None,
                        help="Specifies POMDP to create (default None). \
                                Types include remove_velocity, \
                                flickering, random_noise, \
                                random_sensor_missing, or some combo \
                                (refer to POMDPWrapper() constructor \
                                for more).")
    parser.add_argument("-f", "--hyperparameters_file",
                        type=str, default=None,
                        metavar="F", help="Name of the file to load from \
                                for a new agent's hyperparameter settings \
                                (default None).")

    # seed for reproducibility
    parser.add_argument("-s", "--seed",
                        type=int, default=10,
                        metavar="S", help="Seed for training RNG \
                                (default 10). Eval seed is seed + 1.")

    # options for optimization duration
    parser.add_argument("--num_timesteps",
                        type=int, default=10000,
                        metavar="N", help="Number of timesteps to train for \
                                in each trial (default 10000).")
    parser.add_argument("--num_trials",
                        type=int, default=27,
                        metavar="N", help="Number of trials to optimize \
                                for (default 27).")

    # return the parsed args
    return parser.parse_args()
