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
    USAGE:      python optimizer.py <options>
    EXAMPLES:   (1) python optimizer.py -a ppo -env Ant-v5
                    - Runs the optimizer on an ant ppo agent
                      where timesteps per trial is 10000 (default)
                      and the number of trials to optimize for is
                      10000 (default).
    """

    # create argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for optimizer
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-env", "--env_type",
                        type=str, default=None,
                        help="Which environment to put agent in (default None).")
    parser.add_argument("--num_timesteps",
                        type=int, default=10000,
                        metavar="N", help="Number of timesteps to train for \
                                in each trial (default 10000).")
    parser.add_argument("--num_trials",
                        type=int, default=10000,
                        metavar="N", help="Number of trials to optimize \
                                for (default 10000).")

    # return the parsed args
    return parser.parse_args()
