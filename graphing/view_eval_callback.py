# view_eval_callback.py

import argparse
from argparse import Namespace
import sys
import numpy as np
import os

def read_command(argv) -> Namespace:
    """
    Prints results from an EvalCallback.
    """

    usage_str = """
    USAGE: python view_callback.py -x ppo_hopper_mdp_2026-03-28_19-42-15/agent{i}_seed{i}
    """
    parser = argparse.ArgumentParser(usage=usage_str)

    parser.add_argument("-x", "--instructions",
                        type=str, default=None,
                        metavar="X", help="Path to npz file to \
                                load evaluation callback results from.")

    return parser.parse_args(argv)

# read command-line args
args = read_command(sys.argv[1:])

# build path from graphing/ to outputs/eval_callbacks/
base_dir = os.path.dirname(__file__)
npz_path = os.path.join(base_dir,
                        os.pardir,
                        "outputs",
                        "eval_callbacks",
                        args.instructions,
                        "evaluations.npz"
                        )

# load the data
data = np.load(npz_path)

# print the results out
print("timesteps:", data["timesteps"])
print("results:", data["results"])
print("ep_lengths:", data["ep_lengths"])
