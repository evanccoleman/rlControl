# find_best_agent.py

import os
import numpy as np
import argparse
import sys
from argparse import Namespace

def read_command(argv) -> Namespace:
    """
    Reads in instructions to find best agent from a set of saved agents.
    """

    usage_str = """
    USAGE: python find_best_agent.py -x customddpg_halfcheetah_mdp_2026-02-25_20-37-11
    """

    parser = argparse.ArgumentParser(usage=usage_str)

    parser.add_argument("-x", "--agent",
                        type=str, default=None,
                        metavar="X", help="Name of agent folder to \
                                load agent averages from.")

    return parser.parse_args(argv)

# read which agent to evaluate
args = read_command(sys.argv[1:])

# get path to that agent's each_agent_avgs.txt file
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir,
                         os.pardir,
                         'outputs/runs/',
                         args.agent,
                         'each_agent_avgs.txt')

# load the each_agent_avgs.txt file
scores = np.loadtxt(f"{file_path}")

# find the number of evaluations per agent
num_evals = len(scores[0])

# find where save points happened (assume it was every 10th evaluation)
save_indices = np.arange(9, num_evals, 10)

# check only scores at save points for each agent
scores_at_save_points = scores[:, save_indices] # shape (4, len(save_indices))
best_agent_idx, best_agent_score_idx = np.unravel_index(np.argmax(scores_at_save_points),
                                          scores_at_save_points.shape)
best_save_idx = save_indices[best_agent_score_idx]
best_ver = best_save_idx + 1

best_score = scores_at_save_points[best_agent_idx, best_agent_score_idx]

print(f"Best agent: {best_agent_idx}")
print(f"Best ver: ver_{best_ver}.zip")
print(f"Best score: {best_score}")
