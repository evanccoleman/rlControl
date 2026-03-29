# graphit.py

import json
import os
import matplotlib.pyplot as plt
import argparse
import sys
from argparse import Namespace
import matplotlib.ticker as ticker

def read_command(argv) -> Namespace:
    """
    Reads in graphing instructions from the command line.
    """

    usage_str = """
    USAGE: python graphit.py -x instructions/test_two_ppo.json
    """

    parser = argparse.ArgumentParser(usage=usage_str)

    parser.add_argument("-x", "--instructions",
                        type=str, default=None,
                        metavar="X", help="Name of JSON file to \
                                load graphing instructions from.")

    return parser.parse_args(argv)

args = read_command(sys.argv[1:])

# get current directory
base_dir = os.path.dirname(__file__)

config_path = os.path.join(base_dir, args.instructions)

with open(config_path) as inFile:
    config = json.load(inFile)

# get runs directories to graph
runs_dir = os.path.join(base_dir, os.pardir, 'outputs', 'runs')

# get output/graphs directory
graphs_dir = os.path.join(base_dir, os.pardir, 'outputs', 'graphs')

# set the steps
steps = list(range(config["step_start"],
                   config["step_max"] + config["step_interval"],
                   config["step_interval"]))

# make a plot
plt.figure(figsize=(10, 6))

# fill the plot with cross agent avgs
for label, dirname in config['runs'].items():
    avgs_path = os.path.join(runs_dir, dirname, 'cross_agent_avgs.txt')
    with open(avgs_path) as f:
        values = [float(line.strip()) for line in f if line.strip()]
    plt.plot(steps[:len(values)], values, label=label)

# format the plot
plt.title(config['title'])
plt.xlabel('Number of Steps')
plt.ylabel('Average Rewards')
plt.ylim((-1000, 2500))
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=11, integer=True))
ax.tick_params(axis='x', rotation=45)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, config['output_filename']), dpi=150)
