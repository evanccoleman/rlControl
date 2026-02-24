import json
import os
import matplotlib.pyplot as plt
import argparse
import sys
from argparse import Namespace

def read_command(argv) -> Namespace:
    """
    Reads in graphing instructions from the command line.
    """

    usage_str = """
    USAGE: python graphit.py test_two_ppo.json
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

# get config instructions for graphing
config_path = os.path.join(base_dir,
                           os.pardir,
                           'graph_instructions',
                           args.instructions)
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
    plt.plot(steps[:len(values)], values, marker='o', label=label)

# format the plot
plt.title(config['title'])
plt.xlabel('Number of Steps')
plt.ylabel('Average Rewards')
labels = [str(s) if i % 5 == 0 or i == len(steps) - 1 else '' for i, s in enumerate(steps)]
major = [i for i, l in enumerate(labels) if l]
minor = [i for i, l in enumerate(labels) if not l]
ax = plt.gca()
ax.set_xticks([steps[i] for i in major])
ax.set_xticklabels([labels[i] for i in major], rotation=300)
ax.set_xticks([steps[i] for i in minor], minor=True)
ax.tick_params(axis='x', which='major', length=8)
ax.tick_params(axis='x', which='minor', length=4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, config['output_filename']), dpi=150)
