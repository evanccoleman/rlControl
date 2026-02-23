# view_callback.py

import sys
import numpy as np

"""
USAGE: python view_callback.py ../outputs/eval_callbacks/{output_filename}/ \
        agent{i}_seed{i}/evaluations.npz
"""

data = np.load(sys.argv[1])
print("timesteps:", data["timesteps"])
print("results:", data["results"])
print("ep_lengths:", data["ep_lengths"])
