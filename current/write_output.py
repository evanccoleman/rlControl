# output.py

import os
import numpy as np

def write_output(output_filename: str = None,
                 the_dict: dict = None,
                 oned_nparray: np.ndarray = None,
                 twod_nparray: np.ndarray = None,
                 ):
    """
    Writes the given variables to output files in a run directory.

    the_dict: dictionary of program run -> details.txt
    oned_nparray: average of each agent's averages -> cross_agent_avgs.txt
    twod_nparray: each agent's averages -> each_agent_avgs.txt
    """

    # create run directory
    run_dir = f"../outputs/runs/{output_filename}"
    os.makedirs(run_dir, exist_ok=True)

    # write run details
    with open(f"{run_dir}/details.txt", mode="w", encoding="utf-8") as f:
        for key, value in the_dict.items():
            f.write(f"{key} : {value}\n")

    # write cross-agent averages
    with open(f"{run_dir}/cross_agent_avgs.txt", mode="w", encoding="utf-8") as f:
        for element in oned_nparray:
            f.write(f"{element}\n")

    # write each agent's averages
    with open(f"{run_dir}/each_agent_avgs.txt", mode="w", encoding="utf-8") as f:
        for arr in twod_nparray:
            line = " ".join(str(element) for element in arr)
            f.write(f"{line}\n")
