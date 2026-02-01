# output.py

import numpy as np

def write_output(output_filename: str = None,
                 the_dict: dict = None,
                 oned_nparray: np.ndarray = None,
                 twod_nparray: np.ndarray = None,
                 ):
    """
    Writes the given variables to an output file.
    """

    output_path = "runs/" + output_filename + ".txt"
    with open(output_path, mode="w", encoding="utf-8") as out_file:

        for key, value in the_dict.items():
            out_file.write(str(key) + " : " + str(value) + "\n")
        out_file.write("*****\n")

        for element in oned_nparray:
            out_file.write(str(element) + " ")
        out_file.write("\n*****\n")

        for arr in twod_nparray:
            for element in arr:
                out_file.write(str(element) + " ")
            out_file.write("\n")
        out_file.write("*****")
