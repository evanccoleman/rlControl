# config.py

import argparse
from argparse import Namespace

def read_command(argv) -> Namespace:
    """
    Reads in command line options that set
    the environment and the agent.

    Note to self: program defaults for testing are...
    -n 4 -i 1000 -j 10000 -k 10 -m 1000000
    """

    # instructions for how to run walkers_v5.py found using -h
    usage_str = """
    USAGE:      python walkers_v5.py -a ppo -e Hopper-v5 <options>
    NOTE:       POMDP envs should only have POMDP agents.
                Basically, you can either use -a or -l, not both.
    EXAMPLES:   (1) python walkers_v5.py
                        -a {agent_type}
                        -e {env_type}
                        -n {num_agent_env_pairs}
                        -i {num_training}
                        -j {training_interval}
                        -k {num_testing}
                        -m {max_steps}
                        -p {pomdp_type}
                        -f {../param_files/hyperparameters_file}
                        -q (quiet)
                (2) python walkers_v5.py
                        -l {../saved_agents/filename}
                        -e {env_type}
                        -n {num_agent_env_pairs}
                        -i {num_training}
                        -j {training_interval}
                        -k {num_testing}
                        -m {max_steps}
                        -p {pomdp_type}
                        -q (quiet)
    """

    # create the argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for creating agent and the env
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-l", "--load_agent",
                        type=str, default=None,
                        metavar="L", help="Zip file to load agent from \
                                (default None).")
    parser.add_argument("-e", "--env_type",
                        type=str, default=None,
                        help="Which environment to put agent in.")
    parser.add_argument("-p", "--pomdp_type",
                        type=str, default=None,
                        help="Specifies POMDP to create (default None). \
                                Types include remove_velocity,\
                                flickering, random_noise, \
                                random_sensor_missing, or some combo \
                                (refer to POMDPWrapper() constructor \
                                for more)")
    parser.add_argument("-f", "--hyperparameters_file",
                        type=str, default=None,
                        metavar="F", help="Name of the file to load from \
                                for a new agent's hyperparameter settings \
                                (default None).")

    # options for training and testing
    parser.add_argument("-n", "--num_agent_env_pairs",
                        type=int, default=2,
                        metavar="N", help="The number of agent/env \
                                pairs to train and test (default 2). \
                                Seeds are randomly generated [0, 100).")
    parser.add_argument("-i", "--num_train",
                        type=int, default=1_000,
                        metavar="I", help="The number of steps to \
                                train for before testing (default 1_000).")
    parser.add_argument("-j", "--training_interval",
                        type=int, default=1_000,
                        metavar="J", help="The number of steps to \
                                train for between testing (default 1_000).")
    parser.add_argument("-k", "--num_test",
                        type=int, default=5,
                        metavar="K", help="The number of episodes to \
                                test for (default 5).")
    parser.add_argument("-m", "--max_steps",
                        type=int, default=5_000,
                        metavar="M", help="The max number of steps to \
                                train for, then program ends (default \
                                5_000")

    # other program options
    parser.add_argument("-q", "--quiet",
                        action="store_true",
                        help="Whether to render env (default False, \
                                True when option is present).")
    parser.add_argument("-d", "--discount",
                        action="store_true",
                        help="Whether to return discounted rewards \
                                during testing (default False, True \
                                when option is present).")

    # return the parsed arguments
    return parser.parse_args()

def read_params_file(params_file: str = None) -> dict:
    """
    Parses hyperparameters settings form a file.
    """

    param_settings = {}
    count_delimiters = 0
    with open(params_file, mode="r", encoding="utf-8") as inFile:

        # loop through each line of the file
        for line in inFile:

            line = line.strip()
            param = line.split(" : ")

            # type cast numbers
            if "auto" in param[1]:
                # is specifically an sac agent param
                # it stays a string
                pass
            elif "." in param[1]:
                param[1] = float(param[1])
            else:
                param[1] = int(param[1])
            param_settings.update({param[0]: param[1]})

    return param_settings
