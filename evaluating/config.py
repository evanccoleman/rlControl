# config.py

import argparse
from argparse import Namespace

def read_command(argv) -> Namespace:
    """
    Reads in command line options that specify which agent
    to load, which environment to put it in, and how many
    evaluation episodes to run.
    """

    # instructions for how to run eval_walkers.py found using -h
    usage_str = """
    USAGE:      python eval_walkers.py -l {agent_zip} -e {env} <options>
    NOTE:       agent_type is parsed from the loaded agent's
                grandparent directory name (same scheme as
                walkers_v5.py).

    EXAMPLE:    python eval_walkers.py
                    -l {../outputs/saved_agents/filename/agent_seed/ver_N.zip}
                    -e {env_type}
                    -k {num_episodes}
                    -p {pomdp_type}
                    -q (quiet)
                    -v (save video of eval episodes)
                    --trackbodyid {body id for tracking camera}
                    -d (discount)
    """

    # create the argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for loading agent and creating the env
    parser.add_argument("-l", "--load_agent",
                        type=str, required=True,
                        metavar="L", help="Path to the agent zip \
                                file to evaluate.")
    parser.add_argument("-e", "--env_type",
                        type=str, required=True,
                        metavar="E", help="Which environment to \
                                evaluate the agent in.")
    parser.add_argument("-p", "--pomdp_type",
                        type=str, default=None,
                        metavar="P", help="Specifies POMDP to create \
                                (default None). Types include \
                                remove_velocity, flickering, \
                                random_noise, random_sensor_missing, \
                                or some combo (refer to POMDPWrapper() \
                                constructor for more).")

    # options for evaluation
    parser.add_argument("-k", "--num_test",
                        type=int, default=10,
                        metavar="K", help="Number of evaluation \
                                episodes to run (default 10).")

    # other program options
    parser.add_argument("-q", "--quiet",
                        action="store_true",
                        help="Whether to render env (default False, \
                                True when option is present).")
    parser.add_argument("-d", "--discount",
                        action="store_true",
                        help="Whether to return discounted rewards \
                                during evaluation (default False, \
                                True when option is present).")
    parser.add_argument("-v", "--save_video",
                        action="store_true",
                        help="If present, record mp4 videos of \
                                evaluation episodes to \
                                ../outputs/videos/ (default False). \
                                Requires the moviepy package.")
    parser.add_argument("--trackbodyid",
                        type=int, default=1,
                        metavar="T", help="Body id for the tracking \
                                camera used when saving videos \
                                (default 1, which is typically the \
                                torso in MuJoCo walker envs).")

    # return the parsed arguments
    return parser.parse_args(argv)
