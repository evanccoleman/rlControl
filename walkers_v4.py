# walkers_v4.py

# random module and PyTorch for seeding
import random
import torch
import array

# good ol' numpy
import numpy as np

# gymnasium
import gymnasium as gym                     
from gymnasium.spaces import Box

# parser stuff
import argparse 
from argparse import Namespace
import sys 

# stable_baselines3 (and contrib) agents 
from stable_baselines3 import PPO, DDPG, SAC, TD3 

# stable_baselines3 (and contrib) noise
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

# stable_baselines3 logger
from stable_baselines3.common.logger import configure

# POMDPWrapper
from pomdp_wrapper import POMDPWrapper

# custom agents
# from custom_ddpg import CustomDDPG, ActionNormalizer

MAX_STEPS_TO_TRAIN = 10_000

def read_command(argv) -> Namespace:
    """
    Reads in command line options that set
    the environment and the agent.
    """

    # instructions for how to run walkers_v1.py found using -h
    usage_str = """
    USAGE:      python walkers_v4.py <options>
    EXAMPLES:   (1) python walkers_v4.py -a ppo -e Ant-v5 -i 10000 \
                    -k 10 -sq -p remove_velocity
                      - trains ppo agent in Ant-v5 for 10000 steps and \
                        tests for 10 episodes 
                      - also saves the agent and runs without rendering
                      - env is pomdp where velocity is removed
                (2) python walkers_v4.py -l agents_walkers/ppo_ant_10000\
                    .zip -e Ant-v5 -k 10 
                      - loads a ppo agent into Ant-v5 and tests for 10 \
                        episodes with rendering 
                      - if pomdp, can only load pomdp agents
    """

    # create the argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for creating/saving agent and the env
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-l", "--load_agent",
                        type=str, default=None,
                        metavar="L", help="Zip file to load agent from \
                                (default None).")
    parser.add_argument("-s", "--save_agent",
                        action="store_true",
                        help="Whether to save the agent (default False, \
                                True when option is present). \
                                Seed is not saved. \
                                If True, a save name is auto-generated \
                                and the save directory is automatically \
                                determined. Looks like \
                                'agents_ant/ppo_ant_10000.zip'.")
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
                                pairs to train and test (default 4). \
                                Seeds are randomly generated [0, 100).")
    parser.add_argument("-i", "--num_train",
                        type=int, default=5_000,
                        metavar="I", help="The number of steps to \
                                train for before testing (default 0).")
    parser.add_argument("-j", "--training_interval",
                        type=int, default=1_000,
                        metavar="J", help="The number of steps to \
                                train for between testing (default 0).")
    parser.add_argument("-k", "--num_test",
                        type=int, default=5,
                        metavar="K", help="The number of episodes to \
                                test for (default 0).")
    parser.add_argument("-q", "--quiet",
                        action="store_true",
                        help="Whether to render env (default False, \
                                True when option is present).")
    parser.add_argument("-d", "--no_discount",
                        action="store_true",
                        help="Whether to return discounted rewards \
                                during testing (default False, True \
                                when option is present).")

    # return the parsed arguments
    return parser.parse_args()

def set_seed(seed):
    """
    Sets the seed for Python packages.

    Ensures consistent seeding.
    """

    # for Python's built-in random module
    random.seed(seed)

    # for numpy
    np.random.seed(seed)

    # for PyTorch and PyTorch with GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def turn_off_training_mode(agent):
    """
    Turns off training mode for an agent.

    Sets the agent's learning rate and action noise
    (if any) to 0.
    """
    agent.learning_rate = 0
    agent.action_noise = None

def restore_training_mode(agent, alpha_and_noise):
    """
    Restores training mode for an agent.

    Restores the agent's learning rate and action noise
    (if any) to the originals.
    """
    agent.learning_rate = alpha_and_noise[0]
    agent.action_noise = alpha_and_noise[1]

def run_episode(agent,
                env: gym.Env,
                no_discount: bool = False,
                ) -> int:
    """
    Executes a single episode for an agent without LSTM.

    Returns the episode rewards.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0 # track episode returns
    total_discount = 1
    is_episode_over = False # loop control variable

    # take actions and update agent until episode termination
    while not is_episode_over:

        # agent chooses action
        action, _states = agent.predict(obs, deterministic=True)

        # environment applies action
        next_obs, reward, terminated, trunc, info = env.step(action)

        # move to the next state 
        obs = next_obs
        is_episode_over = terminated or trunc

        # update episode returns
        episode_rewards += reward * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if no_discount:
        return info["episode"]["r"]
    else:
        return episode_rewards 

def run_episode_lstm(agent,
                     env: gym.Env,
                     no_discount: bool = False,
                     ) -> int:
    """
    Executes a single episode for an agent with LSTM.

    Returns the episode rewards.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0 # track episode returns
    total_discount = 1
    is_episode_over = False # loop control variable
    lstm_states = None # track hidden state of LSTM stuff
    episode_starts = np.array([True]) # helps reset lstm_states

    # take actions and update agent until episode termination
    while not is_episode_over:

        # agent chooses action
        action, lstm_states = agent.predict(obs,
                                            state=lstm_states,
                                            episode_start=episode_starts,
                                            deterministic=True)

        # environment applies action
        next_obs, reward, terminated, trunc, info = env.step(action)

        # move to the next state 
        obs = next_obs
        is_episode_over = terminated or trunc
        episode_starts = np.array([terminated or trunc])

        # update episode returns
        episode_rewards += reward * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if no_discount:
        return info["episode"]["r"]
    else:
        return episode_rewards 

def run_many_episodes(agent,
                      env,
                      num_episodes: int = 0,
                      agent_type: str = None,
                      no_discount: bool = False,
                      ) -> None: 
    """
    Executes episodes in testing mode for an agent.

    Returns the average returns from the testing run.
    """

    # turn off training mode and begin exploitation
    training_settings = (agent.learning_rate, agent.action_noise)
    turn_off_training_mode(agent)

    # track episodic returns
    rewards = []

    # run the episodes
    for i in range(1, num_episodes + 1):
        episode_rewards = 0 # initialize and set this in scope

        # decide whether to run LSTM episode
        if agent_type == "rppo":
            episode_rewards = run_episode_lstm(agent, env,
                                             no_discount=no_discount)
        else:
            episode_rewards = run_episode(agent, env,
                                         no_discount=no_discount)

        # add episode returns to running list
        rewards.append(episode_rewards)

    # restore training mode
    restore_training_mode(agent, training_settings)

    # calculate performance
    avg_reward = np.mean(rewards)

    return avg_reward

def read_params_file(params_file: str = None) -> dict:
    """
    Parses hyperparameters settings form a file.
    """

    param_settings = {}
    count_delimiters = 0
    with open(params_file, mode="r", encoding="utf-8") as inFile:

        # loop through each line of the file
        for line in inFile:

            # skip over info before the first two delimiters "*****"
            if count_delimiters != 2:
                if line.strip() == "*****":
                    count_delimiters += 1

            # start reading parameters
            else:
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

def create_agent(agent_type: str = None,
                 load_agent: str = None,
                 env=None,
                 params_file: str = None,
                 seed: int = 0,
                 ): 
    """
    Creates a new agent or loads a pre-existing one.
    """

    if load_agent:

        # load agent from a zip file
        if agent_type == "ppo":
            agent = PPO.load(load_agent, env=env, seed=seed)
        elif agent_type == "ddpg":
            agent = DDPG.load(load_agent, env=env, seed=seed)
        elif agent_type == "td3":
            agent = TD3.load(load_agent, env=env, seed=seed)
        elif agent_type == "sac":
            agent = SAC.load(load_agent, env=env, seed=seed)
        elif agent_type == "rppo":
            agent = RecurrentPPO.load(load_agent, env=env, seed=seed)
        else:
            raise Exception(f"Agent {agent_type} not implemented.")

    else:

        # get settings if creating new agent
        param_settings = {} 
        if params_file is not None:
            param_settings = read_params_file(params_file)

        # create a new agent with the given hyperparameters
        if agent_type == "ppo":
            n_steps = 2 ** param_settings["n_steps_exponent"]
            del param_settings["n_steps_exponent"]
            agent = PPO("MlpPolicy",
                        env,
                        seed=seed,
                        **param_settings,
                        )

        elif agent_type == "ddpg":
            # noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=0.1*np.ones(n_actions),
                                             )
            agent = DDPG("MlpPolicy",
                         env, 
                         action_noise=action_noise,
                         seed=seed,
                         **param_settings,
                         )

        elif agent_type == "td3":
            # noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=0.1*np.ones(n_actions),
                                             )
            agent = TD3("MlpPolicy",
                        env, 
                        action_noise=action_noise,
                        seed=seed,
                        **param_settings,
                        )

        elif agent_type == "sac":
            agent = SAC("MlpPolicy", 
                        env, 
                        seed=seed,
                        **param_settings,
                        )

        elif agent_type == "rppo":
            agent = RecurrentPPO("MlpLstmPolicy",
                                 env, 
                                 seed=seed,
                                 **param_settings,
                                 )
            
#       elif agent_type == "customddpg":
#           agent = CustomDDPG(env=env)
#           agent_type = "customddpg"

    return agent 

def save_agent(agent=None,
               env_type: str = None,
               load: str = None,
               agent_type: str = None,
               num_train=None,
               ) -> None:
    """
    Saves a new agent or a loaded agent.

    The agent is saved under a name according to
    a pattern like: 'agents_ant/ppo_ant_10000.zip'
    """

    # saving a fresh agent
    if load is None:
        # create save name
        the_env = env_type.split("-")[0]
        the_env = the_env.lower()
        save_name = "agents_" + the_env + "/" + \
                agent_type + "_" + the_env + "_" + str(num_train) + ".zip" 

        print(f"\nSAVING AGENT TO '{save_name}'...")
        agent.save(save_name)

    # saving a loaded agent
    else:
        # update number of steps trained in save name
        old_num_train = int(load.split("/")[1].split("_")[2].split(".")[0])
        new_num_train = num_train + old_num_train

        # create save name
        the_env = env_type.split("-")[0]
        the_env = the_env.lower()
        save_name = "agents_" + the_env + "/" + \
                agent_type + "_" + the_env + "_" + str(new_num_train) + ".zip"

        print(f"\nSAVING AGENT TO '{save_name}'...")
        agent.save(save_name)

def create_env(env_type: str,
               quiet: bool,
               pomdp_type: str,
               ):
    """
    Creates a Gymnasium environment.

    Can make the environment partially observable
    using the POMDPWrapper class.
    """

    # decide if environment is POMDP and/or is rendered
    if pomdp_type != None:
        if quiet:
            env = POMDPWrapper(env_type,
                               pomdp_type=pomdp_type,
                               render_mode=None,
                               )
        else:
            env = POMDPWrapper(env_type,
                               pomdp_type=pomdp_type,
                               render_mode="human",
                               )
    else:
        if quiet:
            env = gym.make(env_type,
                           render_mode=None,
                           )
        else:
            env = gym.make(env_type,
                           render_mode="human",
                           )

    # track non-discounted returns automatically
    env = gym.wrappers.RecordEpisodeStatistics(env) 

    return env

def write_output(output_filename: str = None,
                 the_dict: dict = None,
                 oned_nparray: np.ndarray = None,
                 twod_nparray: np.ndarray = None,
                 ):
    """
    Writes the given variables to an output file.
    """

    output_path = "avgs/" + output_filename
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

def main() -> None:
    """
    Runs walkers_v4.py.
    """

    # read in the options from the command line
    args = read_command(sys.argv[1:])

    # convert agent_type to be lowercase
    if args.load_agent:
        agent_type = args.load_agent.split("/")[1].split("_", 1)[0]
    else:
        agent_type = args.agent_type.lower()

    # randomly generate seeds
    # first half for training, second half for testing
    seeds = array.array("i",
                        random.sample(range(100),
                                      k=2*args.num_agent_env_pairs),
                        )

    details = {"seeds" : seeds,
               "agent_type" : args.agent_type,
               "load_agent" : args.load_agent,
               "save_agent" : args.save_agent,
               "env_type" : args.env_type,
               "pomdp_type" : args.pomdp_type,
               "hyperparameters_file" : args.hyperparameters_file,
               "num_agent_env_pairs" : args.num_agent_env_pairs,
               "num_train" : args.num_train,
               "training_interval" : args.training_interval,
               "num_test" : args.num_test,
               "quiet" : args.quiet,
               "no_discount" : args.no_discount,
               }

    # prepare array to hold averages of each agent
    num_entries = (MAX_STEPS_TO_TRAIN - args.num_train) // \
            args.training_interval + 1
    all_agent_avgs = np.zeros([args.num_agent_env_pairs, num_entries],
                              dtype=float,
                              )
    
    # train and test agents one at a time
    for i in range(len(seeds) // 2):

        # create and seed packages, environment, and agent
        set_seed(seeds[i])
        env = create_env(env_type=args.env_type,
                         quiet=args.quiet,
                         pomdp_type=args.pomdp_type,
                         )
        env.reset(seed=seeds[i])
        agent = create_agent(agent_type=agent_type,
                             load_agent=args.load_agent,
                             env=env,
                             seed=seeds[i],
                             params_file=args.hyperparameters_file,
                             )

        # change where rollout output from stable_baselines3 logger goes
        agent.set_logger(configure("", []))

#    # add action normalizer wrapper to env if agent is custom ddpg
#    if agent_type == "customddpg":
#        env = ActionNormalizer(env)

        # first round of training
        # already reset env after creation
        agent.learn(total_timesteps=args.num_train)
        training_rng_state = env.np_random.bit_generator.state

        # first round of testing
        env.reset(seed=seeds[i + len(seeds) // 2])
        ep_eval_avg = run_many_episodes(agent,
                                        env,
                                        num_episodes=args.num_test,
                                        agent_type=agent_type,
                                        no_discount=args.no_discount,
                                        )
        j = 0
        all_agent_avgs[i][j] = ep_eval_avg
        j += 1
        testing_rng_state = env.np_random.bit_generator.state

        # continue to alternate between training and testing
        while j < len(all_agent_avgs[i]):

            # round of training
            env.np_random.bit_generator.state = training_rng_state
            env.reset()
            agent.learn(total_timesteps=args.training_interval)
            training_rng_state = env.np_random.bit_generator.state

            # round of testing
            env.np_random.bit_generator.state = testing_rng_state
            # run_many_episodes() handles resetting
            ep_eval_avg = run_many_episodes(agent,
                                            env,
                                            num_episodes=args.num_test,
                                            agent_type=agent_type,
                                            no_discount=args.no_discount,
                                            )
            all_agent_avgs[i][j] = ep_eval_avg
            j += 1
            testing_rng_state = env.np_random.bit_generator.state

       # close env
        env.close()

    # calculate average of performance across randomly generated seeds
    final_avgs = np.mean(all_agent_avgs, axis=0)

    # write to output file
    write_output(output_filename="ppo_ant",
                 the_dict=details,
                 oned_nparray=final_avgs,
                 twod_nparray=all_agent_avgs,
                 )

if __name__ == "__main__":

    """
    Note to self: by defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
