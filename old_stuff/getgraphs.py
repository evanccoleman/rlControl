# getgraphs.py

# good ol' numpy and plt
import numpy as np
import matplotlib.pyplot as plt

# gymnasium
import gymnasium as gym                     
from gymnasium.spaces import Box

# parser stuff
import argparse 
from argparse import Namespace
import sys 

# stable_baselines3 (and contrib) agents and noise
from stable_baselines3 import PPO, DDPG, SAC, TD3 
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

def getMovingAvgs(arr, window, convolution_mode):
     """
     Compute moving average to smooth noisy data.
     """
     return np.convolve(
         np.array(arr).flatten(),
         np.ones(window),
         mode=convolution_mode
     ) / window

def getPerformancePlots(envs: dict = None,
                        env_type: str = None,
                        roll_length: int = 0,
                        ):
    """
    Creates a plot of agent performance (5 subplots)
    by looping through the envs dictionary.
    
    Rewards for x-many episodes are averaged for each 
    data point on the graph.

    For instance, if k=10000 and roll_length=500,
    then the 500th point on the plot represents the average
    of episodes 1-500.
    """

    # smooth over a x-episode window
    rolling_length = roll_length
    fig, axs = plt.subplots(ncols=5, figsize=(12, 5))

    # create a subplot of agent performance for each env
    i = 0
    for agent_type, env in envs.items():
        axs[i].set_title(f"Episode Rewards ({agent_type})")
        reward_moving_avg = getMovingAvgs(env.return_queue,
                                         rolling_length,
                                         "valid",
                                         )
        axs[i].plot(range(len(reward_moving_avg)), reward_moving_avg)
        axs[i].set_ylabel("Average Reward")
        axs[i].set_xlabel("Episode")
        i += 1

    # save the plot
    plt.suptitle(f"Rewards in {env_type}")
    plt.tight_layout()
    env_str = env_type.split("-")[0].lower()
    plt.savefig(f"graphs/{env_str}_env.png")

def readCommand(argv) -> Namespace:
    """
    Reads in command line options that set
    the environment and the agent.
    """

    # instructions for how to run getgraphs.py found using -h
    usage_str = """
    USAGE:      python getgraphs.py <options>
    EXAMPLES:   (1) python getgraphs.py -l test_agents/ant_env \
            -env Ant-v5 -k 10 -q -roll 10
                    - loads any agents in the ant_env file into \
                            Ant-v5 environments and tests them in \
                            quiet mode for 10 episodes each. The \
                            rolling length gets set to 10.
    """

    # create the argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for creating/saving agent and the env
    parser.add_argument("-l", "--load_agents_file",
                         type=str, default=None,
                         metavar="L", help="File with names of zip \
                                 files to load agents from \
                                 (default None).")
    parser.add_argument("-env", "--env_type",
                         type=str, default=None,
                         help="Which environment to put agent in.")
 
    # options for training and testing
    parser.add_argument("-k", "--num_test",
                        type=int, default=0,
                        metavar="K", help="The number of episodes to \
                                test for (default 0).")
    parser.add_argument("-roll", "--roll_length",
                         type=int, default=0,
                         metavar="R", help="The rolling length in \
                                 the subplot (default 0).")
    parser.add_argument("-q", "--quiet",
                        action="store_true",
                        help="Whether to render env (default False, \
                                True when option is present).")
    parser.add_argument("--no_discount",
                        action="store_true",
                        help="Whether to discount rewards in testing \
                                (default False, yes discounting).")

    # options for agent hyperparameters
    parser.add_argument("-p", "--params_file",
                        type=str, default=None,
                        metavar="P", help="Name of the file to load from for a new \
                                agent's hyperparameter settings")

    # return the parsed arguments
    return parser.parse_args()

def turnOffTrainingMode(agent):
    """
    Sets the agent's learning rate and action noise 
    (if any) to 0.
    """
    agent.learning_rate = 0
    agent.action_noise = None

def restoreTrainingMode(agent, alpha_and_noise):
    """
    Restores the agent's learning rate and action noise
    (if any) to the originals.
    """
    agent.learning_rate = alpha_and_noise[0]
    agent.action_noise = alpha_and_noise[1]

def runEpisode(agent,
               env: gym.Env,
               no_discount: bool = False,
               ) -> int:
    """
    Runs a single episode for an agent without LSTM.
    Returns the episode returns.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0
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

        # update episode rewards
        episode_rewards += reward * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if no_discount:
        return info["episode"]["r"]
    else:
        return episode_rewards

def runEpisodeLSTM(agent,
                   env: gym.Env,
                   no_discount: bool = False,
                   ) -> int:
    """
    Runs a single episode for an agent with LSTM.
    Returns the episode returns.
    """

    obs, info = env.reset() # reset env
    episode_rewards = 0
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

        # update episode rewards
        episode_rewards += rewards * total_discount
        total_discount *= agent.gamma

    # return episode returns
    if no_discount:
        return info["episode"]["r"]
    else:
        return episode_rewards

def runManyEpisodes(agent,
                    env,
                    num_episodes: int = 0,
                    agent_type: str = None,
                    no_discount: bool = False,
                    ) -> None: 
    """
    Runs all the episodes for testing mode.
    Reports the average returns from testing.
    """

    print(f"\nBEGINNING TESTING FOR {num_episodes} EPISODES...")

    # turn off training mode and begin exploitation
    training_settings = (agent.learning_rate, agent.action_noise)
    turnOffTrainingMode(agent)

    # track episodic returns
    rewards = []

    # run the episodes
    for i in range(1, num_episodes + 1):
        print(f"\nEPISODE {i}...")
        episode_rewards = 0 # initialize and set this in scope

        # decide whether to run LSTM episode
        if agent_type == "rppo":
            episode_rewards = runEpisodeLSTM(agent, env, no_discount)
        else:
            episode_rewards = runEpisode(agent, env, no_discount)

        # add episode returns to running list
        rewards.append(episode_rewards)

    # restore training mode
    restoreTrainingMode(agent, training_settings)

    # calculate performance
    avg_reward = np.mean(rewards)

    # report performance
    print(f"\nTESTING PERFORMANCE FOR {num_episodes} EPISODES...")
    print(f"avg reward : {avg_reward:.3f}")

def loadAgent(agent_to_load: str = None,
                env=None,
                ) -> tuple: 
    """
    Loads a pre-trained agent.
    Returns the agent and the type of agent.
    If the specified agent is not defined, an error is
    raised saying so.
    """

    # store agent to return here
    agent = None 

    # determine agent type first
    agent_type = agent_to_load.split("/")[1].split("_", 1)[0]

    # actually load in agent now
    if agent_type == "ppo":
        print(f"\nLOADING PPO AGENT '{agent_to_load}'...\n")
        agent = PPO.load(agent_to_load, env=env)
    elif agent_type == "ddpg":
        print(f"\nLOADING DDPG AGENT '{agent_to_load}'...\n")
        agent = DDPG.load(agent_to_load, env=env)
    elif agent_type == "td3":
        print(f"\nLOADING TD3 AGENT '{agent_to_load}'...\n")
        agent = TD3.load(agent_to_load, env=env)
    elif agent_type == "sac":
        print(f"\nLOADING SAC AGENT '{agent_to_load}'...\n")
        agent = SAC.load(agent_to_load, env=env)
    elif agent_type == "rppo":
        print(f"\nLOADING RPPO AGENT '{agent_to_load}'...\n")
        agent = RecurrentPPO.load(agent_to_load, env=env)
    else:
        raise Exception(f"Agent {agent_type} not implemented.")

    # return tuple
    return agent, agent_type.upper()

def createEnv(env_type: str,
              quiet: bool,
              ):
    """
    Creates a gymnasium env.
    """
    env = None

    # decide whether to render
    if quiet:
        env = gym.make(env_type)
    else:
        env = gym.make(env_type, render_mode="human")
    
    # track non-discounted returns automatically
    env = gym.wrappers.RecordEpisodeStatistics(env) 

    return env


def main() -> None:
    """
    Runs getgraphs.py
    """

    # read in the options from the command line
    args = readCommand(sys.argv[1:])

    # user must specify an agent
    if args.load_agents_file is None:
        raise Exception("Must specify an a file of agents to load.")
    
    # user must specify an environment
    if args.env_type is None:
        raise Exception("Must specify an environment to create.")

    if args.roll_length == 0:
        raise Exception("Must specify a nonzero rolling length.")

    # track five envs in a dictionary
    five_envs = {}

    # read in five agents and train them in the env
    with open(args.load_agents_file, mode="r", encoding="utf-8") as inFile:
        
        # loop through each line of the file
        for line in inFile:

            line = line.strip()

            # create the environment
            print(f"\n\nCREATING ENVIRONMENT IN {args.env_type}...")
            env = createEnv(env_type=args.env_type,
                            quiet=args.quiet,
                            )

            # create new agent and remember agent type
            agent, agent_type = loadAgent(agent_to_load=line,
                                          env=env,
                                          )

            # test agent
            if args.num_test > 0:
                print(f"\nTESTING AGENT FOR {args.num_test} EPISODES...")
                runManyEpisodes(agent,
                                env,
                                num_episodes=args.num_test,
                                agent_type=agent_type,
                                no_discount=args.no_discount,
                                ) 

            # add the env to the five_envs we're tracking
            five_envs.update({agent_type: env})

            print("\nCLOSING WALKERS...\n\n")
            env.close()

    # create a plot of the env performances
    getPerformancePlots(envs=five_envs,
                        env_type=args.env_type,
                        roll_length=args.roll_length,
                        )


if __name__ == "__main__":
    """
    By defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
