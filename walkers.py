# copy_walkers.py

# good ol' numpy
import numpy as np

# gymnasium
import gymnasium as gym                     

# parser stuff
import argparse 
import sys 

# stable_baselines3 (and contrib) agents and noise
from stable_baselines3 import PPO, DDPG, SAC, TD3 
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

# custom agents
# from custom_ddpg import CustomDDPG, ActionNormalizer

def readCommand(argv) -> list:
    """
    Reads in command line options that set
    the environment and the agent.
    """

    # instructions for how to run copy_walkers.py found using -h
    usage_str = """
    USAGE:      python copy_walkers.py <options>
    EXAMPLES:   (1) python copy_walkers.py
                    - starts walkers with default settings
                (2) python copy_walkers.py -n PPO -k 15 -s
                    - starts walkers by creating new PPO agent,
                      testing it for 15 episodes, and saving it
                (3) python copy_walkers.py -l agents_walkers/ppo_ant_10000.zip \
                        -k 10 -q
                    - runs walkers by loading a PPO training agent
                      and testing its performance without rendering
    """

    # create the argument parser
    parser = argparse.ArgumentParser(usage=usage_str,
                                     description="Change settings for \
                                             walkers from the defaults")

    # options for creating/saving agent and the env
    parser.add_argument("-n", "--new_agent",
                        type=str, default=None,
                        metavar="N", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-l", "--load_agent",
                        type=str, default=None,
                        metavar="L", help="Zip file to load agent from \
                                (default None).")
    parser.add_argument("-s", "--save_agent",
                        action="store_true",
                        help="Whether to save the agent (default None). \
                                If True, a save name is auto-generated \
                                and the save directory is automatically \
                                determined. Looks like \
                                'agents_ant/ppo_ant_10000.zip'.")
    parser.add_argument("--env",
                        type=str, default=None,
                        help="Which environment to put agent in.")

    # options for training and testing
    parser.add_argument("-i", "--numTrain",
                        type=int, default=0,
                        metavar="I", help="The number of steps to \
                                train for (default 0).")
    parser.add_argument("-k", "--numTest",
                        type=int, default=0,
                        metavar="K", help="The number of episodes to \
                                test for (default 0).")
    parser.add_argument("-q", "--quiet",
                        action="store_true",
                        help="Whether to render env (default True).")

    # options for agent hyperparameters
    parser.add_argument("--alpha",
                        type=int, default=0.001,
                        metavar="A", help="The learning rate (default 0.001).")
    parser.add_argument("--gamma",
                        type=int, default=0.99,
                        metavar="G", help="The discount factor (default 0.99).")
    parser.add_argument("--buffer_size",
                        type=int, default=10**6,
                        metavar="BUFFER", help="The size of the experience replay \
                                buffer (default 10^6).")
    parser.add_argument("--batch_size",
                        type=int, default=256,
                        metavar="BATCH", help="The size of minibatches (default 256).")
    parser.add_argument("--policy_delay", # not implemented
                        type=int, default=2,
                        metavar="DELAY", help="How often the policy should be updated \
                        relative to Q-function updates (default 2).")
    parser.add_argument("--target_policy_noise", # not implemented
                        type=int, default=0.2,
                        metavar="NOISE", help="Standard deviation of Guassian noise \
                                added to target policy, the smoothing noise \
                                (default 0.2).")

    # return the parsed arguments
    return parser.parse_args()

def runEpisode(agent, env) -> int:
    """
    Runs a single episode for an agent without LSTM.
    Returns the episode returns.
    """

    obs, info = env.reset() # reset env
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

        # Note: environment wrapper automatically tracks episode returns

    # return episode returns
    return info["episode"]["r"] 

def runEpisodeLSTM(agent, env) -> int:
    """
    Runs a single episode for an agent with LSTM.
    Returns the episode returns.
    """

    obs, info = env.reset() # reset env
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

        # Note: environment wrapper automatically tracks episode returns

    # return episode returns
    return info["episode"]["r"] 

def runManyEpisodes(agent,
                    env,
                    num_episodes: int = 0,
                    agent_type: str = None,
                    ) -> None: 
    """
    Runs all the episodes for testing mode.
    Reports the average returns from testing.
    """

    print(f"\nBEGINNING TESTING FOR {num_episodes} EPISODES...")

    # track episodic returns
    rewards = []

    # run the episodes
    for i in range(1, num_episodes + 1):
        print(f"\nEPISODE {i}...")
        episode_rewards = 0 # initialize and set this in scope

        # decide whether to run LSTM episode
        if agent_type == "rppo":
            episode_rewards = runEpisodeLSTM(agent, env)
        else:
            episode_rewards = runEpisode(agent, env)

        # add episode returns to running list
        rewards.append(episode_rewards)

    # calculate performance
    avg_reward = np.mean(rewards)

    # report performance
    print(f"\nTESTING PERFORMANCE FOR {num_episodes} EPISODES...")
    print(f"avg reward : {avg_reward:.3f}")

def createAgent(new_agent: str = None,
                load_agent: str = None,
                env=None,
                alpha: int = None,
                gamma: int = None,
                buffer_size: int = None,
                batch_size: int = None,
                ) -> tuple: 
    """
    Returns a tuple of (agent, agent_type) where an agent is
    created fresh or a pre-existing one is loaded and 
    the type of agent is stored for program purposes.
    If the specified agent is not defined, an error is
    raised saying so.
    """

    agent = None        # store agent to return here
    agent_type = None   # store type of agent here

    # create a new agent with the given hyperparameters
    if new_agent == "ppo":
        print("\nCREATING NEW PPO AGENT...\n")
        agent = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=alpha,
                    gamma=gamma,
                    batch_size=batch_size,
                    )
        agent_type = "ppo"
    elif new_agent == "ddpg":
        print("\nCREATING NEW DDPG AGENT...\n")
        # noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=0.1 * np.ones(n_actions)
                                         )
        agent = DDPG("MlpPolicy", env, verbose=1,
                     action_noise=action_noise,
                     learning_rate=alpha,
                     gamma=gamma,
                     batch_size=batch_size,
                     buffer_size=buffer_size,
                     )
        agent_type = "ddpg"
    elif new_agent == "td3":
        print("\nCREATING NEW TD3 AGENT...\n")
        # noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=0.1 * np.ones(n_actions)
                                         )
        agent = TD3("MlpPolicy", env, verbose=1,
                    action_noise=action_noise,
                    learning_rate=alpha,
                    gamma=gamma,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    )
        agent_type = "td3"
    elif new_agent == "sac":
        print("\nCREATING NEW SAC AGENT...\n")
        agent = SAC("MlpPolicy", env, verbose=1,
                    learning_rate = alpha,
                    gamma = gamma,
                    batch_size = batch_size,
                    buffer_size = buffer_size,
                    )
        agent_type = "sac"
    elif new_agent == "rppo":
        print("\nCREATING NEW RPPO AGENT...\n")
        agent = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
        agent_type = "rppo"
#    elif new_agent == "customddpg":
#        print("\nCREATING NEW CUSTOMDDPG AGENT...\n")
#        agent = CustomDDPG(env=env,
#                           learning_rate=alpha,
#                           gamma=gamma,
#                           batch_size=batch_size,
#                           buffer_size=buffer_size,
#                           )
#        agent_type = "customddpg"

    # load agent in from zip file as is
    else:

        # determine agent type first
        agent_type = load_agent.split("/")[1].split("_", 1)[0]

        # actually load in agent now
        if agent_type == "ppo":
            print(f"\nLOADING PPO AGENT '{load_agent}'...\n")
            agent = PPO.load(load_agent, env=env)
        elif agent_type == "ddpg":
            print(f"\nLOADING DDPG AGENT '{load_agent}'...\n")
            agent = DDPG.load(load_agent, env=env)
        elif agent_type == "td3":
            print(f"\nLOADING TD3 AGENT '{load_agent}'...\n")
            agent = TD3.load(load_agent, env=env)
        elif agent_type == "sac":
            print(f"\nLOADING SAC AGENT '{load_agent}'...\n")
            agent = SAC.load(load_agent, env=env)
        elif agent_type == "rppo":
            print(f"\nLOADING RPPO AGENT '{load_agent}'...\n")
            agent = RecurrentPPO.load(load_agent, env=env)
        else:
            raise Exception("Agent not implemented.")

    # return tuple
    return agent, agent_type

def saveAgent(agent=None,
              env=None,
              load: str = None,
              agent_type: str = None,
              num_train=None,
              ) -> None:
    """
    Save a loaded agent or a fresh agent and
    auto-generate the save name according to
    a pattern like:
    'agents_ant/ppo_ant_10000.zip'
    """

    # saving a fresh agent
    if load is None:
        # create save name
        the_env = env.split("-")[0]
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
        the_env = env.split("-")[0]
        the_env = the_env.lower()
        save_name = "agents_" + the_env + "/" + \
                agent_type + "_" + the_env + "_" + str(new_num_train) + ".zip"

        print(f"\nSAVING AGENT TO '{save_name}'...")
        agent.save(save_name)

def main() -> None:
    """
    Runs copy_walkers.py
    """

    # read in the options from the command line
    args = readCommand(sys.argv[1:])

    # user must specify an agent
    if (args.new_agent is None) and (args.load_agent is None):
        raise Exception("Must specify an agent to create or load.")
    
    # user cannot specify more than one agent
    if args.new_agent and args.load_agent:
        raise Exception("Can only run program with one agent.")

    # user must specify an environment
    if args.env is None:
        raise Exception("Must specify an environment to create.")

    # create the environment
    print(f"\n\nCREATING ENVIRONMENT IN {args.env}...")
    env = None
    if args.quiet:
        env = gym.make(args.env) # no rendering
    else:
        env = gym.make(args.env, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env) # track returns

    # create new agent and remember agent type
    agent, agent_type = createAgent(new_agent=args.new_agent,
                                    load_agent=args.load_agent,
                                    env=env,
                                    alpha=args.alpha,
                                    gamma=args.gamma,
                                    buffer_size=args.buffer_size,
                                    batch_size=args.batch_size
                                    )

#    # add action normalizer wrapper to env if agent is custom ddpg
#    if agent_type == "customddpg":
#        env = ActionNormalizer(env)

    # train agent
    if args.numTrain > 0:
        print(f"\nTRAINING AGENT FOR AT LEAST {args.numTrain} STEPS...")
        agent.learn(total_timesteps=args.numTrain,
                    log_interval=5,
                    progress_bar=True
                    )
   
    # save agent
    if args.save_agent:
        print(f"\nSAVING AGENT...")
        saveAgent(agent=agent,
                  env=args.env,
                  load=args.load_agent,
                  agent_type=agent_type,
                  num_train=args.numTrain
                  )

    # test agent
    if args.numTest > 0:
        print(f"\nTESTING AGENT FOR {args.numTest} EPISODES...")
        runManyEpisodes(agent,
                        env,
                        num_episodes=args.numTest,
                        agent_type=agent_type
                        ) 

    print("\nCLOSING WALKERS...\n\n")
    env.close()


if __name__ == "__main__":
    """
    By defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
