# agents.py

import numpy as np
from datetime import datetime as dt

from customddpg import CustomDDPG
from frameddpg impot FrameDDPG

from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

from config import read_params_file

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
        elif agent_type == "customddpg":
            agent = CustomDDPG.load(load_agent, env=env, seed=seed)
        elif agent_type == "frameddpg":
            agent = FrameDDPG.load(load_agent, env=env, seed=seed)
        else:
            raise Exception(f"Agent {agent_type} not implemented.")

    else:

        # get settings if creating new agent
        param_settings = {}
        if params_file is not None:
            param_settings = read_params_file(params_file)

        # create a new agent with the given hyperparameters
        if agent_type == "ppo":
            # n_steps = 2 ** param_settings["n_steps_exponent"]
            # del param_settings["n_steps_exponent"]
            agent = PPO("MlpPolicy",
                        env,
                        seed=seed,
                        **param_settings,
                        )

        elif agent_type == "ddpg":
            # noise objects for DDPG
            std = param_settings["action_noise"] # just sigma
            del param_settings["action_noise"]
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=std*np.ones(n_actions),
                                             )
            agent = DDPG("MlpPolicy",
                         env,
                         action_noise=action_noise,# noise obj
                         seed=seed,
                         **param_settings,
                         )

        elif agent_type == "td3":
            # noise objects for DDPG
            std = param_settings["action_noise"] # just sigma
            del param_settings["action_noise"]
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=std*np.ones(n_actions),
                                             )
            agent = TD3("MlpPolicy",
                        env,
                        action_noise=action_noise, # noise obj
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

        elif agent_type == "customddpg":
            # noise objects for DDPG
            std = param_settings["action_noise"] # just sigma
            del param_settings["action_noise"]
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=std*np.ones(n_actions),
                                             )
            agent = CustomDDPG(env,
                               action_noise=action_noise, # noise obj
                               seed=seed,
                               **param_settings,
                               )
        elif agent_type == "customddpg":
            # noise objects for DDPG
            std = param_settings["action_noise"] # just sigma
            del param_settings["action_noise"]
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=std*np.ones(n_actions),
                                             )
            agent = FrameDDPG(env,
                              action_noise=action_noise, # noise obj
                              seed=seed,
                              **param_settings,
                              )


    return agent

def save_agent(agent=None,
               env_type_short: str = None,
               load: str = None,
               agent_type: str = None,
               ispomdp: str = None,
               num_train=None,
               ) -> None:
    """
    Saves a new agent or a loaded agent.

    The agent is saved under a name like:
    '../outputs/saved_agents/ppo_ant_mdp_10000_2026-01-31_14-30-22.zip'

    This function is no longer used. Stablebaselines3 save() used instead.
    """

    current_datetime = dt.now().strftime("%Y-%m-%d_%H-%M-%S")

    # if loading, add old steps to new steps
    if load is not None:
        # filename format: {agent_type}_{env_type}_{ispomdp}_{steps}_{datetime}.zip
        old_num_train = int(load.split("/")[-1].split("_")[3])
        num_train = num_train + old_num_train

    save_name = r"../outputs/saved_agents/" + \
            agent_type + \
            "_" + env_type_short + \
            "_" + ispomdp + \
            "_" + str(num_train) + \
            "_" + current_datetime + \
            ".zip"

    print(f"\nSAVING AGENT TO '{save_name}'...")
    agent.save(save_name)
