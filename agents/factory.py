# agents.py

import numpy as np
from datetime import datetime as dt

from agents.customddpg import CustomDDPG
from agents.frameddpg import FrameDDPG

from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

import json

def read_params_file(params_file: str = None) -> dict:
    """
    Parses hyperparameters settings from a file.
    """

    with open(params_file) as inFile:
        param_settings = json.load(inFile)

    return param_settings

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

def process_action_noise(params, action_space_shape):
    """
    If params contains a scalar action_noise, converts it to a
    NormalActionNoise object in-place. If action_noise is already
    a NormalActionNoise or is absent, does nothing.
    """
    if "action_noise" in params:
        noise = params["action_noise"]
        if not isinstance(noise, NormalActionNoise):
            n_actions = action_space_shape
            params["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise * np.ones(n_actions),
            )

def create_agent(agent_type: str = None,
                 load_agent: str = None,
                 env=None,
                 params_file: str = None,
                 param_settings: dict = None,
                 seed: int = 0,
                 ):
    """
    Creates a new agent or loads a pre-existing one.

    Params can come from a file (params_file) or a pre-built dict
    (param_settings). Either way, action_noise scalars are converted
    to NormalActionNoise objects and the correct policy string is
    selected automatically.
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

        # get settings from file or pre-built dict
        if param_settings is not None:
            settings = param_settings
        elif params_file is not None:
            settings = read_params_file(params_file)
        else:
            settings = {}

        # convert scalar action_noise to NormalActionNoise if needed
        process_action_noise(settings, env.action_space.shape[-1])

        # create agent with the correct policy string
        if agent_type == "ppo":
            agent = PPO("MlpPolicy", env, seed=seed, **settings)
        elif agent_type == "ddpg":
            agent = DDPG("MlpPolicy", env, seed=seed, **settings)
        elif agent_type == "td3":
            agent = TD3("MlpPolicy", env, seed=seed, **settings)
        elif agent_type == "sac":
            agent = SAC("MlpPolicy", env, seed=seed, **settings)
        elif agent_type == "rppo":
            agent = RecurrentPPO("MlpLstmPolicy", env, seed=seed, **settings)
        elif agent_type == "customddpg":
            agent = CustomDDPG(env, seed=seed, **settings)
        elif agent_type == "frameddpg":
            agent = FrameDDPG(env, seed=seed, **settings)
        else:
            raise Exception(f"Agent {agent_type} is not implemented.")

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
