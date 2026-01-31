# agents.py

import numpy as np

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
