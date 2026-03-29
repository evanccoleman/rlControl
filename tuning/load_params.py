# params_to_sample.py

# basic imports
import sys
import os
import json
import numpy as np
import optuna

# action noise object
from stable_baselines3.common.noise import NormalActionNoise

# add project root to path so shared packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# parse parameters from file
from training.config import read_params_file

def load_search_space(agent_type: str) -> dict:
    """
    Load the search space for an agent from its JSON file.
    """
    search_space_path = os.path.join(os.path.dirname(__file__),
                                     "search_spaces",
                                     f"search_space_{agent_type}.json")
    with open(search_space_path) as f:
        return json.load(f)

def load_param_settings(agent_type: str = None,
                        param_file: str = None,
                        action_space_shape: int = 0,
                        trial: optuna.Trial = None,
                        ):
    """
    Generate a dictionary of parameter settings for an agent.
    """

    # parse settings from file
    more_settings = read_params_file(param_file)
   
    # sample hyperparameters to tune 
    if agent_type == "ppo":
        more_settings.update(sample_ppo_params(trial))

    elif agent_type == "ddpg":
        if "action_noise" in more_settings:
            sigma = more_settings["action_noise"] # just sigma
            del more_settings["action_noise"]
            n_actions = action_space_shape
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=sigma*np.ones(n_actions))
            more_settings.update({"action_noise": action_noise})
        more_settings.update(sample_ddpg_params(trial))

    elif agent_type == "td3":
        if "action_noise" in more_settings:
            sigma = more_settings["action_noise"] # just sigma
            del more_settings["action_noise"]
            n_actions = action_space_shape
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=sigma*np.ones(n_actions))
            more_settings.update({"action_noise": action_noise})
        more_settings.update(sample_td3_params(trial))

    elif agent_type == "sac":
        more_settings.update(sample_sac_params(trial))

    elif agent_type == "rppo":
        more_settings.update({"policy": "MlpLstmPolicy"})
        more_settings.update(sample_rppo_params(trial))

    elif agent_type == "customddpg":
        if "action_noise" in more_settings:
            sigma = more_settings["action_noise"]
            del more_settings["action_noise"]
            n_actions = action_space_shape
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=sigma*np.ones(n_actions))
            more_settings["action_noise"] = action_noise
        more_settings.update(sample_customddpg_params(trial))

    elif agent_type == "frameddpg":
        if "action_noise" in more_settings:
            sigma = more_settings["action_noise"]
            del more_settings["action_noise"]
            n_actions = action_space_shape
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=sigma*np.ones(n_actions))
            more_settings["action_noise"] = action_noise
        more_settings.update(sample_frameddpg_params(trial))

    return more_settings

def sample_ppo_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for PPO agent.
    """
    search_space = load_search_space("ppo")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    n_steps = trial.suggest_categorical("n_steps",
                                        search_space["n_steps"])
    clip_range = trial.suggest_categorical("clip_range",
                                           search_space["clip_range"])

    return {"learning_rate": learning_rate,
            "n_steps": n_steps,
            "clip_range": clip_range,
            }

def sample_ddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for DDPG agent.
    """
    search_space = load_search_space("ddpg")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    batch_size = trial.suggest_categorical("batch_size",
                                           search_space["batch_size"])
    tau = trial.suggest_categorical("tau",
                                    search_space["tau"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }

def sample_td3_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    search_space = load_search_space("td3")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    batch_size = trial.suggest_categorical("batch_size",
                                           search_space["batch_size"])
    tau = trial.suggest_categorical("tau",
                                    search_space["tau"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }

def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for SAC agent.
    """
    search_space = load_search_space("sac")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    batch_size = trial.suggest_categorical("batch_size",
                                           search_space["batch_size"])
    tau = trial.suggest_categorical("tau",
                                    search_space["tau"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }

def sample_rppo_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for RPPO agent.
    """
    search_space = load_search_space("rppo")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    n_steps = trial.suggest_categorical("n_steps",
                                        search_space["n_steps"])
    clip_range = trial.suggest_categorical("clip_range",
                                           search_space["clip_range"])

    return {"learning_rate": learning_rate,
            "n_steps": n_steps,
            "clip_range": clip_range,
            }

def sample_customddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for CustomDDPG agent.
    """
    search_space = load_search_space("customddpg")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    batch_size = trial.suggest_categorical("batch_size",
                                           search_space["batch_size"])
    tau = trial.suggest_categorical("tau",
                                    search_space["tau"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }

def sample_frameddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for FrameDDPG agent.
    """
    search_space = load_search_space("frameddpg")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    batch_size = trial.suggest_categorical("batch_size",
                                           search_space["batch_size"])
    stack_size = trial.suggest_categorical("stack_size",
                                           search_space["stack_size"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "stack_size": stack_size,
            }
