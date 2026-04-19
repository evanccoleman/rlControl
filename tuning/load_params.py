# params_to_sample.py

# basic imports
import sys
import os
import json
import optuna

# add project root to path so shared packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# parse parameters and process action noise
from agents.factory import read_params_file, process_action_noise

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
        more_settings.update(sample_ddpg_params(trial))

    elif agent_type == "td3":
        more_settings.update(sample_td3_params(trial))

    elif agent_type == "sac":
        more_settings.update(sample_sac_params(trial))

    elif agent_type == "rppo":
        more_settings.update(sample_rppo_params(trial))

    elif agent_type == "customddpg":
        more_settings.update(sample_customddpg_params(trial))

    elif agent_type == "frameddpg":
        more_settings.update(sample_frameddpg_params(trial))

    # convert scalar action_noise to NormalActionNoise if present
    process_action_noise(more_settings, action_space_shape)

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
    action_noise = trial.suggest_categorical("action_noise",
                                              search_space["action_noise"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "action_noise": action_noise,
            }

def sample_td3_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    search_space = load_search_space("td3")

    learning_rate = trial.suggest_categorical("learning_rate",
                                              search_space["learning_rate"])
    policy_delay = trial.suggest_categorical("policy_delay",
                                              search_space["policy_delay"])
    target_policy_noise = trial.suggest_categorical("target_policy_noise",
                                                     search_space["target_policy_noise"])

    return {"learning_rate": learning_rate,
            "policy_delay": policy_delay,
            "target_policy_noise": target_policy_noise,
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
    ent_coef = trial.suggest_categorical("ent_coef",
                                          search_space["ent_coef"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
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
    policy_kwargs = trial.suggest_categorical("policy_kwargs",
                                               search_space["policy_kwargs"])

    return {"learning_rate": learning_rate,
            "n_steps": n_steps,
            "policy_kwargs": policy_kwargs,
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
    action_noise = trial.suggest_categorical("action_noise",
                                              search_space["action_noise"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "action_noise": action_noise,
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
    action_noise = trial.suggest_categorical("action_noise",
                                              search_space["action_noise"])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "action_noise": action_noise,
            }
