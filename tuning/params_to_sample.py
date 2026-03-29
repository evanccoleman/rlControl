# params_to_sample.py

import optuna

def sample_ppo_params(trial: optuna.Trial) -> dict: 
    """
    Sample some hyperparameters for PPO agent.
    """

    learning_rate = trial.suggest_cateogrical("learning_rate",
                                              [0.0001, 0.001, 0.01])
    n_steps = trial.suggest_categorical("n_steps",
                                        [1024, 2048, 4096])
    clip_range = trial.suggest_categorical("clip_range",
                                           [0.1, 0.2, 0.3])

    return {"learning_rate": learning_rate,
            "n_steps": n_steps,
            "clip_range": clip_range,
            }

def sample_ddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for DDPG agent.
    """

    learning_rate = trial.suggest_categorical("learning_rate",
                                              [0.0001, 0.001, 0.01])
    batch_size = trial.suggest_categorical("batch_size",
                                           [50, 100, 150])
    tau = trial.suggest_categorical("tau",
                                    [0.0005, 0.005, 0.05])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
   
def sample_td3_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    learning_rate = trial.suggest_categorical("learning_rate",
                                              [0.0001, 0.001, 0.01])   
    batch_size = trial.suggest_categorical("batch_size",
                                           [50, 100, 150])
    tau = trial.suggest_categorical("tau",
                                    [0.0005, 0.005, 0.05])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
 
def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for SAC agent.
    """

    learning_rate = trial.suggest_categorical("learning_rate",
                                              [0.0001, 0.001, 0.01])
    batch_size = trial.suggest_categorical("batch_size",
                                           [50, 100, 150])
    tau = trial.suggest_categorical("tau",
                                    [0.0005, 0.005, 0.05])

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
 
def sample_rppo_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for RPPO agent.
    """

    learning_rate = trial.suggest_cateogrical("learning_rate",
                                              [0.0001, 0.001, 0.01])
    n_steps = trial.suggest_categorical("n_steps",
                                        [1024, 2048, 4096])
    clip_range = trial.suggest_categorical("clip_range",
                                           [0.1, 0.2, 0.3])

    return {"learning_rate": learning_rate,
            "n_steps": n_steps,
            "clip_range": clip_range,
            }

def sample_action_noise_sigma(trial: optuna.Trial) -> float:
    """
    Sample the sigma for action noise in off-policy algorithms,
    """

    return trial.suggest_categorical("action_noise_sigma",
                                     [0.05, 0.1, 0.15])
