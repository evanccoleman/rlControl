# params_to_sample.py

import optuna

def sample_ppo_params(trial: optuna.Trial) -> dict: 
    """
    Sample some hyperparameters for PPO agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    # Note: since n_env is always 1, the rollout buffer size == n_steps
    
    # check for rollout buffer size and batch size compatibility
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    return {"learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_steps": n_steps,
            }

def sample_ddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for DDPG agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
   
def sample_td3_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
 
def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for SAC agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

    return {"learning_rate": learning_rate,
            "batch_size": batch_size,
            "tau": tau,
            }
 
def sample_rppo_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for RPPO agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    # Note: since n_env is always 1, the rollout buffer size == n_steps
    
    # check for rollout buffer size and batch size compatibility
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    return {"learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_steps": n_steps,
            }

def sample_action_noise_sigma(trial: optuna.Trial) -> float:
    """
    Sample the sigma for action noise in off-policy algorithms,
    """

    return trial.suggest_float("action_noise_sigma", 0.01, 0.3, log=False)
