# params_to_sample.py

import optuna

def sample_ppo_params(trial: optuna.Trial) -> dict: 
    """
    Sample some hyperparameters for PPO agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=False)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, log=False)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.999)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)

    # Note: since n_env is always 1, the rollout buffer size == n_steps
    
    # check for rollout buffer size and batch size compatibility
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "n_steps": n_steps,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            }

def sample_ddpg_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for DDPG agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=False)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 50000, 1000000, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    learning_starts = trial.suggest_int("learning_starts", 1000, 10000, log=True)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16])

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            }
   
def sample_td3_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=False)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 50000, 1000000, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    learning_starts = trial.suggest_int("learning_starts", 1000, 10000, log=True)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16])
    policy_delay = trial.suggest_categorical("policy_delay", [1, 2, 3])
    target_policy_noise = trial.suggest_float("target_policy_noise", 0.1, 0.5)
    target_noise_clip = trial.suggest_float("target_noise_clip", 0.3, 1.0)

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "policy_delay": policy_delay,
            "target_policy_noise": target_policy_noise,
            "target_noise_clip": target_noise_clip,
            }
 
def sample_sac_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for SAC agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=False)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 50000, 1000000, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", "auto_0.1"])
    learning_starts = trial.suggest_int("learning_starts", 1000, 10000, log=True)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 4, 8, 16])

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "ent_coef": ent_coef,
            "learning_starts": learning_starts,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            }
 
def sample_rppo_params(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for RPPO agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=False)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, log=False)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.999)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2.0)

    # Note: since n_env is always 1, the rollout buffer size == n_steps
    
    # check for rollout buffer size and batch size compatibility
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "n_steps": n_steps,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            }

def sample_action_noise_sigma(trial: optuna.Trial) -> float:
    """
    Sample the sigma for action noise in off-policy algorithms,
    """

    return trial.suggest_float("action_noise_sigma", 0.01, 0.3, log=False)
