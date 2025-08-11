# optimizer.py

# good ol' numpy
import numpy as np

# gymnasium
import gymnasium as gym

# parser
import argparse
from argparse import Namespace
import sys

# stable_baselines3 (and contrib) agents and noise
from stable_baselines3 import PPO, DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import RecurrentPPO

# callbacks
from stable_baselines3.common.callbacks import EvalCallback 
from stable_baselines3.common.monitor import Monitor

# hyperparameter tuner
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

# check if directories exist
import pathlib

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(self,
                 eval_env: gym.Env,
                 trial: optuna.Trial,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 deterministic: bool = True,
                 verbose: int = 0,
                 ):
        """
        Initialize a TrialEvalCallBack.
        """

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Every so eval_freq, make a new
        report and prune trial if necessary.
        """

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def readCommand(argv) -> Namespace:
    """
    Reads in command line options that set
    the optimizer.
    """

    # instructions for how to run optimizer.py
    usage_str = """
    USAGE:      python optimizer.py <options>
    EXAMPLES:   (1) python optimizer.py -a ppo -env Ant-v5
                    - Runs the optimizer on an ant ppo agent
                      where timesteps per trial is 10000 (default)
                      and the number of trials to optimize for is
                      10000 (hard-coded).
    """

    # create argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for optimizer
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("-env", "--env_type",
                        type=str, default=None,
                        help="Which environment to put agent in (default None).")
    parser.add_argument("--num_timesteps",
                        type=int, default=10000,
                        metavar="N", help="Number of timesteps to train for \
                                in each trial (default 10000).")
    parser.add_argument("--num_trials",
                        type=int, default=10000,
                        metavar="N", help="Number of trials to optimize \
                                for (default 10000).")

    # return the parsed args
    return parser.parse_args()

def createAgent(agent_type, **kwargs):
    """
    Returns a new agent.
    """

    agent = None
    if agent_type == "ppo":
        agent = PPO(**kwargs)
    elif agent_type == "ddpg":
        agent = DDPG(**kwargs)
    elif agent_type == "td3":
        agent = TD3(**kwargs)
    elif agent_type == "sac":
        agent = SAC(**kwargs)
    elif agent_type == "rppo":
        agent = RecurrentPPO(**kwargs)
    else:
        raise Exception(f"Agent {agent_type} is not implemented.")

    # return fresh agent
    return agent

def sampleParamsPPO(trial: optuna.Trial) -> dict: 
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

def sampleParamsDDPG(trial: optuna.Trial) -> dict:
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
   
def sampleParamsTD3(trial: optuna.Trial) -> dict:
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
 
def sampleParamsSAC(trial: optuna.Trial) -> dict:
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
 
def sampleParamsRPPO(trial: optuna.Trial) -> dict:
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

def sampleActionNoiseSigma(trial: optuna.Trial) -> float:
    """
    Sample the sigma for action noise in off-policy algorithms,
    """

    return trial.suggest_float("action_noise_sigma", 0.01, 0.3, log=False)

def objective(trial: optuna.Trial) -> float:
    """
    The objective function to define how to optimize.
    """
    
    # read in the options from the command line
    args = readCommand(sys.argv[1:])

    # user must specify an agent
    if args.agent_type is None:
        raise Exception("Must specify an agent to create.")

    # user must specify an environment
    if args.env_type is None:
        raise Exception("Must specify an environment to create.")

    # create environment
    try:
        eval_env = Monitor(gym.make(args.env_type, render_mode=None))
    except Exception as e:
        print(f"Failed to create environment {args.env_type}: {e}")
        raise optuna.exceptions.TrialPruned()

    # initialize hyperparameters all agent constructors need
    kwargs = {"policy": "MlpPolicy", "env": eval_env}
   
    # get hyperparameters dependent on agent type
    if args.agent_type == "ppo":
        kwargs.update(sampleParamsPPO(trial))

    elif args.agent_type == "ddpg":
        sigma = sampleActionNoiseSigma(trial) 
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=sigma*np.ones(n_actions),
                                         )
        kwargs.update({"action_noise": action_noise})
        kwargs.update(sampleParamsDDPG(trial))

    elif args.agent_type == "td3":
        sigma = sampleActionNoiseSigma(trial)
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=sigma*np.ones(n_actions),
                                         )
        kwargs.update({"action_noise": action_noise})
        kwargs.update(sampleParamsTD3(trial))

    elif args.agent_type == "sac":
        kwargs.update(sampleParamsSAC(trial))

    elif args.agent_type == "rppo":
        kwargs.update({"policy": "MlpLstmPolicy"})
        kwargs.update(sampleParamsRPPO(trial))

    # create agent
    try:
        agent = createAgent(args.agent_type, **kwargs)
    except Exception as e:
        print(f"Failed to create agent {args.agent_type}: {e}")
        raise optuna.exceptions.TrialPruned() 

    # create callback to periodically evaluate and report the performance
    eval_callback = TrialEvalCallback(eval_env,
                                      trial,
                                      deterministic=True,
                                      )
    # train agent
    nan_encountered = False
    try:
        agent.learn(total_timesteps=args.num_timesteps,
                    log_interval=5,
                    progress_bar=True,
                    callback=eval_callback,
                    )
    except AssertionError as e:
        # sometimes, random hyperparams can generate NaN
        print(f"Trial failed with AssertionError:")
        print(e)
        nan_encountered = True
    except ValueError as e:
        # sometimes, invalid hyperparameters cause crashes
        print(f"Trial failed with ValueError:")
        print(e)
        raise optuna.exceptions.TrialPruned()
    finally:
        # free memory
        agent.env.close()
        eval_env.close()

    # tell the optimizer that the trial failed
    if nan_encountered:
        raise optuna.exceptions.TrialPruned()

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # evaluate performance
    return eval_callback.last_mean_reward

def saveParams(study: optuna.Study,
               agent_type: str = None,
               env_type: str = None,
               ):
    """
    Saves the parameters found to an output file.
    """

    # make save file directory if it does not exist
    pathlib.Path("paramFiles/").mkdir(exist_ok=True)
    
    # name the output file
    agent_str = agent_type.lower()
    env_str = env_type.split("-")[0].lower()
    outputFile = f"paramFiles/{agent_str}_{env_str}"
    print(f"THE NAME OF THE FILE IS: {outputFile}")

    with open(outputFile, mode="w", encoding="utf-8") as outFile:
        outFile.write(f"AGENT TYPE : {agent_type}\n")
        outFile.write(f"ENV TYPE : {env_type}\n")
        outFile.write("\n")
        outFile.write(f"Number of finished trials : {len(study.trials)}\n")
        outFile.write(f"Value of best trial : {study.best_trial.value}\n")
        outFile.write("\n")
        outFile.write(f"...PARAMS...\n")
        for key, value in study.best_trial.params.items():
            outFile.write(f"{key} : {value}\n")
    
def main():
    """
    By defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.

    Note:
    - Trial optimization starts at trial 0
    - Number of finished trials includes interrupted one

    Things to tweak:
    - range of values possible for hyperparameters
    - n_trials to optimize for
    - num_timesteps
    
    Reference:
    https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py#L79
    """

    # read in the options from the command line
    args = readCommand(sys.argv[1:])
    
    # user must specify an agent
    if args.agent_type is None:
        raise Exception("Must specify an agent to create.")

    # user must specify an environment
    if args.env_type is None:
        raise Exception("Must specify an environment to create.")

    # create sampler and pruner
    sampler = TPESampler(n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5)

    # create a study to optimize
    study = optuna.create_study(sampler=sampler,
                                pruner=pruner,
                                direction="maximize",
                                )

    # optimize the hyperparameters
    try:
        study.optimize(objective, n_trials=args.num_trials)
    except KeyboardInterrupt:
        pass

    # write results as std out to a file
    print(f"\n\nSAVING RESULTS TO A FILE...")
    saveParams(study,
               agent_type=args.agent_type,
               env_type=args.env_type,
               )

    print("\n\n")

if __name__ == "__main__":

    # run the main program
    main()
