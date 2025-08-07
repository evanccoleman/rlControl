# optimizer.py

# good ol' numpy
import numpy as np

# gymnasium
import gymnasium as gym

# parser
import argparse
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

def readCommand(argv) -> list:
    """
    Reads in command line options that set
    the optimizer.
    """

    # instructions for how to run optimizer.py
    usage_str = """
    USAGE:      python optimizer.py <options>
    EXAMPLES:   (1) python optimizer.py -a ppo -e Ant-v5
                    Runs the optimizer on an ant ppo agent.
    """

    # create argument parser
    parser = argparse.ArgumentParser(usage=usage_str)

    # options for optimizer
    parser.add_argument("-a", "--agent_type",
                        type=str, default=None,
                        metavar="A", help="The type of new agent to create \
                                (default None).")
    parser.add_argument("--env",
                        type=str, default=None,
                        help="Which environment to put agent in.")
    parser.add_argument("-s", "--num_timesteps",
                        type=int, default=10000,
                        metavar="N", help="Number of timesteps to train for.")

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

    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    # Note: since n_env is always 1, the rollout buffer size == n_steps
    
    # check for rollout buffer size and batch size compatibility
    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned()

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "vf_coef":vf_coef,
            "n_steps": n_steps,
            }

def sampleParamsDDPG(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for DDPG agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000, log=True)
    tau = trial.suggest_float("tau", 0.003, 0.5, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "n_steps": n_steps,
            }
   
def sampleParamsTD3(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for TD3 agent.
    """
    
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "n_steps": n_steps,
            }
 
def sampleParamsSAC(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparameters for SAC agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000, log=True)
    tau = trial.suggest_float("tau", 0.003, 0.5, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "tau": tau,
            "ent_coef": ent_coef,
            "n_steps": n_steps,
            }
 
def sampleParamsRPPO(trial: optuna.Trial) -> dict:
    """
    Sample some hyperparamters for RPPO agent.
    """

    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, log=True)
    n_steps = 2 ** trial.suggest_int("n_steps_exponent", 3, 11)

    return {"learning_rate": learning_rate,
            "gamma": gamma,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "vf_coef":vf_coef,
            "n_steps": n_steps,
            }

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
    if args.env is None:
        raise Exception("Must specify an environment to create.")

    # create environment
    eval_env = Monitor(gym.make(args.env))

    # initialize hyperparameters all agent constructors need
    kwargs = {"agent_type": args.agent_type,
              "env": eval_env,
              "policy": "MlpPolicy",
              }

    # get hyperparameters dependent on agent type
    if args.agent_type == "ppo":
        kwargs.update(sampleParamsPPO(trial))

    elif args.agent_type == "ddpg":
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=0.1*np.ones(n_actions),
                                         )
        kwargs.update(action_noise=action_noise)
        kwargs.update(sampleParamsDDPG(trial))

    elif args.agent_type == "td3":
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=0.1*np.ones(n_actions),
                                         )
        kwargs.update(action_noise=action_noise)
        kwargs.update(sampleParamsTD3(trial))

    elif args.agent_type == "sac":
        kwargs.update(sampleParamsSAC(trial))

    if args.agent_type == "rppo":
        kwargs.update(policy="MlpLstmPolicy")
        kwargs.update(sampleParamsRPPO(trial))
    
    # create agent
    agent = createAgent(**kwargs) 

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
        print(e)
        nan_encountered = True
    finally:
        # free memory
        agent.env.close()
        eval_env.close()

    # tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # evaluate performance
    return eval_callback.last_mean_reward

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
    
    Reference:
    https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py#L79
    """

    # set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

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
        study.optimize(objective, n_trials=10000)
    except KeyboardInterrupt:
        pass

    # report best hyperparameters
    trial = study.best_trial
    print(f"\n\n...RESULTS...")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print(f"\tValue: {trial.value}")
    print(f"\tParams:")
    for key, value in trial.params.items():
        print(f"\t\t{key} : {value}")
    print(f"\tUser attrs:")
    for key, value in trial.user_attrs.items():
        print(f"\t\t{key} : {value}")

    print("\n\n")

if __name__ == "__main__":
    main()
