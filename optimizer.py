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
from trialevalcallback import TrialEvalCallback
from stable_baselines3.common.monitor import Monitor

# hyperparameter tuner
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

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
    Sample some PPO hyperparameters.
    """

    # hyperparamters to tune
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 1, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 1000000, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)

    # other info for this trial to track
    trial.set_user_attr("learning_rate_", learning_rate)
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("batch_size_", batch_size)
    trial.set_user_attr("ent_coef", ent_coef)
    trial.set_user_attr("vf_coef", vf_coef)
    trial.set_user_attr("n_steps_", n_steps)

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

    # create action noise for DDPG and TD3 agents
    n_actions = eval_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1*np.ones(n_actions),
                                     )

    # create agent
    kwargs = {"agent_type": args.agent_type,
              "env": eval_env,
              "policy": "MlpPolicy",
              }
    if args.agent_type == "ddpg" or args.agent_type == "td3":
        kwargs.update(action_noise=action_noise)
    if args.agent_type == "rppo":
        kwargs.update(policy="MlpLstmPolicy")
    kwargs.update(sampleParamsPPO(trial))
    agent = createAgent(**kwargs) 

    # create callback to periodically evaluate and report the performance
    eval_callback = TrialEvalCallback(eval_env,
                                      trial,
                                      deterministic=True,
                                      )

    nan_encountered = False
    try:
        agent.learn(args.num_timesteps, callback=eval_callback)
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

    return eval_callback.last_mean_reward

def main():
    """
    By defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
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
        study.optimize(objective)
    except KeyboardInterrupt:
        # print results so far even if force quit process
        pass

    # report best hyperparameters
    best = study.best_trial
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    print(f"\tValue: {best.value}")
    print(f"\tParams:")
    for key, value in trial.params.items():
        print(f"\t\t{key} : {value}")
    print(f"\tUser attrs:")
    for key, value in trial.user_attrs.items():
        print(f"\t\t{key} : {value}")

if __name__ == "__main__":
    main()
