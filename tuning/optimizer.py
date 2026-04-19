# optimizer.py

# numpy, gymnasium, and Monitor
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# handle parsing command line args
import sys
import os

# add project root to path so shared packages are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# get other classes and custom functions
from tuning.config import read_command
from tuning.write_tuned_output import save_tuned_params, save_details
from tuning.evaluate_agents import evaluate_agent
from tuning.load_params import load_param_settings, load_search_space

# agent creation
from agents.factory import create_agent

# hyperparameter tuner
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
import random
import torch

# pomdp wrapper
from envs.pomdp_wrapper import POMDPWrapper

def set_seed(seed):
    """
    Seeds Python, NumPy, and PyTorch RNGs for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_env(env_type: str = None,
               pomdp_type: str = None
               ):
    """
    Returns an MDP or POMDP environment.
    """

    # check whether to make POMDP
    if pomdp_type is not None:
        eval_env = POMDPWrapper(env_type,
                                pomdp_type=pomdp_type,
                                render_mode=None,
                                )
    else:
        eval_env = gym.make(env_type,
                            render_mode=None)

    # add Monitor wrapper
    eval_env = Monitor(eval_env)
    return eval_env
 

def make_objective(args):
    """
    Returns an objective function that uses args from a closure.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        The objective function to define how to optimize.

        Train the agent for num_timesteps. The EvalCallback
        pauses training periodically to evaluate performance.

        Each trial uses the same training seed and eval seed
        so that the only variable is the hyperparameters.
        """

        train_seed = args.seed
        eval_seed = args.seed + 1

        # seed global RNGs for reproducibility
        set_seed(train_seed)

        # create training environment with training seed
        train_env = create_env(args.env_type, args.pomdp_type)
        train_env.reset(seed=train_seed)

        # create eval environment with eval seed
        eval_env = create_env(args.env_type, args.pomdp_type)
        eval_env.reset(seed=eval_seed)

        # get parameter settings for an agent
        is_custom = args.agent_type in ("customddpg", "frameddpg")
        param_settings = load_param_settings(
            agent_type=args.agent_type,
            param_file=args.hyperparameters_file,
            action_space_shape=train_env.action_space.shape[-1],
            trial=trial,
        )

        # create agent with training seed
        agent = create_agent(args.agent_type,
                             env=train_env,
                             param_settings=param_settings,
                             seed=train_seed,
                             )

        # train and evaluate
        try:
            if is_custom:
                # custom agents don't support callbacks,
                # so train in chunks and evaluate periodically
                eval_freq = 10000
                best_mean_reward = -np.inf
                steps_remaining = args.num_timesteps

                while steps_remaining > 0:
                    chunk = min(eval_freq, steps_remaining)
                    agent.learn(total_timesteps=chunk)
                    reward = evaluate_agent(agent, eval_env,
                                            agent_type=args.agent_type)
                    best_mean_reward = max(best_mean_reward, reward)
                    steps_remaining -= chunk

                mean_reward = best_mean_reward
            else:
                eval_callback = EvalCallback(eval_env,
                                             eval_freq=10000,
                                             n_eval_episodes=5,
                                             deterministic=True,
                                             )
                agent.learn(total_timesteps=args.num_timesteps,
                            log_interval=5,
                            progress_bar=True,
                            callback=eval_callback,
                            )
                mean_reward = eval_callback.best_mean_reward
        except (AssertionError, ValueError) as e:
            print(f"Trial failed: {e}")
            raise optuna.exceptions.TrialPruned()
        finally:
            # free memory
            train_env.close()
            eval_env.close()

        return mean_reward

    return objective

def main():
    """
    Do a sweep of a set of pre-defined parameter settings.
    
    Reference:
    https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py#L79
    """

    # read in the options from the command line
    args = read_command(sys.argv[1:])

    # ensure output directory structure exists
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(project_root, "outputs", "tuning_runs"),
                exist_ok=True)
    os.makedirs(os.path.join(project_root, "param_files"), exist_ok=True)

    # load search space and create grid sampler
    search_space = load_search_space(args.agent_type)
    sampler = GridSampler(search_space)

    # create a study to optimize
    study = optuna.create_study(sampler=sampler,
                                direction="maximize",
                                )

    # calculate number of trials
    num_trials = 1
    for i in search_space:
        num_trials *= len(search_space[i])

    # optimize the hyperparameters
    try:
        study.optimize(make_objective(args), n_trials=num_trials)
    except KeyboardInterrupt:
        pass

    # print best results
    print(f"\n\nBEST TRIAL:")
    print(f"  Performance: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    print("\n")

    # write results to a file
    output_file = save_tuned_params(study, args.agent_type, args.env_type, args.hyperparameters_file, args.pomdp_type)

    # save run details
    search_space_file = os.path.join(os.path.dirname(__file__),
                                     "search_spaces",
                                     f"search_space_{args.agent_type}.json")
    save_details(args, output_file, search_space_file)

if __name__ == "__main__":
    # run the main program
    main()
