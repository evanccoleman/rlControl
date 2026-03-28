# optimizer.py

# numpy, gymnasium, and Monitor
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

# handle parsing command line args
import sys

# get other classes and custom functions
from config import read_command
from trialevalcallback import TrialEvalCallback
from params_to_sample import sample_ppo_params,\
                             sample_ddpg_params,\
                             sample_td3_params,\
                             sample_sac_params,\
                             sample_rppo_params
from write_tuned_output import save_tuned_params

# hyperparameter tuner
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

# pomdp wrapper
from pomdp_wrapper import POMDPWrapper

def objective(trial: optuna.Trial) -> float:
    """
    The objective function to define how to optimize.
    """
    
    # read in the options from the command line
    # potential errors in option parsing handled in main()
    args = read_command(sys.argv[1:])

    # create environment
    try:
        eval_env = Monitor(gym.make(args.env_type, render_mode=None))

        # debug for statemaskingwrapper
        # print(eval_env.observation_space.shape)

        if args.mask_indices is not None:
            num_masks = args.mask_indices
            eval_env = StateMaskingWrapper(eval_env,
                                           indices_to_mask=np.arange(num_masks),
                                           )
        
        # debug for statemaskingwrapper
        # obs, info = eval_env.reset()
        # print(obs.shape)
        # exit()

    except Exception as e:
        print(f"Failed to create environment {args.env_type}: {e}")
        raise optuna.exceptions.TrialPruned()

    # initialize hyperparameters all agent constructors need
    kwargs = {"policy": "MlpPolicy", "env": eval_env}
   
    # get hyperparameters dependent on agent type
    if args.agent_type == "ppo":
        kwargs.update(sample_ppo_params(trial))

    elif args.agent_type == "ddpg":
        sigma = sample_action_noise_sigma(trial) 
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=sigma*np.ones(n_actions),
                                         )
        kwargs.update({"action_noise": action_noise})
        kwargs.update(sample_ddpg_params(trial))

    elif args.agent_type == "td3":
        sigma = sample_action_noise_sigma(trial)
        n_actions = eval_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=sigma*np.ones(n_actions),
                                         )
        kwargs.update({"action_noise": action_noise})
        kwargs.update(sample_td3_params(trial))

    elif args.agent_type == "sac":
        kwargs.update(sample_sac_params(trial))

    elif args.agent_type == "rppo":
        kwargs.update({"policy": "MlpLstmPolicy"})
        kwargs.update(sample_rppo_params(trial))

    # create agent
    try:
        agent = create_agent(args.agent_type, **kwargs)
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
    args = read_command(sys.argv[1:])
    
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
    save_tuned_params(study,
                      agent_type=args.agent_type,
                      env_type=args.env_type,
                      )

    print("\n\n")

if __name__ == "__main__":

    # run the main program
    main()
