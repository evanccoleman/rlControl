# walkers_v5.py

import random
import array

import numpy as np
from datetime import datetime as dt
import sys

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from config import read_command
from agents import create_agent, save_agent
from environments import create_env
from episodes import run_many_episodes, set_seed
from write_output import write_output

def main() -> None:
    """
    Runs walkers_v5.py.
    """

    # read in the options from the command line
    args = read_command(sys.argv[1:])

    # filename format: {agent_type}_{env_type}_{ispomdp}_{steps}_{datetime}.zip
    # parse: agent_type, env_type, env_type_short, and ispomdp
    if args.load_agent:
        filename = args.load_agent.split("/")[-1]
        parts = filename.split("_")
        agent_type = parts[0]
        env_type_short = parts[1]
        ispomdp = parts[2]
    else:
        agent_type = args.agent_type.lower()
        env_type_short = args.env_type.split("-")[0].lower()
        ispomdp = "pomdp" if args.pomdp_type is not None else "mdp"

    # create output filename
    current_datetime = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{agent_type}_{env_type_short}_{current_datetime}"

    # randomly generate seeds
    # first half for training, second half for testing
    seeds = array.array("i",
                        random.sample(range(100),
                                      k=2*args.num_agent_env_pairs),
                        )

    # note the train seeds and test seeds
    train_seeds = seeds[:len(seeds)//2]
    test_seeds = seeds[len(seeds)//2:]

    # note the details of this program run
    details = {"current_datetime": current_datetime,
               "seeds" : seeds,
               "train_seeds" : train_seeds,
               "test_seeds" : test_seeds,
               "agent_type" : args.agent_type,
               "load_agent" : args.load_agent,
               "env_type" : args.env_type,
               "pomdp_type" : args.pomdp_type,
               "hyperparameters_file" : args.hyperparameters_file,
               "num_agent_env_pairs" : args.num_agent_env_pairs,
               "num_train" : args.num_train,
               "training_interval" : args.training_interval,
               "num_test" : args.num_test,
               "max_steps" : args.max_steps,
               "quiet" : args.quiet,
               "no_discount" : args.no_discount,
               "save_agent" : args.save_agent,
               }

    # prepare array to hold averages of each agent
    num_entries = (args.max_steps - args.num_train) // \
            args.training_interval + 1
    all_agent_avgs = np.zeros([args.num_agent_env_pairs, num_entries],
                              dtype=float,
                              )

    # train and test agents one at a time
    for i in range(len(seeds) // 2):

        # create and seed packages, environment, and agent
        set_seed(seeds[i])
        env = create_env(env_type=args.env_type,
                         quiet=args.quiet,
                         pomdp_type=args.pomdp_type,
                         )
        env.reset(seed=seeds[i])
        agent = create_agent(agent_type=agent_type,
                             load_agent=args.load_agent,
                             env=env,
                             seed=seeds[i],
                             params_file=args.hyperparameters_file,
                             )

        # change where rollout output from stable_baselines3 logger goes
        agent.set_logger(configure(None, []))

        # create eval environment and callback for logging
        # seed the eval_env with a seed outside the [0, 100)
        # train/test seed rgn
        eval_env = create_env(env_type=args.env_type,
                              quiet=True,
                              pomdp_type=args.pomdp_type,
                              )
        eval_env = Monitor(eval_env)
        eval_env.reset(seed=seeds[i] + 100)

        # set up paths for callbacks
        callback_path = (f"../outputs/callbacks/{output_filename}/"
                         f"agent{i}_seed{seeds[i]}")
        checkpoint_path = (f"../outputs/checkpoints/{output_filename}/"
                           f"agent{i}_seed{seeds[i]}")
        best_model_path = (f"../outputs/checkpoints/{output_filename}/"
                           f"agent{i}_seed{seeds[i]}/best")

        # checkpoint callback: saves model periodically
        # save every 10 training intervals
        checkpoint_callback = CheckpointCallback(
            save_freq=args.training_interval * 10,
            save_path=checkpoint_path,
            name_prefix="model",
            verbose=0
        )

        # eval callback: evaluates and saves best model
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=best_model_path,
                                     log_path=callback_path,
                                     eval_freq=args.training_interval,
                                     n_eval_episodes=args.num_test,
                                     verbose=0
                                     )

        # combine callbacks
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        # first round of training
        # already reset env after creation
        agent.learn(total_timesteps=args.num_train,
                    callback=callbacks)
        training_rng_state = env.np_random.bit_generator.state

        # first round of testing
        env.reset(seed=seeds[i + len(seeds) // 2])
        ep_eval_avg = run_many_episodes(agent,
                                        env,
                                        num_episodes=args.num_test,
                                        agent_type=agent_type,
                                        no_discount=args.no_discount,
                                        )
        j = 0
        all_agent_avgs[i][j] = ep_eval_avg
        j += 1
        testing_rng_state = env.np_random.bit_generator.state

        # continue to alternate between training and testing
        while j < len(all_agent_avgs[i]):

            # round of training
            env.np_random.bit_generator.state = training_rng_state
            env.reset()
            agent.learn(total_timesteps=args.training_interval,
                        callback=callbacks)
            training_rng_state = env.np_random.bit_generator.state

            # round of testing
            env.np_random.bit_generator.state = testing_rng_state
            # run_many_episodes() handles resetting
            ep_eval_avg = run_many_episodes(agent,
                                            env,
                                            num_episodes=args.num_test,
                                            agent_type=agent_type,
                                            no_discount=args.no_discount,
                                            )
            all_agent_avgs[i][j] = ep_eval_avg
            j += 1
            testing_rng_state = env.np_random.bit_generator.state

            # periodically save output (every 10 intervals, matching checkpoints)
            if j % 10 == 0:
                final_avgs = np.mean(all_agent_avgs, axis=0)
                write_output(output_filename=output_filename,
                             the_dict=details,
                             oned_nparray=final_avgs,
                             twod_nparray=all_agent_avgs,
                             )

       # close envs
        eval_env.close()
        env.close()

        # save agents if applicable
        if args.save_agent:
            save_agent(agent=agent,
                       env_type_short=env_type_short,
                       load=args.load_agent,
                       agent_type=agent_type,
                       ispomdp=ispomdp,
                       num_train=args.num_train,
                       )

    # last call to write to output file
    final_avgs = np.mean(all_agent_avgs, axis=0)
    write_output(output_filename=output_filename,
                 the_dict=details,
                 oned_nparray=final_avgs,
                 twod_nparray=all_agent_avgs,
                 )

if __name__ == "__main__":

    """
    Note to self: by defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
