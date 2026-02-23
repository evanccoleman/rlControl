# walkers_v5.py

import random
import array

import numpy as np
from datetime import datetime as dt
import sys
from tqdm import tqdm

from stable_baselines3.common.callbacks import EvalCallback
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
    output_filename = f"{agent_type}_{env_type_short}_{ispomdp}_{current_datetime}"

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
        # EvalCallback -> evaluates agent in separate test env and saves best one

        # seed the eval_env with a seed outside the [0, 100)
        # train/test seed rgn
        eval_env = create_env(env_type=args.env_type,
                              quiet=True,
                              pomdp_type=args.pomdp_type,
                              )
        eval_env = Monitor(eval_env)
        eval_env.reset(seed=seeds[i] + 100)

        # set up paths for callbacks
        saved_agents_path = (f"../outputs/saved_agents/"
                           f"{output_filename}/"
                           f"agent{i}_seed{seeds[i]}/")
        evalcallback_path = (f"../outputs/eval_callbacks/"
                             f"{output_filename}/"
                             f"agent{i}_seed{seeds[i]}")
        best_model_path = (f"../outputs/eval_callbacks/"
                           f"{output_filename}/"
                           f"agent{i}_seed{seeds[i]}/best")

        # eval callback: evaluates and saves best model
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=best_model_path,
                                     log_path=evalcallback_path,
                                     eval_freq=args.training_interval,
                                     n_eval_episodes=args.num_test,
                                     verbose=0
                                     )

        # first round of training
        # already reset env after creation
        agent.learn(total_timesteps=args.num_train,
                    callback=eval_callback)
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
        for j in tqdm(range(1, len(all_agent_avgs[i])),
                      desc=f"Agent {i+1}/{len(seeds)//2}",
                      unit="interval"):

            # round of training
            env.np_random.bit_generator.state = training_rng_state
            env.reset()
            # set reset_num_timesteps to False so that gradients update
            agent.learn(total_timesteps=args.training_interval,
                        callback=eval_callback,
                        reset_num_timesteps=False,
                        )
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
            testing_rng_state = env.np_random.bit_generator.state

            # periodically save rewards and agent (every 10 training intervals)
            # overwrites filename until successfully reached last iteration
            if (j + 1) % 10 == 0:
                final_avgs = np.mean(all_agent_avgs, axis=0)
                write_output(output_filename=output_filename,
                             the_dict=details,
                             oned_nparray=final_avgs,
                             twod_nparray=all_agent_avgs,
                             )
                agent.save(f"{saved_agents_path}/ver_{j}")

        # close envs
        eval_env.close()
        env.close()

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
