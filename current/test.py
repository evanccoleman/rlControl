# test.py

def main() -> None:
    """
    Runs walkers_v5.py.
    """

    # read in the options from the command line
    args = read_command(sys.argv[1:])

    # filename format: {agent_type}_{env_type}_{ispomdp}_{datetime}/...
    # parse: agent_type, env_type, env_type_short, and ispomdp
    if args.load_agent:
        filename = args.load_agent.split("/")[-3]
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
               "discount" : args.discount,
               }

    # prepare array to hold averages of each agent
    num_entries = (args.max_steps - args.num_train) // \
            args.training_interval + 1
    all_agent_avgs = np.zeros([args.num_agent_env_pairs, num_entries],
                              dtype=float,
                              )

if __name__ == "__main__":

    """
    Note to self: by defining a main() for this file,
    other main functions in the same directory
    are isolated from each other, so I can run
    particular .py files when I want.
    """
    main()
