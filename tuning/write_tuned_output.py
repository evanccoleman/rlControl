# write_tuned_output.py

import json
import os
from datetime import datetime
import optuna

def save_tuned_params(study: optuna.Study,
                      agent_type: str = None,
                      param_file: str = None,
                      pomdp_type: str = None,
                      ):
    """
    Saves tuned parameters to a JSON file in param_files/.

    Merges the best swept params back into the full preset defaults
    so the output JSON has all keys (matching ddpg_params.json format).
    """

    # load preset defaults if a param file was used
    full_params = {}
    if param_file is not None:
        with open(param_file) as f:
            full_params = json.load(f)

    # override with the best swept values
    full_params.update(study.best_trial.params)

    # name the output file
    agent_str = agent_type.lower()
    obs_str = "pomdp" if pomdp_type is not None else "mdp"
    output_path = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               "param_files",
                               f"tuned_{agent_str}_{obs_str}.json")

    print(f"SAVING TUNED PARAMS TO:\n {output_path}")

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(full_params, f, indent=4)
        f.write("\n")

    return output_path


def save_details(args, output_file: str, search_space_file: str):
    """
    Saves the details of an optimizer run to a JSON file in details/.
    """

    obs_str = "pomdp" if args.pomdp_type is not None else "mdp"
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"opt_{args.agent_type}_{args.env_type}_{obs_str}_{now}.json"
    output_path = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               "outputs",
                               "tuning_runs",
                               filename,
                               )



    details = {
        "agent_type": args.agent_type,
        "env_type": args.env_type,
        "pomdp_type": args.pomdp_type,
        "hyperparameters_file": args.hyperparameters_file,
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "num_trials": args.num_trials,
        "output_file": output_file,
        "search_space_file": search_space_file,
    }

    print(f"SAVING RUN DETAILS TO:\n {output_path}")

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(details, f, indent=4)
        f.write("\n")
