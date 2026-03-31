# write_tuned_output.py

import json
import os
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
