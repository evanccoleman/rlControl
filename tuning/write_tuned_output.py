# write_tuned_output.py

import json
import os
import optuna

def save_tuned_params(study: optuna.Study,
                      agent_type: str = None,
                      env_type: str = None,
                      ):
    """
    Saves tuned parameters to a JSON file in param_files/.
    """

    # name the output file
    agent_str = agent_type.lower()
    output_path = os.path.join(os.path.dirname(__file__),
                               os.pardir,
                               "param_files",
                               f"tuned_{agent_str}_params.json")

    print(f"SAVING TUNED PARAMS TO: {output_path}")

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=4)
        f.write("\n")
