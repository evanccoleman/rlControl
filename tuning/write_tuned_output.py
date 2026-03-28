# write_tuned_output.py

import optuna
import pathlib

def save_tuned_params(study: optuna.Study,
                      agent_type: str = None,
                      env_type: str = None,
                      ):
    """
    Saves tuned parameters to an output file.
    """

    # make save file directory if it does not exist
    pathlib.Path("tuned_params/").mkdir(exist_ok=True)
    
    # name the output file
    agent_str = agent_type.lower()
    env_str = env_type.split("-")[0].lower()
    outputFile = f"paramFiles/optim_{agent_str}_{env_str}"
    print(f"THE NAME OF THE FILE IS: {outputFile}")

    with open(outputFile, mode="w", encoding="utf-8") as outFile:
        outFile.write(f"AGENT TYPE : {agent_type}\n")
        outFile.write(f"ENV TYPE : {env_type}\n")
        outFile.write("*****\n")
        outFile.write(f"Number of finished trials : {len(study.trials)}\n")
        outFile.write(f"Value of best trial : {study.best_trial.value}\n")
        outFile.write("*****\n")
        for key, value in study.best_trial.params.items():
            outFile.write(f"{key} : {value}\n")
