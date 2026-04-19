# rlControl

Research code investigating whether memory (LSTM) improves reinforcement learning agent performance in continuous control environments and POMDPs.

## Research Questions

1. Does memory (LSTM) improve agent performance in continuous control environments?
2. Does memory (LSTM) improve agent performance in POMDPs?

## What's Implemented

- Various Stable-Baselines3 agents
- Gymnasium environments
- `POMDPWrapper` class
- `CustomDDPG` and `FrameDDPG` agents

## Setup

```bash
pip install -r requirements.txt
```

To regenerate the requirements file from the current environment:

```bash
pip list --format=freeze > requirements.txt
```

## Project Layout

### Factories
- `agents/` — factory to create, save, and load agents
- `envs/` — factory to create envs; POMDP wrapper

### Inputs / Outputs
- `param_files/` — hyperparameters to try when training new agents
- `outputs/` — eval callbacks, run info, saved agents from training, and graphs from graphing
- `find_best_agent/` — script that inspects multiple saves from a single run and determines which save is best

### Training
- `training/` — run the `walkers_v5.py` training script

### Tuning
- `tuning/` — run the `optimizer.py` hyperparameter tuning script

### Graphing
- `graphing/` — run `graphit.py` to graph agent results

### Miscellaneous
- `old/` — old agents, walker files, optimizer, and graph scripts from Aug 2025
- `notes/` — personal notes

## Notes

- Files pulled from the cluster may have a 1-hour time difference.
