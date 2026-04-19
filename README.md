# rlControl

This repo is part of a project called "Reinforcement Learning: Does Memory Improve AI's Performance?" This research investigated whether memory (framestacking and LSTM) improve a reinforcement learning agent's performance in continuous contorl environments and POMDPs.

## Installation 

First activate a virtual environment of your choice. This project used conda.

```bash
git clone https://github.com/evanccoleman/rlControl.git
cd rlControl
pip install -r requirements.txt
```

## Run

There are several different scripts in this repo, each divided into their own folder. Run `python {script.py} -h` to the instructions for a given script and refer to the section "Project Layout" for what scripts are available.

## What's Implemented

- Various Stable-Baselines3 agents
- Gymnasium environments
- Custom `POMDPWrapper` class
- Custom `CustomDDPG` and `FrameDDPG` agents

## Project Layout

### Factories
- `agents/` — factory to create, save, and load agents
- `envs/` — factory to create envs; POMDP wrapper

### Inputs / Outputs
- `param_files/` — hyperparameters to try when training new agents
- `outputs/` — eval callbacks, run info, saved agents from training, and graphs from graphing
- `find_best_agent/` — run the `find_best_agent.sh` script to inspect multiple saves from a single run and determine which save is best

### Training
- `training/` — run the `walkers_v5.py` script to train an agent in one of the Gymnasium walker environments

### Tuning
- `tuning/` — run the `optimizer.py` hyperparameter script to tune settings for an agent in a Gymnasium walker environment

### Graphing
- `graphing/` — run `graphit.py` to graph agent results according to averages in `outputs/{training/tuning}_runs/`

### Miscellaneous
- `old/` — old agents, walker files, optimizer, and graph scripts from initial research in Aug 2025
- `notes/` — personal notes
