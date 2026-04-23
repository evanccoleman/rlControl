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

There are several different scripts in this repo. Be sure to navigate to the same directory as the script before running it. Run the help command to learn how to use it. Refer to the "Project Layout" section for an overview of scripts and directories. Below is a list of this project's scripts:
- Training: `python walkers_v5.py -h`
- Training: `python resave_without_buffer.py -h`
- Evaluating: `python eval_walkers.py -h`
- Tuning: `python optimizer.py -h`
- Find Best Agent: `python find_best_agent.sh -h`
- Graphing: `python graphit.py -h` and `python view_eval_callback.py -h`

## What's Implemented

- Stable-Baselines3 agents: DDPG, PPO, RPPO, SAC, and TD3
- Gymnasium environments: Ant-v5, HalfCheetah-v5, and Hopper-v5
- Custom `POMDPWrapper` class
- Custom agents: `CustomDDPG` and `FrameDDPG` 

## Project Layout

### Factories
- `agents/` — factory to create, save, and load agents; CustomDDPG and FrameDDPG implementations
- `envs/` — factory to create envs, POMDP wrapper

### Inputs / Outputs
- `param_files/` — hyperparameters to try when training new agents
- `outputs/` — eval callbacks, run info, saved agents from training, and graphs from graphing; not tracked by git because it constantly gets written to
- `saved_outputs/` - copy of `outputs/` as of 4/19/2026, tracked by git
- `find_best_agent/` — run the `find_best_agent.sh` script to inspect multiple saves from a single run and determine which save is best

### Training
- `training/` — run the `walkers_v5.py` script to train an agent in one of the Gymnasium walker environments, run the `resave_without_buffer.py` script to create a save of CustomDDPG or FrameDDPG without the memory-intensive filled replay buffer

### Evaluating
- `evaluating/` — run the `eval_walkers.py` script to load a saved agent and run it for a specified number of episodes, optionally saving mp4 videos of each episode to `outputs/videos/`

### Tuning
- `tuning/` — run the `optimizer.py` hyperparameter script to tune settings for an agent in a Gymnasium walker environment

### Graphing
- `graphing/` — run `graphit.py` to graph agent results according to averages in `outputs/training_runs/`
- `graphing\` - run `view_eval_callback.py` to view an eval callback's data

### Miscellaneous
- `old/` — old agents, walker files, optimizer, and graph scripts from initial research in Aug 2025
- `notes/` — personal notes
