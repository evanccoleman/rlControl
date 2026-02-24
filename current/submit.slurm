#!/bin/bash

# #NECESSARY JOB SPECIFICATIONS  
#SBATCH --job-name=ddpg_mdp
#SBATCH --time=3-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --output=/dev/null
#SBATCH --partition=cpu 
# #SBATCH --gres=gpu:h100:1 

# redirect standard output to ../outputs/jobs/{FILENAME}
LOGFILE="../outputs/jobs/ddpg_mdp_$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOB_ID}.out"
exec > "$LOGFILE" 2>&1

# run the Python script(s)
python walkers_v5.py -a ddpg -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/ddpg_params.json -q
# python walkers_v5.py -a ddpg -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/ddpg_params.json -p remove_velocity -q

# python walkers_v5.py -a ppo -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/ppo_params.json -q
# python walkers_v5.py -a ppo -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/ppo_params.json -p remove_velocity -q

# python walkers_v5.py -a td3 -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/td3_params.json -q
# python walkers_v5.py -a td3 -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/td3_params.json -p remove_velocity -q

# python walkers_v5.py -a sac -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/sac_params.json -q
# python walkers_v5.py -a sac -e HalfCheetah-v5 -n 4 -i 1000 -j 1000 -k 10 -m 1000000 -f ../param_files/sac_params.json -p remove_velocity -q
