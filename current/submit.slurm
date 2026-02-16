#!/bin/bash

# #NECESSARY JOB SPECIFICATIONS  
#SBATCH --job-name=rppo_halfcheetah
#SBATCH --time=2-12:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --output=/dev/null
#SBATCH --partition=cpu 
# #SBATCH --gres=gpu:h100:1 

# redirect standard output to ../outputs/jobs/{FILENAME}
LOGFILE="../outputs/jobs/rppo_halfcheetah_$(date +%Y-%m-%d_%H-%M-%S)_${SLURM_JOB_ID}.out"
exec > "$LOGFILE" 2>&1

# run the Python script(s)
python walkers_v5.py -a rppo -e HalfCheetah-v5 -n 5 -i 50000 -j 10000 -k 10 -m 1000000 -p remove_velocity -sq
