#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --array=0-20

# SBATCH --mem-per-cpu=2G  # NOTE DO NOT USE THE --mem= OPTION 

# When running a large number of tasks simultaneously, it may be
# necessary to increase the user process limit
source activate leabra

python simulation.py $SLURM_ARRAY_TASK_ID