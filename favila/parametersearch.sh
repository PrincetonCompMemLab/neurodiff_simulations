#!/bin/bash
#SBATCH --job-name=favila
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:30:00
#  SBATCH --mail-type=ALL
#SBATCH --mail-user=alexn@uni.minerva.edu
source activate leabra
./favila --mode=batch --saveDirName=/scratch/qanguyen/favila/alex_test1_may_25_2022_3/results_--same_diff_flag=Different --runs=50 --trncyclog=false --tstcyclog=false                         --same_diff_flag=Different  
python -u Post_analyses.py /scratch/qanguyen/favila/alex_test1_may_25_2022_3/results_--same_diff_flag=Different cmd
python -u read_task_parameters_into_csv.py /scratch/qanguyen/favila/alex_test1_may_25_2022_3/results_--same_diff_flag=Different cmd