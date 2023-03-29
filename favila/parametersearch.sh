#!/bin/bash
#SBATCH --job-name=favila
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:30:00
conda activate leabra
jupyter nbconvert cross_pair_analysis.ipynb --to python
./main --mode=batch --saveDirName=./figs/test/results_--same_diff_flag=Different --runs=2 --trncyclog=false --tstcyclog=false                         --same_diff_flag=Different  
python -u Post_analyses.py ./figs/test/results_--same_diff_flag=Different cmd
python -u read_task_parameters_into_csv.py ./figs/test/results_--same_diff_flag=Different cmd
python cross_pair_analysis.py ./figs/test cmd