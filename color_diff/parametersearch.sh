#!/bin/bash
#SBATCH --job-name=color_diff
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:50:00
source activate leabra
./main --mode=batch --saveDirName=./figs/test3/results_--HiddNumOverlapUnits=5 --runs=2 --trncyclog=false --tstcyclog=false                         --HiddNumOverlapUnits=5  
python Post_analyses.py 2 ./figs/test3/results_--HiddNumOverlapUnits=5 cmd
python read_task_parameters_into_csv.py ./figs/test3/results_--HiddNumOverlapUnits=5 cmd
python cross_pair_analysis.py ./figs/test3 cmd