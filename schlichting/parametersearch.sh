#!/bin/bash
#SBATCH --job-name=schlichting
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:30:00
conda activate leabra
jupyter nbconvert cross_pair_analysis.ipynb --to python
./main --mode=batch --saveDirName=./figs/test/results_--LRateOverAll=2 --runs=2 --trncyclog=false --tstcyclog=false                         --LRateOverAll=2  
python Post_analyses.py ./figs/test/results_--LRateOverAll=2 cmd
python read_task_parameters_into_csv.py ./figs/test/results_--LRateOverAll=2 cmd
python boundary_condition_plots.py ./figs/test cmd