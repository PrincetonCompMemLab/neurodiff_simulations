#!/bin/bash
#SBATCH --job-name=schlichting
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:30:00
conda activate leabra
jupyter nbconvert cross_pair_analysis.ipynb --to python
./main --mode=batch --saveDirName=./figs/test/results_--blocked_interleave_flag=Interleave --runs=2 --trncyclog=false --tstcyclog=false                         --blocked_interleave_flag=Interleave  
python Post_analyses.py ./figs/test/results_--blocked_interleave_flag=Interleave cmd
python read_task_parameters_into_csv.py ./figs/test/results_--blocked_interleave_flag=Interleave cmd
python cross_pair_analysis.py ./figs/test cmd