#!/bin/bash
#SBATCH --job-name=schlichting
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=2:30:00
#  SBATCH --mail-type=ALL
#SBATCH --mail-user=alexn@uni.minerva.edu
source activate leabra
./schlichting --mode=batch --saveDirName=/scratch/qanguyen/schlichting/alex_mar_23_lrate_interleave/results_--LRateOverAll=2 --runs=50 --trncyclog=false --tstcyclog=false                         --LRateOverAll=2  
python Post_analyses.py /scratch/qanguyen/schlichting/alex_mar_23_lrate_interleave/results_--LRateOverAll=2 cmd
python read_task_parameters_into_csv.py /scratch/qanguyen/schlichting/alex_mar_23_lrate_interleave/results_--LRateOverAll=2 cmd