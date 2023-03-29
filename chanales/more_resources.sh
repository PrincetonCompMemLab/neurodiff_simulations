#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
#SBATCH --time=2:00:00


# Used if I get "killed" alert. Because only allowed 5G,
#to call this script, in terminal run sbatch more_resources.#!/bin/sh

#to see slurm output as it's happening, do tail -f slurm....
source activate leabra

jupyter nbconvert cross_pair_analysis.ipynb --to python
jupyter nbconvert Post_analyses.ipynb --to python
# python ./Post_analyses.py /scratch/vej/color_diff/2021_09_27_interleaved/results_--HiddNumOverlapUnits\=0/ cmd
python $@
# ./Post_analyses.py /scratch/vej/color_diff/alex_11_10_2021/results_--HiddNumOverlapUnits\=0/ cmd
