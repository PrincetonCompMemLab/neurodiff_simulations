#!/bin/bash
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=1
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
For more info and how to connect from windows,
   see https://docs.ycrc.yale.edu/clusters-at-yale/guides/jupyter/

MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@spock.princeton.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: spock.princeton.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# uncomment the following two lines to use your conda environment called notebook_env
# module load miniconda
# source activate rtsynth

# DON'T USE ADDRESS BELOW.
# DO USE TOKEN BELOW
jupyter-notebook --no-browser --port=${port} --ip=${node}
