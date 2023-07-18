#!/bin/bash

#SBATCH --job-name=SLI-with-SLURM
#SBATCH --output=SLIWithSlurm.out
#SBATCH --error=SLIWithSlurm.err

# The number of tasks per node should always be one, as we already parallize within the pipeline.
####SBATCH --ntasks-per-node=6
# The number of nodes, which should be <= number of holdouts
# This is generally the same as `NUMBER_OF_SLURM_NODES` from the `run_pipeline.py` script
# but may be much higher when you are running some other layer of parallelization, such
# as when you are running a grid search.
#SBATCH --nodes=1
# RAM to be used, just set a reasonable amount for your task
#SBATCH --mem=64GB
# Computation time to be used, just set a reasonable amount for your task
#SBATCH --time 48:00:00
# Number of processing cores to be used per node, just set a reasonable amount for your task
#SBATCH --cpus-per-task=24
# We want to wait for the script to complete running.
#SBATCH --wait

# GRAPE requires at least version 3.8 of Python
# If you need to load a specific version of Python on your cluster with the module
# system, uncomment and adjust the following line as appropriate.
#module load python37

#conda init bash
#conda activate py38

# Before running this script, create a virtual environment called 'venv'
# use pip to install grape and pandas in venv
# The following command makes the packages
# available to the runSLi.py script
#source ./venv/bin/activate

echo "running Sli"
python runSli.py 



