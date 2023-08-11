#!/bin/bash

#SBATCH -J sample
#SBATCH --mem-per-cpu=64G
#SBATCH --time=20-00:00:00
#SBATCH --partition=wsu_gen_gpu.q
#SBATCH --cpus-per-task=5 # maximum 11 for BeoShock
##SBATCH --nodes=4 --ntasks-per-node=1
##SBATCH --tasks=8
#SBATCH --output=sample_out%j.txt
#SBATCH --error=sample_err%j.txt
##SBATCH --mail-type=ALL # NONE, BEGIN, END, FAIL, REQUEUE
##SBATCH --mail-user fxyan@shockers.wichita.edu

# module purge        # clean up loaded modules 
module load Python
module load TensorFlow/1.13.1-foss-2019a-Python-3.7.2
source ~/venv/bin/activate


python3 ~/sample_hpc.py

# srun -n1
