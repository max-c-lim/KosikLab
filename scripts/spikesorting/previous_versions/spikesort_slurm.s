#!/bin/bash
#SBATCH --job-name=alessio_spikesort
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maxlim@ucsb.edu
#SBATCH --output=SpikeSorting/data/maxone/200123/2953/alessio/concat_spikesort_slurm_%j.out

python SpikeSorting/scripts/spikesort_drug_experiment_test.py