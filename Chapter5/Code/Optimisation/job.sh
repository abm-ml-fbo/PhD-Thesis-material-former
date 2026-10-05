#!/bin/bash
#SBATCH --job-name=GA
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

eval "$(conda shell.bash hook)"
conda activate abm_env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python optimisation_run.py "$@"

echo "End time: $(date)"