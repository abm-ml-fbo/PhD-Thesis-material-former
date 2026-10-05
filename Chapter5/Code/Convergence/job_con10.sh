#!/bin/bash
#SBATCH --job-name=Con10
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

eval "$(conda shell.bash hook)"
conda activate abm_env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python convergence_10.py \
    --n-samples 10 \
    --n-reps 150 \
    --n-ticks 12 \
    --n-workers 32 \
    --n-fbo 1000 \
    --output results/

echo "End time: $(date)"