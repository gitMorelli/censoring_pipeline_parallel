#!/bin/bash
#SBATCH --job-name=censoring_parallel_test
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --ntasks=1                    # 1 task per array job
#SBATCH --cpus-per-task=14           # Use 2xnum_workers+2 CPUs per node
#SBATCH --mem=8G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=00:10:00               # Estimated time for 500 images
#SBATCH --partition=shortq
#SBATCH --output=/mnt/beegfs01/scratch/a_morelli/test_parallel_censoring/logs/slurm/job_%A_%a.out
#SBATCH --error=/mnt/beegfs01/scratch/a_morelli/test_parallel_censoring/logs/slurm/job_%A_%a.err

# 1. Load necessary modules (this varies by cluster)
# module load python/3.10
# module load cuda/12.1
#module load nvidia/cuda/12.2.2-535.104.05

# --- Environment Setup ---
# Project Root
PROJECT_ROOT="/home/a_morelli/vscode_projects/censoring_pipeline_parallel"
ENV_PYTHON="/home/a_morelli/.conda/envs/CensoringEnv/bin/python"

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# 1. Move into the project root directory
cd $PROJECT_ROOT

# 2. Run using the -m flag (No .py extension, use dots for path)
$ENV_PYTHON -m src.scripts.censoring_parallel \
    --n_workers 12 \
    --save_debug_times
#$ENV_PYTHON -m src.scripts.censoring_parallel_advanced