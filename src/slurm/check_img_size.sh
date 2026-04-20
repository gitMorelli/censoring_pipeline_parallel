#!/bin/bash
#SBATCH --job-name=censoring_parallel_q1
#SBATCH --array=0-29%4              # Divide IDs into chunks; limit the number of parallel arrays to 20
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --cpus-per-task=10          # Use 2xnum_workers+2 CPUs per node
#SBATCH --mem=10G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=00:30:00               # Estimated time for 500 images
#SBATCH --partition=shortq
#SBATCH --output=/home/a_morelli/temporary_data/check_img_size/job_%A_%a.out
#SBATCH --error=/home/a_morelli/temporary_data/check_img_size/job_%A_%a.err

# Set SLURM_ARRAY_COUNT manually if not provided by your version of Slurm
export SLURM_ARRAY_COUNT=30

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
$ENV_PYTHON -m src.scripts.check_img_size \
    --n_workers 8