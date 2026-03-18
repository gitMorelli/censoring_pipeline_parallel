#!/bin/bash
#SBATCH --job-name=update_ref 
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4           
#SBATCH --mem=8G                     
#SBATCH --time=00:02:00               
#SBATCH --partition=shortq
#SBATCH --output=/home/a_morelli/temporary_data/other_logs/job_%A.out
#SBATCH --error=/home/a_morelli/temporary_data/other_logs/job_%A.out

# Set SLURM_ARRAY_COUNT manually if not provided by your version of Slurm

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
$ENV_PYTHON -m src.scripts.update_ref 