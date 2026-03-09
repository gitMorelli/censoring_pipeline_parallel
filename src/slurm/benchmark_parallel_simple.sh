#!/bin/bash
#SBATCH --job-name=bench_parallel_simple
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16    # Request the max number of CPUs you want to test
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/a_morelli/vscode_projects/censoring_pipeline_parallel/results/benchmark_parallel_simple/job_%j.out
#SBATCH --error=/home/a_morelli/vscode_projects/censoring_pipeline_parallel/results/benchmark_parallel_simple/job_%j.err


# 1. Load necessary modules (this varies by cluster)
# module load python/3.10
# module load cuda/12.1
#module load nvidia/cuda/12.2.2-535.104.05

# --- Environment Setup ---
#source ~/anaconda3/etc/profile.d/conda.sh
#conda activate yolo_env
ENV_PYTHON="/home/a_morelli/.conda/envs/CensoringEnv/bin/python"

# Add this line to resolve the libiomp5 conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# --- Execution ---
# You can run the script from any location using its full path
$ENV_PYTHON /home/a_morelli/vscode_projects/censoring_pipeline_parallel/src/scripts/benchmark_parallel_simple.py
