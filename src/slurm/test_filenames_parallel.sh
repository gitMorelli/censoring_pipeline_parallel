#!/bin/bash
#SBATCH --job-name=filenames_test_parallel
#SBATCH --array=0-9                 # Divide 100 files into 10 chunks (10 per task)
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --cpus-per-task=8            # Use 16 CPUs per node
#SBATCH --mem=4G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=00:10:00               # Estimated time for 500 images
#SBATCH --partition=shortq
#SBATCH --output=/home/a_morelli/vscode_projects/censoring_pipeline_parallel/results/test_filenames_simple/job_%A_%a.out
#SBATCH --error=/home/a_morelli/vscode_projects/censoring_pipeline_parallel/results/test_filenames_simple/job_%A_%a.err

# Set SLURM_ARRAY_COUNT manually if not provided by your version of Slurm
export SLURM_ARRAY_COUNT=10

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
$ENV_PYTHON /home/a_morelli/vscode_projects/censoring_pipeline_parallel/src/scripts/test_filenames_parallel.py 