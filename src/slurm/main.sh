#!/bin/bash
#SBATCH --job-name=img_parallel
#SBATCH --array=0-199                 # Divide 100k IDs into 200 chunks (500 per task)
#SBATCH --nodes=1                     # 1 Node per array task
#SBATCH --cpus-per-task=16            # Use 16 CPUs per node
#SBATCH --mem=64G                     # Request enough RAM for 16 parallel processes
#SBATCH --time=12:00:00               # Estimated time for 500 images
#SBATCH --output=logs/job_%a.out

# Set SLURM_ARRAY_COUNT manually if not provided by your version of Slurm
export SLURM_ARRAY_COUNT=200

# Load your environment (conda, modules, etc.)
# module load python/3.9
# source activate my_env

python your_script.py --templates_path "/path/to/templates" --pdf_load_path "/path/to/pdfs" ...

$ENV_PYTHON /home/a_morelli/vscode_projects/censoring_pipeline_parallel/src/scripts/benchmark_parallel_simple.py \
    --source /home/a_morelli/vscode_projects/yolo_test/data \
    --output /home/a_morelli/vscode_projects/yolo_test/results/test_yolo \
    --name job_$SLURM_JOB_ID \
    --model yolo11m.pt