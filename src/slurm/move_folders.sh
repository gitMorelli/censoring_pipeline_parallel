#!/bin/bash
#SBATCH --job-name=move_40k_tars
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Increase to 12 or 16 for this many files
#SBATCH --mem=12G               # More memory helps with the large file list
#SBATCH --time=02:00:00         # Give it a long window
#SBATCH --output=move_folders.log
#SBATCH --partition=shortq

module load parallel/20250222

SOURCE="/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/10"
DEST="/home/a_morelli/datasets/censored_pdfs/archived_10"

# 1. Create destination
mkdir -p "$DEST"

echo "Starting move of files at: $(date)"

# 2. Parallel Transfer with Grouping
# -m: "Multiple" - tells parallel to put as many filenames as possible in one command
# This prevents the '40,000 process' overhead
find "$SOURCE" -maxdepth 1 -name "*.tar" | parallel -j $SLURM_CPUS_PER_TASK -m \
    rsync -a --remove-source-files {} "$DEST/"

# 3. Final Cleanup
# We only delete the source folder if it is completely empty
if [ -z "$(ls -A $SOURCE)" ]; then
   rmdir "$SOURCE"
   echo "All files moved and source directory removed."
else
   echo "Warning: Some files were not moved. Check logs."
fi

echo "Finished at: $(date)"