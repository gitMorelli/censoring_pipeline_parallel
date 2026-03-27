#!/bin/bash
#SBATCH --job-name=parallel_rm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G                # Increased RAM for the file list
#SBATCH --time=01:00:00
#SBATCH --output=parallel_rm.out
#SBATCH --partition=shortq

TARGETS=(
    "/mnt/beegfs01/scratch/a_morelli/datasets/pdfs/Q9"
)

module load parallel/20250222

for dir in "${TARGETS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Starting deletion of: $dir at $(date)"
        
        # 1. Faster Listing: 
        # Using 'ls -U' (unsorted) is MUCH faster than 'find' on BeeGFS/Lustre
        # because it doesn't try to sort the files alphabetically.
        
        cd "$dir" || continue
        
        # 2. Bulk Parallel Delete:
        # -m: Groups files into long command lines (rm file1 file2 ... file100)
        # This is 100x faster than running 'rm' 40,000 times.
        ls -U1 | parallel -j $SLURM_CPUS_PER_TASK -m rm -rf
        
        cd - > /dev/null
        
        # 3. Final Cleanup of the root dir
        rm -rf "$dir"
        echo "Completed: $dir at $(date)"
    else
        echo "Skipping $dir: Not found."
    fi
done