#!/bin/bash
#SBATCH --job-name=tar_files
#SBATCH --output=archive_out.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --partition=shortq

# Run the tar command
tar -cf /mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/Q13.tar /mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/13