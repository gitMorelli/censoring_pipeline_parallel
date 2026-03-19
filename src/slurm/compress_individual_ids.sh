#!/bin/bash
#SBATCH --job-name=tar_folders
#SBATCH --output=tar_folders.out
#SBATCH --error=tar_folders.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=shortq

set -euo pipefail

module load parallel/20250222
# =========================
# User settings
# =========================
SRC_ROOT="/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/13"
DST_ROOT="/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/archived_13"

# Optional: set to 1 to overwrite existing tar files, 0 to skip them
OVERWRITE=0

# =========================
# Checks
# =========================
if [ ! -d "$SRC_ROOT" ]; then
    echo "Source directory does not exist: $SRC_ROOT" >&2
    exit 1
fi

mkdir -p "$DST_ROOT"

if ! command -v parallel >/dev/null 2>&1; then
    echo "GNU parallel is not available in PATH" >&2
    exit 1
fi

CPUS="${SLURM_CPUS_PER_TASK:-1}"

export SRC_ROOT
export DST_ROOT
export OVERWRITE

echo "Source:      $SRC_ROOT"
echo "Destination: $DST_ROOT"
echo "CPUs:        $CPUS"
echo "Overwrite:   $OVERWRITE"
echo "Started:     $(date)"

# =========================
# Archive function
# =========================
archive_one() {
    local src_dir="$1"
    local base_name
    local out_tar
    local parent_dir

    base_name="$(basename "$src_dir")"
    out_tar="${DST_ROOT}/${base_name}.tar"
    parent_dir="$(dirname "$src_dir")"

    if [ -f "$out_tar" ] && [ "$OVERWRITE" -ne 1 ]; then
        echo "SKIP  $src_dir -> $out_tar"
        return 0
    fi

    # Create tar in destination, preserving the folder as the top-level entry
    tar -cf "$out_tar" -C "$parent_dir" "$base_name"

    echo "DONE  $src_dir -> $out_tar"
}

export -f archive_one

# =========================
# Find first-level folders and archive them in parallel
# =========================
find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 \
  | parallel -0 -j "$CPUS" archive_one {}

echo "Finished: $(date)"