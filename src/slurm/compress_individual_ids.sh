#!/bin/bash
#SBATCH --job-name=tar_folders
#SBATCH --output=tar_folders.out
#SBATCH --error=tar_folders.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --partition=shortq

set -euo pipefail

module load parallel/20250222

# =========================
# User settings
# =========================
SRC_ROOT="/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/12"
DST_ROOT="/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/archived_12"

# Optional: set to 1 to overwrite existing tar files, 0 to skip them
OVERWRITE=0

# =========================
# Checks & Setup
# =========================
if [ ! -d "$SRC_ROOT" ]; then
    echo "Source directory does not exist: $SRC_ROOT" >&2
    exit 1
fi

mkdir -p "$DST_ROOT"

# Cleanup: Remove any partial .tmp files from a previous crashed run
echo "Cleaning up any stale temporary files..."
rm -f "$DST_ROOT"/*.tar.tmp

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
    local tmp_tar
    local parent_dir

    base_name="$(basename "$src_dir")"
    out_tar="${DST_ROOT}/${base_name}.tar"
    tmp_tar="${DST_ROOT}/${base_name}.tar.tmp"
    parent_dir="$(dirname "$src_dir")"

    # Skip if final archive already exists
    if [ -f "$out_tar" ] && [ "$OVERWRITE" -ne 1 ]; then
        #echo "SKIP  $src_dir"
        return 0
    fi

    # Archive to a temporary file first
    if tar -cf "$tmp_tar" -C "$parent_dir" "$base_name"; then
        # Success: Rename temp file to final filename
        mv "$tmp_tar" "$out_tar"
        #echo "DONE  $src_dir"
    else
        # Failure: Clean up the broken temp file
        echo "ERROR archiving $src_dir" >&2
        rm -f "$tmp_tar"
        return 1
    fi
}

export -f archive_one

# =========================
# Find first-level folders and archive them in parallel
# =========================
# We use -print0 and parallel -0 to safely handle folder names with spaces
find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 \
  | parallel -0 -j "$CPUS" archive_one {}

echo "Finished: $(date)"