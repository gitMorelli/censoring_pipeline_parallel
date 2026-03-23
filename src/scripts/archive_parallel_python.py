import os
import subprocess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# =========================
# User Settings
# =========================
QUESTIONNAIRE = "11"
SRC_ROOT = f"/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/{QUESTIONNAIRE}"
DST_ROOT = f"/mnt/beegfs01/scratch/a_morelli/parallel_censoring/censored_images/archived_{QUESTIONNAIRE}"
LIST_FILE = os.path.join("/mnt/beegfs01/scratch/a_morelli/datasets/pdfs/ref_pdf_Qx", f"ref_pdf_Q{QUESTIONNAIRE}.csv")
OVERWRITE = False
ID_COL = 'e3n_id_hand'

def archive_one(folder_name):
    """ Archives a single folder using a temporary file to prevent corruption. """
    folder_name = str(folder_name).strip()
    if not folder_name or folder_name == 'nan':
        return "ERROR: Empty ID"
    
    src_path = os.path.join(SRC_ROOT, folder_name)
    final_tar = os.path.join(DST_ROOT, f"{folder_name}.tar")
    temp_tar = os.path.join(DST_ROOT, f"{folder_name}.tar.tmp")

    if os.path.exists(final_tar) and not OVERWRITE:
        return f"SKIP: {folder_name}"

    if not os.path.isdir(src_path):
        return f"ERROR: Source {folder_name} not found"

    try:
        subprocess.run(
            ["tar", "-cf", temp_tar, "-C", SRC_ROOT, folder_name],
            check=True,
            capture_output=True
        )
        os.rename(temp_tar, final_tar)
        return f"DONE: {folder_name}"
    except Exception as e:
        if os.path.exists(temp_tar):
            os.remove(temp_tar)
        return f"FAILED: {folder_name} -> {str(e)}"

def main():
    os.makedirs(DST_ROOT, exist_ok=True)

    # 1. Get Slurm Array Info
    # Default to 0 and 1 if running locally for testing
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    num_tasks = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))

    # 2. Load and slice the data
    if not os.path.exists(LIST_FILE):
        print(f"List file {LIST_FILE} not found.")
        return

    df = pd.read_csv(LIST_FILE)
    all_folders = df[ID_COL].unique().tolist()
    
    # Logic to split the list among nodes
    # Each task_id takes every N-th folder (Staggered approach)
    my_folders = all_folders[task_id::num_tasks]

    if not my_folders:
        print(f"Task {task_id}: No folders assigned.")
        return

    # 3. Parallel processing within the node
    MAX_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Task {task_id}/{num_tasks}: Processing {len(my_folders)} folders with {MAX_WORKERS} CPUs...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(archive_one, my_folders))

    # 4. Summary for this specific node
    successes = sum(1 for r in results if r.startswith("DONE"))
    skips = sum(1 for r in results if r.startswith("SKIP"))
    errors = len(results) - successes - skips
    print(f"Task {task_id} finished. Success: {successes} | Skipped: {skips} | Errors: {errors}")

if __name__ == "__main__":
    main()