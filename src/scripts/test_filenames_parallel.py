import multiprocessing as mp
import time
import os
import math
import shutil
import os
import socket

def clean_results_dir(target_dir):
    if os.path.exists(target_dir):
        print(f"Cleaning directory: {target_dir}")
        # Option A: Delete everything inside but keep the folder
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path) # Deletes file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) # Deletes subfolders
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory {target_dir} does not exist. Creating it...")
        os.makedirs(target_dir, exist_ok=True)

# Usage
path = "/home/a_morelli/vscode_projects/censoring_pipeline_parallel/results/test_filenames_simple"

FOLDER_POSITION_flamingo = "/mnt/beegfs01/scratch/a_morelli/"

'''def get_slurm_info():
    info = {
        "partition": os.environ.get('SLURM_JOB_PARTITION', 'local'),
        "node":      os.environ.get('SLURM_NODENAME', 'localhost'),
        "job_id":    os.environ.get('SLURM_ARRAY_JOB_ID', 'N/A'),
        "task_id":   os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')
    }
    return info'''

def get_slurm_info():
    return {
        # Standard Python way (Reliable)
        "node": socket.gethostname(), 
        # Slurm-specific ways
        "partition": os.environ.get('SLURM_JOB_PARTITION', 'local'),
        "job_id": os.environ.get('SLURM_ARRAY_JOB_ID', 'N/A'),
        "task_id": os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')
    }

# 1. THE MOCK TASK
def process_task(task_id,task_data):
    """Simulates processing one ID: Math + Disk I/O"""
    start = time.time()
    
    # Simulate CPU work (Math)
    result = 0
    for i in range(10**6):
        result += math.sqrt(i)
    end = time.time() - start
    
    slurm_meta = get_slurm_info()

    # Simulate I/O work (Writing a small file)
    folder = os.path.join(FOLDER_POSITION_flamingo,"test_filenames_parallel",f"node_{slurm_meta['node']}")
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/task_{task_id}.txt", "w") as f:
        f.write(f"Task data is {task_data}\n")
        f.write(f"Time required for this single task was {end-start:.4f}\n")
        f.write(f"Processing ID on {slurm_meta['node']} in queue {slurm_meta['partition']}\n")
    
    return 

if __name__ == "__main__":
    TOTAL_TASKS = 100  # Total items to process
    TOTAL_ARRAYS = 10
    N_WORKERS = 4

    ids = [i for i in range(1,TOTAL_TASKS+1)]
    data = [i**2 for i in range(1,TOTAL_TASKS+1)]
    # Determine which IDs THIS specific Slurm task should handle
    # Usage: sbatch --array=0-199 ... (for 200 chunks)
    chunk_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    chunk_size = TOTAL_TASKS // int(os.environ.get('SLURM_ARRAY_COUNT', 1))
    
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < TOTAL_ARRAYS-1 else TOTAL_ARRAYS
    
    my_ids = ids[start_idx:end_idx]
    my_data = data[start_idx:end_idx]

    task_data = []
    for i,ind in enumerate(my_ids):
        task_data.append((ind,my_data[i]))

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    slurm_meta = get_slurm_info()
    print(f"Task {chunk_idx}: Processing {TOTAL_TASKS} IDs using {cpus} CPUs")
    print(f"Processing ID on {slurm_meta['node']} in queue {slurm_meta['partition']}")
    
    with mp.Pool(processes=N_WORKERS) as pool:
        # starmap allows passing multiple arguments to the function
        results = pool.starmap(process_task, task_data)
    