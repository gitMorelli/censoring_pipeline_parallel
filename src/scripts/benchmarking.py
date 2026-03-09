import time
import os
import multiprocessing as mp

# FORCE SINGLE-THREADING for libraries like NumPy/OpenCV 
# This prevents different processes from fighting over the same CPU cores
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def benchmark_worker(unique_id):
    start = time.time()
    # Call your original processing logic here
    # result = process_subject(unique_id, ...) 
    duration = time.time() - start
    return duration

if __name__ == "__main__":
    sample_ids = [f"ID_{i}" for i in range(20)]
    
    # Test 1: Serial (1 at a time)
    start_time = time.time()
    for uid in sample_ids:
        benchmark_worker(uid)
    print(f"Serial Total Time: {time.time() - start_time:.2f}s")

    # Test 2: Parallel (e.g., 4 at a time)
    start_time = time.time()
    with mp.Pool(processes=4) as pool:
        pool.map(benchmark_worker, sample_ids)
    print(f"Parallel (4 CPUs) Total Time: {time.time() - start_time:.2f}s")