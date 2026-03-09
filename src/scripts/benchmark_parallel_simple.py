import multiprocessing as mp
import time
import os
import math

FOLDER_POSITION_flamingo = "/home/a_morelli/temporary_data"

# 1. THE MOCK TASK
def heavy_task(task_id):
    """Simulates processing one ID: Math + Disk I/O"""
    start = time.time()
    
    # Simulate CPU work (Math)
    result = 0
    for i in range(10**6):
        result += math.sqrt(i)
    
    # Simulate I/O work (Writing a small file)
    folder = os.path.join(FOLDER_POSITION_flamingo,"benchmark_temp")
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/task_{task_id}.txt", "w") as f:
        f.write(f"Result was {result}")
    
    return time.time() - start

# 2. THE RUNNER
def run_benchmark(num_tasks, num_cpus):
    print(f"---> Testing with {num_cpus} CPU(s) for {num_tasks} tasks...")
    
    start_time = time.time()
    
    if num_cpus == 1:
        # Serial execution
        results = [heavy_task(i) for i in range(num_tasks)]
    else:
        # Parallel execution
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.map(heavy_task, range(num_tasks))
    
    total_time = time.time() - start_time
    avg_task_time = sum(results) / len(results)
    
    print(f"      Total Wall Time: {total_time:.2f}s")
    print(f"      Avg Time per Task: {avg_task_time:.4f}s")
    return total_time

if __name__ == "__main__":
    TOTAL_TASKS = 40  # Total items to process
    CPU_COUNTS = [1, 2, 4, 8, 16] # Try these parallel levels
    
    performance_log = {}

    print(f"Starting Benchmark: Processing {TOTAL_TASKS} total items.\n")
    
    for count in CPU_COUNTS:
        # Don't try to use more CPUs than the machine actually has
        if count > mp.cpu_count():
            continue
            
        wall_time = run_benchmark(TOTAL_TASKS, count)
        performance_log[count] = wall_time

    # 3. RESULTS SUMMARY
    print("\n" + "="*30)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*30)
    serial_time = performance_log[1]
    for cpus, t in performance_log.items():
        speedup = serial_time / t
        efficiency = (speedup / cpus) * 100
        print(f"CPUs: {cpus:2d} | Time: {t:6.2f}s | Speedup: {speedup:5.2f}x | Efficiency: {efficiency:5.1f}%")