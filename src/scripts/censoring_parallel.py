#!/usr/bin/env python3
import argparse
import logging
import logging.handlers
#### FORCE SINGLE THREADING for some libraries ########
import os
import time
from threadpoolctl import threadpool_limits
#import subprocess
import tarfile
import shutil
os.environ["OMP_NUM_THREADS"] = "1" #numpy
os.environ["MKL_NUM_THREADS"] = "1" #numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from time import perf_counter
from pathlib import Path
import sys
import copy
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import csv
import signal

import pandas as pd
import numpy as np
import cv2
from pympler import asizeof
import psutil

from src.utils.convert_utils import process_pdf_files
#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_files_with_extension, load_template_info
from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

from src.utils.json_parsing import get_page_list, get_roi_boxes
from src.utils.json_parsing import get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region, preprocess_blank_roi, censor_image_with_boundary

from src.utils.matching_utils import pre_load_image_properties, initialize_page_dictionary, initialize_template_dictionary
from src.utils.matching_utils import initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base, perform_orb_matching

from src.utils.alignment_utils import compute_misalignment, roi_blank_decision, adjust_boundary_boxes, orb_matching,convert_to_axis_aligned_box,is_geometry_valid, rescale_box_coords_given_resolutions

from src.utils.censor_utils import map_to_smallest_containing, save_as_is_no_censoring, get_transformation_from_dictionaries, apply_transformation_to_boxes, save_censored_image, map_to_all_containing

from src.utils.debug_utils import visualize_templates_w_annotations, save_w_boxes, superimpose_images, save_these_boxes

from src.utils.logging import FileWriter, initialize_logger
 
'''logging.basicConfig(
    filename='parallel_processing.log',
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(processName)s: %(message)s'
)
logger = logging.getLogger(__name__)'''

PDF_LOAD_PATH= "/mnt/beegfs01/scratch/a_morelli/datasets/pdfs"#"/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization" 
CSV_LOAD_PATH= "/mnt/beegfs01/scratch/a_morelli/datasets/pdfs/ref_pdf_Qx" #"/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization/ref_pdf"# "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"#additional"#100263_template"
TEMPLATES_PATH="/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization/current_template" #"//vms-e34n-databr/2025-handwriting\\data\\annotations\current_template"#additional"#100263_template"
SAVE_PATH= "/mnt/beegfs01/scratch/a_morelli/parallel_censoring" #"/mnt/beegfs01/scratch/a_morelli/test_parallel_censoring" #"//vms-e34n-databr/2025-handwriting\\data\\test_censoring_pipeline"#additional"#100263_template" 

#names for debug subfolders
debug_images_name='debug_images\\'
debug_images_templates_name=debug_images_name+'debug_images_templates'
debug_images_original_name=debug_images_name+'original'
debug_images_w_boxes_name=debug_images_name+'original_w_boxes'
debug_images_superimposed_name=debug_images_name+'superimposed'
debug_images_adjusted_boxes = debug_images_name+'adjusted_for_boundaries'
time_logs_name = 'time_logs'
debug_temporary_name = 'debug_temporary'
partial_results_name = 'partial_results'
#temlates path
template_images_path = "/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization/png_templates/current_template/"#"//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\current_template"

# script parameters
#TOTAL_TASKS = 100  # Total items to process

# other variables
SAVE_ARCHIVED = True #if true all results from an id will be stored in a tar folder with the id name
PNG_COMPRESSION_LEVEL = 6 #for png compression when saving debug images
QUESTIONNAIRE = "11"
ID_COL = 'e3n_id_hand'
FILENAME_COL = 'object_name'
#SAVE_ANNOTATED_TEMPLATES=True
#additional cols
USED_COL = 'Used'
WARNING_ORDERING_COL_NAME = 'Warning_ordering'
WARNING_CENSORING_COL_NAME = 'Warning_censoring'

# matching parameters
N_ALIGN_REGIONS = 2 #minimum number of align boxes needed for matching
#template matching
SCALE_FACTOR_MATCHING = 1.5 # used during page ordering and to compute alignements
SCALE_FACTOR_RESIZING = 1.2 # used to correct alignement
#matchTemplate
MATCHING_THRESHOLD = 0.7
#phash
GAP_THRESHOLD_PHASH = 5
MAX_DIST_PHASH = 18
#ocr
GAP_THRESHOLD_OCR = 0.1
MAX_DIST_OCR = 0.2
DISCARD_DIST_OCR = 0.7
TEXT_SIMILARITY_METRIC = 'similarity_jaccard_tokens'
#orb
GAP_THRESHOLD_ORB = 5 # i shoudl modify this and the other value (i fixed to the same value as phash but makes no sense)
MAX_DIST_ORB = 18
ORB_GOOD_MATCH = 50 #if more than 50 matches then the page/roi is a good match
ORB_top_n_matches = 50 # how many matches to keep for alignement
ORB_match_threshold = 15
ORB_method_to_find_matches= 'brute_force'#'knn' #'brute_force' #knn
ORB_match_filtering_method= 'lowe_ratio' #"best_n" #"lowe_ratio" #all
ORB_lowe_threshold = 0.5 #0-1, smaller more restrictive
ORB_decision_procedure = 'simple' #simple or homography
ORB_parameters={
    'orb_match_threshold': ORB_match_threshold, 
    'orb_top_n_matches': ORB_top_n_matches,
    'orb_lowe_threshold': ORB_lowe_threshold,
    'orb_match_filtering_method': ORB_match_filtering_method,
    'orb_method_to_find_matches': ORB_method_to_find_matches,
    'orb_decision_procedure' : ORB_decision_procedure
}
#white regions
N_BLACK_THRESH=0.1
BLANK_REGION_TESTING_THRESHOLD = 1

# pipeline parameters
# extending censor regions to boundaries
EPSILON_EDGE_MATCHING = 2.0
# computing alignement
ALIGNEMENT_METHOD = 'orb_page_level_homography' #'orb_page_level_homography'  # 'pre_computed'# 'orb_page_level_homography', 'orb_page_level_affine'
#Checking alignement
ANGLE_TOLERANCE = 10 #this is the upper limit for the increas in the largest angle of the rectangle after transformation (if it is over 90+angle_tolerange degrees -> not good)
#correcting after alignement
RESCALE_CENSOR_WITH = 'skip_assume_aligned'#'template_align_and_extra' #'orb_extra', 'skip_assume_aligned','skip_assume_misaligned'
#applying alignement
TRANSFORMATION_OPTION = 'standard'#'no_rotation' #options are 'no_rotation', 'standard'
FORCE_AXIS_ALIGNED_BOXES = True #if false the censor boxes are rotated after alygnemetn -> they are polygon; If true the smallest 
#rectangle that is axi aligned and contains the polygon is drawn instead
# page ordering
ORDERING_METHOD_STEP_2 = 'orb' #orb or phash
CHECKING_FIRST_STAGE = 'template' #options are 'template' or 'orb'
CHECKING_SECOND_STAGE = 'orb' #options are 'template' or 'orb'
#censoring (sriped regions)
THICKNESS_PCT = 0.1
SPACING_MULT = 0.1
PAGE_FRAC = 0.04 #2 characters are in a range of 0.02-0.06 page width
AREA_FRAC = 2/3 #covered fraction in partially censored boxes

########### main async approach ########################
def main(test_size=-1,timeout_lim=120,test=False):
    args = parse_args()

    ####### INITIALIZING PATHS #########
    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path
    save_debug_times = args.save_debug_times
    save_debug_images = args.save_debug_images
    #total_arrays = args.total_arrays
    n_workers = args.n_workers

    #folder for the csv table
    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    #folder for the global logging
    log_path=os.path.join(save_path,'logs')
    #folder per il debug
    debug_path = os.path.join(save_path,'debug')
    #load path for the pdfs
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    #saving the censored images here
    censored_images_path = os.path.join(save_path,'censored_images')


    ####### CLEANING FOLDERS (only in debugging) ###########
    '''if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(censored_images_path):
            remove_folder(censored_images_path)
        if os.path.exists(debug_path):
            remove_folder(debug_path)'''
    
    ########## Initializing loggers ##################
    #console logger
    #logger.setLevel(logging.DEBUG)
    #logging.getLogger().setLevel(logging.DEBUG)

    #file logger (global logger for the execution)
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_logger.txt"))
    # create folder to save debug results
    create_folder(debug_path, parents=True, exist_ok=True)

    ########## LOAD the DATFRAME with the subject ids and filenames ###############
    csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    #csv_modified_path = os.path.join(csv_load_path,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    
    #for testing i am using used_col in the opposite way 
    #df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df
    #file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct
    df=pd.read_csv(csv_modified_path)
    #select only the lines with used=False
    df = df[df[USED_COL]==False] 
    print(f"Total number of unique ids to process: {df[ID_COL].nunique()}")
    #return 0

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)
    #print("selected templates: ",selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one
    print("pages in annotations:", pages_in_annotation)

    ##### PREPARE SHARED RESOURCES #############
    shared_resources = {
        'templates_path': templates_path,
        'npy_data': npy_data,
        'annotation_roots': annotation_roots,
        'pages_in_annotation': pages_in_annotation,
        'selected_templates': selected_templates,
        # paths
        'questionnairres_log_path' : questionnairres_log_path,
        'debug_path': debug_path,
        'censored_images_path': censored_images_path,
        'updated_csv_paths': updated_csv_paths,
        #args
        'save_debug_times': save_debug_times,
        'save_debug_images': save_debug_images,
        'verbose': args.verbose,
    }

    # 2. SLURM CHUNKING LOGIC
    # Get all unique IDs
    all_unique_ids = df[ID_COL].unique()
    
    #debug
    if test_size > 0:
        all_unique_ids=all_unique_ids[:test_size]
    print(f"Total unique IDs after test size limit: {len(all_unique_ids)}")

    # Determine which IDs THIS specific Slurm task should handle
    # Usage: sbatch --array=0-199 ... (for 200 chunks)
    total_arrays = int(os.environ.get('SLURM_ARRAY_COUNT', 1))
    chunk_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    chunk_size = len(all_unique_ids) // total_arrays
    
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < total_arrays-1 else len(all_unique_ids) #if i consider the last chinck i have to take all the remaining ids to avoid losing data
    
    my_ids = all_unique_ids[start_idx:end_idx]
    print("Chunck of ids has length: ", len(my_ids))
    
    # Prepare the arguments for the multiprocessing pool
    tasks = []
    for uid in my_ids:
        group = df[df[ID_COL] == uid]
        tasks.append((uid, group, shared_resources))
    
    if test:
        print("Running in test mode")
        tasks = build_test_tasks()

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Task {chunk_idx}: Processing {len(tasks)} IDs using {cpus} CPUs")
    
    # initialize process parameters
    # 1. Define your Column Names
    headers = [ID_COL, FILENAME_COL, USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME, 
               'time','status','error']
    if test:
        headers = ['task_id', 'value', 'time', 'status', 'error']
    ref_Qx_updated_path = os.path.join(updated_csv_paths,f"final_ref_pdf_Q{QUESTIONNAIRE}_chunk_{chunk_idx}.csv")
    # 2. Initialize the file: Write the header once
    # This will overwrite any existing file at that path
    with open(ref_Qx_updated_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    ctx = mp.get_context("spawn")  # safer on many platforms / libraries

    buffer_size = 50  # Write every 50 tasks
    poll_interval = 1
    if test:
        timeout_lim = 10
        buffer_size = 1
        poll_interval = 0.5
    results_buffer = []
    pending_tasks = list(tasks)
    completed_task_ids = set()
    timed_out_task_ids = set()

    def new_pool():
        return ctx.Pool(processes=n_workers, maxtasksperchild=1)

    pool = new_pool()

    # active_jobs: async_result -> metadata
    active_jobs = {}

    def submit_until_full():
        while pending_tasks and len(active_jobs) < n_workers:
            task = pending_tasks.pop(0)
            if test:
                async_result = pool.apply_async(process_wrapper_test, (task,))
            else:
                async_result = pool.apply_async(process_wrapper, (task,))
            active_jobs[async_result] = {
                "task": task,
                "task_id": task[0],
                "submitted_at": time.monotonic(),
            }
    total_tasks = len(tasks)
    last_reported = 0 # To avoid spamming the same number
    try:
        submit_until_full()

        while active_jobs or pending_tasks:
            finished = []
            timed_out = []

            now = time.monotonic()

            # 1. Check each active job
            for async_result, meta in list(active_jobs.items()):
                elapsed = now - meta["submitted_at"]

                if async_result.ready():
                    finished.append((async_result, meta))
                elif elapsed > timeout_lim:
                    timed_out.append((async_result, meta))

            # 2. Handle finished jobs
            for async_result, meta in finished:
                task_info = meta["task"]
                task_id = meta["task_id"]

                try:
                    item = async_result.get()   # no timeout needed; already ready()

                    sub_df = item_to_dataframe_row(item, headers)
                    append_result_row(results_buffer, sub_df, headers)

                    completed_task_ids.add(task_id)

                except Exception as e:
                    # Worker-level exception, crash, serialization problem, etc.
                    print(f"[Worker exception] task_id={task_id}: {e}")

                    item = make_worker_exception_item(task_info, e)
                    sub_df = item_to_dataframe_row(item, headers)
                    append_result_row(results_buffer, sub_df, headers)

                    completed_task_ids.add(task_id)

                del active_jobs[async_result]

            # 3. Handle timeouts
            if timed_out:
                print(
                    f"[Timeout] {len(timed_out)} task(s) exceeded {timeout_lim}s. "
                    "Terminating pool and resubmitting unfinished work."
                )
                ids_to_show = [meta["task_id"] for _, meta in timed_out]
                print(f"-> Timed out task IDs: {ids_to_show}")
                # Mark timed-out tasks as timed out
                timed_out_ids_this_round = set()
                for async_result, meta in timed_out:
                    task_info = meta["task"]
                    task_id = meta["task_id"]

                    item = make_timeout_item(task_info)
                    sub_df = item_to_dataframe_row(item, headers)
                    append_result_row(results_buffer, sub_df, headers)

                    completed_task_ids.add(task_id)
                    timed_out_task_ids.add(task_id)
                    timed_out_ids_this_round.add(task_id)

                # Tasks still active but not timed out yet were killed too when pool dies.
                # We need to resubmit them.
                survivors_to_resubmit = []
                for async_result, meta in active_jobs.items():
                    task_id = meta["task_id"]
                    if task_id not in timed_out_ids_this_round:
                        survivors_to_resubmit.append(meta["task"])

                # Kill the whole pool to stop the hanging worker(s)
                pool.terminate()
                pool.join()

                # Reset state
                active_jobs.clear()

                # Resubmit killed-but-not-timed-out tasks first
                pending_tasks = survivors_to_resubmit + pending_tasks

                # Create a fresh pool
                pool = new_pool()

            # 4. Flush buffer if needed
            if len(results_buffer) >= buffer_size:
                flush_buffer(results_buffer, ref_Qx_updated_path)

            # 5. Top up the pool
            submit_until_full()

            current_done = len(completed_task_ids)
            if current_done > last_reported:
                print(f"Progress: {current_done}/{total_tasks} ({(current_done/total_tasks)*100:.1f}%)")
                last_reported = current_done

            # 6. Avoid busy waiting
            if active_jobs:
                time.sleep(poll_interval)

        # Final flush
        flush_buffer(results_buffer, ref_Qx_updated_path)

    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass

    if test:
        print("\n=== SUMMARY ===")
        print(f"CSV written to: {ref_Qx_updated_path}")

        print("\n=== READ CSV BACK WITH PANDAS ===")
        final_df = pd.read_csv(ref_Qx_updated_path)
        print(final_df)

        print("\n=== STATUS COUNTS ===")
        print(final_df["status"].value_counts(dropna=False))

        # Optional sanity checks
        assert len(final_df) == len(tasks), (
            f"Expected {len(tasks)} rows, found {len(final_df)}"
        )

        expected_statuses = {"success", "failed", "timeout", "worker_exception"}
        found_statuses = set(final_df["status"].unique())
        print("\nExpected statuses subset:", expected_statuses)
        print("Found statuses:", found_statuses)
        return 
    '''#open the csv and print a summary of the results
    final_df = pd.read_csv(ref_Qx_updated_path)
    #print the number of failed and timed out tasks
    print("Summary of results:")
    print(final_df['status'].value_counts())
    #for each unique_id that wa succesfully completed print the execution time
    successful_tasks = final_df[final_df['status']=='success']
    for i, row in successful_tasks.iterrows():
        print(f"Result for task {row[ID_COL]}: {row['time']:.2f} seconds, status: {row['status']}, error: {row['error']}")
    for i, row in final_df.iterrows():
        print(f"Result for task {row[ID_COL]}: {row['time']:.2f} seconds, status: {row['status']}, error: {row['error']}")

    return {
        "completed_task_ids": completed_task_ids,
        "timed_out_task_ids": timed_out_task_ids,
    }'''
    return

def append_result_row(results_buffer, sub_df, headers):
    sub_df = sub_df.copy()
    sub_df = sub_df[headers]
    results_buffer.extend(sub_df.values.tolist())

def flush_buffer(results_buffer, output_path):
    if results_buffer:
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(results_buffer)
        results_buffer.clear()

def make_timeout_item(task_info):
    task_id = task_info[0]
    sub_df = task_info[1].copy()
    return {
        "task_id": task_id,
        "success": False,
        "result": sub_df,
        "time": np.nan,
        "error": {
            "type": "Timeout",
            "message": "Task exceeded allowed runtime",
            "traceback": "",
        },
        "status": "timeout",
    }

def make_worker_exception_item(task_info, exc):
    task_id = task_info[0]
    sub_df = task_info[1].copy()
    return {
        "task_id": task_id,
        "success": False,
        "result": sub_df,
        "time": np.nan,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "",
        },
        "status": "worker_exception",
    }

def item_to_dataframe_row(item, headers):
    sub_df = item["result"].copy()
    sub_df["time"] = item["time"]

    if "status" in item:
        sub_df["status"] = item["status"]
    else:
        sub_df["status"] = "success" if item["success"] else "failed"

    sub_df["error"] = "" if item.get("error") is None else str(item["error"])
    sub_df = sub_df[headers]
    return sub_df

### tests ####
def process_subject_test(task_id, group, mode):
    """
    Simulated task behavior.

    Parameters
    ----------
    task_id : int
    group   : pd.DataFrame
    mode    : str
        One of: ok, py_exception, hang, worker_crash, sigkill
    """
    if mode == "ok":
        out = group.copy()
        out["processed"] = 1
        return out

    elif mode == "py_exception":
        raise ValueError(f"Python exception for task {task_id}")

    elif mode == "hang":
        time.sleep(999999)

    elif mode == "worker_crash":
        os._exit(1)   # this will likely end up in your timeout path

    elif mode == "unpicklable_result":
        return lambda x: x   # triggers get()/pickling failure in parent

    else:
        raise RuntimeError(f"Unknown mode: {mode}")

def make_group(task_id):
    return pd.DataFrame({"task_id": [task_id], "value": [task_id * 10]})

def build_test_tasks():
    # task tuple: (task_id, group_df, mode)
    return [
        (1, make_group(1), "ok"),
        (2, make_group(2), "py_exception"),
        (3, make_group(3), "hang"),
        (4, make_group(4), "worker_crash"),
        (5, make_group(5), "unpicklable_result"),
        (6, make_group(6), "ok"),
    ]

def process_wrapper_test(task_tuple):
    task_id = task_tuple[0]
    group = task_tuple[1]
    try:
        #compute time for the process 
        t_0 = perf_counter()
        # Pass the rest of the tuple to your actual function
        result = process_subject_test(*task_tuple) 
        t_end = perf_counter()
        return {
            "task_id": task_id,
            "success": True,
            "result": result,
            "time": t_end - t_0,
            "error": None,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "result": group.copy(), #return the group of the subject that caused the error to facilitate debugging
            "time": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }
########### main process Pool executor style #############
def main_processPoolExecutor_style(test_size=-1,chunk_size_mp=10,timeout_lim=120):
    args = parse_args()

    ####### INITIALIZING PATHS #########
    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path
    save_debug_times = args.save_debug_times
    save_debug_images = args.save_debug_images
    #total_arrays = args.total_arrays
    n_workers = args.n_workers

    #folder for the csv table
    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    #folder for the global logging
    log_path=os.path.join(save_path,'logs')
    #folder per il debug
    debug_path = os.path.join(save_path,'debug')
    #load path for the pdfs
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    #saving the censored images here
    censored_images_path = os.path.join(save_path,'censored_images')


    ####### CLEANING FOLDERS (only in debugging) ###########
    '''if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(censored_images_path):
            remove_folder(censored_images_path)
        if os.path.exists(debug_path):
            remove_folder(debug_path)'''
    
    ########## Initializing loggers ##################
    #console logger
    #logger.setLevel(logging.DEBUG)
    #logging.getLogger().setLevel(logging.DEBUG)

    #file logger (global logger for the execution)
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_logger.txt"))
    #memory logger (global logger for the execution)
    memory_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"memory_logger.txt"))
    #memory logger (global logger for the execution)
    global_time_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_time_logger.txt"))
    # create folder to save debug results
    create_folder(debug_path, parents=True, exist_ok=True)

    '''#parallel loggign
    # 1. Setup the Queue and the File Handler
    manager = mp.Manager()
    log_queue = manager.Queue()
    # Define the format and where the file goes
    threaded_log_path = os.path.join(log_path, "parallel_run.log")
    file_handler = logging.FileHandler(threaded_log_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s: %(message)s')
    file_handler.setFormatter(formatter)
    # 2. Start the Listener
    # This stays in the MAIN process but runs in the background
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()'''

    ########## LOAD the DATFRAME with the subject ids and filenames ###############
    #csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    csv_modified_path = os.path.join(csv_load_path,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    
    #for testing i am using used_col in the opposite way 
    #df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df
    #file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct
    df=pd.read_csv(csv_modified_path)
    #select only the lines with used=False
    df = df[df[USED_COL]==True] 
    print(f"Total number of unique ids to process: {df[ID_COL].nunique()}")

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)
    #print("selected templates: ",selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one
    print("pages in annotations:", pages_in_annotation)

    ##### PREPARE SHARED RESOURCES #############
    shared_resources = {
        'templates_path': templates_path,
        'npy_data': npy_data,
        'annotation_roots': annotation_roots,
        'pages_in_annotation': pages_in_annotation,
        'selected_templates': selected_templates,
        # paths
        'questionnairres_log_path' : questionnairres_log_path,
        'debug_path': debug_path,
        'censored_images_path': censored_images_path,
        'updated_csv_paths': updated_csv_paths,
        #args
        'save_debug_times': save_debug_times,
        'save_debug_images': save_debug_images,
        'verbose': args.verbose,
    }

    # 2. SLURM CHUNKING LOGIC
    # Get all unique IDs
    all_unique_ids = df[ID_COL].unique()
    
    all_unique_ids=all_unique_ids[:test_size]

    # Determine which IDs THIS specific Slurm task should handle
    # Usage: sbatch --array=0-199 ... (for 200 chunks)
    total_arrays = int(os.environ.get('SLURM_ARRAY_COUNT', 1))
    chunk_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    chunk_size = len(all_unique_ids) // total_arrays
    
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < total_arrays-1 else len(all_unique_ids) #if i consider the last chinck i have to take all the remaining ids to avoid losing data
    
    my_ids = all_unique_ids[start_idx:end_idx]
    print("Chunck of ids has length: ", len(my_ids))
    
    # Prepare the arguments for the multiprocessing pool
    tasks = []
    for uid in my_ids:
        group = df[df[ID_COL] == uid]
        tasks.append((uid, group, shared_resources))

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Task {chunk_idx}: Processing {len(tasks)} IDs using {cpus} CPUs")
    
    successful_results = []
    failed_results = []
    timed_out_count = 0
    received_task_ids = set()
    buffer_size = 50  # Write every 50 tasks
    results_buffer = []

    # 1. Define your Column Names
    headers = [ID_COL, FILENAME_COL, USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME, 
               'time','status','error']
    
    # 2. Initialize the file: Write the header once
    # This will overwrite any existing file at that path
    with open(ref_Qx_updated_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Map futures to their original task data for recovery during timeout
        future_to_task = {executor.submit(process_wrapper, t): t for t in tasks}
        
        for future in as_completed(future_to_task):
            try:
                # result() will raise TimeoutError if it exceeds timeout_lim
                item = future.result(timeout=timeout_lim)
                
                sub_df = item["result"] 
                sub_df['time'] = item["time"]
                sub_df['status'] = "success" if item["success"] else "failed"
                sub_df['error'] = item.get("error", "")
                sub_df=sub_df[headers] #reorder columns to match the header order
                
                # Convert row to list format for CSV writing
                results_buffer.extend(sub_df.values.tolist())
            
            except Exception as e:
                # Recover task info from the mapping
                task_info = future_to_task[future]

                sub_df = task_info[1] 
                sub_df['time'] = np.nan
                sub_df['status'] = "timeout"
                sub_df['error'] = ""
                sub_df=sub_df[headers] #reorder columns to match the header order
                
                # Convert row to list format for CSV writing
                results_buffer.extend(sub_df.values.tolist())
                print(f"Generated an exception: {e}")

            # 2. Check buffer size and write
            if len(results_buffer) >= buffer_size:
                with open(ref_Qx_updated_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(results_buffer)
                results_buffer.clear()

        # 3. FINAL FLUSH: Write remaining items
        if results_buffer:
            with open(ref_Qx_updated_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(results_buffer)
            results_buffer.clear()

    # Find tasks that never produced any result
    all_task_ids = {task[0] for task in tasks}
    missing_task_ids = all_task_ids - received_task_ids
    missing_groups = [group[1] for group in tasks if group[0] in missing_task_ids]

    '''return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "timed_out_count": timed_out_count,
        "missing_task_ids": list(missing_task_ids),
    }'''
    
    print(f"Task {chunk_idx} completed. Successful: {len(successful_results)}, Failed: {len(failed_results)}")
    
    print(f"Timed out: {timed_out_count}, Missing due to timeout: {len(missing_task_ids)}")

    for i, res in enumerate(successful_results):
        print(f"Result for task {res[0]}: {res[1]:.2f} seconds")
    
    # 4. Cleanup
    #listener.stop()
    #this saves all results in a single csv
    final_df = assemble_final_csv(successful_results, failed_results, missing_groups,
                          filename_col=FILENAME_COL, id_col=ID_COL, used_col_name=USED_COL,
                          warning_ordering_col_name=WARNING_ORDERING_COL_NAME, warning_censoring_col_name=WARNING_CENSORING_COL_NAME)

    ref_Qx_updated_path = os.path.join(updated_csv_paths,f"final_ref_pdf_Q{QUESTIONNAIRE}_chunk_{chunk_idx}.csv")
    final_df.to_csv(ref_Qx_updated_path, index=False)

    '''tot = 0
    for r in results:
        if r != 1:
            print("Error in processing a subject, check the logs for details.")
        tot+=r
    print(f"Task {chunk_idx} completed. Successfully processed {tot}/{len(tasks)} subjects.")'''

    return 0

def process_wrapper(task_tuple):
    task_id = task_tuple[0]
    group = task_tuple[1]
    try:
        #compute time for the process 
        t_0 = perf_counter()
        # Pass the rest of the tuple to your actual function
        result = process_subject(*task_tuple) 
        t_end = perf_counter()
        return {
            "task_id": task_id,
            "success": True,
            "result": result,
            "time": t_end - t_0,
            "error": None,
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "result": group.copy(), #return the group of the subject that caused the error to facilitate debugging
            "time": None,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }

def assemble_final_csv(successful_results, failed_results, missing_groups, 
                       filename_col,id_col,used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring'):
    #iterate on the successful results and concatenate the dataframes 
    sub_dfs = [res[2] for res in successful_results] 
    if sub_dfs:
        final_df = pd.concat(sub_dfs, ignore_index=True)
    else:
        final_df = pd.DataFrame() #empty df if no successful results
    final_df['time'] = [res[1] for res in successful_results] 
    final_df['status'] = 'success'
    column_names = final_df.columns.tolist()
    expected_columns = {filename_col, id_col, used_col_name, warning_ordering_col_name, warning_censoring_col_name}
    assert set(column_names) >= expected_columns, f"Missing expected columns in final dataframe. Found columns: {column_names}"

    sub_dfs_failed = [res[2] for res in failed_results]
    if sub_dfs_failed:
        df_failed = pd.concat(sub_dfs_failed, ignore_index=True)
    else:
        df_failed = pd.DataFrame() #empty df if no successful results
    df_failed['time'] = [res[1] for res in failed_results]
    df_failed['status'] = 'failed'

    sub_dfs_missing = [group for group in missing_groups]
    if sub_dfs_missing:
        df_missing = pd.concat(sub_dfs_missing, ignore_index=True)
    else:
        df_missing = pd.DataFrame() #empty df if no successful results
    df_missing['time'] = np.nan
    df_missing['status'] = 'Timeout'

    final_df = pd.concat([final_df, df_failed, df_missing], ignore_index=True)

    final_df[USED_COL]==True

    return final_df

############################################################

def multi_threading_test():
    args = parse_args()

    ####### INITIALIZING PATHS #########
    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path
    save_debug_times = args.save_debug_times
    save_debug_images = args.save_debug_images
    #total_arrays = args.total_arrays
    n_workers = args.n_workers

    #folder for the csv table
    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    #folder for the global logging
    log_path=os.path.join(save_path,'logs')
    #folder per il debug
    debug_path = os.path.join(save_path,'debug')
    #load path for the pdfs
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    #saving the censored images here
    censored_images_path = os.path.join(save_path,'censored_images')


    ####### CLEANING FOLDERS (only in debugging) ###########
    '''if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(censored_images_path):
            remove_folder(censored_images_path)
        if os.path.exists(debug_path):
            remove_folder(debug_path)'''
    
    ########## Initializing loggers ##################
    #console logger
    #logger.setLevel(logging.DEBUG)
    #logging.getLogger().setLevel(logging.DEBUG)

    #file logger (global logger for the execution)
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_logger.txt"))
    #memory logger (global logger for the execution)
    memory_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"memory_logger.txt"))
    #memory logger (global logger for the execution)
    global_time_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_time_logger.txt"))
    # create folder to save debug results
    create_folder(debug_path, parents=True, exist_ok=True)

    #parallel loggign
    # 1. Setup the Queue and the File Handler
    manager = mp.Manager()
    log_queue = manager.Queue()
    # Define the format and where the file goes
    threaded_log_path = os.path.join(log_path, "parallel_run.log")
    file_handler = logging.FileHandler(threaded_log_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s: %(message)s')
    file_handler.setFormatter(formatter)
    # 2. Start the Listener
    # This stays in the MAIN process but runs in the background
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()

    ########## LOAD the DATFRAME with the subject ids and filenames ###############
    #csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    csv_modified_path = os.path.join(csv_load_path,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    
    #for testing i am using used_col in the opposite way 
    #df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df
    #file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct
    df=pd.read_csv(csv_modified_path)
    #select only the lines with used=False
    df = df[df[USED_COL]==True] 
    print(f"Total number of unique ids to process: {df[ID_COL].nunique()}")

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)
    print("selected templates: ",selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one
    print("pages in annotations:", pages_in_annotation)

    ##### PREPARE SHARED RESOURCES #############
    shared_resources = {
        'templates_path': templates_path,
        'npy_data': npy_data,
        'annotation_roots': annotation_roots,
        'pages_in_annotation': pages_in_annotation,
        'selected_templates': selected_templates,
        # paths
        'questionnairres_log_path' : questionnairres_log_path,
        'debug_path': debug_path,
        'censored_images_path': censored_images_path,
        'updated_csv_paths': updated_csv_paths,
        #args
        'save_debug_times': save_debug_times,
        'save_debug_images': save_debug_images,
        'verbose': args.verbose,
        #loggers
        'file_logger': file_logger,
        'memory_logger': memory_logger,
        'global_time_logger': global_time_logger,
        # ... add all other necessary variables/objects ...
    }

    # 2. SLURM CHUNKING LOGIC
    # Get all unique IDs
    all_unique_ids = df[ID_COL].unique()
    
    my_ids=all_unique_ids[:15]
    
    # Prepare the arguments for the multiprocessing pool
    tasks = []
    for uid in my_ids:
        group = df[df[ID_COL] == uid]
        tasks.append((uid, group, shared_resources))

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Processing {len(tasks)} IDs using {cpus} CPUs")


    # Warm-up run (to ignore initialization overhead)
    '''with threadpool_limits(limits=1):
        benchmark_task()'''
    results=[]
    runs = [1]+[cpus for cpus in range(1, 17) ] 
    for num_cpus in runs:
        # 1. THE "HARD" WAY (For Stubborn OpenCV/OCR)
        # Note: This only works if the library hasn't 'locked in' yet.
        # It's a safety net for any subprocesses your code might spawn.
        os.environ["OMP_NUM_THREADS"] = str(num_cpus)
        cv2.setNumThreads(num_cpus)

        with threadpool_limits(limits=num_cpus, user_api='blas'):
            start_time = perf_counter()
            print("-"*50)
            print(f"CPUs: {num_cpus}")
            # --- YOUR TASK HERE ---
            dtimes = []
            for task in tasks:
                start_time_id = perf_counter()
                process_subject(*task)
                end_time_id = perf_counter()
                dtimes.append(end_time_id - start_time_id)
                print(f"Time for id {task[0]}: {dtimes[-1]:.4f}s")
            
            end_time = perf_counter()
            duration = end_time - start_time
            results.append((duration,dtimes))
            print(f"Time: {duration:.4f}s")
    print("\n" + "="*30)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*30)
    reference_time = results[1][0]
    for i,result in enumerate(results[1:]):
        cpus=i+1
        speedup = reference_time / result[0]
        efficiency = (speedup / cpus) * 100
        print(f"CPUs: {cpus:2d} | Time: {result[0]:6.2f}s | Speedup: {speedup:5.2f}x | Efficiency: {efficiency:5.1f}%")
    
    # 4. Cleanup
    listener.stop()

    '''tot = 0
    for r in results:
        if r != 1:
            print("Error in processing a subject, check the logs for details.")
        tot+=r
    print(f"Task {chunk_idx} completed. Successfully processed {tot}/{len(tasks)} subjects.")'''

    return 0


def multi_node_test(n_ids):
    args = parse_args()

    ####### INITIALIZING PATHS #########
    templates_path = args.templates_path
    pdf_load_path = args.pdf_load_path
    csv_load_path = args.csv_load_path
    save_path = args.save_path
    save_debug_times = args.save_debug_times
    save_debug_images = args.save_debug_images
    #total_arrays = args.total_arrays
    n_workers = args.n_workers

    #folder for the csv table
    updated_csv_paths = os.path.join(save_path,"ref_pdf")
    #folder for the global logging
    log_path=os.path.join(save_path,'logs')
    #folder per il debug
    debug_path = os.path.join(save_path,'debug')
    #load path for the pdfs
    questionnairres_log_path=os.path.join(pdf_load_path, f"Q{QUESTIONNAIRE}")
    #saving the censored images here
    censored_images_path = os.path.join(save_path,'censored_images')


    ####### CLEANING FOLDERS (only in debugging) ###########
    '''if args.delete_previous_results:
        if os.path.exists(updated_csv_paths):
            remove_folder(updated_csv_paths)
        if os.path.exists(log_path):
            remove_folder(log_path)
        if os.path.exists(censored_images_path):
            remove_folder(censored_images_path)
        if os.path.exists(debug_path):
            remove_folder(debug_path)'''
    
    ########## Initializing loggers ##################
    #console logger
    #logger.setLevel(logging.DEBUG)
    #logging.getLogger().setLevel(logging.DEBUG)

    #file logger (global logger for the execution)
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_logger.txt"))
    '''
    #memory logger (global logger for the execution)
    memory_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"memory_logger.txt"))
    #memory logger (global logger for the execution)
    global_time_logger=FileWriter(enabled=False,path=os.path.join(log_path,f"global_time_logger.txt"))
    # create folder to save debug results'''
    create_folder(debug_path, parents=True, exist_ok=True)

    '''#parallel loggign
    # 1. Setup the Queue and the File Handler
    manager = mp.Manager()
    log_queue = manager.Queue()
    # Define the format and where the file goes
    threaded_log_path = os.path.join(log_path, "parallel_run.log")
    file_handler = logging.FileHandler(threaded_log_path)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s: %(message)s')
    file_handler.setFormatter(formatter)
    # 2. Start the Listener
    # This stays in the MAIN process but runs in the background
    listener = logging.handlers.QueueListener(log_queue, file_handler)
    listener.start()'''

    ########## LOAD the DATFRAME with the subject ids and filenames ###############
    #csv_modified_path = os.path.join(updated_csv_paths,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    csv_modified_path = os.path.join(csv_load_path,f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    if os.path.exists(csv_modified_path)==False: #if the csv has not been preprocessed yet
        df = preprocess_df(os.path.join(csv_load_path,f"ref_pdf_Q{QUESTIONNAIRE}.csv"),FILENAME_COL, ID_COL,USED_COL,WARNING_ORDERING_COL_NAME,WARNING_CENSORING_COL_NAME)
        create_folder(updated_csv_paths, parents=True, exist_ok=True)
        df.to_csv(csv_modified_path)
    
    
    #for testing i am using used_col in the opposite way 
    #df = load_preprocessed_df(csv_modified_path,used_col_name=USED_COL,id_col_name=ID_COL) #load the preprocessed df
    #file_logger.write(df.head(10).to_string()) #log the first 10 lines of the df to check it is correct
    df=pd.read_csv(csv_modified_path)
    #select only the lines with used=False
    df = df[df[USED_COL]==True] 
    print(f"Total number of unique ids to process: {df[ID_COL].nunique()}")

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)
    print("selected templates: ",selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, npy_data = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    pages_in_annotation = get_page_list(annotation_roots[0]) #is the same for both templates in Q1 case 
    #so i can just take it from the first one
    print("pages in annotations:", pages_in_annotation)

    ##### PREPARE SHARED RESOURCES #############
    shared_resources = {
        'templates_path': templates_path,
        'npy_data': npy_data,
        'annotation_roots': annotation_roots,
        'pages_in_annotation': pages_in_annotation,
        'selected_templates': selected_templates,
        # paths
        'questionnairres_log_path' : questionnairres_log_path,
        'debug_path': debug_path,
        'censored_images_path': censored_images_path,
        'updated_csv_paths': updated_csv_paths,
        #args
        'save_debug_times': save_debug_times,
        'save_debug_images': save_debug_images,
        'verbose': args.verbose,
        #loggers
        #'file_logger': file_logger,
        #'memory_logger': memory_logger,
        #'global_time_logger': global_time_logger,
        # ... add all other necessary variables/objects ...
    }

    # 2. SLURM CHUNKING LOGIC
    # Get all unique IDs
    all_unique_ids = df[ID_COL].unique()
    
    my_ids=all_unique_ids[:n_ids]
    
    # Prepare the arguments for the multiprocessing pool
    tasks = []
    for uid in my_ids:
        group = df[df[ID_COL] == uid]
        tasks.append((uid, group, shared_resources))

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Processing {len(tasks)} IDs using {cpus} CPUs")


    # Warm-up run (to ignore initialization overhead)
    '''with threadpool_limits(limits=1):
        benchmark_task()'''
    results=[]
    tests=[4]+[1,2,4,6,8,10,12,14,16]
    for num_cpus in tests:
        start_time = perf_counter()
        print("-"*50)
        print(f"CPUs: {num_cpus}")
        '''with mp.Pool(processes=num_cpus, initializer=worker_init, initargs=(log_queue,)) as pool:
            # starmap allows passing multiple arguments to the function
            parallel_results = pool.starmap(process_subject, tasks)
            # --- YOUR TASK HERE ---'''
        with mp.Pool(processes=num_cpus) as pool:
            # starmap allows passing multiple arguments to the function
            parallel_results = pool.starmap(process_subject, tasks)
        times_per_id=[parallel_results[i][0] for i in range(len(parallel_results))]
        end_time = perf_counter()
        duration = end_time - start_time
        results.append((duration,times_per_id))
        print(f"Times per ID: {times_per_id}")
        print(f"Time: {duration:.4f}s")
    print("\n" + "="*30)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*30)
    reference_time = results[1][0]
    for i,result in enumerate(results[1:]):
        cpus=tests[i+1]
        speedup = reference_time / result[0]
        efficiency = (speedup / cpus) * 100
        print(f"CPUs: {cpus:2d} | Time: {result[0]:6.2f}s | Speedup: {speedup:5.2f}x | Efficiency: {efficiency:5.1f}%")
    
    # 4. Cleanup
    #listener.stop()

    '''tot = 0
    for r in results:
        if r != 1:
            print("Error in processing a subject, check the logs for details.")
        tot+=r
    print(f"Task {chunk_idx} completed. Successfully processed {tot}/{len(tasks)} subjects.")'''

    return 0

def process_subject(unique_id, group, shared_resources):
    cv2.setNumThreads(0)

    """
    This function processes ONE subject. 
    shared_resources: A dictionary containing templates, paths, and settings 
                      loaded once at the start of the script. 
    """
    #print("Entered  process for id: ", unique_id, flush=True)
    #parallel_logger = logging.getLogger("parallel_logger")
    #parallel_logger.info(f"Started processing ID: {unique_id}")
    #parallel_logger.debug(f"Trying to print debug data for ID: {unique_id}")
    ######## PAGE SORTING ##############

    #### UNPACK shared RESOURCES ####
    pages_in_annotation = shared_resources['pages_in_annotation']
    annotation_roots = shared_resources['annotation_roots']
    npy_data = shared_resources['npy_data']
    selected_templates = shared_resources['selected_templates']
    #paths
    questionnairres_log_path = shared_resources['questionnairres_log_path']
    debug_path = shared_resources['debug_path']
    censored_images_path = shared_resources['censored_images_path']
    updated_csv_paths = shared_resources['updated_csv_paths']
    #args
    save_debug_times = shared_resources['save_debug_times']
    save_debug_images = shared_resources['save_debug_images']
    verbose = shared_resources['verbose']

    #save path
    results_for_id_save_path = os.path.join(censored_images_path, QUESTIONNAIRE, f"{unique_id}")
    #loggers
    patient_file_log_path=os.path.join(results_for_id_save_path,'global_logger.txt')
    create_folder(os.path.dirname(patient_file_log_path), parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=verbose,path=patient_file_log_path)
    #file_logger = shared_resources['file_logger']
    #global_time_logger = shared_resources['global_time_logger']
    # memory_logger = shared_resources['memory_logger']

    #INITIALIZE LOGGERS
    patient_time_log_path=os.path.join(results_for_id_save_path,time_logs_name)
    create_folder(patient_time_log_path, parents=True, exist_ok=True)
    questionnaire_time_logger=FileWriter(save_debug_times,os.path.join(patient_time_log_path,f"questionnaire_level_time_log.txt"))

    #### LOAD filenames for selected ID #####
    filenames = group[FILENAME_COL].tolist()
    # i sort the filenames by name (expected page ordering is absed on alphabetical ordering)
    filenames.sort()  
    #print("filenames: ",filenames, flush=True)
    #checks both for .pdf and for .tif.pdf
    pdf_paths = get_file_paths(filenames,questionnairres_log_path,file_logger) 
    file_logger.write(filenames)
    #file_logger.write(pdf_paths)

    #i extract the images and order them based on the expeected ordering
    #in some cases pdf_paths is a single multipage pdfs in others are multiple one page pdfs files
    questionnaire_time_logger.call_start('process_pdfs')
    list_of_images, extraction_log = process_pdf_files(QUESTIONNAIRE,pdf_paths,None,save=False, test_log={})
    questionnaire_time_logger.call_end('process_pdfs')
    pages_to_consider = [i+1 for i in range(len(list_of_images))]
    test_log = initialize_warning_log(pages_to_consider)
    test_log = test_log | extraction_log

    #DEBUG, remove after, disorder the pages 
    #new_order=[3,2,1,0]
    #list_of_images = [list_of_images[i] for i in new_order] 
    #DEBUG
    if save_debug_images:
        save_list_of_images(list_of_images, debug_path, debug_images_original_name,unique_id, verbose)

    #### GET correct template for Q1 (uses phash)#####
    #i select the annotation root and npy data corresponding to the correct template 
    # #(in Q1 case i have two templates, in the other cases only one so it is straightforward)
    questionnaire_time_logger.call_start('select_template')
    report,selected_template_index,selected_confidence, root , npy_dict = select_template(pages_in_annotation,QUESTIONNAIRE,
    annotation_roots,npy_data, list_of_images, file_logger) 
    if report:
        test_log['Choosen template'] = selected_templates[selected_template_index]
        test_log['Confidence on template choice'] = selected_confidence
        test_log['report_template_choice'] = copy.deepcopy(report)
    file_logger.write(selected_templates[selected_template_index])
    file_logger.write(report)
    questionnaire_time_logger.call_end('select_template')
    
    #### DICTIONARY INITIALIZATION ####
    questionnaire_time_logger.call_start('dictionary_initialization')
    #i should avoid to re-initialize if q1 (but i spare a negligible amount of time)
    page_dictionary,template_dictionary = initialize_sorting_dictionaries(list_of_images, root, input_from_file=False)
    #i will consider all template pages from the beginning and all images of course
    templates_to_consider = pages_in_annotation[:]
    #pre_load_template_info
    template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, 
                                                        root, template_dictionary)
    #DEBUG
    total_bytes = get_deep_size(template_dictionary)
    #memory_logger.write(f"Current memory usage: {get_process_memory():.2f} MB")
    #memory_logger.write(f"Total deep size of template_dictionary: {total_bytes/ 1024:.2f} kbytes")
    questionnaire_time_logger.call_end('dictionary_initialization')
    
    ##### CHECK if PAGES MATCH expected ORDER (can choose orb or template matchign) #########
    questionnaire_time_logger.call_start("total_sorting_time")
    questionnaire_time_logger.call_start("page_sorting_stage_1")
    test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_first_stage_check(pages_to_consider, templates_to_consider, 
                                                                                                                                        page_dictionary, template_dictionary, test_log,
                                                                                                                                        file_logger, 
                                                                                                                                        method_checking=CHECKING_FIRST_STAGE,
                                                                                                                                        orb_parameters=ORB_parameters)
    questionnaire_time_logger.call_end("page_sorting_stage_1")
    test_log['problematic_pages_step_1'] = problematic_pages[:]
    test_log['test_passed_step_1'] = test_passed
    #DEBUG
    for img_id in pages_to_consider:
        page = page_dictionary[img_id]
        shifts = page['shifts']
        file_logger.write(f"Page {img_id} has these shifts {shifts}")
    file_logger.write(f"problematic pages {problematic_pages}")
    file_logger.write(f"test passed: {test_passed}")

    if not test_passed: 
        questionnaire_time_logger.call_start("page_sorting_stage_2")
        ##### SORT PAGES (with orb or phash) AND then CHECK if PAGES MATCH expected ORDER (can choose orb or template matchign) #########
        #sort with orb matching and check if the association is correct via template matching
        #pre_load orb keypoints for images
        questionnaire_time_logger.call_start("pre_load_image_properties_stage_2")
        page_dictionary = pre_load_image_properties(problematic_pages,page_dictionary,
                                                    template_dictionary,properties=[ORDERING_METHOD_STEP_2])
        questionnaire_time_logger.call_end("pre_load_image_properties_stage_2")
        test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates = perform_second_stage_check(problematic_pages, problematic_templates, 
                                                                                                                                        page_dictionary, template_dictionary, test_log,
                                                                                                                                        file_logger, questionnaire_time_logger,
                                                                                                                                        method_checking=CHECKING_SECOND_STAGE, 
                                                                                                                                        orb_parameters=ORB_parameters,
                                                                                                                                        method_ordering=ORDERING_METHOD_STEP_2)
        test_log['problematic_pages_step_2'] = problematic_pages[:]
        test_log['test_passed_step_2'] = test_passed
        #DEBUG
        file_logger.write(f"problematic pages step 2: {problematic_pages}")
        file_logger.write(f"test passed 2: {test_passed}")
        debug_print_associations(pages_to_consider,page_dictionary,file_logger)
        questionnaire_time_logger.call_end("page_sorting_stage_2")
    
        if not test_passed:
            ###### SORT PAGES WITH OCR #########
            questionnaire_time_logger.call_start("ocr_matching")
            template_dictionary, page_dictionary, report = perform_ocr_matching(problematic_pages,problematic_templates, 
                                                    page_dictionary, template_dictionary,text_similarity_metric=TEXT_SIMILARITY_METRIC, compute_report=True,
                                                    gap_threshold=GAP_THRESHOLD_OCR, max_dist=MAX_DIST_OCR)
            #update the test log with the ocr results
            for img_id in problematic_pages:
                test_log[img_id]['matched_ocr'] = page_dictionary[img_id]['match_ocr']
                test_log[img_id]['confidences_ocr'] = page_dictionary[img_id]['confidence_template']
                #i give a warning if the match computed with ocr is  below the threshold
                test_log[img_id]['warning_ocr'] = -1*test_log[img_id]['confidences_ocr']+1>MAX_DIST_OCR #i convert back to the cost for comparing with the threshold
                #i give a warnign if the cost is over the maximum cost defined by the threshold
            test_log['report_ocr'] = copy.deepcopy(report)
            debug_print_associations(pages_to_consider,page_dictionary,file_logger)
            questionnaire_time_logger.call_end("ocr_matching")
    questionnaire_time_logger.call_end("total_sorting_time")
    
    page_times = {}
    partial_boxes_coords_path = initialize_csv(results_for_id_save_path)
    ########## PAGE CENSORING ##########
    for img_id in pages_to_consider:
        ### load image properties and ignore those non matched ######
        page = page_dictionary[img_id]
        matched_id = page['matched_page']
        test_log[img_id]['matched_template_final'] = matched_id
        if matched_id == None:
            test_log[img_id]['not_matched_ignored'] = True
            continue
        if test_log[img_id]['warning_ocr']: # i discard pages that were mathed with ocr but very very unreliably 
            #(eg case in which i have less pages than templates and one or more pages are also wrong)
            if -1*test_log[img_id]['confidences_ocr']+1>DISCARD_DIST_OCR:
                test_log[img_id]['ocr_too_low_ignored'] = True
                continue
        template = template_dictionary[matched_id]
        test_log[img_id]['template_image_resolutions_final'] = list([template['template_size'], page['img_size']])
        img_size = page['img_size']
        #template_size = template_dictionary[matched_id]['template_size']
        img=page['img'] #all pages are already loaded

        #DEBUG
        questionnaire_time_logger.call_start('total_time_page',block=True)
        #create time logger for the current page
        #image_time_logger=FileWriter(save_debug_times,os.path.join(patient_time_log_path,f"page_{matched_id}.txt"))
        #image_time_logger.call_start('total_time_page')

        #### DEAL WITH PAGES NOT TO CENSOR (and special questionnairres?) ###### 
        if template['type']=='N':
            file_logger.write(f"Page {img_id} considered as N, no censoring applied, saved as is")
            save_as_is_no_censoring(file_logger,questionnaire_time_logger,img_id,page_dictionary,dest_folder=results_for_id_save_path,n_page=matched_id,
                                    compression_level=PNG_COMPRESSION_LEVEL)
            test_log[img_id]['not_to_censor_ignored'] = True
            questionnaire_time_logger.write(f"Page {matched_id} considered as N, no censoring applied, saved as is")
            dtime = questionnaire_time_logger.call_end('total_time_page')
            page_times[matched_id] = dtime
            continue
        
        #### load pre_computed data and boxes (and rescale them according to tempalte and image resolutions #######
        pre_computed = npy_dict[matched_id]
        roi_boxes, pre_computed_rois = get_roi_boxes(root,pre_computed,matched_id) #first one is the extra roi, second one is the blank
        #prev_values = check_blank_and_extra(roi_boxes, pre_computed_rois, page, img_size) #uncomment if you want to use the roi for template matching and the blank for checking
        align_boxes=template['align_boxes'][:]
        pre_computed_align = template['pre_computed_align'][:]
        # I preprocess the censor regions to extend their dimensions to page limits
        censor_boxes,partial_coverage = get_censor_boxes(root,matched_id) #we need to refer to the correct id of the template
        censor_close_boxes,_ = get_censor_close_boxes(root,matched_id)


        #DEBUG (superimposed images)
        if save_debug_images:
            img2_path = os.path.join(template_images_path, f"q_{QUESTIONNAIRE}",f"page_{matched_id}.png")
            output_path = os.path.join(debug_path, unique_id, f"{QUESTIONNAIRE}", debug_images_superimposed_name,f"page_{matched_id}.png")
            create_folder(os.path.dirname(output_path), parents=True, exist_ok=True)
            superimpose_images(img, img2_path, output_path,file_logger)

        ##### ADJUST LARGE CENSOR BOXES TO PAGE LIMITS ######
        questionnaire_time_logger.call_start('adjust_censor_boxes_to_boundaries')
        if save_debug_images:
            rescaled_censor_boxes = rescale_box_coords_given_resolutions(censor_boxes, template['template_size'], img_size)
            adjusted_censor_boxes = adjust_boundary_boxes(rescaled_censor_boxes, template['template_size'], img_size , epsilon=EPSILON_EDGE_MATCHING)
            output_path = os.path.join(debug_path, unique_id, f"{QUESTIONNAIRE}", debug_images_adjusted_boxes,f"page_{matched_id}.png")
            create_folder(os.path.dirname(output_path), parents=True, exist_ok=True)
            save_these_boxes(output_path,img,[censor_boxes,adjusted_censor_boxes],list_of_colors=['red','green'])
        #i also rescale to thecorrect resolution (image scale instead of template scale)
        censor_boxes = rescale_box_coords_given_resolutions(censor_boxes, template['template_size'], img_size)
        censor_boxes = adjust_boundary_boxes(censor_boxes, template['template_size'], img_size , epsilon=EPSILON_EDGE_MATCHING)
        questionnaire_time_logger.call_end('adjust_censor_boxes_to_boundaries')

        #### RESCALE BOXES based on resolution of image and template ##########
        selected_alignement_method = ALIGNEMENT_METHOD
        if page['shifts'] is None:#if the alignement method is pre_computed but matching with template regions
            #was not possible in the first phase i need to backup to a weaker methods
            selected_alignement_method = 'orb_page_level_homography' #as an alternative i can add a method that tries to recalculate the tempalte matches
        if selected_alignement_method not in ['orb_page_level_affine', 'orb_page_level_homography']:
            #I dn't resize the boxes if the transformation parameters are computed from the pages
            #if instead they are computed from the boxes the boxes have to be rescaled to the image resolution for computing the values
            align_boxes = rescale_box_coords_given_resolutions(align_boxes, template['template_size'], img_size)
            roi_boxes = rescale_box_coords_given_resolutions(roi_boxes, template['template_size'], img_size)
            censor_close_boxes = rescale_box_coords_given_resolutions(censor_close_boxes, template['template_size'], img_size)

        ###### COMPUTE TRANSFORMATION PARAMETERS FOR ALIGNEMENT ######
        questionnaire_time_logger.call_start('compute_transformation_parameters')
        if test_log[img_id]['warning_ocr']:
            selected_alignement_method = 'orb_page_level_homography' #i force this method because in this case i cannot rely on align regions (were not matched)
            file_logger.write(f"Warning for page {img_id} because OCR matching confidence is low, alignement will be computed with orb_page_level_homography")
        test_log[img_id]['selected_alignement_method'] = selected_alignement_method
        scale_factor, shift_x, shift_y, angle_degrees,reference = get_transformation_from_dictionaries(page, template, questionnaire_time_logger, 
                                                                                                        scale_factor=SCALE_FACTOR_MATCHING, 
                                                                                                        method=selected_alignement_method,orb_parameters=ORB_parameters)#orb_page_level_affine
        transformation = {'reference': reference, 'scale_factor': scale_factor, 'shift_x': shift_x, 'shift_y': shift_y, 'angle_degrees': angle_degrees}
        questionnaire_time_logger.call_end('compute_transformation_parameters')
        
        ##### CHECK THAT COMPUTED TRANSFORMATION IS GOOD ####
        if reference:
            resize_factor_area = (img_size[0]*img_size[1])/(template['template_size'][0]*template['template_size'][1])
            flag,error = is_geometry_valid(test_w=100,test_h=100, transformation=transformation, angle_tolerance=ANGLE_TOLERANCE,
                                        resize_factor_area=resize_factor_area)
            test_log[img_id]['valid_geometry_for_transformation'] = flag
            test_log[img_id]['geometry_error'] = error
        test_log[img_id]['alignement_transformation'] = copy.deepcopy(transformation)
        # Censor iimage if alignement failed
        if (np.isscalar(scale_factor) and scale_factor == -1) or (flag==False): 
            test_log[img_id]['is_failure'] = True
            test_log[img_id]['censored_with_large_boxes'] = True
            #censor_boxes,partial_coverage = get_censor_boxes(root,matched_id) #we need to refer to the correct id of the template
            #in this case i censor with the extended regions because i cannot be sure of the alignement in any way
            save_censored_image(img, censor_boxes, results_for_id_save_path,matched_id,
                                warning='',partial_coverage=partial_coverage,
                                thickness_pct=THICKNESS_PCT, spacing_mult=SPACING_MULT,logger=questionnaire_time_logger,
                                page_w=img_size[0],page_frac=PAGE_FRAC,area_frac=AREA_FRAC, compression_level=PNG_COMPRESSION_LEVEL)   
            save_partial_regions_to_csv(censor_boxes,partial_coverage, matched_id, partial_boxes_coords_path)
            questionnaire_time_logger.write(f"Page {matched_id} considered as N, no censoring applied, saved as is")
            dtime=questionnaire_time_logger.call_end('total_time_page')
            page_times[matched_id] = dtime
            continue
        #DEBUG
        #force parameters of alingement to debug
        #scale_factor=1.0
        #angle_degrees=0.0
        #shift_x*=-1
        #shift_y*=-1
        file_logger.write(f"Alignement parameters for page {matched_id}: {transformation}")
        if save_debug_images:
            #save_w_boxes(debug_images_w_boxes_path,unique_id,QUESTIONNAIRE,matched_id,img,root,
                #            pre_computed,questionnaire_time_logger,which_boxes=['align','roi','censor','censor_close'], transformation=None)
            output_path = os.path.join(debug_path, unique_id, f"{QUESTIONNAIRE}", debug_images_w_boxes_name,f"page_{matched_id}.png")
            create_folder(os.path.dirname(output_path), parents=True, exist_ok=True)
            save_w_boxes(output_path,matched_id,img,root,
                            pre_computed,questionnaire_time_logger,which_boxes=['align','roi','censor','censor_close','transformed'], transformation=transformation)


        #### RESCALE CLOSE CENSOR BOXES BASED ON ALIGNEMENT OF THE ROI, for security (with orb or template matching) ######
        questionnaire_time_logger.call_start('rescale_using_roi')
        #i need to give the rescale parameters because rescaling relies on template matching -> templates have to be resized
        rescale_x_y=(template['template_size'][0]/img_size[0], template['template_size'][1]/img_size[1]) 
        is_match, extra_shift_x, extra_shift_y, extra_scale, extra_angle = rescale_censor_with_alignement(img,img_size,align_boxes,roi_boxes,pre_computed_align,pre_computed_rois,
                                transformation,questionnaire_time_logger,rescale_censor_with=RESCALE_CENSOR_WITH,rescale_x_y=rescale_x_y)
        extra_transformation = {'extra_shift_x': extra_shift_x, 'extra_shift_y': extra_shift_y, 'extra_scale': extra_scale, 'extra_angle': extra_angle}
        test_log[img_id]['adjust_transformation'] = copy.deepcopy(extra_transformation)
        #DEBUG
        file_logger.write(f"Alignement parameters after orb matching for page {matched_id}: shift_x: {extra_shift_x}, shift_y: {extra_shift_y}, Match: {is_match}")
        questionnaire_time_logger.call_end('rescale_using_roi')


        ###### CENSOR IMAGES ######   
        questionnaire_time_logger.call_start('censoring')
        test_log = censor_the_page(is_match, transformation, extra_transformation, censor_boxes, censor_close_boxes, partial_coverage, 
                questionnaire_time_logger, save_debug_images, debug_path, 
                unique_id, QUESTIONNAIRE, matched_id, img, results_for_id_save_path, test_log=test_log, warning_string='', debug_images_name=debug_images_name, 
                page_w=img_size[0],page_frac=PAGE_FRAC,area_frac=AREA_FRAC, partial_boxes_coords_path = partial_boxes_coords_path,
                compression_level=PNG_COMPRESSION_LEVEL) # i don't add warning strings to pages
        questionnaire_time_logger.call_end('censoring')
        questionnaire_time_logger.write(f"Page {matched_id} considered as N, no censoring applied, saved as is")
        dtime=questionnaire_time_logger.call_end('total_time_page')
        page_times[matched_id] = dtime
    #save global and page level warning
    save_warning_log(test_log, results_for_id_save_path)
    group = update_warning_cols(group,unique_id,test_log,pages_to_consider,id_col=ID_COL,ordering_warning_col=WARNING_ORDERING_COL_NAME,
                                censoring_warning_col=WARNING_CENSORING_COL_NAME)
    
    if SAVE_ARCHIVED:
        archive_result = archive_one(results_for_id_save_path, overwrite=True)
        #debug archiviation failures by returning the output of the archive function
        # 4. Summary for this specific node
        archive_flag = False if archive_result.startswith("DONE") else True 
        if archive_flag:
            raise Exception(f"Archiving failed for ID {unique_id}: {archive_result}")

    # Save a partial CSV for THIS chunk
    '''chunk_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    output_csv = os.path.join(updated_csv_paths,partial_results_name,f"partial_results_{chunk_id}.csv")
    create_folder(os.path.dirname(output_csv), parents=True, exist_ok=True)
    group.to_csv(output_csv)'''
    total_time=questionnaire_time_logger.call_end('complete_process')

    return group.copy()


def archive_one(file_path, overwrite=True):
    """ 
    Archives a folder and replaces the original folder with the .tar file 
    in the same directory. 
    """
    folder_name = os.path.basename(file_path)
    src_root = os.path.dirname(file_path)

    folder_name = str(folder_name).strip()
    if not folder_name or folder_name == 'nan':
        return "ERROR: Empty ID"
    
    # Pathing: everything happens within src_root
    src_path = os.path.join(src_root, folder_name)
    final_tar = os.path.join(src_root, f"{folder_name}.tar")
    temp_tar = os.path.join(src_root, f"{folder_name}.tar.tmp")

    # 1. Validation & Skip Logic
    if os.path.exists(final_tar) and not overwrite:
        return f"SKIP: Archive already exists for {folder_name}"

    if not os.path.isdir(src_path):
        return f"ERROR: Source {folder_name} not found"

    try:
        # 2. Create the archive as a temporary file first
        with tarfile.open(temp_tar, "w") as tar:
            tar.add(src_path, arcname=folder_name)
        
        # 3. Substitution Logic (The "Swap")
        # First, remove the original directory
        shutil.rmtree(src_path)
        
        # Then, rename the temp archive to the final name
        os.rename(temp_tar, final_tar)
        
        return f"DONE: {folder_name} replaced by archive"

    except Exception as e:
        # Cleanup: if archiving failed, remove the partial temp file
        # Note: shutil.rmtree hasn't been called yet if we hit this, 
        # so the original data remains safe.
        if os.path.exists(temp_tar):
            os.remove(temp_tar)
        return f"FAILED: {folder_name} -> {str(e)}"

########## INDIVIDUAL FUNCTIONS #################


def debug_print_associations(pages_to_consider,page_dictionary,logger):
    for img_id in pages_to_consider:
        page = page_dictionary[img_id]
        matched_id = page['matched_page']
        logger.write(f"Page {img_id} is matched with template page {matched_id}")
    logger.write("-"*50 + "\n")
    return 

def initialize_result_df(pdf_names,template_type):
    if template_type == "Q10":
        #add code to modify the list
        pass
    df = pd.DataFrame(pdf_names, columns=['pdf_name'])
    # 3. Add two empty columns
    # Using None or np.nan is standard for "empty" data
    df['saved_file_name'] = pdf_names
    df['id'] = None
    df['matched_template_page'] = None
    return df

def preprocess_df(source_path,filename_col,id_col,used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring'):
    df=pd.read_csv(source_path)
    log_filename = 'log_'+get_basename(source_path,remove_extension=True).split('_')[-1]
    log_path = os.path.join(os.path.dirname(source_path),log_filename+'.txt')
    def log(message):
        # 'a' (append) creates the file if it doesn't exist
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(str(message) + "\n")

    # 1. Drop lines with at least one missing values
    df = df.dropna()

    # 2. Remove file extensions from filenames (remove everything after the first point)
    df[filename_col] = df[filename_col].str.rsplit('.', n=1).str[0]

    # 3. Remove lines with filenames that are associated with more than one ID
    fname_id_counts = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = fname_id_counts[fname_id_counts > 1].index.tolist()
    df = df[~df[filename_col].isin(multi_id_filenames)]
    log(f"Removed these filenames because associated to multiple ids: {multi_id_filenames}")
    
    length_before = len(df)
    df.drop_duplicates(inplace=True) #by default it keep the first occurrence
    length_after = len(df)
    log(f"Before eliminating duplicates the row length is {length_before} after it is {length_after} -> {length_before-length_after} rows were eliminated")

    df = df.sort_values(id_col).reset_index(drop=True)
    #add columns
    df[used_col_name] = False
    df[warning_ordering_col_name] =''
    df[warning_censoring_col_name] =''

    return df

def load_preprocessed_df(file_path,id_subset = None, used_col_name='Used',id_col_name='e3n_id_hand'):
    df=pd.read_csv(file_path)
    #select only the lines with used=False
    df = df[df[used_col_name]==False] 
    #Select a subset of ids if provided (eg PD subjects)
    if id_subset is not None:
        df = df[df[id_col_name].isin(id_subset)]

    return df


def get_file_paths(filenames,pdf_load_path,logger):
    file_paths = []

    def get_filepath(path):
        if os.path.exists(path):
            return path
        else:
            #try adding .tif before the extension 
            # (eg if the path is /folder/doc_5_page_1.pdf it will try /folder/doc_5_page_1.tif.pdf)
            base, ext = os.path.splitext(path)
            new_path = base + '.tif' + ext
            if os.path.exists(new_path):
                return new_path
            else:
                return logger.write(f"Neither {path} nor {new_path} exist.")

    for filename in filenames:
        try:
            # check both for file.pdf and for file.tif.pdf
            file_path = get_filepath(os.path.join(pdf_load_path, filename+'.pdf'))
            file_paths.append(file_path)
        except FileNotFoundError as e:
            logger.warning(str(e))
            continue
    return file_paths

def select_specific_annotation_file(questionnaire):
    #i will select only one annotation file from the library
    if questionnaire in [f"{i}" for i in range(2,14)]:
        selected_templates = [f"q_{questionnaire}"]
    elif questionnaire == "1":
        selected_templates = ["q_1","q_1v2"]
    return selected_templates

def select_template(pages_in_annotation,questionnaire,annotation_roots,npy_data, list_of_images, logger):
    if questionnaire == "1":
        #i have two templates for Q1 so i will select the one that has better matches with the images
        #i will do this by performing a simple phash matching between the images 
        # and the templates and selecting the template with more matches
        templates_to_consider = pages_in_annotation[:]
        pages_to_consider = [i+1 for i in range(len(list_of_images))]

        #i load image information
        page_dictionary = initialize_page_dictionary(list_of_images,input_from_file=False)

        #set the cost to infinity
        max_cost = float('inf')
        selected_confidence = 0

        for i in range(len(annotation_roots)):
            root, npy_dict = annotation_roots[i], npy_data[i]

            #initialize the template dictionaries i will use to store info on the sorting process
            template_dictionary = initialize_template_dictionary(root)

            #pre_load_template_info
            template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, 
                                                              root, template_dictionary)
            
            #pre_load phash for images, i can do only once but i have to do after i have the template
            #dictionary because it contains info on page pre_processing for phash
            if i==0:
                page_dictionary = pre_load_image_properties(pages_to_consider,page_dictionary,
                                                            template_dictionary,properties=['phash'])
            
            #perform phash matching
            page_dictionary, report = perform_phash_matching(page_dictionary,template_dictionary, templates_to_consider, templates_to_consider, 
                            gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH, compute_report=True)
            total_cost = report['total_cost']

            if total_cost < max_cost:
                selected_template_index = i
                max_cost = total_cost
                selected_confidence = report["is_confident"]
        logger.write(f"Selected template: {i} with total cost {max_cost}")
        
        return report,selected_template_index,selected_confidence,annotation_roots[selected_template_index],npy_data[selected_template_index]
            
    else:
        return None,0, None, annotation_roots[0],npy_data[0]

def perform_first_stage_check(pages_to_consider, templates_to_consider, page_dictionary, template_dictionary, test_log, logger,method_checking='template',**kwargs):
    #i test if the pages are already in place
    pairs_to_consider = []
    #i need to prepare the list of pairs to check considering that the extracted pages can be less or more than the pages in the template, 
    min_length = min(len(pages_to_consider),len(templates_to_consider))
    range_to_check = range(min_length)
    for i in range_to_check: 
        img_id = pages_to_consider[i]
        matched_id = templates_to_consider[i]
        pairs_to_consider.append([img_id,matched_id])
    
    if method_checking == 'template':
        selected_metric = "matchTemplate"
    elif method_checking == 'orb':
        selected_metric = "orb"

    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, metric=selected_metric,
                                n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING,matching_threshold=MATCHING_THRESHOLD, compute_report=True,**kwargs)
    
    problematic_pages = pages_to_consider[:]
    problematic_templates = templates_to_consider[:]
    n_matches=0 
    for i in range_to_check:
        img_id = pages_to_consider[i]
        t_id = templates_to_consider[i]
        matched_id_template = page_dictionary[img_id]['matched_page']
        logger.write(f"Checking page {img_id} against template {t_id}: matched with template {matched_id_template} with confidence {page_dictionary[img_id]['confidence_template']}")
        test_log[img_id]['confidences_template_1'] = page_dictionary[img_id]['confidence_template'][:]
        if matched_id_template: #there was match with the expected page -> remove it from the problematic list
            n_matches+=1
            problematic_pages.remove(img_id)
            problematic_templates.remove(t_id)
            test_log[img_id]['matched_template_step_1'] = matched_id_template
        else:
            test_log[img_id]['failed_test_1'] = True 
        test_log[img_id]['template_page_resolutions_step_1'] = list([template_dictionary[t_id]['template_size'], page_dictionary[img_id]['img_size']])
    test_passed=False
    if n_matches==min_length:
        test_passed=True
    
    return test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates

#probably i can make more modular by making the "check_with_template_matching" a function that can be used after phash or after orb or others ...
def perform_second_stage_check(pages_to_consider, templates_to_consider, page_dictionary, template_dictionary, test_log,logger,time_logger,method_checking='template',method_ordering='orb',**kwargs):
    
    if method_checking == 'template':
        selected_metric = "matchTemplate"
    elif method_checking == 'orb':
        selected_metric = "orb"

    if method_ordering == 'orb':
        key_to_check = 'match_orb'
    elif method_ordering == 'phash':
        key_to_check = 'match_phash'

    time_logger.call_start(f"sorting_step_2_{method_ordering}")
    if method_ordering == 'orb':
        page_dictionary, report = perform_orb_matching(page_dictionary,template_dictionary, pages_to_consider, templates_to_consider, 
                                gap_threshold=GAP_THRESHOLD_ORB,max_dist=MAX_DIST_ORB, orb_good_match=ORB_GOOD_MATCH,compute_report=True)
        test_log['report_sorting_step_2'] = copy.deepcopy(report)
    elif method_ordering == 'phash':
        page_dictionary, report = perform_phash_matching(page_dictionary,template_dictionary, pages_to_consider, templates_to_consider, 
                            gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH, compute_report=True)
        test_log['report_sorting_step_2'] = copy.deepcopy(report)
    time_logger.call_end(f"sorting_step_2_{method_ordering}")

    #i test if the pages are already in place
    pairs_to_consider = []
    for img_id in pages_to_consider: 
        if page_dictionary[img_id][key_to_check] is not None:
            orb_matched_id = page_dictionary[img_id][key_to_check]
            #print(orb_matched_id)
            pairs_to_consider.append([img_id,orb_matched_id])

    time_logger.call_start(f"checking_step_2_{selected_metric}")
    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING,
                                matching_threshold=MATCHING_THRESHOLD, compute_report=True,metric=selected_metric,**kwargs)
    problematic_pages = pages_to_consider[:]
    problematic_templates = templates_to_consider[:]
    n_matches=0
    for i in range(len(pairs_to_consider)):
        img_id = pairs_to_consider[i][0]
        orb_match = pairs_to_consider[i][1]
        matched_id_template = page_dictionary[img_id]['matched_page']
        logger.write(f"Checking page {img_id} against template {orb_match}: matched with template {matched_id_template} with confidence {page_dictionary[img_id]['confidence_template']}")
        test_log[img_id]['confidences_template_2'] = page_dictionary[img_id]['confidence_template'][:]
        if matched_id_template == orb_match: #there was match with the expected page -> remove it from the problematic list
            n_matches+=1
            problematic_pages.remove(img_id)
            problematic_templates.remove(orb_match)
            test_log[img_id]['matched_template_step_2'] = matched_id_template
        else:
            test_log[img_id]['failed_test_2'] = True 
        test_log[img_id]['template_page_resolutions_step_2'] = list([template_dictionary[orb_match]['template_size'], page_dictionary[img_id]['img_size']])
    test_passed=False
    if n_matches==len(pairs_to_consider):
        test_passed=True
    time_logger.call_end(f"checking_step_2_{selected_metric}")
    
    return test_passed,page_dictionary, template_dictionary, test_log, problematic_pages, problematic_templates

def check_blank_and_extra(roi_boxes, pre_computed_rois, page, img_size):
    shifts, centers,_,confidences= compute_misalignment(page['img'], roi_boxes[:1], img_size, pre_computed_rois[:1], scale_factor=SCALE_FACTOR_MATCHING,
                            matching_threshold=MATCHING_THRESHOLD, pre_computed_rois=None,return_confidences=True)
    #blank region
    f_roi = preprocess_blank_roi(page['img'], roi_boxes[-1])
    decision, black_diff_to_template,cc_difference_to_template = roi_blank_decision(f_roi,pre_computed_roi=pre_computed_rois[-1], return_features = True ,
                                                                                    n_black_thresh=N_BLACK_THRESH,threshold_test=BLANK_REGION_TESTING_THRESHOLD)
    prev_values = {'shifts': shifts, 'centers': centers, 'confidences': confidences, 
                    'black_diff_to_template': black_diff_to_template, 'cc_difference_to_template': cc_difference_to_template}
    return prev_values
        
def adjust_close_censor(censor_close_boxes,orb_shift_x, orb_shift_y, orb_scale, orb_angle,is_aligned=True):
    def transform_box(box, scale, shift_x, shift_y):
        #i suppose the box is saved as [x_tl, y_tl, x_br, y_br]
        x1,y1,x2,y2 = box
        center_x = x1 + (x2-x1)/2
        center_y = y1 + (y2-y1)/2
        w = x2-x1
        h = y2-y1

        if scale>1:
            w *= scale
            h *= scale
            x1 = center_x - w/2
            x2 = center_x + w/2
            y1 = center_y - h/2
            y2 = center_y + h/2
        
        # Shift X: Affects 'right' or 'left' side of the box
        if shift_x >= 0:
            x2 += shift_x
        elif shift_x < 0:
            x1 += shift_x
        
        # Shift Y: Affects 'bottom' or 'top' side of the box
        if shift_y >= 0:
            y2 += shift_y
        elif shift_y < 0:
            y1 += shift_y
        
        return [int(x1), int(y1), int(x2), int(y2)]
    def transform_polygon(box, scale, shift_x, shift_y):
        pts=box.copy() #without this line the censor_close_boxes list is modified in place
        #i suppose the box is saved as a polygon: points in clockwise order starting from the top left corner (pt0, pt1, pt2, pt3)
        # 1. Scaling around the center
        center = np.mean(pts, axis=0)

        if scale>1:
            pts = center + (pts - center) * scale
        
        # 2. Define Local Axes (Direction vectors)
        # Unit vector along the 'width' (from pt0 to pt1)
        v_w = pts[1] - pts[0]
        dist_w = np.linalg.norm(v_w)
        u_w = v_w / dist_w
        
        # Unit vector along the 'height' (from pt0 to pt3)
        v_h = pts[3] - pts[0]
        dist_h = np.linalg.norm(v_h)
        u_h = v_h / dist_h

        # 3. Apply Shifts
        # Shift X: Affects 'right' (pts 1,2) or 'left' (pts 0,3)
        if shift_x > 0:
            pts[1] += u_w * shift_x
            pts[2] += u_w * shift_x
        elif shift_x < 0:
            pts[0] += u_w * shift_x # shift_x is negative, so it moves 'backwards'
            pts[3] += u_w * shift_x

        # Shift Y: Affects 'bottom' (pts 2,3) or 'top' (pts 0,1)
        # Note: Logic depends on if your Y-axis grows 'down' (images) or 'up' (math)
        if shift_y > 0:
            pts[2] += u_h * shift_y
            pts[3] += u_h * shift_y
        elif shift_y < 0:
            pts[0] += u_h * shift_y
            pts[1] += u_h * shift_y

        return pts
    if isinstance(orb_shift_x,list) and len(orb_shift_x)>1:
        shift_x = [max(orb_shift_x), min(orb_shift_x)]
        shift_y = [max(orb_shift_y), min(orb_shift_y)]
    else:
        shift_x = [orb_shift_x]
        shift_y = [orb_shift_y]
    new_censor_close_boxes = []
    for i in range(len(censor_close_boxes)):
        source_box = censor_close_boxes[i]
        for j in range(len(shift_x)):
            if is_aligned:
                source_box = transform_box(source_box, orb_scale, shift_x[j], shift_y[j])
            else:
                new_censor_close_boxes.append(transform_polygon(censor_close_boxes[i], orb_scale, orb_shift_x, orb_shift_y) )
        new_censor_close_boxes.append(source_box)
    
    return new_censor_close_boxes

def rescale_censor_with_alignement(img,img_size,align_boxes,roi_boxes,pre_computed_align,pre_computed_rois,
                                   transformation,image_time_logger,rescale_censor_with='template_align_and_extra',rescale_x_y=None):

    shift_x=transformation['shift_x']
    shift_y=transformation['shift_y']
    reference=transformation['reference']
    scale_factor=transformation['scale_factor']
    angle_degrees=transformation['angle_degrees']

    if rescale_censor_with == 'orb_extra':
        new_roi_boxes = apply_transformation_to_boxes(roi_boxes, image_time_logger, reference, scale_factor, 
                                                            shift_x, shift_y, angle_degrees,name='roi',option=TRANSFORMATION_OPTION) #you can set no_rotation
        #new_values = check_blank_and_extra(new_roi_boxes, pre_computed_rois, page, img_size) #uncomment if you want to use the roi for template matching and the blank for checking
        # I check alignement on the extra roi with orb matching
        is_match, n_matches_orb,extra_shift_x, extra_shift_y, extra_scale, extra_angle = orb_matching(img,new_roi_boxes[0],pre_computed_rois[0], shift_wr_tl = (0,0),
                                                                                            top_n_matches=ORB_top_n_matches, match_threshold=ORB_match_threshold)
    
    elif rescale_censor_with == 'template_align_and_extra':
        check_boxes = align_boxes + [roi_boxes[0]] 
        check_pre_computed = pre_computed_align + [pre_computed_rois[0]]
        check_boxes = apply_transformation_to_boxes(check_boxes, image_time_logger, reference, scale_factor, 
                                                            shift_x, shift_y, angle_degrees,name='check',option=TRANSFORMATION_OPTION)
        #axis align
        check_boxes = convert_to_axis_aligned_box(check_boxes)
        #print(check_boxes)
        check_shifts, check_centers,_,check_confidences= compute_misalignment(img, check_boxes, img_size, check_pre_computed, scale_factor=SCALE_FACTOR_RESIZING,
                    matching_threshold=MATCHING_THRESHOLD, pre_computed_rois=None,return_confidences=True,metric='matchTemplate',rescale_x_y=rescale_x_y)
        extra_shift_x = [shift[0] for shift in check_shifts]
        extra_shift_y = [shift[1] for shift in check_shifts]
        extra_angle = 0.0
        extra_scale = 1.0
        is_match=True
        if len(check_shifts) < N_ALIGN_REGIONS: #or N_ALIGN_REGIONS+1 ??
            is_match=False
        #i take the largest and smallest values for shift_x and shift_y
    elif rescale_censor_with == 'skip_assume_aligned':
        is_match = True
        extra_shift_x = 0
        extra_shift_y = 0
        extra_scale = 1.0
        extra_angle = 0.0
    elif rescale_censor_with == 'skip_assume_misaligned':
        is_match = True
        extra_shift_x = 0
        extra_shift_y = 0
        extra_scale = 1.0
        extra_angle = 0.0
    return is_match, extra_shift_x, extra_shift_y, extra_scale, extra_angle

def censor_the_page(is_match, transformation, extra_transformation, censor_boxes, censor_close_boxes, partial_coverage, 
                    image_time_logger, save_debug_images, debug_path, 
                    unique_id, QUESTIONNAIRE, matched_id, img, censored_images_path, test_log, warning_string, debug_images_name,partial_boxes_coords_path,**kwargs):

    shift_x=transformation['shift_x']
    shift_y=transformation['shift_y']
    reference=transformation['reference']
    scale_factor=transformation['scale_factor']
    angle_degrees=transformation['angle_degrees']

    extra_shift_x=extra_transformation['extra_shift_x']
    extra_shift_y=extra_transformation['extra_shift_y']
    extra_scale=extra_transformation['extra_scale']
    extra_angle=extra_transformation['extra_angle']

    if is_match:
        #i create an extanded list with all the censor and censor-close boxes in superposition of more than 0.
        #i rescale the dimensions of the censor-close boxes
        new_censor_close_boxes = apply_transformation_to_boxes(censor_close_boxes, image_time_logger, reference, scale_factor, 
                                                            shift_x, shift_y, angle_degrees,name='censor_close', option=TRANSFORMATION_OPTION)
        # i can convert back to axis aligned rectangles -> covers more space
        if FORCE_AXIS_ALIGNED_BOXES:
            new_censor_close_boxes = convert_to_axis_aligned_box(new_censor_close_boxes)
        #these two operations have to be performed before associating with the censor boxes because before conversion censor and censor-close don't have
        #the same scale in general
        
        #i associate each close box with the container box and create an ordered list of the containers that match the censoring boxes
        #map_to_container = map_to_smallest_containing(censor_boxes,censor_close_boxes)
        censor_boxes,new_censor_close_boxes, partial_coverage = map_to_all_containing(censor_boxes,new_censor_close_boxes,
                                                                                    partial_coverage, percentage_threshold=0.5) #to deal with multiple censor boxes for one censor-close and vicevers
        
        # I enlarge close-censor regions based on alignement results 
        # #if i am in one of two skip modes the transformation is actually an identity
        censor_close_boxes_orb = adjust_close_censor(new_censor_close_boxes,extra_shift_x, extra_shift_y, 
                                                        extra_scale, extra_angle, is_aligned=FORCE_AXIS_ALIGNED_BOXES)
        if save_debug_images:
            output_path = os.path.join(debug_path, unique_id, f"{QUESTIONNAIRE}", debug_images_name+'RE_adjusted_boxes',f"page_{matched_id}.png")
            create_folder(os.path.dirname(output_path), parents=True, exist_ok=True)
            save_these_boxes(output_path,img,[new_censor_close_boxes,censor_close_boxes_orb,censor_boxes],list_of_colors=['red','green','black'])
        

        # apply censoring considering the boundary boxes and the close censor boxes
        save_censored_images_path=os.path.join(censored_images_path,f"censored_page_w{warning_string}_{matched_id}.png")#, f"censored_page_{n_page}.png")
        create_folder(os.path.dirname(save_censored_images_path), parents=True, exist_ok=True)
        censored_img = censor_image_with_boundary(img, censor_close_boxes_orb, censor_boxes, 
                                                partial_coverage=partial_coverage,logger=image_time_logger,
                                                thickness_pct=THICKNESS_PCT, spacing_mult=SPACING_MULT,**kwargs)
        clipped_boxes=get_contained_censor_boxes(censor_close_boxes_orb, censor_boxes, partial_coverage)
        save_partial_regions_to_csv(clipped_boxes,[True for _ in clipped_boxes], matched_id, partial_boxes_coords_path)
        image_time_logger and image_time_logger.call_start(f'writing_to_memory')
        compression_level = kwargs.get('compression_level', None)  # Default to maximum compression if not provided
        if compression_level:
            cv2.imwrite(save_censored_images_path, censored_img,[cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        else:
            cv2.imwrite(save_censored_images_path, censored_img)
        image_time_logger and image_time_logger.call_end(f'writing_to_memory')
    else:
        test_log['censored_with_large_boxes']=True
        # i censor considering the large regions
        save_censored_image(img, censor_boxes, censored_images_path,matched_id,
                            warning=warning_string,partial_coverage=partial_coverage,
                            thickness_pct=THICKNESS_PCT, spacing_mult=SPACING_MULT,logger=image_time_logger,**kwargs) 
        save_partial_regions_to_csv(censor_boxes,partial_coverage, matched_id, partial_boxes_coords_path)
    return test_log

def initialize_csv(save_path):
    """Creates the file and writes the header."""
    filepath = os.path.join(save_path, "partial_boxes_coords.csv")
    create_folder(os.path.dirname(filepath), parents=True, exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'xtl', 'ytl', 'xbr', 'ybr'])
    return filepath
def save_partial_regions_to_csv(censor_boxes,partial_list, page, save_path):
    """Appends a list of boxes for a specific page."""
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i,box in enumerate(censor_boxes):
            if partial_list[i]:
                # box is expected to be [xtl, ytl, xbr, ybr]
                writer.writerow([page] + box)
def get_contained_censor_boxes(censor_close_boxes, container_boxes, partial_list):
    """
    Clips censor_close_boxes to the boundaries of container_boxes.
    
    Args:
        censor_close_boxes: List of [xtl, ytl, xbr, ybr]
        container_boxes: List of [xtl, ytl, xbr, ybr]
        
    Returns:
        List of clipped [xtl, ytl, xbr, ybr]
    """
    clipped_boxes = []

    for i in range(len(censor_close_boxes)):
        if partial_list[i]:
            # Unpack coordinates
            c_xtl, c_ytl, c_xbr, c_ybr = censor_close_boxes[i]
            b_xtl, b_ytl, b_xbr, b_ybr = container_boxes[i]

            # Clip Top-Left: Must be at least the container's TL, 
            # but no further than the container's BR
            xtl = max(b_xtl, c_xtl)
            ytl = max(b_ytl, c_ytl)

            # Clip Bottom-Right: Must be no further than the container's BR,
            # but at least the (potentially clipped) xtl
            xbr = min(c_xbr, b_xbr)
            ybr = min(c_ybr, b_ybr)

            clipped_boxes.append([xtl, ytl, xbr, ybr])

    return clipped_boxes
#### Debug ####

def logger_init(log_path):
    """The 'Listener' that stays in the main process area."""
    root = logging.getLogger()
    # Clean existing handlers to avoid duplicates
    for h in root.handlers[:]: root.removeHandler(h)
    
    h = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s [%(name)s]: %(message)s')
    h.setFormatter(fmt)
    root.addHandler(h)
    root.setLevel(logging.DEBUG)

    # This loop keeps the listener alive until it receives 'None'
    # Use this if you aren't using the 'logging.handlers.QueueListener' class
    # But for simplicity, we'll use the QueueListener class below.

def worker_init(queue):
    root = logging.getLogger()
    root.handlers = [] # Clear existing handlers
    
    # Direct everything to the queue
    handler = logging.handlers.QueueHandler(queue)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

def save_list_of_images(list_of_images, debug_images_original,folder_name, unique_id, verbose):
    if not verbose:
        return 0
    else:
        for i in range(len(list_of_images)):
            #save the images in the debug folder to check they are correctly extracted and ordered
            debug_img_path = os.path.join(debug_images_original ,f"{unique_id}",str(QUESTIONNAIRE),folder_name,f"page_{i+1}.png")
            create_folder(os.path.dirname(debug_img_path), parents=True, exist_ok=True)
            cv2.imwrite(debug_img_path, list_of_images[i])

def initialize_warning_log(pages_in_annotation):
    # I want to find the best match for the 
        # load dictionary to store warning messages on pages
        test_log = {
            # template choice
            'Choosen template': QUESTIONNAIRE,
            'Confidence on template choice': None,
            'report_template_choice': None,
            # first stage check
            'problematic_pages_step_1':None,
            'test_passed_step_1':None,
            # second stage check
            'problematic_pages_step_2':None,
            'test_passed_step_2':None,
            'report_sorting_step_2': None,
            # ocr stage results
            'report_ocr': None, 
            }
        for p in pages_in_annotation:
            test_log[p]={
                #first stage check
                'failed_test_1': False, 'matched_template_step_1': None, 'confidences_template_1': None, 'template_page_resolutions_step_1': None,
                #second stage check
                'failed_test_2': False, 'matched_template_step_2': None, 'confidences_template_2': None, 'template_page_resolutions_step_2': None,
                #ocr stage check
                'warning_ocr': False, 'matched_ocr': None, 'confidences_ocr': None,
                # final match
                'matched_template_final': None, 'template_image_resolutions_final': None ,
                #is the page ignored because it was not matched?
                'not_matched_ignored': False,
                'ocr_too_low_ignored' : False,
                #is the page saved as is because not to censor?
                'not_to_censor_ignored': False,
                # alignement calculation
                'selected_alignement_method': None, #should be the global one or homography if warning_ocr is True
                'alignement_transformation': None,
                'is_failure': False, 
                'valid_geometry_for_transformation': None,  #if false the transformation check was not passe -> (eg orb page matching failed)
                'geometry_error': None,
                # is the page censored with the large censor boxes in the end?
                'censored_with_large_boxes': False,
                # adjust transformation part
                'adjust_transformation': None
            }
        return test_log

def save_warning_log(test_log, save_path):
    create_folder(save_path, parents=True, exist_ok=True)
    log_file_path = os.path.join(save_path, "warning_log.txt")
    with open(log_file_path, 'w', encoding='utf-8') as f:
        for key, value in test_log.items():
            if isinstance(key, int):
                f.write(f"Page {key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"  {sub_key}: {sub_value}\n")
            else:
                f.write(f"{key}: {value}\n")

def update_warning_cols(df,unique_id,test_log,pages_to_consider,id_col,ordering_warning_col,censoring_warning_col):
    ordering_warning=''
    censoring_warning=''
    if test_log['test_passed_step_1']:
        ordering_warning ='First stage'
    elif test_log['test_passed_step_2']:
        ordering_warning ='Second stage'
    else:
        n_warnings = 0
        for img_id in pages_to_consider:
            if test_log[img_id]['warning_ocr']:
                n_warnings+=1
        ordering_warning = f'OCR stage, pages with warning:   {n_warnings}/{len(pages_to_consider)}'
    n_warnings=0
    for img_id in pages_to_consider:
        if test_log[img_id]['censored_with_large_boxes']:
            n_warnings+=1
    if n_warnings>0:
        censoring_warning = f'pages censored with large boxes:   {n_warnings}/{len(pages_to_consider)}'
    else:
        censoring_warning = 'No warning'
    # Ensure the warning column is capable of holding strings
    if ordering_warning_col in df.columns:
        df[ordering_warning_col] = df[ordering_warning_col].astype(object)
    if censoring_warning_col in df.columns:
        df[censoring_warning_col] = df[censoring_warning_col].astype(object)
    df.loc[df[id_col] == unique_id, ordering_warning_col] = ordering_warning
    df.loc[df[id_col] == unique_id, censoring_warning_col] = censoring_warning
    return df
def get_process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert bytes to MB

def get_deep_size(obj, seen=None):
    """Recursively finds the actual size of a python object"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_deep_size(v, seen) for v in obj.values()])
        size += sum([get_deep_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_deep_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_deep_size(i, seen) for i in obj])
        
    return size


def set_threads(n):
    n_str = str(n)
    # The "Big Five" for Math/Image libraries
    os.environ["OMP_NUM_THREADS"] = n_str
    os.environ["MKL_NUM_THREADS"] = n_str
    os.environ["OPENBLAS_NUM_THREADS"] = n_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_str
    os.environ["NUMEXPR_NUM_THREADS"] = n_str
    # Specifically for OpenCV if it's compiled with TBB
    try:
        import cv2
        cv2.setNumThreads(n)
    except ImportError:
        pass
######### PARSING #####################
def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-l", "--pdf_load_path",
        default=PDF_LOAD_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-c", "--csv_load_path",
        default=CSV_LOAD_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-t", "--templates_path",
        default=TEMPLATES_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SAVE_PATH,
        help="Directory to save the converted PNG images",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--save_debug_times",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )

    '''parser.add_argument(
        "-d",
        "--delete_previous_results",
        action="store_true",
        help="Delete the results from the prvious directories to test the pipeline from scratch",
    )'''

    parser.add_argument(
        "-i",
        "--save_debug_images",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers for parallel processing",
    )
    '''parser.add_argument(
        "--total_arrays",
        type=int,
        default=100,
        help="Number of arrays in which to split the indexes",
    )'''
    return parser.parse_args()

if __name__ == "__main__":
    main(test_size=-1,timeout_lim=180,test=False)
    #multi_threading_test()
    #multi_node_test(90)
    # A huge matrix multiplication that takes ~5-10 seconds
    '''N = 10000
    A = np.random.rand(N, N).astype('float64')
    B = np.random.rand(N, N).astype('float64')

    print("Starting 16-core burst...")
    with threadpool_limits(limits=16):
        start = perf_counter()
        _ = A @ B  # This should make 16 bars in htop go bright green
        print(f"Done in {perf_counter() - start:.2f}s")
    print("Starting 8-core burst...")
    with threadpool_limits(limits=8):
        start = perf_counter()
        _ = A @ B  # This should make 16 bars in htop go bright green
        print(f"Done in {perf_counter() - start:.2f}s")
    print("Starting 2-core burst...")
    with threadpool_limits(limits=2):
        start = perf_counter()
        _ = A @ B  # This should make 16 bars in htop go bright green
        print(f"Done in {perf_counter() - start:.2f}s")'''