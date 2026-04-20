import argparse
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import tarfile
import multiprocessing as mp

from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree
from src.utils.file_utils import list_files_with_extension, load_template_info
from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_censor_boxes, get_censor_close_boxes

from src.utils.logging import FileWriter, initialize_logger

QUESTIONNAIRE="2"
def main():
    args = parse_args()
    n_workers = args.n_workers

    log_path = "/home/a_morelli/temporary_data/other_logs/check_img_size.txt"
    file_logger=FileWriter(enabled=False,path=os.path.join("/home/a_morelli/temporary_data/other_logs/",f"global_logger.txt"))
    #main_path = "/mnt/beegfs01/scratch/a_morelli/parallel_censoring"
    censored_docs_path = "/home/a_morelli/datasets/censored_pdfs"
    templates_path="/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization/current_template"
    files_with_ids = "/home/a_morelli/datasets/pdfs/ref_pdf_final"

    questionnaire_path = os.path.join(censored_docs_path, f"{QUESTIONNAIRE}")
    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(file_logger, templates_path)
    #i select only the template of interest (eg for Q5 only doc_5.json)
    selected_templates = select_specific_annotation_file(QUESTIONNAIRE)
    #print("selected templates: ",selected_templates)

    # I open the jsons for the selected templates and save them in a list, i also open the corresponding pre_computed data
    # #this list is a single element for QX X>1 and two elements for X=1 
    annotation_roots, _ = load_template_info(file_logger,annotation_files,annotation_file_names,
                                                    templates_path, selected_files=selected_templates)
    
    root = annotation_roots[0] #i will consider only the first template for the moment, even for Q1 where i have two templates
    template_dictionary = initialize_template_dictionary(root)

    shared_resources = {
        'template_dictionary': template_dictionary,
        'questionnaire_path': questionnaire_path
    }

    
    ids_path = os.path.join(files_with_ids, f"combined_success_ids_q{QUESTIONNAIRE}.csv")
    df = pd.read_csv(ids_path)

    #get unique ids from the column e3n_id_hand and select 10 randomly
    all_unique_ids = df["e3n_id_hand"].unique()

    # Determine which IDs THIS specific Slurm task should handle
    # Usage: sbatch --array=0-199 ... (for 200 chunks)
    total_arrays = int(os.environ.get('SLURM_ARRAY_COUNT', 1))
    chunk_idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    chunk_size = len(all_unique_ids) // total_arrays
    
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < total_arrays-1 else len(all_unique_ids) #if i consider the last chinck i have to take all the remaining ids to avoid losing data
    
    my_ids = all_unique_ids[start_idx:end_idx]
    
    # Prepare the arguments for the multiprocessing pool
    tasks = []
    for uid in my_ids:
        tasks.append((uid, shared_resources))

    # 3. RUN MULTIPROCESSING
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Task {chunk_idx}: Processing {len(tasks)} IDs using {cpus} CPUs")
    
    with mp.Pool(processes=n_workers) as pool:
        # starmap allows passing multiple arguments to the function
        results = pool.starmap(process_id, tasks)
    
    total_pages_to_redo=0
    total_ids_to_redo=0
    total_ratio_over_threshold=0
    for entry,to_redo,ratio_over in results:
        total_pages_to_redo+=entry
        if to_redo:
            total_ids_to_redo+=1
        total_ratio_over_threshold+=ratio_over
    print(f"Task {chunk_idx}: Total pages to redo: {total_pages_to_redo}")
    #open the log file and write the total number of pages to redo and the total number of ids to redo
    with open(log_path, 'a') as log_file:
        log_file.write(f"Task {chunk_idx}: Total pages to redo: {total_pages_to_redo}\n")
        log_file.write(f"Task {chunk_idx}: Total IDs to redo: {total_ids_to_redo}\n")
        log_file.write(f"Task {chunk_idx}: Total ratio over threshold: {total_ratio_over_threshold}\n")
        log_file.write(f"Task {chunk_idx}: Processed IDs: {len(my_ids)}\n")
        log_file.write(f"------------------------------\n")

def process_id(id,shared_resources):
    questionnaire_path = shared_resources['questionnaire_path']
    template_dictionary = shared_resources['template_dictionary']
    tar_path = os.path.join(questionnaire_path, f"{id}.tar")

    to_redo=False
    pages_to_redo = 0
    ratio_over_threshold = 0
    with tarfile.open(tar_path, 'r:*') as tar:
        for member in tar.getmembers():
            # Process only PNG files
            if member.isfile() and member.name.lower().endswith('.png'):
                
                # 1. Process the filename
                base_name = os.path.basename(member.name)
                name_no_ext = os.path.splitext(base_name)[0]
                page_number = int(name_no_ext.split('_')[-1])

                # 2. Get dimensions without extracting to disk
                with tar.extractfile(member) as f:
                    with Image.open(f) as img:
                        width, height = img.size
                
                width_0_x = template_dictionary[page_number]['template_size'][0]
                height_0_y = template_dictionary[page_number]['template_size'][1]

                ratio_x = width_0_x / width
                ratio_y = height_0_y / height

                x_0 = template_dictionary[page_number]['worse_coords'][0]
                y_0 = template_dictionary[page_number]['worse_coords'][1]

                #print('x,y=',x_0,y_0,'width,height=',width,height,'width_0_x,height_0_y=',width_0_x,height_0_y,'ratio_x,ratio_y=',ratio_x,ratio_y)

                if x_0>width_0_x*ratio_x or y_0>height_0_y*ratio_y:
                    pages_to_redo+=1
                    #print(f"ID: {id} | Page: {page_number} | Original Dimensions: {width}x{height} | Template Dimensions: {width_0_x}x{height_0_y} | Worse Coords: ({x_0}, {y_0}) | Ratios: (x: {ratio_x:.2f}, y: {ratio_y:.2f})")
                
                if ratio_x>1.5 or ratio_y>1.5:
                    ratio_over_threshold+=1
                #print(f"ID: {page_number} | Dimensions: {width}x{height}")
    if pages_to_redo>0:
        to_redo=True
    return pages_to_redo,to_redo,ratio_over_threshold


def initialize_template_dictionary(root):
    pages_in_annotation = get_page_list(root)
    template_dictionary = {} 
    for p in pages_in_annotation:
        template_dictionary[p]={}
    #iterate on the pages in a document and initialize their parameters
    for img_id in pages_in_annotation:
        #type
        censor_type=get_censor_type(root,img_id) 
        template_dictionary[img_id]['type']=censor_type

        width,height = get_page_dimensions(root, img_id)
        template_dictionary[img_id]['template_size'] = (width,height)

        #censoring boxes
        #pre_computed = npy_dict[img_id]
        censor_boxes,_ = get_censor_boxes(root,img_id) 
        min_distance_x = float('inf')
        x_0 = 0
        min_distance_y = float('inf')
        y_0 = 0
        for box in censor_boxes:
            #get the xtl that is closest to the width of the page
            distance_x = width - box[0]
            if distance_x < min_distance_x:
                min_distance_x = distance_x
                x_0 = box[0]
            #get the ytl that is closest to the height of the page
            distance_y = height - box[1]
            if distance_y < min_distance_y:
                min_distance_y = distance_y
                y_0 = box[1]
        template_dictionary[img_id]['worse_coords']=(x_0,y_0)
    return template_dictionary

def select_specific_annotation_file(questionnaire):
    #i will select only one annotation file from the library
    if questionnaire in [f"{i}" for i in range(2,14)]:
        selected_templates = [f"q_{questionnaire}"]
    elif questionnaire == "1":
        selected_templates = ["q_1","q_1v2"]
    return selected_templates

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
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
    main()