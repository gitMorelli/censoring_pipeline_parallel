#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import json

from PIL import Image
import numpy as np 
import cv2

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import save_as_is_no_censoring, save_original_w_boxes, get_transformation_to_match_to_template, apply_transformation_to_boxes
from src.utils.censor_utils import enlarge_censor_regions, save_pre_post_boxes, save_censored_image, generate_warning_string, censor_page_base

from src.utils.debug_utils import visualize_templates_w_annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#SOURCE = "//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests" #"Z:\\vscode\\censor_e3n\\data\\q5_tests" # "C:\\Users\\andre\\VsCode\\censoring project\\data\\rimes_tests"
SOURCE = "//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests_ordering"

# thresholds
MIN_TO_CHECK_TEMPLATE = 4
THRESHOLD_MATCHING = 0.7
SCALE_FACTOR_MATCHING = 2 

GAP_THRESHOLD_PHASH = 5
MAX_DIST_PHASH = 18
TEXT_SIMILARITY_METRIC = 'similarity_jaccard_tokens'

#global vars
mode = 'cv2'
N_ALIGN_REGIONS=3 #number of align boxes used for template matching
GLOBAL_INCREASE_CENSORING = 1.2

def main():
    args = parse_args()

    # initialize the variables 
    skip_checking_1 = args.skip_checking_1
    skip_checking_2 = args.skip_checking_2 
    skip_aligning = args.skip_aligning
    save_path = args.save_path
    annotation_path = args.annotation_path
    filled_path = args.filled_path
    enlarge_censor_boxes = args.enlarge_censor_boxes
    save_debug_images = args.save_debug_images
    save_debug_times = args.save_debug_times

    #Initialize folders: remove files and folders to generate, add time_logs folder if needed
    remove_folder(save_path)
    if save_debug_images : remove_folder(os.path.join(SOURCE,'debug'))
    if save_debug_times : remove_folder(os.path.join(SOURCE,'time_logs'))
    log_path=os.path.join(SOURCE,'time_logs')
    create_folder(log_path, parents=True, exist_ok=True)

    #initialize error logger and global time logger
    initialize_logger(args.verbose,logger)
    global_time_logger=FileWriter(save_debug_times,
                                    os.path.join(log_path,f"global_time_logger.txt"))

    #check for mistakes in the filenames (you can trasform in a single logging function)
    logger.debug("Output folder: %s", save_path)
    logger.debug("Annotation folder: %s", annotation_path)
    logger.debug("Filled folder: %s", filled_path)
    logger.debug("Skip checking %s, Skip aligning %s",skip_checking_1,skip_checking_2,skip_aligning)

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(logger, annotation_path)
    # I open the json and save them in a list (ordered as annotation_files), i also open the pre_computed data from the npy files
    annotation_roots, npy_data = load_template_info(logger,annotation_files,annotation_file_names,annotation_path)
    # M: add a function that stores these objects in a dictionary that has the questionnnaire id as key (eg Q_5, Q_6, ..)
    # ---> HERE

    # load subjects tree
    warning_map, filled_folder_names, filled_folders = load_subjects_tree(logger, filled_path) 

    # i iterate on the filled_folders (study subjects)
    for j, filled_folder in enumerate(filled_folders): #subject level
        warning_map[j]=[[] for _ in range(len(annotation_files))]
        subj_id=filled_folder_names[j]
        #I load the documents for the ith subject
        documents = list_subfolders(filled_folder, recursive=False) # the document paths for the jth subject
        documents_folder_names = [get_basename(p, remove_extension=False) for p in documents]
        logger.debug("Document folder names for subject %s: %s", j, documents_folder_names)

        # I match them with the annotation file names (will be a more complex function, in this test the names are the same)
        #check that names match
        if check_name_matching(annotation_file_names, documents_folder_names, logger) == 1:
            logger.error(f"Mismatch between annotation files and document folders for subject {j}. Exiting.")
            return 1
        #check that they are sorted in the same way
        assert annotation_file_names == documents_folder_names, "Annotation files and documents folders are not in the same order."
        
        print(f"considering subject {subj_id}")
        #i can access them by index since they are sorted in the same way
        for i, annotation_file in enumerate(annotation_files): #document level

            doc_path = documents[i] #the file path for the ith document of the jth subject
            doc_files = list_files_with_extension(doc_path, ['png','tif'], recursive=False)
            sorted_files = sort_files_by_page_number(doc_files)

            #load the json file
            root = annotation_roots[i]
            npy_dict = npy_data[i]
            pages_in_annotation = get_page_list(root)

            page_dictionary,template_dictionary, test_log = ordering_scheme_base(pages_in_annotation, root, sorted_files, npy_dict, 
                                                                                 N_ALIGN_REGIONS, SCALE_FACTOR_MATCHING, GAP_THRESHOLD_PHASH,
                                                                                 MAX_DIST_PHASH, TEXT_SIMILARITY_METRIC, mode=mode)

            #show summary of results to the user
            for t_id in pages_in_annotation:
                if page_dictionary[t_id]['type']!='N':
                    #warning_map[j][i][img_id]['actual_position']=page_dictionary[img_id]['matched_page']
                    #warning_map[j][i][img_id]['was_moved'] = (page_dictionary[img_id]['matched_page'] == img_id)
                    print(f"template {t_id} is matched to page {template_dictionary[t_id]['final_match']}, and log is {test_log[t_id]}")
            print("\n \n","--"*50)
            #if the test is passed they are orthered correctly -> i match with corresponding index 
            
            #at this stage I have ordered the pages in the best possible way and identified the documents for which 
            # i had to re shuffle and for which i am not sure of the re-ordering
            for img_id in pages_in_annotation:

                log_path=os.path.join(SOURCE,'time_logs', f"patient_{subj_id}", f"document_{i}")
                create_folder(log_path, parents=True, exist_ok=True)
                image_time_logger=FileWriter(save_debug_times,
                                             os.path.join(log_path,f"time_logger_page_{img_id}.txt"))
                
                img, censor_boxes, partial_coverage, decision_1, decision_2 = censor_page_base(page_dictionary, img_id, root, npy_dict,logger, 
                                            image_time_logger, save_path, subj_id, i, 
                                            skip_checking_1, skip_checking_2, save_debug_images, skip_aligning,enlarge_censor_boxes, 
                                            GLOBAL_INCREASE_CENSORING, SOURCE, mode='csv')

                #save the censored image
                warning_string=generate_warning_string(decision_1,decision_2,test_log,img_id)
                warning_map[j][i]=test_log.copy()
                save_censored_image(img, censor_boxes, save_path,subj_id,i,img_id,
                                    warning=warning_string,partial_coverage=partial_coverage,
                                    thickness_pct=0.2, spacing_mult=0.5,logger=image_time_logger)
                image_time_logger.call_end('censoring',block=True)
                image_time_logger.call_end('complete_process',block=True) 
    #save warning_map as npy file
    '''warning_map_path=os.path.join(save_path, "warning_map.npy")
    np.save(warning_map_path, np.array(warning_map, dtype=object))'''
    # warning_map = np.load(warning_map_path, allow_pickle=True)'''

    #debug
    visualize_templates_w_annotations(annotation_files,annotation_roots,npy_data,SOURCE,align=True,censor=False,roi=False, mode=mode)

    logger.debug("Warning map: %s", warning_map)
    logger.info("Conversion finished")
    global_time_logger.call_end('complete_process')
    return 0



# main blocks --------------------------------------------------------------------------------

# parsing
def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-a", "--annotation_path",
        default=SOURCE+"\\annotazioni",
        help="Directory with the annotation files from cvat for each image",
    )
    parser.add_argument(
        "-f", "--filled_path",
        #default=SOURCE+"\\filled\\rimes",
        default=SOURCE+"\\filled",
        help="Directory with the files to censor",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SOURCE+"\\censored",
        help="Directory where I save the final censored files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--skip_checking_1",
        action="store_true",
        help="Skip checking for matching annotation and numpy files",
    )
    parser.add_argument(
        "--skip_checking_2",
        action="store_true",
        help="Skip checking for matching annotation and numpy files",
    )
    parser.add_argument(
        "--skip_aligning",
        action="store_true",
        help="Skip alignment for matching annotation and numpy files",
    )
    parser.add_argument(
        "--enlarge_censor_boxes",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )

    parser.add_argument(
        "--save_debug_times",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()