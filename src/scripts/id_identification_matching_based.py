#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path

import pandas as pd

from src.utils.convert_utils import process_pdf_files
#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_files_with_extension, load_template_info
from src.utils.file_utils import get_basename, create_folder, remove_folder, load_annotation_tree

from src.utils.json_parsing import get_page_list
from src.utils.json_parsing import get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.matching_utils import pre_load_image_properties
from src.utils.matching_utils import initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import map_to_smallest_containing

from src.utils.debug_utils import visualize_templates_w_annotations


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOAD_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival"#additional"#100263_template"
TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\current_template"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\results"#additional"#100263_template" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-l", "--load_path",
        default=LOAD_PATH,
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
    return parser.parse_args()

QUESTIONNAIRE = "Q5"
N_ALIGN_REGIONS = 2 #minimum number of align boxes needed for matching
SCALE_FACTOR_MATCHING = 2 
GAP_THRESHOLD_PHASH = 5
MAX_DIST_PHASH = 18
TEXT_SIMILARITY_METRIC = 'similarity_jaccard_tokens'

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.templates_path
    load_path = args.load_path
    save_path = args.save_path
    pdf_paths = list_files_with_extension(load_path, ["pdf",'tif'], recursive=False)
    pdf_names = [get_basename(p, remove_extension=True) for p in pdf_paths]

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(logger, templates_path)
    #i will select only one annotation file from the library
    if QUESTIONNAIRE == "Q5":
        selected_templates = ["doc_5"]
    elif QUESTIONNAIRE == "Q10":
        selected_templates = ["doc_10"]
    # I open the json and save it in a (one element) list, i also open the corresponding pre_computed data 
    annotation_roots, npy_data = load_template_info(logger,annotation_files,annotation_file_names,templates_path, selected_files=selected_templates)
    #load the json file
    root = annotation_roots[0]
    npy_dict = npy_data[0]
    pages_in_annotation = get_page_list(root)

    '''#i create a pd dataframe for storing the exected results and the actual result
    extraction_results = initialize_result_df(pdf_names,QUESTIONNAIRE)'''

    data_rows = []
    for i,pdf_path in enumerate(pdf_paths): #iterate on the folders Q_1,Q_2,..
        groups=[
            ["Q5"],
            ["Q10"],
            []
        ]
        list_of_images = process_pdf_files(QUESTIONNAIRE,[pdf_path],None,save=False,groups=groups)

        # I want to find the best match for the 
        # load dictionary to store warning messages on pages
        test_log = {'doc_level_warning':None}
        for p in pages_in_annotation:
            test_log[p]={'failed_test_1': False, 'phash_1': None, 'template_1': None,
                            'OCR_WARNING': None, 'OCR': None}
        
        #initialize the dictionaries i will use to store info on the sorting process
        page_dictionary,template_dictionary = initialize_sorting_dictionaries(list_of_images, root, input_from_file=False)
        #i will consider all template pages from the beginning and all images of course
        templates_to_consider = pages_in_annotation[:]
        pages_to_consider = [i+1 for i in range(len(list_of_images))]

        #pre_load_template_info
        template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, root, template_dictionary)
        #pre_load phash for images
        page_dictionary = pre_load_image_properties(pages_to_consider,page_dictionary,template_dictionary,properties=['phash'])
        

        #perform phash matching
        page_dictionary = perform_phash_matching(page_dictionary,template_dictionary, templates_to_consider, templates_to_consider, 
                        gap_threshold=GAP_THRESHOLD_PHASH,max_dist=MAX_DIST_PHASH)
        
        #I will assume phash matching is correct and check if the template matching agrees
        pairs_to_consider = []
        for img_id in pages_to_consider:
            matched_id = page_dictionary[img_id]['match_phash']
            pairs_to_consider.append([img_id,matched_id])
        #perform template_matching and update the matching keys in the dictionaries
        page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                    n_align_regions=N_ALIGN_REGIONS,scale_factor=SCALE_FACTOR_MATCHING)
        problematic_pages = []
        problematic_templates = []
        for img_id in pages_to_consider:
            matched_id_phash = page_dictionary[img_id]['match_phash']
            matched_id_template = page_dictionary[img_id]['matched_page']
            if matched_id_template==None: #there was no match with the page proposed by phas method
                problematic_pages.append(img_id)
                problematic_templates.append(matched_id_phash)        

        template_dictionary, page_dictionary = perform_ocr_matching(problematic_pages,problematic_templates, 
                                                        page_dictionary, template_dictionary,text_similarity_metric=TEXT_SIMILARITY_METRIC)
        
        #save matches
        for img_id in pages_to_consider:
            new_row = {"pdf_name": pdf_names[i], "saved_file_name": pdf_names[i]+f'_page_{img_id}', 
                         "extracted_id": "", "matched_template_page": page_dictionary[img_id]['matched_page']}
            data_rows.append(new_row)
        
        text_list=[]
        for img_id in pages_to_consider:
            matched_id = page_dictionary[img_id]['matched_page']
            censor_boxes, _ = get_censor_boxes(root,matched_id)
            censor_close_boxes, id_boxes = get_censor_close_boxes(root,matched_id)
            map_to_container = map_to_smallest_containing(censor_boxes,id_boxes)
            id_box = id_boxes[0]
            margin_box = map_to_smallest_containing[id_box]
            patch = preprocess_text_region(page_dictionary[img_id]['img'], template_dictionary[matched_id]['text_box'])
            page_text = extract_features_from_text_region(patch, psm=template_dictionary[matched_id]['psm'])['text']
            #i need to add a preprocessing here to extract the actual id (eg function that finds 6 consecutive numbers or that
            #exploits tesseract bounding boxes)
            text_list.append(page_text)
        
        for i,img_id in enumerate(pages_to_consider):
            data_rows[i]['extracted_id']=text_list[i]
        
    extracted_results = pd.DataFrame(data_rows)
    extracted_results.to_csv(SAVE_PATH+'\\extracted_results.csv', index=False)

    logger.info("Conversion finished")
    return 0

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


if __name__ == "__main__":
    main()