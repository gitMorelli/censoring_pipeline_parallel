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

from src.utils.matching_utils import pre_load_image_properties, extract_target_numeric, extract_special_id
from src.utils.matching_utils import initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base, discover_template

from src.utils.censor_utils import map_to_smallest_containing

from src.utils.debug_utils import visualize_templates_w_annotations


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOAD_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\mixed_2"#additional"#100263_template"
TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\current_template"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\results"#additional"#100263_template" 
ANNOTATION_PATH = "//vms-e34n-databr/2025-handwriting\\data\\\\annotations\\current_template"

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
        "-a", "--annotation_path",
        default=ANNOTATION_PATH,
        help="Directory to save the converted PNG images",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

N_ALIGN_REGIONS = 2 #minimum number of align boxes needed for matching
SCALE_FACTOR_MATCHING = 2 
GAP_THRESHOLD_PHASH = 5
MAX_DIST_PHASH = 18
TEXT_SIMILARITY_METRIC = 'similarity_jaccard_tokens'
CONFIG = f"--oem 3 --psm 3" #6, 3, 11 see segmentation modes at https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
LANG = "fra"

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.templates_path
    load_path = args.load_path
    save_path = args.save_path
    annotation_path = args.annotation_path
    pdf_paths = list_files_with_extension(load_path, ["pdf",'tif'], recursive=False)
    pdf_names = [get_basename(p, remove_extension=True) for p in pdf_paths]

    #load the annotation files (full paths and names)
    annotation_file_names, annotation_files = load_annotation_tree(logger, annotation_path)
    # I open the json and save them in a list (ordered as annotation_files), i also open the pre_computed data from the npy files
    annotation_roots, npy_data = load_template_info(logger,annotation_files,annotation_file_names,annotation_path)

    '''#i create a pd dataframe for storing the exected results and the actual result
    extraction_results = initialize_result_df(pdf_names,QUESTIONNAIRE)'''

    data_rows = []
    for i,pdf_path in enumerate(pdf_paths): #iterate on the folders Q_1,Q_2,..
        list_of_images = process_pdf_files(-1,[pdf_path],None,save=False) #i extract all pages in order from the pdf

        n_images = len(list_of_images)
        q_from_page_number = None
        if n_images == 32:
            q_from_page_number = 8
        elif n_images == 12:
            q_from_page_number = 10
        print("-----"*50)
        print("Processing file: ", get_basename(pdf_path))
        
        for j,image in enumerate(list_of_images):
            height = image.shape[0]
            width = image.shape[1]
            if q_from_page_number == 8:
                box = [0,height*3/4,width,height]
            else:
                box = [0,0,width,height/4]
            print("extrqcting text")
            patch = preprocess_text_region(image,box,verbose=False, aggressive=False)
            extracted_id=extract_special_id(patch) 
            
            print("performing matching")
            if q_from_page_number == None:
                gray_image=preprocess_page(image)
                matched_q,matched_p = discover_template(gray_image,annotation_file_names,annotation_roots,npy_data)
            else:
                matched_q = q_from_page_number
                matched_p = j+1
            
            # extract ground truths from file name
            count = pdf_names[i].count('_')
            pieces = pdf_names[i].split('_') 
            if count == 2:
                true_q = pieces[0]
                true_id = pieces[1]
                true_p = pieces[2]
            else:
                true_q = pieces[0]
                true_id = pieces[1]
                true_p="multi"

            new_row = {"pdf_name": pdf_names[i], "saved_file_name": pdf_names[i]+f'_page_{j}', 
                         "extracted_id": extracted_id,"true_id":true_id,"full_text": None, "matched_template": matched_q, 
                         "true_template":true_q,"matched_template_page": matched_p,"true_page":true_p}
            '''print(pdf_names[i])
            print(text)
            print("-=-="*50)'''
            data_rows.append(new_row)

            if q_from_page_number == 8: #i only inspect the first page (i don't have it on other pages)
                break

        
    extracted_results = pd.DataFrame(data_rows)
    extracted_results.to_csv(SAVE_PATH+'\\extracted_results.csv', index=False,sep=';')

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