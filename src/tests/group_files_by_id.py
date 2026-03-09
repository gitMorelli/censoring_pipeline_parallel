#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import shutil

import pandas as pd

from src.utils.convert_utils import pdf_to_images,save_as_is,extract_images, process_pdf_files
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PDF_PATH="//vms-e34n-databr/2025-handwriting\\data\\manually_downloaded_by_Q" #additional"#100263_template"
METADATA_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\manually_downloaded_by_Q_and_id"#additional"#100263_template" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-f", "--pdf_path",
        default=PDF_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-m", "--metadata_path",
        default=METADATA_PATH,
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


QUESTIONNAIRE = "10"

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path = args.pdf_path
    metadata_path = args.metadata_path
    save_path = args.save_path
    
    #open metadata
    metadata_file_path = metadata_path+f"\\ref_pdf_Q{QUESTIONNAIRE}.csv"
    questionnaire_folder_path = pdf_path+f"\\q_{QUESTIONNAIRE}"

    raw_metadata = pd.read_csv(metadata_file_path)

    # i check for nans
    print("Missing values per column:")
    print(raw_metadata.isna().sum())

    # I drop rows with a missing object_name
    metadata = raw_metadata.dropna(subset=['object_name'])

    duplicates = metadata[metadata['object_name'].duplicated()]['object_name'].unique()
    print("There are ", len(duplicates), " duplicate files; Please inspect the data")
    if len(duplicates)>0:
        first_group = metadata[metadata['object_name'] == duplicates[0]]
        print(f"Investigating first duplicate ID: {duplicates[0]}")
        print(first_group)

    meta_dict = dict(zip(metadata['object_name'], metadata['e3n_id_hand'])) #i convert the pd df to a dict for fast retrieval
    # if there are duplicate object_id the last one overwrites the others

    questionnaire_file_list = list_files_with_extension(questionnaire_folder_path,extension = ['tiff','pdf'])
    questionnaire_file_names = [get_basename(p, remove_extension=True) for p in questionnaire_file_list]
    #files_already_read = {name: False for name in questionnaire_file_names}


    for i,file_path in enumerate(questionnaire_file_list):
        file_name = questionnaire_file_names[i]
        '''if files_already_read[file_name]:
            continue'''
        subj_id = meta_dict.get(file_name)
        if subj_id:
            destination_folder = os.path.join(save_path,"q_"+QUESTIONNAIRE,subj_id)
            destination_path = os.path.join(destination_folder,file_name+'.pdf')
            create_folder(destination_folder)
            shutil.copy(file_path,destination_path)
        else:
            print(f"File {file_name} is not present in the metadata")

    return 0

if __name__ == "__main__":
    main()