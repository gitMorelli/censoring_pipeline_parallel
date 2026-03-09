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

PDF_PATH="//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers"#additional"#100263_template"
METADATA_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_read_shared_files\\"#additional"#100263_template" 

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


QUESTIONNAIRE = "Q10"

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path = args.pdf_path
    metadata_path = args.metadata_path
    save_path = args.save_path
    
    #open metadata
    metadata_file_path = metadata_path+f"\\ref_pdf_{QUESTIONNAIRE}.csv"
    questionnaire_folder_path = pdf_path+f"\\{QUESTIONNAIRE}"

    metadata = pd.read_csv(metadata_file_path)

    # count the number of elements in the df
    '''print(len(metadata))
    print(metadata.columns)'''

    # Count of unique entries
    '''count = metadata['e3n_id_hand'].nunique()
    print(f"There are {count} unique entries.")'''

    # 1. Group and get the count for each group
    # 2. Extract the unique count values
    '''unique_counts = metadata.groupby('e3n_id_hand').size().value_counts()
    print(unique_counts)'''
    

    #read files and get info
    '''
    print("start reading files: ")
    time_1=perf_counter()
    questionnaire_file_list = list_files_with_extension(questionnaire_folder_path,extension = ['tiff','pdf'])
    print("finished reading, it took ", perf_counter()-time_1, " seconds")
    questionnaire_file_names = [get_basename(p, remove_extension=False) for p in questionnaire_file_list]

    print("there are ", len(questionnaire_file_list), " files")
    print(f"Memory occupied by the list object: {sys.getsizeof(questionnaire_file_list)} bytes")
    print("example name = ", questionnaire_file_names[0])
    print(metadata.head())'''

    #count how many tif files
    '''count = sum(1 for name in questionnaire_file_names if name.endswith(".tif"))
    print(f"Number of .tif files: {count}")'''
    # clean filenames from .tif
    '''clean_quest_names = [name.replace(".tif", "") for name in questionnaire_file_names]'''

    # select some ids at random to convert and observe
    # 1. Ottieni i valori unici della colonna
    unique_ids = metadata['e3n_id_hand'].unique()
    # 2. Seleziona casualmente N valori da quelli unici
    # (assicurati che N non sia superiore alla lunghezza di valori_unici)
    N_selected = 10
    selected_ids = pd.Series(unique_ids).sample(n=N_selected).tolist()

    # 3. Filtra il DataFrame originale usando i valori selezionati
    selected_metadata = metadata[metadata['e3n_id_hand'].isin(selected_ids)]

    n=0
    for id in selected_ids: #iterate on the folders Q_1,Q_2,..
        file_names = selected_metadata.loc[selected_metadata['e3n_id_hand'] == id, 'object_name'].tolist()
        for file_name in file_names:
            load_path = os.path.join(questionnaire_folder_path,file_name+'.pdf')
            if Path(load_path).exists():
                destination_folder = os.path.join(save_path,QUESTIONNAIRE,id)
                destination_path = os.path.join(destination_folder,file_name+'.pdf')
                create_folder(destination_folder)
                shutil.copy(load_path,destination_path)
            else:
                n+=1
                print(f"Non existing file number {n}, id: {id}, name: {load_path}")

    return 0




if __name__ == "__main__":
    main()