#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import shutil

import pandas as pd


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#METADATA_PATH = "/mnt/beegfs01/scratch/a_morelli/datasets/pdfs/ref_pdf_Qx"
FILENAME_PATH = "/home/a_morelli/datasets/pdfs/list_filenames"
OUTPUT_PATH = "/home/a_morelli/temporary_data/pdf_transfer"
PDF_PATH = "/home/a_morelli/datasets/pdfs"


def remove_after_first_dot(text):
    """
    Removes all characters following the first '.' in a string.
    If no dot is found, returns the original string.
    """
    return text.split('.', 1)[0]

def preprocess_df(df_main,filename_col,id_col,used_col_name='Used',warning_ordering_col_name='Warning_ordering',warning_censoring_col_name='Warning_censoring'):
    df=df_main.copy()
    # 1. Drop lines with at least one missing values
    df = df.dropna()

    unique_ids_before = df[id_col].nunique()
    # 2. Remove file extensions from filenames (remove everything after the first point)
    df[filename_col] = df[filename_col].str.rsplit('.', n=1).str[0]

    # 3. Remove lines with filenames that are associated with more than one ID
    fname_id_counts = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = fname_id_counts[fname_id_counts > 1].index.tolist()
    df = df[~df[filename_col].isin(multi_id_filenames)]
    print(f"Removed filenames because associated to multiple ids: {len(multi_id_filenames)}")
    
    length_before = len(df)
    df.drop_duplicates(inplace=True) #by default it keep the first occurrence
    length_after = len(df)
    print(f"Before eliminating duplicates the row length is {length_before} after it is {length_after} -> {length_before-length_after} rows were eliminated")

    unique_ids_after = df[id_col].nunique()
    print(f"Unique ids before: {unique_ids_before} after: {unique_ids_after} -> {unique_ids_before-unique_ids_after} unique ids were eliminated")

    df = df.sort_values(id_col).reset_index(drop=True)
    #add columns
    df[used_col_name] = False
    df[warning_ordering_col_name] =''
    df[warning_censoring_col_name] =''

    return df



def main():
    questionnaire = "2"

    filename_dir = FILENAME_PATH
    #output_dir = OUTPUT_PATH
    pdf_path = PDF_PATH
    #os.makedirs(output_dir, exist_ok=True) # Create folder if it doesn't exist

    #questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}",f"archived_{questionnaire}")
    questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}")
    #questionnaire_folder_path = os.path.join(pdf_path,f"Q{questionnaire}",f"{questionnaire}")
    filename_path = os.path.join(filename_dir, f"pdf_filepaths_{questionnaire}.txt")
    #report_file_path = os.path.join(output_dir, f"audit_report_Q{questionnaire}.txt")

    if not os.path.exists(filename_path):
        print(f"Error: The file '{filename_path}' was not found.")
        return

    # Check if the target directory exists
    if not os.path.isdir(questionnaire_folder_path):
        print(f"Error: The directory '{questionnaire_folder_path}' does not exist.")
        return

    found = 0
    total = 0
    # Open the text file and read it line by line
    with open(filename_path, 'r') as f:
        for line in f:
            # Remove leading/trailing whitespace and newline characters
            filename = line.strip()
            
            # Skip empty lines
            if not filename:
                continue
            
            # Construct the full path using the os library
            file_to_check = os.path.join(questionnaire_folder_path, filename)
            
            # Check if the file exists at that path
            if os.path.exists(file_to_check):
                found += 1
            total += 1
            if total % 2000 == 0:
                print(f"Checked {total} files, found {found} so far.")
    print(f"Finished checking. Found {found} out of {total} files.")

    return 0

if __name__ == "__main__":
    main()