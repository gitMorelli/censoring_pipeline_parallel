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

LOAD_PATH="//vms-e34n-databr/2025-handwriting\\data\\test_read_shared_files\\"#additional"#100263_template" 
SAVE_PATH = "//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\mixed_2"

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-l", "--load_path",
        default=LOAD_PATH,
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

def get_done_files(directory_path):
    """
    Returns a list of all file paths containing '_done' 
    within the directory and its subfolders.
    """
    root = Path(directory_path)
    
    # .rglob('*') iterates recursively through all files and folders
    # we filter for files only, and check if '_done' is in the full string path
    files = [
        str(p.absolute()) for p in root.rglob('*') 
        if p.is_file() and "_done" in str(p)
    ]
    
    return files

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    load_path = args.load_path
    save_path = args.save_path

    files_to_move=get_done_files(load_path)
    file_names = [get_basename(f, remove_extension=True) for f in files_to_move]

    
    for i,file_path in enumerate(files_to_move):
        destination_path = os.path.join(save_path,file_names[i]+'.pdf')
        shutil.copy(file_path,destination_path)
    

    return 0




if __name__ == "__main__":
    main()