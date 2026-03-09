#!/usr/bin/env python3
import argparse
import logging
import os
from src.utils.convert_utils import pdf_to_images,save_as_is,extract_images, process_pdf_files
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number
from time import perf_counter
from pathlib import Path

logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\data\\training_obj_det_model_pdf"#additional"#100263_template"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\data\\training_obj_det_model_png"#additional"#100263_template" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-f", "--folder_path",
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


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.folder_path
    save_path = args.save_path
    pdf_paths = list_subfolders(templates_path)

    for pdf_path in pdf_paths: #iterate on the folders Q_1,Q_2,..
        folder_name = Path(pdf_path).name
        pdf_files = list_files_with_extension(pdf_path, ["pdf",'tif'], recursive=False)
        logger.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_path)
        if not pdf_files:
            logger.warning("No PDF files found. Exiting.")
            return 0

        save_folder=save_path+'\\'+folder_name
        n_template=get_page_number(pdf_path) #get the number of the questionnaire
        process_pdf_files(n_template,pdf_files,save_folder)

    logger.info("Conversion finished")
    return 0




if __name__ == "__main__":
    main()