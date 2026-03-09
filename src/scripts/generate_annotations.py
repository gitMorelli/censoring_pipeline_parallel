#!/usr/bin/env python3
import argparse
import logging
import os
import json

from PIL import Image
import numpy as np

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import load_annotation_tree, load_templates_tree, create_folder

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.logging import FileWriter, initialize_logger

from src.utils.annotation_utils import precompute_and_store_template_properties


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

#"//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\" # "Z:\\vscode\\censor_e3n\\data\\q5_tests\\" #C:\\Users\\andre\\VsCode\\censoring project\\data\\rimes_tests\\
CROP_PATCH_PCTG = 0.02
OCR_PSM=6

mode="cv2"


def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-t", "--template_path",
        default="//vms-e34n-databr/2025-handwriting\\data\\e3n_templates_png\\current_template",
        help="Path to the template files, saved as folders containing PNG images",
    )
    parser.add_argument(
        "-a", "--annotation_path",
        default="//vms-e34n-databr/2025-handwriting\\data\\annotations\\current_template",
        help="Directory with the annotation files from cvat for each image",
    )
    parser.add_argument(
        "-s", "--save_path",
        default="//vms-e34n-databr/2025-handwriting\\data\\annotations\\current_template\\precomputed_features",
        help="Directory where I save the final annotation files",
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

    template_path = args.template_path
    save_path = args.save_path
    annotation_path = args.annotation_path 

    initialize_logger(args.verbose,logger)
    
    log_path = os.path.join(save_path, "logs")
    create_folder(log_path, parents=True, exist_ok=True)
    file_logger=FileWriter(enabled=args.verbose,path=os.path.join(log_path,f"global_logger.txt"))

    logger.info("Starting PDF -> PNG conversion")
    logger.debug("Input folder: %s", template_path)
    logger.debug("Output folder: %s", save_path)
    logger.debug("Annotation folder: %s", annotation_path)

    annotation_file_names, annotation_files = load_annotation_tree(file_logger, annotation_path)

    template_folder_names, template_folders = load_templates_tree(file_logger,template_path)

    precompute_and_store_template_properties(annotation_files, template_folders, file_logger, save_path, 
                                             annotation_file_names,template_folder_names,OCR_PSM,CROP_PATCH_PCTG, mode=mode)

    logger.info("Conversion finished")
    return 0


if __name__ == "__main__":
    main()