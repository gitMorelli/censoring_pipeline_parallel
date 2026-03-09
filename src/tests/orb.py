#!/usr/bin/env python3
import argparse
import logging
import json
import os
import io
from time import perf_counter
import pikepdf
import fitz
from PIL import Image
from PyPDF2 import PdfReader
import re
import math
import cv2
import pytesseract #for ocr
pytesseract.pytesseract.tesseract_cmd = r'//vms-e34n-databr/2025-handwriting\programs\tesseract\tesseract.exe'

from src.utils.json_parsing import get_attributes_by_page 
from src.utils.convert_utils import process_pdf_files
from src.utils.feature_extraction import preprocess_text_region, preprocess_page



#JSON_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\censor_e3n\data\q5_tests\\annotazioni" 

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Test script")
    '''parser.add_argument(
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
    )'''
    return parser.parse_args()




#experiments on using pytesseract for identifying the id: https://gemini.google.com/share/54f0575cafcb
def main():
    args = parse_args()
    path_file = "//vms-e34n-databr/2025-handwriting\\data\\test_read_shared_files\\Q5\\A9Y0H8E8\\ISP00JLX_ISP01RGX.tif.pdf"
    path_file = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers\\Q5\\ISP00JLX_ISP01RGX.tif.pdf"
    path_file ="//intra.igr.fr/profils$/UserCtx_Data$/A_MORELLI\\Downloads\\ISP00DLO_ISP013FX.tif.pdf"
    path_file = "//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\mixed\\5_4999742_4.pdf"

    list_of_images = process_pdf_files(-1,[path_file],None,save=False)
    image = list_of_images[0]

    height = image.shape[0]
    width = image.shape[1]
    t_1=perf_counter()
    orb = cv2.ORB_create(nfeatures=2000)
    patch = preprocess_page(image)
    kp_unkn, des_unkn = orb.detectAndCompute(patch, None)
    t_2 = perf_counter()
    print(t_2-t_1)



    return 0

if __name__ == "__main__":
    main()