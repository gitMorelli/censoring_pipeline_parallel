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
import pytesseract #for ocr
pytesseract.pytesseract.tesseract_cmd = r'//vms-e34n-databr/2025-handwriting\programs\tesseract\tesseract.exe'

from src.utils.json_parsing import get_attributes_by_page 
from src.utils.convert_utils import process_pdf_files
from src.utils.feature_extraction import preprocess_text_region



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

def extract_target_numeric(text):
    
    # 2. Search for the primary pattern: 6 digits + optional spaces + 1 digit
    # Pattern: \d{6} (six digits) \s* (zero or more spaces) \d (one digit)
    primary_pattern = r'\d{6}\s*\d'
    primary_match = re.search(primary_pattern, text)
    
    if primary_match:
        # Found the specific target! Return it (stripping extra spaces)
        return primary_match.group(0).replace(" ", "")

    # 3. Fallback: Search for the longest numeric string in the entire text
    # \d+ finds any consecutive sequence of digits
    all_numeric_strings = re.findall(r'\d+', text)
    
    if all_numeric_strings:
        # Sort by length and take the longest
        longest_string = max(all_numeric_strings, key=len)
        return longest_string
    
    return None

def get_numeric_boxes(image):
    """
    Parses image_to_boxes and returns a list of tuples: 
    [('1', (x1, y1, x2, y2)), ...]
    """
    # Get raw box data from Tesseract
    raw_data = pytesseract.image_to_boxes(image)
    
    numeric_results = []
    
    for line in raw_data.splitlines():
        # Split line: ['char', 'x1', 'y1', 'x2', 'y2', '0']
        parts = line.split(' ')
        char = parts[0]
        
        # Check if the character is a digit
        if char.isdigit():
            # Convert coordinates to integers
            coords = tuple(map(int, parts[1:5]))
            numeric_results.append((char, coords))
            
    return numeric_results

def get_numeric_data(image, config,min_conf=50):
    """
    Returns a list of numbers found in the image with their 
    top-left (OpenCV style) coordinates and confidence scores.
    """
    # image_to_data returns a TSV-style string by default
    # Output_type=Output.DICT is the cleanest way to handle this
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT,config=config)
    
    numeric_results = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        # Filter: Is it a number and is the confidence high enough?
        # This regex-free check handles digits; use .replace('.','') for floats
        if text.isdigit() and conf > min_conf:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Format: (Text, (x, y, width, height), confidence)
            numeric_results.append((text, (x, y, w, h), conf))
            
    return numeric_results

def extract_special_id(numeric_data):
    # 1. Helper to get the connection points
    # We want distance from the RIGHT edge of the 6-digit number 
    # to the LEFT edge of the single digit.
    
    # --- Step 1: Check for exactly 6 elements ---
    for i, (text6, (x6, y6, w6, h6), conf6) in enumerate(numeric_data):
        if len(text6) == 6:
            r_center_x = x6 + w6
            r_center_y = y6 + (h6 / 2)
            
            closest_digit = None
            min_dist = float('inf')
            
            for j, (text1, (x1, y1, w1, h1), conf1) in enumerate(numeric_data):
                # Must be a single digit and NOT the same box
                if i == j or len(text1) != 1:
                    continue
                
                # Target point: Left-center of the single digit
                l_center_x = x1
                l_center_y = y1 + (h1 / 2)
                
                # Calculate Euclidean Distance
                dist = math.sqrt((l_center_x - r_center_x)**2 + (l_center_y - r_center_y)**2)
                
                # Requirement: Must be to the right (x1 > x6)
                if x1 >= x6 + w6 and dist < min_dist:
                    min_dist = dist
                    closest_digit = text1
            
            if closest_digit:
                return f"{text6}{closest_digit}"

    # --- Step 2: Look for exactly 7 elements ---
    for text, coords, conf in numeric_data:
        if len(text) == 7:
            if text != "1234567":
                return text
            
    return None

#experiments on using pytesseract for identifying the id: https://gemini.google.com/share/54f0575cafcb
def main():
    args = parse_args()
    path_file = "//vms-e34n-databr/2025-handwriting\\data\\test_read_shared_files\\Q5\\A9Y0H8E8\\ISP00JLX_ISP01RGX.tif.pdf"
    path_file = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\Fichiers\\Q5\\ISP00JLX_ISP01RGX.tif.pdf"
    path_file ="//intra.igr.fr/profils$/UserCtx_Data$/A_MORELLI\\Downloads\\ISP00DLO_ISP013FX.tif.pdf"
    path_file = "//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\mixed\\2_3554546_1.pdf"

    list_of_images = process_pdf_files(-1,[path_file],None,save=False)
    image = list_of_images[0]

    height = image.shape[0]
    width = image.shape[1]
    box = [0,0,width,height/3]
    patch = preprocess_text_region(image,box,mode='cv2',verbose=False, aggressive=False)

    config = f"--oem 3 --psm 3"
    lang = "fra"

    '''#test different usages of tesseract
    print(pytesseract.image_to_string(patch))
    print('-------------------------00000000000000000-----------------------------')
    print(pytesseract.image_to_boxes(patch))
    print('-------------------------00000000000000000-----------------------------')
    print(pytesseract.image_to_data(patch))'''

    '''text = pytesseract.image_to_string(patch,lang = lang, config = config)
    print(text)
    print( extract_target_numeric(text) )'''
    characters = get_numeric_boxes(patch)
    print(characters)
    boxes=get_numeric_data(patch,config)
    print(boxes)
    print(extract_special_id(boxes))

    return 0

if __name__ == "__main__":
    main()