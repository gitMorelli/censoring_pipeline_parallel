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
import pandas as pd
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


def calculate_accuracy(df, col1, col2):
    """
    Returns the accuracy (matches / total) between two columns.
    """
    # This creates a boolean Series where True = 1 and False = 0
    matches = (df[col1] == df[col2])
    
    # Taking the mean of booleans gives the percentage of True values
    accuracy = matches.mean()
    
    return accuracy

def calculate_grouped_accuracy(df, col1, col2, group_col='pdf_name'):
    """
    Groups by a column, takes the mode of the ID columns, 
    and returns the accuracy of those modes.
    """
    # 1. Group by the PDF name and take the mode (most frequent)
    # Note: .mode() can return multiple values if there's a tie, 
    # so we take .iloc[0] to ensure we get a single value.
    grouped_df = df.groupby(group_col)[[col1, col2]].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    
    # 2. Calculate accuracy on the aggregated results
    accuracy = (grouped_df[col1] == grouped_df[col2]).mean()
    
    return accuracy, grouped_df

def calculate_final_accuracy(df, col1, col2, group_col='pdf_name'):
    # 1. Eliminate rows that have 'ISP' in col1
    df_clean = df[~df[col1].astype(str).str.contains('ISP', na=False)].copy()
    
    # 2. Convert col1 (and col2 for consistency) to float
    # 'coerce' turns non-numeric values into NaN so the code doesn't crash
    df_clean[col1] = pd.to_numeric(df_clean[col1], errors='coerce')
    df_clean[col2] = pd.to_numeric(df_clean[col2], errors='coerce')
    
    # 3. Drop any rows that became NaN during conversion (optional but recommended)
    df_clean = df_clean.dropna(subset=[col1, col2])

    # 4. Group by PDF and take the mode
    grouped_df = df_clean.groupby(group_col)[[col1, col2]].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    
    # 5. Calculate accuracy
    if grouped_df.empty:
        return 0.0
        
    accuracy = (grouped_df[col1] == grouped_df[col2]).mean()
    return accuracy

#experiments on using pytesseract for identifying the id: https://gemini.google.com/share/54f0575cafcb
def main():
    args = parse_args()
    load_path="//vms-e34n-databr/2025-handwriting\\data\\test_id_retrival\\results\\extracted_results.csv"
    df = pd.read_csv(load_path,delimiter=';')
    print("Accuracy id:", calculate_final_accuracy(df,'true_id','extracted_id'))



    return 0

if __name__ == "__main__": 
    main()