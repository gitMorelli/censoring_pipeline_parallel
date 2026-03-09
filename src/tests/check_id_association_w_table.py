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

METADATA_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"
OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\data\\analysis_of_ref_Qx"
CSV_PATH = "//vms-e34n-databr/2025-handwriting\\E3N\\TABLES" #d01_20260216_handwriting

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Script to convert PDF template pages to PNG images.")
    parser.add_argument(
        "-m", "--metadata_path",
        default=METADATA_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-o", "--output_path",
        default=OUTPUT_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-f", "--csv_path",
        default=CSV_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()

def check_dataframe_assembly(df, filename_col='filename', id_col='id', n_sample=5):
    """
    Checks the integrity and composition of a 2-column dataframe.
    
    Parameters:
    - df: The pandas DataFrame to check.
    - filename_col: The name of the column containing filenames.
    - id_col: The name of the column containing IDs.
    
    Returns:
    - A dictionary containing the analysis results.
    """
    
    # 1. Missing values per column
    missing_filename = df[filename_col].isna().sum()
    missing_id = df[id_col].isna().sum()
    
    # 2. Rows with NaNs on both columns
    both_nan = df[df[filename_col].isna() & df[id_col].isna()].shape[0]
    
    # 3. Extension checks (handling NaNs by setting na=False)
    is_tif = df[filename_col].str.endswith('.tif', na=False).sum()
    is_pdf = df[filename_col].str.endswith('.pdf', na=False).sum()
    
    # No extension: No dot in string and not NaN
    no_extension = (df[filename_col].notna() & ~df[filename_col].str.contains(r'\.', na=False)).sum()
    
    # 4. Duplicate filenames
    # Total count of rows that are duplicates of a previous row
    total_duplicate_rows = df[filename_col].duplicated().sum()
    # Number of unique filenames that appear more than once
    duplicate_filenames_count = df[filename_col].value_counts()
    unique_duplicates = (duplicate_filenames_count > 1).sum()
    
    # 5. Duplicate filenames associated with more than one ID
    # Group by filename and count unique IDs for each
    filename_id_mapping = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = (filename_id_mapping > 1).sum()
    
    # 6. Unique counts
    unique_ids = df[id_col].nunique()
    unique_filenames = df[filename_col].nunique()
    
    # 7. Distribution of counts (How many files per ID)
    # Count filenames per ID and then find how many unique count values exist
    counts_per_id = df.groupby(id_col)[filename_col].count()
    unique_frequency_counts = counts_per_id.nunique()
    
    results = {
        "N missing_filename": missing_filename, #expect 0
        "N missing_id": missing_id, #expect 0
        "N rows_both_nan": both_nan, #can be any
        "Count_tif": is_tif, 
        "Count_pdf": is_pdf, 
        "Count_no_extension": no_extension, 
        "N duplicate rows": total_duplicate_rows, # expect one of the three to be the lenght and the others to be 0
        "Has_duplicates": unique_duplicates > 0,
        "N filenames_with_duplicates": unique_duplicates,
        "N filenames_with_multiple_ids": multi_id_filenames,
        "Total_unique_ids": unique_ids,
        "Total_unique_filenames": unique_filenames,
        "unique_counts_per_id": unique_frequency_counts
    }
    
    return results

def audit_dataframe(df_source, output_file,file_names, filename_col='filename', id_col='id', n_sample=5):
    """
    Performs audit and writes results to the provided output_file handle.
    """
    # Helper to write to file instead of print
    def log(message):
        output_file.write(str(message) + "\n")

    df = df_source.copy()

    # 1. Missing values
    missing_filename = df[filename_col].isna().sum()
    missing_id = df[id_col].isna().sum()
    
    # 2. Rows with NaNs on both columns
    both_nan = df[df[filename_col].isna() & df[id_col].isna()].shape[0]
    
    # 3. Extension checks
    is_tif = df[filename_col].str.endswith('.tif', na=False).sum()
    is_pdf = df[filename_col].str.endswith('.pdf', na=False).sum()
    no_ext = (df[filename_col].notna() & ~df[filename_col].str.contains(r'\.', na=False)).sum()

    df[filename_col] = df[filename_col].str.rsplit('.', n=1).str[0]

    # 3 e 1/2
    log(f"--- Step 3 and 1/2: Does each filename in the folder correspond to a df line? ---")
    set_filenames = set(file_names)
    log(f"There are {len(set_filenames)} unique filenames in the folder" )
    filenames_in_column = set(df[filename_col]) #set automatically removes any duplicate values.
    log(f"There are {len(filenames_in_column)} unique filenames in the csv" )
    in_df_but_not_in_folder = filenames_in_column - set_filenames
    in_folder_but_not_in_df = set_filenames - filenames_in_column
    log(f"Names in df but not in folder: {len(in_df_but_not_in_folder)}")
    log(f"Examples {list(in_df_but_not_in_folder)[:min(3,len(in_df_but_not_in_folder))]}")
    log(f"Names in folder but not in df: {len(in_folder_but_not_in_df)}")
    log(f"Examples {list(in_folder_but_not_in_df)[:min(3,len(in_folder_but_not_in_df))]}")

    
    # 4. Duplicate filenames + Sample
    duplicate_mask = df[filename_col].duplicated(keep=False)
    unique_duplicate_filenames = df.loc[duplicate_mask, filename_col].dropna().unique()
    num_duplicates = len(unique_duplicate_filenames)
    
    log(f"--- Step 4: Duplicate Filenames ---")
    if num_duplicates > 0:
        sample = pd.Series(unique_duplicate_filenames).sample(min(n_sample, num_duplicates)).tolist()
        log(f"Total unique filenames with duplicates: {num_duplicates}")
        log(f"Sample of {len(sample)} duplicates: {sample}")
        log("Full table ->")
        log(df[df[filename_col].isin(sample)].to_string()) # Use to_string() for clean file output
    else:
        log("No duplicate filenames found.")

    # 5. Filenames associated with more than one ID
    fname_id_counts = df.groupby(filename_col)[id_col].nunique()
    multi_id_filenames = fname_id_counts[fname_id_counts > 1].index.tolist()
    
    log(f"\n--- Step 5: Filenames associated with multiple IDs ---")
    if multi_id_filenames:
        log(f"Found {len(multi_id_filenames)} filenames: {multi_id_filenames}")
        log("Full table ->")
        log(df[df[filename_col].isin(multi_id_filenames)].to_string())
    else:
        log("No filenames are associated with more than one unique ID.")

    # 6. Unique Totals
    unique_ids = df[id_col].nunique()
    unique_filenames = df[filename_col].nunique()
    
    # 7. Unique counts of filenames per ID
    df_deduplicated = df.drop_duplicates(subset=[id_col, filename_col])
    counts_per_id = df_deduplicated.groupby(id_col)[filename_col].count()
    unique_counts_per_id = counts_per_id.nunique()
    
    log(f"\n--- Step 7: Value counts for the number of filenames associated to an id ---")
    log(counts_per_id.value_counts().to_string())
    
    log(f"\n--- Summary Statistics ---")
    log(f"Missing values: {filename_col}={missing_filename}, {id_col}={missing_id}")
    log(f"Rows with both NaN: {both_nan}")
    log(f"Extensions: .tif={is_tif}, .pdf={is_pdf}, None={no_ext}")
    log(f"Unique IDs: {unique_ids}")
    log(f"Unique Filenames: {unique_filenames}")
    log(f"Number of unique count values (files per ID): {unique_counts_per_id}")

    return {
        "missing_filename": missing_filename,
        "missing_id": missing_id,
        "both_nan": both_nan,
        "tif_count": is_tif,
        "pdf_count": is_pdf,
        "no_ext_count": no_ext,
        "unique_ids": unique_ids,
        "unique_filenames": unique_filenames,
        "unique_frequency_counts": unique_counts_per_id
    }

def remove_after_first_dot(text):
    """
    Removes all characters following the first '.' in a string.
    If no dot is found, returns the original string.
    """
    return text.split('.', 1)[0]

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    metadata_path = args.metadata_path 
    output_dir = args.output_path 
    csv_path = args.csv_path
    os.makedirs(output_dir, exist_ok=True) # Create folder if it doesn't exist

    questionnairres = [str(i) for i in range(1,14)]
    
    covariates_file_path = os.path.join(csv_path, "d01_20260216_handwriting.csv")
    report_file_path = os.path.join(output_dir, f"compare_id_table_w_ref_pdf_indices.txt")
    
    if os.path.exists(covariates_file_path):
        raw_covariates = pd.read_csv(covariates_file_path,encoding='cp1252')
        table_ids = set(raw_covariates['ident_projet'].to_list())
        union_set = set()
        with open(report_file_path, 'w') as f:
            f.write(f"table has {len(table_ids)} unique ids\n")
        for q in questionnairres:
            print(f"--- Comparison for questionnaire Q{q} ---")
            metadata_file_path = os.path.join(metadata_path, f"ref_pdf_Q{q}.csv")
            raw_metadata = pd.read_csv(metadata_file_path)
            metadata_ids = set(raw_metadata['e3n_id_hand'].to_list())
            in_metadata_but_not_in_table = metadata_ids - table_ids
            in_table_but_not_in_metadata = table_ids - metadata_ids
            with open(report_file_path, 'a') as f:
                f.write(f"--- Comparison for questionnaire Q{q} ---\n")
                f.write(f"Names in metadata but not in table: {len(in_metadata_but_not_in_table)}\n")
                f.write(f"Examples {list(in_metadata_but_not_in_table)[:min(3,len(in_metadata_but_not_in_table))]}\n")
                f.write(f"Names in table but not in metadata: {len(in_table_but_not_in_metadata)}\n")
                f.write(f"Examples {list(in_table_but_not_in_metadata)[:min(3,len(in_table_but_not_in_metadata))]}\n")
            union_set = union_set.union(metadata_ids)
        in_union_but_not_in_table = union_set - table_ids
        in_table_but_not_in_union = table_ids - union_set
        with open(report_file_path, 'a') as f:
            f.write(f"--- Comparison for union of all questionnaires ---\n")
            f.write(f"Names in union but not in table: {len(in_union_but_not_in_table)}\n")
            f.write(f"Examples {list(in_union_but_not_in_table)[:min(3,len(in_union_but_not_in_table))]}\n")
            f.write(f"Names in table but not in union: {len(in_table_but_not_in_union)}\n")
            f.write(f"Examples {list(in_table_but_not_in_union)[:min(3,len(in_table_but_not_in_union))]}\n")
        #log(f"Examples {list(in_folder_but_not_in_df)[:min(3,len(in_folder_but_not_in_df))]}")

        #print(raw_covariates.head())
        
        # Open the file for writing
        #with open(report_file_path, 'w') as f:

            #report = audit_dataframe(raw_metadata, f, questionnaire_file_names, 'object_name', 'e3n_id_hand')

            #f.write("\n---------------- START DICTIONARY REPORT -------------------\n")
    else:
        print(f"One of the files was not found: {metadata_file_path}")

    return 0

if __name__ == "__main__":
    main()