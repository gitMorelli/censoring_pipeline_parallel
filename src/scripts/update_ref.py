import argparse
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

QUESTIONNAIRE="13"
def main():
    args = parse_args()
    main_path = "/mnt/beegfs01/scratch/a_morelli/parallel_censoring"
    store_folder = os.path.join(main_path, "csv_results_aggregated")
    ref_path = os.path.join(main_path, "ref_pdf")

    ref_dir = Path(ref_path)
    exclude_name = f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv"

    # f is a Path object; str(f) gives you the full absolute or relative path
    csv_paths = [
        str(f) for f in ref_dir.glob("*.csv") 
        if f.name.lower() != exclude_name.lower()
    ]

    # 2. Read each CSV and store them in a list
    # Use a list comprehension for memory efficiency and speed
    df_list = [pd.read_csv(path) for path in csv_paths]

    # 3. Concatenate all DataFrames in the list into one
    # axis=0 means stack rows (vertically); ignore_index=True resets the row numbers
    if df_list:
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"Successfully combined {len(df_list)} files.")
    else:
        print("No CSV files found to combine.")
        final_df = pd.DataFrame() # Create empty DF to prevent downstream errors
    
    #select only the ids with status="success" amd save to a success_df
    success_df = final_df[final_df['status'] == 'success'].copy()
    #select only the ids with status!="success" and save to a failed_df
    failed_df = final_df[final_df['status'] != 'success'].copy()
    #load the updated_ref_pdf_Q13.csv 
    updated_ref_df = pd.read_csv(os.path.join(ref_path, exclude_name))
    #print the number of unique ids for which Used==true in the updated_ref_df
    print("Number of unique ids in updated_ref_df: ", updated_ref_df['e3n_id_hand'].nunique())
    print(f"Number of unique ids with Used==True in updated_ref_df: {updated_ref_df[updated_ref_df['Used'] == True]['e3n_id_hand'].nunique()}")
    print(f"Number of unique ids with Used==False in updated_ref_df: {updated_ref_df[updated_ref_df['Used'] == False]['e3n_id_hand'].nunique()}")

    #set the Used column of the updated_ref_df to True if the id is in the success_df, False 
    updated_ref_df['Used'] = updated_ref_df['e3n_id_hand'].isin(success_df['e3n_id_hand'])
    print(f"Number of unique ids that are already succesfully processed and set to True is: {updated_ref_df[updated_ref_df['Used'] == True]['e3n_id_hand'].nunique()}")
    #overwrite the updated_ref_df 
    updated_ref_df.to_csv(os.path.join(ref_path, exclude_name), index=False)

    
    #save the success_df to a csv file in a custom folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #create the folder if it doesn't exist
    os.makedirs(store_folder, exist_ok=True)
    success_df.to_csv(os.path.join(store_folder, f"success_ids_{timestamp}.csv"), index=False)

    #eliminate the files in csv_paths list
    for path_str in csv_paths:
        path_obj = Path(path_str)
        # missing_ok=True prevents an error if the file was already moved/deleted
        path_obj.unlink(missing_ok=True)


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

if __name__ == "__main__":
    main()