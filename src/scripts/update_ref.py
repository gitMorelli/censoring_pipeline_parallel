import argparse
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

QUESTIONNAIRE="7"
def main():
    args = parse_args()
    #main_path = "/mnt/beegfs01/scratch/a_morelli/parallel_censoring"
    main_path = "/home/a_morelli/datasets/censored_pdfs"
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

    print(len(df_list), "CSV files read successfully.")

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
    print(f"Number of unique ids with status='success': {success_df['e3n_id_hand'].nunique()}")
    #select only the ids with status!="success" and save to a failed_df
    failed_df = final_df[final_df['status'] != 'success'].copy()
    print(f"Number of unique ids with status!='success': {failed_df['e3n_id_hand'].nunique()}")

    analysis(final_df)

    #load the updated_ref_pdf_Q13.csv 
    updated_ref_df = pd.read_csv(os.path.join(ref_path, exclude_name))
    #print the number of unique ids for which Used==true in the updated_ref_df
    print("Number of unique ids in updated_ref_df: ", updated_ref_df['e3n_id_hand'].nunique())
    print(f"Number of unique ids with Used==True in updated_ref_df: {updated_ref_df[updated_ref_df['Used'] == True]['e3n_id_hand'].nunique()}")
    print(f"Number of unique ids with Used==False in updated_ref_df: {updated_ref_df[updated_ref_df['Used'] == False]['e3n_id_hand'].nunique()}")

    #set the Used column of the updated_ref_df to True if the id is in the success_df, False 
    # 1. Create a boolean mask of rows where the ID exists in success_df
    mask = updated_ref_df['e3n_id_hand'].isin(success_df['e3n_id_hand'])

    # 2. Update 'Used' to True ONLY for those specific rows
    updated_ref_df.loc[mask, 'Used'] = True
    print(f"Number of unique ids that are already succesfully processed and set to True is: {updated_ref_df[updated_ref_df['Used'] == True]['e3n_id_hand'].nunique()}")

    
    #overwrite the updated_ref_df 
    updated_ref_df.to_csv(os.path.join(ref_path, exclude_name), index=False)

    
    #save the success_df to a csv file in a custom folder with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #create the folder if it doesn't exist
    os.makedirs(store_folder, exist_ok=True)
    success_df.to_csv(os.path.join(store_folder, f"success_ids_{timestamp}.csv"), index=False)
    #save the failed_df to a csv file in a custom folder with a timestamp
    failed_df.to_csv(os.path.join(store_folder, f"failed_ids_{timestamp}.csv"), index=False)

    #load all success csv files in the store_folder, concatenate them into a single dataframe and remove duplicate lines, then save it as a single csv file 
    success_csv_paths = [str(f) for f in Path(store_folder).glob("success_ids_*.csv")]
    success_df_list = [pd.read_csv(path) for path in success_csv_paths]
    if success_df_list:
        combined_success_df = pd.concat(success_df_list, axis=0, ignore_index=True)
        combined_success_df = combined_success_df.drop_duplicates()
        combined_success_df.to_csv(os.path.join(store_folder, f"combined_success_ids.csv"), index=False)
        print(f"Successfully combined {len(success_df_list)} success files into one.")
        print(f"Number of unique ids in combined success file: {combined_success_df['e3n_id_hand'].nunique()}")
    else:
        print("No success CSV files found to combine.")

    #eliminate the files in csv_paths list
    for path_str in csv_paths:
        path_obj = Path(path_str)
        # missing_ok=True prevents an error if the file was already moved/deleted
        path_obj.unlink(missing_ok=True)
    

def analysis(df):
    #recall that you have duplicate lines (one per file per each id) and you want to keep only one line per id, for example the last one (the one with the largest time)
    final_df=df.copy()
    #keep a single line per unique id
    final_df = final_df.drop_duplicates(subset='e3n_id_hand', keep='last')
    #get the unique values and their count for the columns "Warning_ordering", "Warning_censoring", "status"
    for col in ["Warning_ordering", "Warning_censoring", "status"]:
        print(f"Unique values for {col}:")
        print(final_df[col].value_counts())
        print("\n")
    # get the 5 ids with larger value of the 'time' column
    top_5_time = final_df.nlargest(5, 'time')
    print("Top 5 IDs with largest time:")
    print(top_5_time[['e3n_id_hand', 'time']])
    # get the list of ids with 'status'=='timeout'
    timeout_ids = final_df[final_df['status'] == 'timeout']['e3n_id_hand'].tolist()
    print(f"IDs with status 'timeout': {timeout_ids[:min(10, len(timeout_ids))]}") # Print only the first 10 for brevity
    #return the average time and the number of ids with time>100 over the total
    avg_time = final_df['time'].mean()
    count_time_gt_100 = (final_df['time'] > 100).sum()
    print(f"Average time: {avg_time}")
    print(f"Number of IDs with time > 100: {count_time_gt_100} over {len(final_df)} total IDs")

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