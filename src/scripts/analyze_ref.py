import argparse
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

QUESTIONNAIRE="13"
def main():
    args = parse_args()
    main_path = "/mnt/beegfs01/scratch/a_morelli/parallel_censoring/run_data_q13/ref_pdf"
    file_path = os.path.join(main_path, f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv")
    
    
    df = pd.read_csv(file_path)
    #get the list of unique ids in the column "e3n_id_hand"
    unique_ids = df['e3n_id_hand'].unique()
    #show the first 10 unique ids
    print("First 10 unique ids: ", unique_ids[:10])
    #get the number of ids that start with ab 
    pattern="A0A"
    count_ab = sum(id.startswith(pattern) for id in unique_ids)
    print(f"Number of unique ids that start with {pattern}: {count_ab} out of {len(unique_ids)} total unique ids")
    

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