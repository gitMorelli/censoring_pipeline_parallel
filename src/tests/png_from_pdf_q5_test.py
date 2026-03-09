#!/usr/bin/env python3
import argparse
import logging
import os
from src.utils.convert_utils import pdf_to_images,save_as_is,extract_images
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number
from time import perf_counter
from src.utils.logging import FileWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_PATH= "//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\filled_non_converted" #"Z:\\vscode\\censor_e3n\\data\\q5_tests\\filled_non_converted" 
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\filled" #"Z:\\vscode\\censor_e3n\\data\\q5_tests\\filled" 

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
    parser.add_argument(
        "--save_debug_times",
        action="store_true",
        help="Enlarge censor boxes before applying censorship",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    templates_path = args.folder_path
    save_path = args.save_path
    save_debug_times = args.save_debug_times
    subject_paths = list_subfolders(templates_path)
    subject_id = [get_basename(s) for s in subject_paths]

    for i,subject in enumerate(subject_paths): #iterate on the subjects
        pdf_files = list_files_with_extension(subject, "pdf", recursive=False)#get the files in the folder (they are the files for doc_5)
        n_template=5
        id=subject_id[i]

        log_path=os.path.join(TEMPLATES_PATH,f"{id}")
        subj_time_logger=FileWriter(save_debug_times,
                                        os.path.join(log_path,f"time_logger.txt"))
        process_pdf_files(n_template,pdf_files,save_path,id,logger=subj_time_logger)
        subj_time_logger.call_end('complete_process',block=True)

    logger.info("Conversion finished") 
    return 0


def process_pdf_files(n_quest,pdf_files,save_path,id,logger=None):
    num_files=len(pdf_files)
    pdf_files.sort()
    if num_files>1: #case in which i have one file per page
        #_t0 = perf_counter()
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            n_page=num_files-i
            doc_path = os.path.join(save_path,id,f"doc_{n_quest}")
            create_folder(doc_path, parents=True, exist_ok=True)
            out_path = os.path.join(doc_path, f"page_{n_page}")
            logger and logger.call_start('extract_images')
            images_data,doc=extract_images(pdf_file)
            logger and logger.call_end('extract_images')
            logger and logger.call_start('save_images')
            save_as_is(doc,0,images_data,out_path) #i always have a single image
            logger and logger.call_end('save_images')
        #_t1 = perf_counter()
        #print(f"time={( _t1 - _t0 ):0.6f}s")
    else: #to change, now i test on q5
        num_files=4
        doc_path = os.path.join(save_path,id,f"doc_{n_quest}")
        create_folder(doc_path, parents=True, exist_ok=True)
        images_data,doc=extract_images(pdf_files[0])
        for i in range(len(images_data)):
            n_page=num_files-i
            out_path = os.path.join(doc_path, f"page_{n_page}")
            save_as_is(doc,i,images_data,out_path) #i always have a single image
    return 0

if __name__ == "__main__":
    main()