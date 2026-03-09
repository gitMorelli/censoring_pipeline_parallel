#!/usr/bin/env python3
import argparse
import logging
import os
from src.utils.convert_utils import pdf_to_images,save_as_is,extract_images
from src.utils.file_utils import list_files_with_extension, get_basename, create_folder,list_subfolders,get_page_number
from time import perf_counter

logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEMPLATES_PATH="//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\pdf_templates" # "Z:\\vscode\\censor_e3n\\data\\q5_tests\\pdf_templates" #"C:\\Users\\andre\\PhD\\Datasets\\e3n\\e3n templates"
SAVE_PATH="//vms-e34n-databr/2025-handwriting\\vscode\\censor_e3n\\data\\q5_tests\\templates" #"Z:\\vscode\\censor_e3n\\data\\q5_tests\\templates" #"C:\\Users\\andre\\PhD\\Datasets\\e3n\\e3n templates\\png_output"

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
        pdf_files = list_files_with_extension(pdf_path, "pdf", recursive=False)
        logger.info("Found %d PDF file(s) in %s", len(pdf_files), pdf_path)
        if not pdf_files:
            logger.warning("No PDF files found. Exiting.")
            return 0

        #pdf_names = [get_basename(pdf_file, remove_extension=True) for pdf_file in pdf_files]
        n_template=get_page_number(pdf_path) #get the number of the questionnaire
        process_pdf_files(n_template,pdf_files,save_path)

    logger.info("Conversion finished")
    return 0


def process_pdf_files(n_quest,pdf_files,save_path):
    #num_files=len(pdf_files)
    group_1=[5] #in this group the templates are saved as separate pdf files, each a tiff image
    group_2=[13] #in this group the templates are saved as single pdf with all the pages
    group_3=[8] #they have both alimentaire and health questionnaire together
    group_4=[]#like group 2 but inverse order of the pages
    if n_quest in group_1:
        #_t0 = perf_counter()
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            n_page=get_page_number(pdf_file)
            doc_path = os.path.join(save_path, f"doc_{n_quest}")
            create_folder(doc_path, parents=True, exist_ok=True)
            out_path = os.path.join(doc_path, f"page_{n_page}")
            images_data,doc=extract_images(pdf_file)
            save_as_is(doc,0,images_data,out_path) #i always have a single image
        #_t1 = perf_counter()
        #print(f"time={( _t1 - _t0 ):0.6f}s")
            
    elif n_quest in group_2: #to change, now i test on q5
        for i, pdf_file in enumerate(pdf_files): #iterate on the pdf files in Q_i
            n_page=get_page_number(pdf_file)
            try:
                images = pdf_to_images(pdf_file)
            except Exception:
                logger.exception("Failed to convert PDF to images: %s", pdf_file)
                continue

            for j, image in enumerate(images):
                doc_path = os.path.join(save_path, f"doc_{n_quest}")
                try:
                    create_folder(doc_path, parents=True, exist_ok=True)
                    out_path = os.path.join(doc_path, f"page_{j + 1}.png")
                    logger.debug("Saving image to %s", out_path)
                    image.save(out_path, "PNG")
                    logger.info("Saved %s", out_path)
                except Exception:
                    logger.exception("Failed to save image for %s page %d", pdf_file, j + 1)
    return 0

if __name__ == "__main__":
    main()