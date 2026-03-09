from pdf2image import convert_from_path
import fitz  # PyMuPDF
import os
import numpy as np
import cv2

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import save_as_is_no_censoring, save_original_w_boxes, get_transformation_to_match_to_template, apply_transformation_to_boxes
from src.utils.censor_utils import enlarge_censor_regions, save_pre_post_boxes, save_censored_image, generate_warning_string, censor_page_base


POPPLER_PATH = "\\vms-e34n-databr\\2025-handwriting\\programs\\Release-25.11.0-0\\poppler-25.11.0\\Library\\bin" #"Z:\\programs\\Release-25.11.0-0\\poppler-25.11.0\\Library\\bin" #"C:\\Program Files\\poppler-25.11.0\\Library\\bin"

def pdf_to_images(pdf_path):
    """Convert PDF pages to PNG images.

    Args:
        args: Command-line arguments containing the PDF file path and poppler path.

    Returns:
        List of PNG images converted from the PDF.
    """
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    return images

def get_n_pages(pdf_path): 
    ''' given a pdf file it returns the number of pages'''
    with fitz.open(pdf_path) as doc:
        return len(doc)

def save_as_is(pdf_path,i,out_path, return_image = False):
    with fitz.open(pdf_path) as doc:
        page = doc[i]
        # get list of images for this specific page
        page_images = page.get_images(full=True)  

        # take first image on this page
        img = page_images[0]
        
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image["ext"]
        if return_image:
            # Convert bytes to a numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode the array into an OpenCV image (BGR format)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return cv_img
        else:
            out_path = out_path+f".{ext}"
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            return 0


def process_pdf_files(n_quest,pdf_files,save_path, save=True, test_log = {}):
    ''' if save is false it returns the list of images instead of saving the pngs to memory
    If groups is provided it overrides the specification of the groups in the function'''
    #num_files=len(pdf_files)
    expected_properties ={
        "1":{"num_pages":4,"order": "alphabetical","stored_as":"multi-page"}, #two layouts
        "2":{"num_pages":8,"order": "random","stored_as":"multi-page"}, #casi visti: often missing pages, p1 e p2 in un file unico chiamato Qx e gli altri in file separati
        #non ho ancora trovato un pattern
        "3":{"num_pages":2,"order": "alphabetical","stored_as":"multi-page"}, 
        "4":{"num_pages":4,"order": "reverse","stored_as":"multi-page"}, #casi: anche presente come singolo pdf con tutte le pagine in ordine inverso (inizia con Qx)
        "5":{"num_pages":4,"order": "reverse","stored_as":"multi-page"}, #casi: anche presente come singolo pdf con tutte le pagine in ordine inverso (inizia con Qx)
        "6":{"num_pages":4,"order": "alphabetical","stored_as":"multi-page"},# casi: pagine mancanti
        "7":{"num_pages":4,"order": "alphabetical","stored_as":"multi-page"},# casi: pagine mancanti
        "8":{"num_pages":32,"order": "alphabetical","stored_as":"single"},# casi: pagine mancanti
        "9":{"num_pages":4,"order": "alphabetical","stored_as":"multi-page"},# casi: pagine mancanti
        "10":{"num_pages":12,"order": "alphabetical","stored_as":"single"},# sia un questionario di 2 pagine con adesione familiari sia uno di 12 (che comprende anche quelle 2)
        "11":{"num_pages":8,"order": "alphabetical","stored_as":"single"},
        "12":{"num_pages":12,"order": "alphabetical","stored_as":"single"},
        "13":{"num_pages":20,"order": "alphabetical","stored_as":"single"},

    }
    properties = expected_properties.get(str(n_quest), None)
    if properties is None:
        raise ValueError(f"Unknown n_quest: {n_quest}. No expected properties defined.")
    #expected  properties
    expected_pages = properties["num_pages"] 
    expected_order = properties["order"]
    expected_storage = properties["stored_as"]
    
    #i sort the pages according to the expected ordering -> the single pages are already ordered correctly
    if expected_order=="alphabetical":
        # sort the files by name
        pdf_files.sort()
    else:
        # sort and reverse
        pdf_files.sort()
        pdf_files.reverse()
    
    individual_pages = []
    multi_pages = []
    for pdf_file in pdf_files:
        n_pages=get_n_pages(pdf_file)
        if n_pages == 1:
            individual_pages.append(pdf_file)
        else:
            multi_pages.append((n_pages, pdf_file))
    
    images_list = []
    already_extracted_pages=0
    if len(multi_pages) > 0 :
        #sort multi_pages in descending order of the first tuple element
        multi_pages.sort(key=lambda x: x[0], reverse=True)
        already_extracted_pages=multi_pages[0][0]
        pdf_file = multi_pages[0][1]
        if save:
            sub_folder_name=get_basename(pdf_file,remove_extension=True)
            doc_path = os.path.join(save_path, sub_folder_name)
            create_folder(doc_path, parents=True, exist_ok=True)
        for j in range(already_extracted_pages):

            if expected_order=="alphabetical":
                page_index = j
            else:
                page_index = already_extracted_pages-j-1
            
            if save:
                out_path = os.path.join(doc_path, f"page_{j+1}")
                save_as_is(pdf_file,page_index,out_path) #i always have a single image
            else:
                images_list.append( save_as_is(pdf_file,page_index,None,return_image=True) ) 
    #passo a quelle individuali solo se non ho già estratto tutte le pagine che mi aspettavo da quelle multi-page
    if len(individual_pages) > 0 and (already_extracted_pages < expected_pages):
        already_extracted_pages += len(individual_pages)
        #remember they are already sorted
        for pdf_file in individual_pages:

            if save:
                file_name = get_basename(pdf_file,remove_extension=True)
                create_folder(save_path, parents=True, exist_ok=True)
                out_path = os.path.join(save_path, file_name)
                save_as_is(pdf_file,0,out_path) #i always have a single image
                
            else:
                images_list.append( save_as_is(pdf_file,0,None,return_image=True) )
    
    test_log["extracted_pages"] = already_extracted_pages 
    test_log["expected_pages"] = expected_pages
    test_log["multi_page_files"] = len(multi_pages)
    test_log["individual_page_files"] = len(individual_pages)
    if len(multi_pages) > 0:
        test_log["multi_page_largest"] = [multi_pages[0][0]] 
    else:
        test_log["multi_page_largest"] = None

    return images_list, test_log

