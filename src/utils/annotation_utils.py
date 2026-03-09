
import argparse
import logging
import os
import json

from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree, load_templates_tree

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


def precompute_features_on_template_page(bb_list, img,img_size, ocr_psm, crop_path_pctg, mode='cvs'): 
    properties_list = []
    #iterate on the selected boxes and precompute features
    for box in bb_list:
        #get coordinates
        box_coords=get_box_coords_json(box,img_size)
        #check what kind of box it is
        #i extract and save features for roi, roi-blank, and align boxes
        pre_comp = None
        if box['sub_attribute']=='align':
            # iextract orbs features from align regions
            patch = preprocess_roi(img, box_coords, mode=mode, verbose=False,target_size=None)
            pre_comp = extract_features_from_roi(patch, mode=mode, 
                                                verbose=False,to_compute=['orb'])
            #i extract the whole image for template matching 
            pre_comp['full']=preprocess_alignment_roi(img, box_coords, mode=mode, verbose=False)
            if pre_comp['orb_des'] is None:
                raise ValueError(f"ORB feature extraction failed for align box")
        elif box['sub_attribute']=='blank':
            patch = preprocess_blank_roi(img, box_coords, mode=mode, verbose=False)
            pre_comp = extract_features_from_blank_roi(patch, mode=mode, verbose=False,to_compute=['cc','n_black'])
        elif box['sub_attribute']=='standard' and box['label']=='roi':
            patch = preprocess_roi(img, box_coords, mode=mode, verbose=False,target_size=None)
            '''plt.imshow(patch, cmap='gray')
            plt.title("Preprocessed Patch") 
            plt.axis('off')
            plt.show()'''
            pre_comp = extract_features_from_roi(patch, mode=mode, 
                                                verbose=False,to_compute=['orb'])
            pre_comp['full']=preprocess_alignment_roi(img, box_coords, mode=mode, verbose=False) #for template matching
        elif box['sub_attribute']=='text' and box['label']=='roi': #i may extract differen features for this wrt to standard roi
            patch = preprocess_roi(img, box_coords, mode=mode, verbose=False, target_size=None)
            pre_comp = extract_features_from_roi(patch, mode=mode, 
                                                verbose=False,to_compute=['orb'])
            pre_comp['full']=preprocess_alignment_roi(img, box_coords, mode=mode, verbose=False) #for template matching
        elif box['sub_attribute']=='ocr':
            patch = preprocess_text_region(img, box_coords, mode=mode, verbose=False)
            pre_comp = extract_features_from_text_region(patch, mode=mode, 
                                                verbose=True,psm=ocr_psm)
        properties_list.append(pre_comp)
    
    #precompute features for the whole page
    preprocessed_img = preprocess_page(img)
    pre_comp = extract_features_from_page(preprocessed_img, mode=mode, verbose=False,to_compute=['page_phash','orb'],border_crop_pct=crop_path_pctg)
    properties_list.append(pre_comp) #i add as -1 element, recall when you perform the censoring

    return properties_list

def precompute_and_store_template_properties(annotation_files, template_folders, logger, save_path, annotation_file_names, template_folder_names,ocr_psm,crop_patch_pctg, mode='csv'):
    def get_corresponding(annotation_name, template_folder_names):
        for j,template_name in enumerate(template_folder_names):
            if template_name == annotation_name:
                return j
    #i can access them by index since they are sorted in the same way
    for i, annotation_file in enumerate(annotation_files):
        logger.write(f"Processing file {i + 1}/{len(annotation_files)}: {annotation_file}")
        i_corr = get_corresponding(annotation_file_names[i],template_folder_names)
        doc_path = template_folders[i_corr]
        #pages = list_files_with_extension(doc_path, "png", recursive=False)
        #load the json file
        with open(annotation_file, 'r') as f: json_data = json.load(f)
        pages_in_annotation = get_page_list(json_data)
        #iterate on the images in the annotation file (page index)
        data_dict = {}
        for img_id in pages_in_annotation:

            img_name=f'page_{img_id}.png'
            img_size = get_page_dimensions(json_data,img_id)
            logger.write(f"Processing image: id={img_id}, name={img_name}, size={img_size}")
            #find the corresponding png image in the template folder
            png_img_path = os.path.join(doc_path, img_name)
            if not os.path.exists(png_img_path):
                logger.write(f"PNG image not found for annotation image {img_name}: expected at {png_img_path}")
                continue
            #load image with cv2
            img=load_image(png_img_path, mode=mode, verbose=False)
            bb_list=get_attributes_by_page(json_data, img_id)

            properties_list = precompute_features_on_template_page(bb_list, img,img_size, ocr_psm, crop_patch_pctg, mode=mode)
            
            data_dict[img_id]=properties_list[:] 

        #save data_dict as npy file
        save_folder = create_folder(save_path, parents=True, exist_ok=True)
        save_file_path = os.path.join(save_folder, f"{annotation_file_names[i]}.npy")
        np.save(save_file_path, data_dict)
    return 0
