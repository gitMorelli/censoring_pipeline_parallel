
import shutil
import os

import cv2
import numpy as np

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes, get_censor_close_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image, convert_to_grayscale
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching, plot_rois_on_image_stackable

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base

from src.utils.censor_utils import save_as_is_no_censoring, save_original_w_boxes, get_transformation_to_match_to_template, apply_transformation_to_boxes


def visualize_templates_w_annotations(annotation_files,annotation_roots,npy_data,source,align=True,censor=False,roi=False, mode='cv2'): 
    for i, annotation_file in enumerate(annotation_files): #document level
        #load the json file
        root = annotation_roots[i]
        pages_in_annotation = get_page_list(root)
        npy_dict = npy_data[i]
        annotation_filename=get_basename(annotation_file, remove_extension=True)

        for img_id in pages_in_annotation:
            pre_computed=npy_dict[img_id]
            align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,img_id)
            debug_boxes=align_boxes
            colors=["red" for i in range(len(align_boxes))]
            parent_path=os.path.join(source,'debug', "templates", f"{annotation_filename}")#, f"censored_page_{n_page}.png")
            create_folder(parent_path, parents=True, exist_ok=True)
            save_debug_path=os.path.join(parent_path, f"{img_id}_template_w_align.png")

            img_path=os.path.join(source,'templates', f"{annotation_filename}",f"page_{img_id}.png")
            img = load_image(img_path, mode=mode, verbose=False)
            plot_rois_on_image(img, debug_boxes, save_debug_path,colors=colors)


def save_w_boxes(save_debug_path,matched_id,img,root,pre_computed,logger,which_boxes=['align','transformed'], transformation=None):
    
    debug_boxes=[]
    box_colors = []
    box_types = []
    colors = {"align":"red", "roi":"red", "censor":"red", "censor_close":"red",
              "new_align":"green", "new_roi":"green", "new_censor":"green", "new_censor_close":"green"}

    if "align" in which_boxes:
        boxes, pre_computed_align = get_align_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["align"] for i in range(len(boxes))]
        box_types+=["align" for i in range(len(boxes))]
    if "roi" in which_boxes:
        boxes, pre_computed_align = get_roi_boxes(root,pre_computed,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["roi"] for i in range(len(boxes))]
        box_types+=["roi" for i in range(len(boxes))]
    if "censor" in which_boxes:
        boxes, pre_computed_align = get_censor_boxes(root,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["censor"] for i in range(len(boxes))]
        box_types+=["censor" for i in range(len(boxes))]
    if "censor_close" in which_boxes:
        boxes, pre_computed_align = get_censor_close_boxes(root,matched_id)
        debug_boxes+=boxes
        box_colors+=[colors["censor_close"] for i in range(len(boxes))]
        box_types+=["censor_close" for i in range(len(boxes))]
    
    if 'transformed' in which_boxes:
        new_image=plot_rois_on_image_stackable(img, debug_boxes, colors=box_colors)

        if (transformation is not None): 
            new_boxes = apply_transformation_to_boxes(debug_boxes, logger, transformation['reference'],transformation['scale_factor'], 
                                                                transformation['shift_x'], transformation['shift_y'], 
                                                                transformation['angle_degrees'],name='align')
            box_colors=[]
            for box_type in box_types:
                box_colors.append(colors[f"new_{box_type}"] )
            plot_rois_on_image_polygons(new_image, new_boxes, save_debug_path,colors=box_colors)
    else:
        plot_rois_on_image(img, debug_boxes, save_debug_path,colors=box_colors)

def save_these_boxes(save_path,img,list_of_boxes,list_of_colors=['red']):
    
    boxes=[]
    colors=[]
    for i in range(len(list_of_boxes)):
        boxes+=list_of_boxes[i]
        colors+=[list_of_colors[i] for j in range(len(list_of_boxes[i]))]

    plot_rois_on_image_polygons(img, boxes, colors=colors, save_path=save_path)

def superimpose_images(img1_path_or_img, img2_path, output_path,logger ,alpha=0.5, threshold_val=200):
    '''superimpose two images and shows the second one in red
    the second is meant to be the template
    '''
    #if the inputs are array i dont need to load
    if not isinstance(img1_path_or_img, np.ndarray):
        img1 = cv2.imread(img1_path_or_img)
    else:
        img1 = img1_path_or_img.copy()
    if not isinstance(img2_path, np.ndarray):
        img2 = cv2.imread(img2_path)
    else:
        img2 = img2_path.copy()

    if img1 is None or img2 is None:
        logger.write("Error: Could not load one or both images. Check file paths.")
        return
    
    img1 = convert_to_grayscale(img1)
    img2 = convert_to_grayscale(img2)

    # 2. Match sizes
    h1, w1 = img1.shape[:2]
    img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)

    # 3. Binarize (Invert so text is White/255 and background is Black/0)
    # This makes it easier to use as a mask
    _, mask1 = cv2.threshold(img1, threshold_val, 255, cv2.THRESH_BINARY_INV)
    _, mask2 = cv2.threshold(img2, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 4. Create a white background canvas
    canvas = np.full((h1, w1, 3), 255, dtype=np.uint8)

    # 5. Apply Colors to the text
    # Image 1 Text -> Cyan (B=255, G=255, R=0)
    # Image 2 Text -> Red (B=0, G=0, R=255)
    
    # We use the mask to 'color' the white canvas where text exists
    canvas[mask1 > 0] = [255, 255, 0]  # Cyan text
    
    # For the second image, we "blend" or overwrite. 
    # To see the overlap, we can use cv2.addWeighted or bitwise logic
    temp_img2 = np.full((h1, w1, 3), 255, dtype=np.uint8)
    temp_img2[mask2 > 0] = [0, 0, 255] # Red text
    
    # Combine: Multiply or Linear Blend
    # Multiply works well for "ink" on paper
    result = cv2.addWeighted(canvas, 0.5, temp_img2, 0.5, 0)

    # 6. Save
    cv2.imwrite(output_path, result)
    logger.write(f"Alignment map saved to {output_path}")