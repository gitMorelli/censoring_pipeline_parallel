import shutil
import os

import cv2

#from src.utils.convert_utils import pdf_to_png_images
from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info, deserialize_keypoints
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree

#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes, get_roi_boxes, get_censor_boxes

from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region

from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching, orb_matching

from src.utils.logging import FileWriter, initialize_logger

from src.utils.matching_utils import update_phash_matches, match_pages_phash, check_matching_correspondence, pre_load_images_to_censor, pre_load_image_properties
from src.utils.matching_utils import compare_pages_same_section, match_pages_text, initialize_sorting_dictionaries, pre_load_selected_templates, perform_template_matching
from src.utils.matching_utils import perform_phash_matching, perform_ocr_matching, ordering_scheme_base





def save_as_is_no_censoring(logger,image_time_logger,img_id,page_dictionary,dest_folder,n_p,n_doc,n_page):
    page=page_dictionary[img_id]
    img_name = page['img_name']
    img_size = page['img_size']
    img = page['img']
    png_img_path = page['img_path']

    logger.write(f"Skip image: id={img_id}, name={img_name}, size={img_size}, no censor regions")
    image_time_logger.call_start('copy_image')

    # Build the destination file path
    dest_path = os.path.join(dest_folder, str(n_p), str(n_doc))
    create_folder(dest_path, parents=True, exist_ok=True)
    save_path=os.path.join(dest_path, f"page_{n_page}.png")

    if img is not None:
        cv2.imwrite(save_path, img)
    else:
        shutil.copy2(png_img_path, save_path)  # copy2 preserves metadata
    image_time_logger.call_end('copy_image')

def save_original_w_boxes(align_boxes, roi_boxes,censor_boxes,source,subj_id,img_id,img,doc_ind):
    debug_boxes=align_boxes+roi_boxes+censor_boxes
    colors=["red" for i in range(len(align_boxes))]+["green" for i in range(len(roi_boxes))]+["blue" for i in range(len(censor_boxes))]
    parent_path=os.path.join(source,'debug', f"patient_{subj_id}", f"document_{doc_ind}")#, f"censored_page_{n_page}.png")
    create_folder(parent_path, parents=True, exist_ok=True)
    save_debug_path=os.path.join(parent_path, f"{img_id}_original_w_boxes.png")
    plot_rois_on_image(img, debug_boxes, save_debug_path,colors=colors)

def save_pre_post_boxes(new_align_boxes, new_roi_boxes, new_censor_boxes, align_boxes,roi_boxes,censor_boxes, source,subj_id, img_id, img, doc_ind):
    debug_boxes=new_align_boxes+new_roi_boxes+new_censor_boxes
    colors=["red" for i in range(len(align_boxes))]+["green" for i in range(len(roi_boxes))]+["blue" for i in range(len(censor_boxes))]
    parent_path=os.path.join(source,'debug', f"patient_{subj_id}", f"document_{doc_ind}")#, f"censored_page_{n_page}.png")
    create_folder(parent_path, parents=True, exist_ok=True)
    save_debug_path=os.path.join(parent_path, f"{img_id}_aligned_w_boxes.png")
    #plot_rois_on_image(img, debug_boxes, save_path,colors=colors)
    plot_both_rois_on_image(img, roi_boxes+align_boxes, new_roi_boxes+new_align_boxes, os.path.join(parent_path, f"{img_id}_both_roi_boxes.png"),color_1="red", color_2="green")
    plot_rois_on_image_polygons(img, debug_boxes, save_debug_path,colors)

def get_transformation_to_match_to_template(page, root, pre_computed, img, img_size, matched_id, image_time_logger):
    align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,matched_id)

    image_time_logger.call_start('compute_misalignement')
    if page['shifts']==None:
        shifts, centers = compute_misalignment(img, align_boxes, img_size,pre_computed_template=pre_computed_align,
                                            scale_factor=2,pre_computed_rois=None)
    else:
        shifts , centers = page['shifts'], page['centers']
    image_time_logger.call_end('compute_misalignement')

    image_time_logger.call_start('compute_transformation')
    scale_factor, shift_x, shift_y, angle_degrees,reference = compute_transformation(shifts, centers)
    image_time_logger.call_end('compute_transformation')
    return scale_factor, shift_x, shift_y, angle_degrees,reference

def get_transformation_from_dictionaries(page, template, image_time_logger, scale_factor=2, method = 'pre_computed',**kwargs):

    if method == 'pre_computed': 
        image_time_logger.call_start('compute_misalignement')
        shifts , centers = page['shifts'], page['centers']
        image_time_logger.call_end('compute_misalignement')

        image_time_logger.call_start('compute_transformation')
        scale_factor, shift_x, shift_y, angle_degrees,reference = compute_transformation(shifts, centers)
        image_time_logger.call_end('compute_transformation')
    elif method in ['orb_page_level_affine','orb_page_level_homography'] :
        # orb properties of the template are already loaded in the template dictionary, also in the page dict if the second step was executed
        template_des = template['orb']
        template_kp = template['orb_kp']
        orb_args = template['orb_args'] #i can take the orb args from any template because they are all the same
        nfeatures = orb_args.get('nfeatures', 500)
        fastThreshold = orb_args.get('fastThreshold', 20)
        edgeThreshold = orb_args.get('edgeThreshold', 31)
        patchSize = orb_args.get('patchSize', 31)
        if page['orb'] is None:
            preprocessed_img = preprocess_page(page['img'])
            pre_comp = extract_features_from_page(preprocessed_img, verbose=False, to_compute=['orb'],
                                                    nfeatures=nfeatures, fastThreshold=fastThreshold, edgeThreshold=edgeThreshold, patchSize=patchSize)
            page_des=pre_comp['orb_des'] #.copy() copy should not be needed if i reinitialize pre_comp in the loop
            page_kp=deserialize_keypoints(pre_comp['orb_kp'])
        else:
            page_des = page['orb']
            page_kp = page['orb_kp']
        if method == 'orb_page_level_affine':
            compute_method = "affine"
        else:
            compute_method = "homography"
        
        orb_parameters = kwargs.get('orb_parameters',{})
        orb_nfeatures = orb_parameters.get('orb_nfeatures',2000)
        orb_match_threshold = orb_parameters.get('orb_match_threshold',10)
        orb_top_n_matches = orb_parameters.get('orb_top_n_matches',50)
        orb_method_to_find_matches = orb_parameters.get('orb_method_to_find_matches','brute_force')
        orb_match_filtering_method = orb_parameters.get('orb_match_filtering_method',"best_n")
        lowe_threshold=orb_parameters.get("orb_lowe_threshold",0.7)
        is_matched, _,shift_x, shift_y, scale_factor, angle_degrees = orb_matching(img=None,box=None,template_properties=None, 
                                                                                 image_kpts=(page_kp,page_des),template_kpts=(template_kp,template_des), 
                                                                                 compute_method=compute_method, 
                                                                                 shift_wr_tl=(0,0), orb_nfeatures=orb_nfeatures, 
                                                                                 match_threshold=orb_match_threshold, top_n_matches=orb_top_n_matches,
                                                                                 method_to_find_matches=orb_method_to_find_matches, 
                                                                                 match_filtering_method=orb_match_filtering_method, lowe_threshold=lowe_threshold)
        reference=(0,0)

        if not is_matched:
            #if the matching fails i can backup to a simpler method that does not use the template matches
            return -1, None, None, None, None

    return scale_factor, shift_x, shift_y, angle_degrees,reference

def apply_transformation_to_boxes(roi_boxes, image_time_logger, reference, scale_factor, shift_x, shift_y, angle_degrees,name='roi',option='standard'):
    if option=='no_rotation' and shift_x!='homography':
        angle_degrees=0
    new_roi_boxes = []
    for coord in roi_boxes: 
        image_time_logger.call_start(f'apply_transformation_{name}') #I should limit the shift/rotation to a certain max value
        new_coord = apply_transformation(reference,coord, scale_factor, shift_x, shift_y, angle_degrees, inverse=False)
        image_time_logger.call_end(f'apply_transformation_{name}')
        new_roi_boxes.append(new_coord)
    return new_roi_boxes

def enlarge_censor_regions(image_time_logger,img_size,scale_factor,censor_boxes):
    image_time_logger.call_start(f'enlarge_{len(censor_boxes)}_censor_regions')
    new_censor_boxes = []
    for coord in censor_boxes:
        new_coord = enlarge_crop_coords(coord, scale_factor=scale_factor, img_shape=img_size)
        new_censor_boxes.append(new_coord)
    censor_boxes = new_censor_boxes[:]
    image_time_logger.call_end(f'enlarge_{len(censor_boxes)}_censor_regions')
    return new_censor_boxes

def save_censored_image(img, censor_boxes, save_path,n_p,n_doc,n_page,warning='00', verbose=False,partial_coverage=None,logger=None,**kwargs):
    parent_path=os.path.join(save_path, f"{n_p}", f"{n_doc}")#, f"censored_page_{n_page}.png")
    create_folder(parent_path, parents=True, exist_ok=True)
    save_path=os.path.join(parent_path, f"censored_page_w{warning}_{n_page}.png")
    censored_img = censor_image(img, censor_boxes, verbose=verbose,partial_coverage=partial_coverage,logger=logger,**kwargs)
    logger and logger.call_start(f'writing_to_memory')
    cv2.imwrite(str(save_path), censored_img)
    logger and logger.call_end(f'writing_to_memory')

def generate_warning_string(decision_1,decision_2,test_log,img_id):
    return str(1-int(decision_1))+str(1-int(decision_2)) #true decision becomes 1 which becomes '0' in the warning


def get_area(rect):
    x1, y1, x2, y2 = rect
    return abs(x2 - x1) * abs(y2 - y1)

def map_to_smallest_containing(list1, list2):
    """
    Maps each rect in list2 to the smallest rect in list1 that contains it.
    Returns a dictionary: {tuple_from_list2: tuple_from_list1_or_None}
    """
    mapping = {}

    for r2 in list2:
        x1b, y1b, x2b, y2b = r2
        best_match = None
        min_area = float('inf')

        for r1 in list1:
            x1a, y1a, x2a, y2a = r1
            
            # Check containment
            if x1a <= x1b and y1a <= y1b and x2a >= x2b and y2a >= y2b:
                area = get_area(r1)
                if area < min_area:
                    min_area = area
                    best_match = r1
        
        mapping[tuple(r2)] = best_match
        
    return mapping

def map_to_all_containing(list1, list2, partial, percentage_threshold=0.5):
    """
    assume list2 is censor-close and list1 is censor. Returns all the censor-close,censor pairs that are superimposed for more than 50%.
    For each it returns the partial keyword copying it from the corresponding censor boox
    """
    new_list_1 = []
    new_list_2 = []
    new_partial = []

    for r2 in list2:
        x1b, y1b, x2b, y2b = r2

        for i,r1 in enumerate(list1):
            x1a, y1a, x2a, y2a = r1
            
            # Compute the fraction of area of r2 that is covered by r1
            intersection_x1 = max(x1a, x1b)
            intersection_y1 = max(y1a, y1b)
            intersection_x2 = min(x2a, x2b)
            intersection_y2 = min(y2a, y2b)
            intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
            r2_area = get_area(r2)
            coverage = intersection_area / r2_area if r2_area > 0 else 0
            if coverage >= percentage_threshold:
                new_list_1.append(r1)
                new_list_2.append(r2)
                new_partial.append(partial[i])
        
    return new_list_1, new_list_2, new_partial

######### CENSORING SCHEMES ####################
def censor_page_base(page_dictionary, img_id, root, npy_dict,logger, image_time_logger, save_path, subj_id, doc_ind, 
                     skip_checking_1, skip_checking_2, save_debug_images, skip_aligning,enlarge_censor_boxes, 
                     global_increase_censoring, source, mode='csv'):
    page=page_dictionary[img_id]
    matched_id=page['matched_page'] #this is different from img_id only if i have reordered the pages
    img_name = page['img_name']
    img_size = page['img_size']
    png_img_path = page['img_path']

    #get censor boxes
    censor_boxes,partial_coverage = get_censor_boxes(root,matched_id) #we need to refer to the correct id of the template
    
    #M: should I also add code to get rid of the very problematic pages that were never matched?
    # the pages matched to templates labelled as N can be kept as they are
    if page_dictionary[matched_id]['type']=='N':
        save_as_is_no_censoring(logger,image_time_logger,img_id,page_dictionary,dest_folder=save_path,
                                n_p=subj_id,n_doc=doc_ind,n_page=matched_id)
    else:
        logger.debug("Processing image: id=%s, name=%s, size=%s", img_id, img_name, img_size)
        #find the corresponding png image in the template folder
        image_time_logger.call_start('load_image')
        if page['img']==None:
            img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
        else:
            img=page['img']
        image_time_logger.call_end('load_image')

        pre_computed = npy_dict[matched_id]
        #logger.debug("Pre-computed data keys for image %s: %s", img_name, pre_computed)

        #check if templates are aligned
        roi_boxes, pre_computed_rois = get_roi_boxes(root,pre_computed,matched_id)
    
        decision_1=False
        if not skip_checking_1:
            decision_1 = page_vote(img, roi_boxes, min_votes=2, template_png=None, pre_computed_rois=pre_computed_rois,logger=image_time_logger)
            #start logging of all nested function if active
        if save_debug_images:
            align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,matched_id)
            save_original_w_boxes(align_boxes, roi_boxes,censor_boxes,source,subj_id,img_id,img,doc_ind)
        
        decision_2=False
        if not skip_aligning and not decision_1:
            image_time_logger.call_start('alignement_and_check',block=True)
            image_time_logger.call_start('alignement_only')
            
            # compute misalignement
            scale_factor, shift_x, shift_y, angle_degrees,reference = get_transformation_to_match_to_template(page, root, pre_computed, img, 
                                                                                                                img_size, matched_id, image_time_logger)
            #new/template , new-template, new-template
            if not skip_checking_2:
                new_roi_boxes = apply_transformation_to_boxes(roi_boxes, image_time_logger, reference, scale_factor, 
                                                                shift_x, shift_y, angle_degrees,name='roi')
                if save_debug_images:
                    new_align_boxes = apply_transformation_to_boxes(align_boxes, image_time_logger, reference, scale_factor, 
                                                                shift_x, shift_y, angle_degrees,name='align')
                
                image_time_logger.call_end('alignement_only')
                decision_2 = page_vote(img, new_roi_boxes, min_votes=2, template_png=None, 
                                        pre_computed_rois=pre_computed_rois,logger=image_time_logger)  
            image_time_logger.call_end('alignement_and_check',block=True)
        
        if decision_1:
            decision_2 = True 

        image_time_logger.call_start('censoring',block=True)
        # increase size of censor boxes // I should increase based on how sure i am of the alignement -> 
        #greater alignement angles imply broader censoring
        new_censor_boxes=censor_boxes[:]
        if enlarge_censor_boxes:
            new_censor_boxes = enlarge_censor_regions(image_time_logger,img_size,new_censor_boxes,scale_factor=global_increase_censoring)
        #transform censor box regions if documents need aligning
        if not skip_aligning and not decision_1:
            new_censor_boxes=apply_transformation_to_boxes(new_censor_boxes, image_time_logger, reference, scale_factor, 
                                                                shift_x, shift_y, angle_degrees,name='censor')
        
        if save_debug_images and not decision_1:
            save_pre_post_boxes(new_align_boxes, new_roi_boxes, new_censor_boxes, align_boxes,roi_boxes,censor_boxes, source,subj_id, img_id, img, doc_ind)
        
        return img, new_censor_boxes, partial_coverage, decision_1, decision_2
        