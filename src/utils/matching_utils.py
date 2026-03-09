import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.optimize import linear_sum_assignment
import re
import unicodedata
from difflib import SequenceMatcher
import os
import math
import pytesseract #for ocr
pytesseract.pytesseract.tesseract_cmd = r'//vms-e34n-databr/2025-handwriting\programs\tesseract\tesseract.exe'

from src.utils.file_utils import list_subfolders,list_files_with_extension,sort_files_by_page_number,get_page_number, load_template_info, deserialize_keypoints
from src.utils.file_utils import get_basename, create_folder, check_name_matching, remove_folder, load_annotation_tree, load_subjects_tree
#from src.utils.xml_parsing import load_xml, iter_boxes, add_attribute_to_boxes
#from src.utils.xml_parsing import save_xml, iter_images, set_box_attribute,get_box_coords

from src.utils.json_parsing import get_attributes_by_page, get_page_list, get_page_dimensions,get_box_coords_json, get_censor_type
from src.utils.json_parsing import get_align_boxes, get_ocr_boxes


from src.utils.feature_extraction import crop_patch, preprocess_alignment_roi, preprocess_roi, preprocess_blank_roi,load_image
from src.utils.feature_extraction import extract_features_from_blank_roi, extract_features_from_roi,censor_image, resize_patch_asymmetric
from src.utils.feature_extraction import extract_features_from_page, preprocess_page, extract_features_from_text_region, preprocess_text_region
from src.utils.alignment_utils import page_vote,compute_transformation, compute_misalignment,apply_transformation,enlarge_crop_coords, rescale_box_coords_given_resolutions
from src.utils.alignment_utils import plot_rois_on_image_polygons,plot_rois_on_image,plot_both_rois_on_image,template_matching
from src.utils.logging import FileWriter, initialize_logger


def check_matching_correspondence(page_dict, pages_list):
    non_corresponding_subset=[]
    for img_id in pages_list:
        if page_dict[img_id]['match_phash']!=page_dict[img_id]['matched_page']:
            non_corresponding_subset.append(img_id)
    return non_corresponding_subset


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    return int(np.count_nonzero(hash1 ^ hash2))

def match_pages_phash(page_dict, template_dict, pages_list, templates_to_consider, 
                      gap_threshold=5, max_dist=18, compute_report = False): #i define this function to avoid breaking old code 
    return match_pages(page_dict, template_dict, pages_list, templates_to_consider, 
                      gap_threshold=gap_threshold, max_dist=max_dist, compute_report = compute_report,type="phash")

def match_pages(page_dict, template_dict, pages_list, templates_to_consider, 
                      gap_threshold=5, max_dist=18,orb_good_match=50, compute_report = False,type="phash"):
    if type=="phash":
        key_value = 'page_phash'
    elif type=="orb":
        key_value = 'orb'
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #assume phash are already computed
    hashes_pages = []
    hashes_templates = []

    #length of pages and templates may be different if templates only has the ones to censor
    for img_id in pages_list:
        hashes_pages.append(page_dict[img_id][key_value])
    for t_id in templates_to_consider:
        hashes_templates.append(template_dict[t_id][key_value])

    # Cost matrix: Hamming distances
    cost = np.zeros((len(hashes_pages), len(hashes_templates)), dtype=np.int32)
    for i in range(len(hashes_pages)):
        for j in range(len(hashes_templates)):
            if type=="phash":
                cost[i, j] = hamming_distance(hashes_pages[i], hashes_templates[j])
            else:
                # Match descriptors
                matches = bf.match(hashes_pages[i], hashes_templates[j])
                # Sort matches by distance (lower distance = better match)
                matches = sorted(matches, key=lambda x: x.distance)
                # Count "good" matches (those with a distance below a threshold)
                good_matches = [m for m in matches if m.distance < orb_good_match]
                cost[i, j] = -len(good_matches) #the more good matches the better, but the Hungarian algorithm minimizes the cost, so I can take the negative of this value


    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement 

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "hamming": int(cost[i, j]),
        })
    
    if compute_report:
        report = assignment_report(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)

        matches_sorted = sorted(matches, key=lambda x: x["page_index"])

        return matches_sorted, cost, report
    else:
        matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost 


def hungarian_min_cost(cost: np.ndarray):
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


#can be used also for other types of matching, just need to change the thresholds
def assignment_report(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - Only evaluates rows that were actually matched.
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    row_ind, col_ind = assignment
    # Create a mapping only for matched rows to avoid KeyErrors
    row_to_col = {r: c for r, c in zip(row_ind, col_ind)}
    
    per_row = []
    all_rows_confident = True

    # Iterate ONLY over the matched row indices
    for r in row_ind:
        row = cost[r].astype(int)
        
        # 1. Calculate best and second best in the row
        if row.size >= 2:
            # partition(1) puts the 2 smallest at indices 0 and 1
            sorted_vals = np.partition(row, 1)
            best_in_row = int(sorted_vals[0])
            second_best = int(sorted_vals[1])
        else:
            best_in_row = int(row[0])
            second_best = 10**9 # Infinity if only one template exists

        gap = second_best - best_in_row

        # 2. Check the actually chosen value (from Hungarian matching)
        chosen_c = row_to_col[r]
        chosen_val = int(cost[r, chosen_c])

        # 3. Confidence Logic
        # A match is confident if it's close enough AND significantly better than the runner-up
        row_ok = (gap >= gap_threshold) and (chosen_val <= max_dist)
        
        if not row_ok:
            all_rows_confident = False

        per_row.append({
            "row": int(r),
            "best_possible": best_in_row,
            "second_best": second_best,
            "gap": gap,
            "chosen_val": chosen_val,
            "ok": row_ok
        })

    # Summary Stats
    total_cost = int(sum(cost[r, c] for r, c in zip(row_ind, col_ind)))
    num_matches = len(row_ind)
    avg_cost = float(total_cost) / num_matches if num_matches > 0 else 0

    report = {
        "is_confident": all_rows_confident,
        "total_cost": total_cost,
        "avg_cost": avg_cost,
        "num_matches": num_matches,
        "per_row": per_row,
    }
    
    return report

############     ################
############ OCR ################
############     ################


# -----------------------------
# Text normalization
# -----------------------------

def normalize_text(s: str) -> str:
    """
    Normalize in a way that tends to help comparisons:
    - Unicode normalize
    - lowercase
    - collapse whitespace
    - remove “weird” punctuation except basic ones (tweak to your needs)
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()

    # keep letters/numbers/basic punctuation; drop the rest
    s = re.sub(r"[^\w\s\.\,\:\;\-\(\)\/%]", " ", s, flags=re.UNICODE)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Similarity measures
# -----------------------------

def sequence_similarity(a: str, b: str) -> float:
    """
    Character-level similarity in [0,1].
    """
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity_tokens(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity in [0,1].
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment_similarity(a: str, b: str) -> float:
    """
    How much of the shorter text's tokens are contained in the longer text.
    Useful when one side has extra boilerplate.
    """
    ta = a.split()
    tb = b.split()
    sa, sb = set(ta), set(tb)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small & big) / max(1, len(small))


def compare_pages_same_section(raw1, raw2):
    n1 = normalize_text(raw1)
    n2 = normalize_text(raw2)

    return {
        "raw_text_1": raw1,
        "raw_text_2": raw2,
        "normalized_text_1": n1,
        "normalized_text_2": n2,
        "similarity_sequence": sequence_similarity(n1, n2),
        "similarity_jaccard_tokens": jaccard_similarity_tokens(n1, n2),
        "similarity_containment": containment_similarity(n1, n2),
    }

def match_pages_text(pages_list,templates_to_consider,similarity, compute_report=False, gap_threshold=0.1, max_dist=0.2):
    #assume phash are already computed
    cost = similarity*(-1)+1
    # Hungarian assignment (minimize total cost)
    assignement = linear_sum_assignment(cost)
    row_ind, col_ind = assignement

    '''if the dimension of the two lists is different -> Every single template will be assigned to a page.
    The function will pick the subset of pages that results in the lowest total Hamming distance.
    The "leftover" pages will be ignored. Because linear_sum_assignment only returns a match for the smaller dimension, 
    the row_ind and col_ind arrays will only have a length equal to the number of templates'''
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "page_index": pages_list[i],
            "template_index": templates_to_consider[j],
            "similarity": int(-1*(cost[i, j]-1)), # i invert again the formula to get the similarity
        })
    
    if compute_report:
        report = assignment_report(cost, assignement, gap_threshold=gap_threshold, max_dist=max_dist)

        matches_sorted = sorted(matches, key=lambda x: x["page_index"])

        return matches_sorted, cost, report
    else:
        matches_sorted = sorted(matches, key=lambda x: x["page_index"])

    return matches_sorted, cost 

def assignment_confidence_text(cost: np.ndarray, assignment, gap_threshold: int, max_dist: int):
    """
    Returns (is_confident, report_dict).
    Confidence logic:
      - per-row gap test: (2nd best - best) >= gap_threshold
      - matched distance <= max_dist
    """
    n = cost.shape[0]
    row_to_col = {r: c for r, c in assignment}

    per_row = []
    confident = True

    for r in range(n):
        row = cost[r].astype(int)
        best = int(row.min())
        # second best: take smallest two
        sorted_vals = np.partition(row, 1)[:2]
        second = int(sorted_vals[1]) if len(sorted_vals) > 1 else 10**9
        gap = second - best

        chosen_c = row_to_col[r]
        chosen = int(cost[r, chosen_c])

        row_ok = (gap >= gap_threshold) and (chosen <= max_dist)
        if not row_ok:
            confident = False

        per_row.append({
            "row": r,
            "best": best,
            "second": second,
            "gap": gap,
            "chosen": chosen,
            "ok": row_ok
        })

    total = int(sum(cost[r, c] for r, c in assignment))
    avg = float(total) / float(n)

    report = {
        "total_cost": total,
        "avg_cost": avg,
        "gap_threshold": gap_threshold,
        "max_dist": max_dist,
        "per_row": per_row,
    }
    return confident, report

def find_corresponding_file(sorted_files, img_name):
    index=get_page_number(img_name)
    if index <= len(sorted_files):
        return sorted_files[index-1]
    return None #the sorted files are supposed to be page_1,page_2, .. (they are sorted by number)

def initialize_page_dictionary(sorted_files,input_from_file=True):
    page_dictionary = {}
    for i,file in enumerate(sorted_files):
        page_dictionary[i+1]={}  
    for i,file in enumerate(sorted_files):
        img_id=i+1
        page_dictionary[img_id]['img_id']=i+1
        img_name=f'page_{img_id}.png'
        page_dictionary[img_id]['img_name']=img_name 
        if input_from_file:
            png_img_path = find_corresponding_file(sorted_files, img_name)
            page_dictionary[img_id]['img_path']=png_img_path
            page_dictionary[img_id]['img']=None #i willload is needed to not waste time
            page_dictionary[img_id]['img_size']=None
        else:
            page_dictionary[img_id]['img_path']=None
            page_dictionary[img_id]['img']=file.copy() #the images are already loaded, i put them in the dictionary 
            height, width = file.shape[:2] 
            page_dictionary[img_id]['img_size'] = (width, height)

        #template matching properties
        page_dictionary[img_id]['template_matches']=0 #how many time this page was matched with a template
        page_dictionary[img_id]['shifts']=None #(shift_x,shift_y) for first qnd second region
        page_dictionary[img_id]['centers']=None #(shift_x,shift_y) for first qnd second region
        page_dictionary[img_id]['stored_template']=None # to store the features extracted from the align regions 
        #for the page (will be overwritten each time i compare with diff template)
        page_dictionary[img_id]['matched_page']=None #initially I assume the page is matched to the same index template
        page_dictionary[img_id]['matched_page_list']=[] #holds the list of all successive matches for the page
        page_dictionary[img_id]['confidence_template']=None

        #phash properties
        page_dictionary[img_id]['page_phash']=None
        page_dictionary[img_id]['match_phash']=None
        page_dictionary[img_id]['report_phash']=None
        #orb properties
        page_dictionary[img_id]['orb']=None
        page_dictionary[img_id]['orb_kp']=None
        page_dictionary[img_id]['report_orb']=None
        page_dictionary[img_id]['match_orb']=None
        #ocr properties
        page_dictionary[img_id]['text']=None
        page_dictionary[img_id]['match_ocr']=None
        page_dictionary[img_id]['report_ocr']=None
        # M: there is some redundancy -> rewrite the keys to make it less crowded
    return page_dictionary

def initialize_template_dictionary(root):
    pages_in_annotation = get_page_list(root)
    template_dictionary = {} 
    for p in pages_in_annotation:
        template_dictionary[p]={}
    #iterate on the pages in a document and initialize their parameters
    for img_id in pages_in_annotation:
        censor_type=get_censor_type(root,img_id) 
        template_dictionary[img_id]['type']=censor_type
        template_dictionary[img_id]['align_boxes']=None #the coordinates of the align boxes for a template
        template_dictionary[img_id]['pre_computed_align']=None #the pre computed values for the align region in the template
        template_dictionary[img_id]['matched_to_this']=0
        template_dictionary[img_id]['page_phash']=None
        template_dictionary[img_id]['final_match']=None
        template_dictionary[img_id]['text']=None
        template_dictionary[img_id]['text_box']=None
        template_dictionary[img_id]['psm']=None
        template_dictionary[img_id]['template_size']=None
        #template_dictionary[img_id]['text']=None
    return template_dictionary

def initialize_sorting_dictionaries(sorted_files, root,mode='cv2',input_from_file=True):
    ''' this function takes one json annotation file called root (for one template) 
    takes a list of file_paths and returns the page_dictionary, and template_dictionary (each initialized to the starting values)'''

    page_dictionary = initialize_page_dictionary(sorted_files,input_from_file=input_from_file)

    template_dictionary = initialize_template_dictionary(root)
    
    return page_dictionary,template_dictionary

def pre_load_images_to_censor(template_dictionary,page_dictionary, mode='csv'):
    ''' takes the initialized dictionaries and return the updated dictionaries (with pre-loaded values) and the 
    list of template ids'''
    templates_to_consider=[]
    pages_in_annotation = list(page_dictionary.keys())
    for img_id in pages_in_annotation:
        censor_type=template_dictionary[img_id]['type']
        png_img_path=page_dictionary[img_id]['img_path']
        #i load in memory only the pages that needs censoring or partial censoring at the beginning
        if censor_type!='N':
            img=load_image(png_img_path, mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
            page_dictionary[img_id]['img']=img.copy()
            height, width = img.shape[:2] 
            page_dictionary[img_id]['img_size'] = (width, height)
            templates_to_consider.append(img_id)
        else:
            page_dictionary[img_id]['img']=None
    return page_dictionary,template_dictionary, templates_to_consider

def pre_load_selected_templates(templates_to_consider,npy_dict, root, template_dictionary):
    for t_id in templates_to_consider:
        pre_computed = npy_dict[t_id]
        align_boxes, pre_computed_align = get_align_boxes(root,pre_computed,t_id) 
        text_boxes, pre_computed_texts = get_ocr_boxes(root,pre_computed,t_id) #i know that i have a single text_box ->
        text_box, pre_computed_text = text_boxes[0], pre_computed_texts[0]['text']
        psm = pre_computed_texts[0]['psm']

        template_dictionary[t_id]['align_boxes']=align_boxes
        template_dictionary[t_id]['pre_computed_align']=pre_computed_align
        template_dictionary[t_id]['page_phash']=pre_computed[-1]['page_phash']
        template_dictionary[t_id]['border_crop_pct']=pre_computed[-1]['border_crop_pct'] 
        template_dictionary[t_id]['text']=pre_computed_text
        template_dictionary[t_id]['text_box']=text_box
        template_dictionary[t_id]['psm']=psm
        template_dictionary[t_id]['orb']=pre_computed[-1]['orb_des']
        template_dictionary[t_id]['orb_kp']=deserialize_keypoints(pre_computed[-1]['orb_kp'])
        template_dictionary[t_id]['orb_args']=pre_computed[-1]['orb_args']
        template_dictionary[t_id]['template_size'] = get_page_dimensions(root, t_id)
    return template_dictionary

def pre_load_image_properties(pages_to_consider,page_dictionary,template_dictionary,properties=[],mode='csv'):
    '''given a list of images and the properties to pre-compute it updates the appropriate 
    keys in the dictionary '''
    CROP_PATCH_PCTG = template_dictionary[1]['border_crop_pct'] #i can get this parameter from any page template really
    for img_id in pages_to_consider:
        if 'img' in properties:
            if page_dictionary[img_id]['img'] is None:
                img=load_image(page_dictionary[img_id]['img_path'], mode=mode, verbose=False) #modify code to manage tiff and jpeg if needed
                page_dictionary[img_id]['img']=img.copy()
                height, width = img.shape[:2] 
                page_dictionary[img_id]['img_size'] = (width, height)
        if 'phash' in properties: #should follow im loading because it requires the image to be in the dictionary already
            if page_dictionary[img_id]['page_phash'] is None:
                preprocessed_img = preprocess_page(page_dictionary[img_id]['img'])
                pre_comp = extract_features_from_page(preprocessed_img, mode=mode, verbose=False,to_compute=['page_phash'],border_crop_pct=CROP_PATCH_PCTG)
                page_dictionary[img_id]['page_phash']=pre_comp['page_phash'] #.copy() copy should not be needed if i reinitialize pre_comp in the loop
        if 'orb' in properties: #should follow im loading because it requires the image to be in the dictionary already
            if page_dictionary[img_id]['orb'] is None:
                preprocessed_img = preprocess_page(page_dictionary[img_id]['img'])
                orb_args = template_dictionary[1]['orb_args'] #i can take the orb args from any template because they are all the same
                nfeatures = orb_args.get('nfeatures', 500)
                fastThreshold = orb_args.get('fastThreshold', 20)
                edgeThreshold = orb_args.get('edgeThreshold', 31)
                patchSize = orb_args.get('patchSize', 31)
                pre_comp = extract_features_from_page(preprocessed_img, mode=mode, verbose=False, to_compute=['orb'],
                                                      nfeatures=nfeatures, fastThreshold=fastThreshold, edgeThreshold=edgeThreshold, patchSize=patchSize)
                page_dictionary[img_id]['orb']=pre_comp['orb_des'] #.copy() copy should not be needed if i reinitialize pre_comp in the loop
                page_dictionary[img_id]['orb_kp']=deserialize_keypoints(pre_comp['orb_kp'])
    return page_dictionary

def perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, n_align_regions,scale_factor, 
                              matching_threshold=0.7,compute_report=False,metric="matchTemplate",**kwargs):
    '''expects a list or a list of pairs, if only a list is provided it means we consider the list of pairs i,i j,j k,k ..
    the first element is the id of a page and the second is the id of the template'''
    if pairs_to_consider and isinstance(pairs_to_consider[0], (int, float)):
        #the input is a list of numbers
        new_pairs_to_consider=[]
        for i in pairs_to_consider:
            new_pairs_to_consider.append([i,i])
        pairs_to_consider = new_pairs_to_consider[:]
    
    for pair in pairs_to_consider:
        img_id,t_id = pair 
        align_boxes = template_dictionary[t_id]['align_boxes']
        pre_computed_align = template_dictionary[t_id]['pre_computed_align']
        #i rescale align boxes based on the resolution of the page and template before checking
        align_boxes = rescale_box_coords_given_resolutions(align_boxes, template_dictionary[t_id]['template_size'], page_dictionary[img_id]['img_size'])
        #this is the rescale facto to resize the align region in the image to the size of the align region in the template
        rescale_x = template_dictionary[t_id]['template_size'][0]/page_dictionary[img_id]['img_size'][0]
        rescale_y = template_dictionary[t_id]['template_size'][1]/page_dictionary[img_id]['img_size'][1]
        if compute_report:
            shifts, centers, processed_rois, report = compute_misalignment(page_dictionary[img_id]['img'], align_boxes, page_dictionary[img_id]['img_size'], 
                                pre_computed_template=pre_computed_align,scale_factor=scale_factor, 
                                matching_threshold=matching_threshold, return_confidences=True,metric=metric,
                                rescale_x_y=(rescale_x,rescale_y), **kwargs)  
        else:
            shifts, centers, processed_rois = compute_misalignment(page_dictionary[img_id]['img'], align_boxes, page_dictionary[img_id]['img_size'], 
                                    pre_computed_template=pre_computed_align,scale_factor=scale_factor, 
                                    matching_threshold=matching_threshold,metric=metric,
                                    rescale_x_y=(rescale_x,rescale_y), **kwargs) #recall this functions returns a shift for each good match
            #thus you expect len=2 for the shift variable, instead processed_rois returns all regions
            report = None
        
        if len(shifts)>=n_align_regions:
            page_dictionary[img_id]['template_matches']+=1 #should be +=1
            #page_dictionary[img_id]['stored_template'] = processed_rois #save the rois so i can re-use them without recomputing
            page_dictionary[img_id]['matched_page']=t_id
            page_dictionary[img_id]['matched_page_list'].append(t_id)
            template_dictionary[t_id]['matched_to_this']+=1
            page_dictionary[img_id]['shifts'] = shifts
            page_dictionary[img_id]['centers'] = centers 
        page_dictionary[img_id]['confidence_template'] = report
    return page_dictionary,template_dictionary

def perform_phash_matching(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                            gap_threshold,max_dist, compute_report = False):
    if compute_report:
        matches_sorted, _,report  = match_pages_phash(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                                gap_threshold=gap_threshold,max_dist=max_dist, compute_report=compute_report) #As of now i don't consider the confidence of the matching, but I may in future versions
        page_dictionary = update_phash_matches(matches_sorted,page_dictionary)
        return page_dictionary, report
    else:
        matches_sorted, _ = match_pages_phash(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                                gap_threshold=gap_threshold,max_dist=max_dist, compute_report=compute_report) #As of now i don't consider the confidence of the matching, but I may in future versions
        page_dictionary = update_phash_matches(matches_sorted,page_dictionary)
    return page_dictionary

def perform_orb_matching(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                            gap_threshold,max_dist,orb_good_match,compute_report = False):
    if compute_report:
        matches_sorted, _,report  = match_pages(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                                gap_threshold=gap_threshold,max_dist=max_dist, compute_report=compute_report, orb_good_match = orb_good_match, type = 'orb') #As of now i don't consider the confidence of the matching, but I may in future versions
        page_dictionary = update_phash_matches(matches_sorted,page_dictionary,type='orb')
        return page_dictionary, report
    else:
        page_dictionary = match_pages(page_dictionary,template_dictionary, pages_list, templates_to_consider, 
                                gap_threshold=gap_threshold,max_dist=max_dist, compute_report=compute_report, orb_good_match = orb_good_match, type = 'orb') #As of now i don't consider the confidence of the matching, but I may in future versions
        page_dictionary = update_phash_matches(matches_sorted,page_dictionary,type='orb')
        return page_dictionary

def perform_ocr_matching(pages_step_3, problematic_templates_step_2,page_dictionary,template_dictionary,
                         text_similarity_metric,mode='cv2', compute_report=False, gap_threshold=0.1, max_dist=0.2):
    similarity = np.zeros((len(pages_step_3), len(problematic_templates_step_2)))
    #i need to iterate on all the remaining temlates and on all the remaining pages that are not the final match of a template
    for jj,t_id in enumerate(problematic_templates_step_2):

        for ii,img_id in enumerate(pages_step_3):
            if page_dictionary[img_id]['text']==None:
                text_box = [template_dictionary[t_id]['text_box']]
                #i rescale the boxes based on image and template resolution
                rescaled_text_box = rescale_box_coords_given_resolutions(text_box, template_dictionary[t_id]['template_size'], page_dictionary[img_id]['img_size'])
                patch = preprocess_text_region(page_dictionary[img_id]['img'], rescaled_text_box[0], mode=mode, verbose=False)
                page_text = extract_features_from_text_region(patch, mode=mode, verbose=False, psm=template_dictionary[t_id]['psm'])['text']
            else:
                page_text = page_dictionary[img_id]['text']
            similarity[ii,jj] = compare_pages_same_section(page_text, template_dictionary[t_id]['text'])[text_similarity_metric]
    #print(similarity) 
    if compute_report:
        matches_sorted, cost, report = match_pages_text(pages_step_3,problematic_templates_step_2,similarity, 
                                                                   compute_report=True, gap_threshold=gap_threshold, max_dist=max_dist)
    else:
        matches_sorted, cost = match_pages_text(pages_step_3,problematic_templates_step_2,similarity)
    for match in matches_sorted:
        #cost is - the number of good matches
        img_id = match["page_index"] 
        t_id = match["template_index"]
        similarity_score = match["similarity"]
        template_dictionary[t_id]['final_match']=img_id 
        page_dictionary[img_id]['matched_page']=t_id
        page_dictionary[img_id]['match_ocr']=t_id
        page_dictionary[img_id]['confidence_template'] = similarity_score
    if compute_report:
         return template_dictionary, page_dictionary, report
    return template_dictionary, page_dictionary

def update_phash_matches(matches_sorted,page_dict,type='phash'):
    if type=='phash':
        key = 'match_phash'
    elif type=='orb':
        key = 'match_orb'
    for match in matches_sorted:
        img_id = match['page_index']
        page_dict[img_id][key]=match['template_index']
    return page_dict

def update_orb_matches(matches_sorted,page_dict):
    for match in matches_sorted:
        img_id = match['page_index']
        page_dict[img_id]['match_orb']=match['template_index']
    return page_dict

def extract_target_numeric(patch, lang, config):
    text = pytesseract.image_to_string(patch,lang = lang, config = config)
    # 2. Search for the primary pattern: 6 digits + optional spaces + 1 digit
    # Pattern: \d{6} (six digits) \s* (zero or more spaces) \d (one digit)
    primary_pattern = r'\d{6}\s*\d'
    primary_match = re.search(primary_pattern, text)
    
    if primary_match:
        # Found the specific target! Return it (stripping extra spaces)
        return primary_match.group(0).replace(" ", ""),text

    # 3. Fallback: Search for the longest numeric string in the entire text
    # \d+ finds any consecutive sequence of digits
    all_numeric_strings = re.findall(r'\d+', text)
    
    if all_numeric_strings:
        # Sort by length and take the longest
        longest_string = max(all_numeric_strings, key=len)
        return longest_string,text
    
    return None,text


def get_numeric_data(image, min_conf=50):
    """
    Returns a list of numbers found in the image with their 
    top-left (OpenCV style) coordinates and confidence scores.
    """
    # image_to_data returns a TSV-style string by default
    # Output_type=Output.DICT is the cleanest way to handle this
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    numeric_results = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        # Filter: Is it a number and is the confidence high enough?
        # This regex-free check handles digits; use .replace('.','') for floats
        if text.isdigit() and conf > min_conf:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Format: (Text, (x, y, width, height), confidence)
            numeric_results.append((text, (x, y, w, h), conf))
            
    return numeric_results

def discover_template(image,annotation_file_names,annotation_roots,npy_data):
    orb = cv2.ORB_create(nfeatures=2000)
    kp_unkn, des_unkn = orb.detectAndCompute(image, None)

    best_match_q = None
    best_match_p = None
    max_good_matches = 0
    
    # 3. Create a Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i, root in enumerate(annotation_roots):
        npy_dict=npy_data[i]
        questionnaire = annotation_file_names[i]
        pages_in_annotation = get_page_list(root)
        for img_id in pages_in_annotation:
            pre_comp = npy_dict[img_id]
            orb_kp=deserialize_keypoints(pre_comp[-1]['orb_kp'])
            orb_des=pre_comp[-1]['orb_des']

            # Match descriptors
            matches = bf.match(des_unkn, orb_des)
            
            # Sort matches by distance (lower distance = better match)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Count "good" matches (those with a distance below a threshold)
            good_matches = [m for m in matches if m.distance < 50]
            
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_q = questionnaire
                best_match_p = img_id
    return best_match_q,best_match_p


def extract_special_id(patch):
    # 1. Helper to get the connection points
    # We want distance from the RIGHT edge of the 6-digit number 
    # to the LEFT edge of the single digit.
    numeric_data=get_numeric_data(patch)
    
    # --- Step 1: Check for exactly 6 elements ---
    for i, (text6, (x6, y6, w6, h6), conf6) in enumerate(numeric_data):
        if len(text6) == 6:
            r_center_x = x6 + w6
            r_center_y = y6 + (h6 / 2)
            
            closest_digit = None
            min_dist = float('inf')
            
            for j, (text1, (x1, y1, w1, h1), conf1) in enumerate(numeric_data):
                # Must be a single digit and NOT the same box
                if i == j or len(text1) != 1:
                    continue
                
                # Target point: Left-center of the single digit
                l_center_x = x1
                l_center_y = y1 + (h1 / 2)
                
                # Calculate Euclidean Distance
                dist = math.sqrt((l_center_x - r_center_x)**2 + (l_center_y - r_center_y)**2)
                
                # Requirement: Must be to the right (x1 > x6)
                if x1 >= x6 + w6 and dist < min_dist:
                    min_dist = dist
                    closest_digit = text1
            
            if closest_digit:
                return f"{text6}{closest_digit}"

    # --- Step 2: Look for exactly 7 elements ---
    for text, coords, conf in numeric_data:
        if len(text) == 7:
            if text != "1234567":
                return text
            
    return None

def identify_questionnaire(unknown_path, templates_folder):
    # 1. Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=2000)
    
    # 2. Load and "fingerprint" the unknown image
    img_unknown = cv2.imread(unknown_path, cv2.IMREAD_GRAYSCALE)
    kp_unkn, des_unkn = orb.detectAndCompute(img_unknown, None)
    
    best_match_name = None
    max_good_matches = 0
    
    # 3. Create a Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 4. Compare against all templates in the folder
    for filename in os.listdir(templates_folder):
        template_path = os.path.join(templates_folder, filename)
        img_temp = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        if img_temp is None: continue
        
        # Detect features in template
        kp_temp, des_temp = orb.detectAndCompute(img_temp, None)
        
        # Match descriptors
        matches = bf.match(des_unkn, des_temp)
        
        # Sort matches by distance (lower distance = better match)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Count "good" matches (those with a distance below a threshold)
        good_matches = [m for m in matches if m.distance < 50]
        
        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_name = filename

    return best_match_name, max_good_matches


########### ORDERING SCHEMES ######################

def ordering_scheme_base(pages_in_annotation, root, sorted_files, npy_dict, 
                         n_align_regions,scale_factor,gap_threshold,max_dist, text_similarity_metric, mode='csv'):
    '''this ordering scheme assumes we know the subject, that we have all 
    the pages from a certain questionnaire. It first checks if the expected ordering is sqtisfied using phqsh qnd templqte mqtching
    if not sqtisfied performs phqsh qnd templqte mqtching on qll pqges (not only pqges to censor)
    finqlly mqtches problemqtic pqges using ocr'''
    # load dictionary to store warning messages on pages
    test_log = {'doc_level_warning':None}
    for p in pages_in_annotation:
        test_log[p]={'failed_test_1': False, 'phash_1': None, 'template_1': None,
                        'failed_test_2': False, 'phash_2': None, 'template_2': None, 
                        'OCR_WARNING': None, 'OCR': None}
    
    #initialize the dictionaries i will use to store info on the sorting process
    page_dictionary,template_dictionary = initialize_sorting_dictionaries(sorted_files, root,mode=mode)
    #pre load the images to be processed (according to the templates that we want to censor)
    page_dictionary,template_dictionary, templates_to_consider = pre_load_images_to_censor(template_dictionary, page_dictionary, mode=mode)

    #pre_load_template_info
    template_dictionary = pre_load_selected_templates(templates_to_consider,npy_dict, root, template_dictionary)
    #pre_load phash for images
    page_dictionary = pre_load_image_properties(templates_to_consider,page_dictionary,template_dictionary,properties=['phash'],mode=mode)
    
    #perform template_matching and update the matching keys in the dictionaries
    page_dictionary,template_dictionary = perform_template_matching(templates_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=n_align_regions,scale_factor=scale_factor)
        
    #perform phash matching
    page_dictionary = perform_phash_matching(page_dictionary,template_dictionary, templates_to_consider, templates_to_consider, 
                    gap_threshold=gap_threshold,max_dist=max_dist)

    #check for which pages at least one test failed (page not matched to expected template for phash or template_matching)
    problematic_pages_step_1 = []
    correct_pages_step_1 = []
    for t_id in templates_to_consider:
        if page_dictionary[t_id]['match_phash']!=t_id or page_dictionary[t_id]['matched_page']!=t_id: #should log which test failed for debugging (eg code 1 ->
            #only phash_failed)
            problematic_pages_step_1.append(t_id)
            test_log[t_id]['failed_test_1'] = True
        else: 
            correct_pages_step_1.append(t_id)
            template_dictionary[t_id]['final_match']=t_id
        test_log[t_id]['phash_1']=page_dictionary[t_id]['match_phash']
        test_log[t_id]['template_1']=page_dictionary[t_id]['matched_page']

    if len(problematic_pages_step_1)>0:
        # i need to load all pages in memory if the first test failed (i don't reload the ones that were already loaded)
        page_dictionary = pre_load_image_properties(pages_in_annotation,page_dictionary,template_dictionary,properties=['img','phash'],mode='csv')

        # prepare the pairs for which to check if there is template matching
        pairs_to_consider = []
        pages_step_2 = []
        for img_id in pages_in_annotation:
            if img_id in correct_pages_step_1:
                continue
            else:
                pages_step_2.append(img_id)

            for t_id in problematic_pages_step_1:
                if t_id==img_id: #skip the pairs that were checked (i have already checked each with itself)
                    continue
                pairs_to_consider.append([img_id,t_id])
                
        #perform template matching on the selected pairs
        page_dictionary,template_dictionary = perform_template_matching(pairs_to_consider,page_dictionary,template_dictionary, 
                                n_align_regions=n_align_regions,scale_factor=scale_factor)
        
        #log the results of the matching
        for img_id in pages_step_2:
            test_log[img_id]['template_2']=page_dictionary[img_id]['matched_page_list']

        # check which pages are problematic for template matching and which are matched correctly instead
        #all templates that are matched to more than one are problematic, also the ones that ar enot matched
        problematic_templates_step_2 = [p for p in problematic_pages_step_1 if template_dictionary[p]['matched_to_this']!=1] 
        matched_templates_step_2 = [p for p in problematic_pages_step_1 if template_dictionary[p]['matched_to_this']==1] 

        #i perform phash matching to check the matched templates
        pages_step_3 = pages_step_2[:]
        if len(matched_templates_step_2)>0:
            matches_sorted, cost = match_pages_phash(page_dictionary,template_dictionary, pages_step_2, matched_templates_step_2, 
                            gap_threshold=gap_threshold,max_dist=max_dist) 
            #As of now i don't consider the confidence of the matching, but I may in future versions
            #page_dictionary = update_phash_matches(matches_sorted,page_dictionary)

            #I look for problematic pages (that were matched with a signle template in template_matching but are
            # now matched to a different template by phash)
            for match in matches_sorted:
                img_id = match["page_index"] 
                t_id = match["template_index"]
                test_log[t_id]['phash_2'] = img_id
                if page_dictionary[img_id]['matched_page']!=t_id:
                    problematic_templates_step_2.append(t_id)
                else:
                    template_dictionary[t_id]['final_match']=img_id
                    #if a page was matched i can remove from the list of pages to pass to step_3
                    pages_step_3.remove(img_id)

        # if there are problematic pages i need to process further; If only one is left out i check it regardless
        # I test with the strongest approach (OCR)
        if len(problematic_templates_step_2)>0: 
            template_dictionary, page_dictionary = perform_ocr_matching(pages_step_3,problematic_templates_step_2, 
                                                        page_dictionary, template_dictionary,text_similarity_metric=text_similarity_metric, mode=mode)  
    
    #reciprocate the matching templates -> pages, pages -> templates
    for t_id in templates_to_consider: 
        img_id = template_dictionary[t_id]['final_match']
        if img_id:
            page_dictionary[img_id]['matched_page'] = t_id 
    #i update the test_log with the ocr results
    for img_id in pages_step_3:
        test_log[img_id]['OCR']=page_dictionary[img_id]['matched_page']

    return page_dictionary,template_dictionary, test_log

