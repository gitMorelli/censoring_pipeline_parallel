from src.utils.feature_extraction import preprocess_roi, preprocess_blank_roi,preprocess_alignment_roi, extract_features_from_roi
from src.utils.feature_extraction import profile_ncc, projection_profiles, edge_iou, ncc, dct_phash,count_black_pixels,count_connected_components
from src.utils.feature_extraction import phash_hamming_distance, binary_crc32, convert_to_grayscale, resize_patch_asymmetric
from src.utils.file_utils import deserialize_keypoints
import cv2  
import numpy as np
import math
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from src.utils.logging import FileWriter
import datetime
import uuid

def rescale_box_coords_given_resolutions(coords, original_resolution, target_resolution):
    '''Rescales box coordinates of a list of boxes from original_resolution to target_resolution. 
    Both resolutions should be in the format (width, height).'''
    x_scale = target_resolution[0] / original_resolution[0]
    y_scale = target_resolution[1] / original_resolution[1]
    
    rescaled_coords = []
    for coord in coords:
        x_min, y_min, x_max, y_max = coord
        new_x_min = int(x_min * x_scale)
        new_y_min = int(y_min * y_scale)
        new_x_max = int(x_max * x_scale)
        new_y_max = int(y_max * y_scale)
        rescaled_coords.append([new_x_min, new_y_min, new_x_max, new_y_max])
    
    return rescaled_coords

def convert_to_axis_aligned_box(coords):
    '''takes a four points polygon and returns the axis aligned bounding box that contains the polygon'''
    new_coords=[]
    for coord_old in coords:
        coord = np.array(coord_old).reshape(-1, 2)
        x_min = np.min(coord[:, 0]).astype(int)
        y_min = np.min(coord[:, 1]).astype(int)
        x_max = np.max(coord[:, 0]).astype(int)
        y_max = np.max(coord[:, 1]).astype(int)
        new_coords.append([x_min, y_min, x_max, y_max])
    return new_coords

def box_to_polygon(coords):
    """
    Converts [xtl, ytl, xbr, ybr] OR a flat list of points 
    into a (N, 2) array of vertices.
    """
    coords = np.array(coords).flatten()
    
    # Case 1: Axis-aligned rectangle [xtl, ytl, xbr, ybr]
    if len(coords) == 4:
        xtl, ytl, xbr, ybr = coords
        
        # Returns a (4, 2) array
        polygon = np.array([
            [xtl, ytl], # Top Left
            [xbr, ytl], # Top Right
            [xbr, ybr], # Bottom Right
            [xtl, ybr]  # Bottom Left
        ], dtype=np.float64) # Float allows for the precision in your example
        
        return polygon
    
    # Case 2: Already a polygon (points), just ensure (N, 2) shape
    else:
        # Reshape to (number_of_points, 2)
        return coords.reshape(-1, 2).astype(np.float64)

## check alignement distorsions ###
def get_angle(p1, p2, p3):
    """Calculates the angle (in degrees) at p2 formed by vectors p2-p1 and p2-p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
        
    cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
    # Clip to avoid float errors outside [-1, 1]
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_geometry_valid(test_w, test_h, transformation,angle_tolerance=10, resize_factor_area=1.0):

    reference=transformation['reference']
    scale_factor = transformation['scale_factor']
    shift_x = transformation['shift_x']
    shift_y = transformation['shift_y']
    angle_degrees = transformation['angle_degrees']

    x_origin=reference[0]
    y_origin=reference[1]
    coords = [x_origin, y_origin,x_origin+test_w,y_origin+test_h]
    test_corners = box_to_polygon(coords)

    transformed_corners = apply_transformation(reference,coords, scale_factor, shift_x, shift_y, angle_degrees, inverse=False)
    
    
    # Flatten the points array from (4, 1, 2) to (4, 2)
    pts = transformed_corners.reshape(4, 2)
    
    # 1. Basic Convexity Check
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False, "Non-convex (bow-tie distortion)"

    # 2. Area Consistency Check
    scanned_area = cv2.contourArea(pts.astype('float32'))
    template_area = test_w * test_h * resize_factor_area 
    if scanned_area < (template_area * 0.4) or scanned_area > (template_area * 1.6):
        return False, f"Area error: {scanned_area/template_area:.2f}x size"

    # 3. Angle Check (The 90-degree test)
    # Indices for the 4 corners: 0, 1, 2, 3
    # We check the angle at each corner p[i] using neighbors p[i-1] and p[i+1]
    for i in range(4):
        p1 = pts[i - 1]          # Previous point
        p2 = pts[i]              # Current corner
        p3 = pts[(i + 1) % 4]    # Next point
        
        angle = get_angle(p1, p2, p3)
        
        # Check if the angle is within (90 - tolerance) and (90 + tolerance)
        if abs(angle - 90) > angle_tolerance:
            return False, f"Bad angle at corner {i}: {angle:.1f}° (Target: 90°±{angle_tolerance}°)"

    return True, "Success"

######### Compute misalignment using ALIGN boxes #########
'''def enlarge_crop_coords(coords, scale_factor=1.2, img_shape=None):
    x_min, y_min, x_max, y_max = coords
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    new_width = width * scale_factor
    new_height = height * scale_factor
    new_x_min = int(max(center_x - new_width / 2, 0))
    new_y_min = int(max(center_y - new_height / 2, 0))
    new_x_max = int(min(center_x + new_width / 2, img_shape[0] - 1)) if img_shape else int(center_x + new_width / 2)
    new_y_max = int(min(center_y + new_height / 2, img_shape[1] - 1)) if img_shape else int(center_y + new_height / 2)
    return (new_x_min, new_y_min, new_x_max, new_y_max)'''
# this version accounts for tilted rectangles (enlarges the axis aligned rectangle that contains the titled rectangle)
def enlarge_crop_coords(box, scale_factor, img_shape):
    # 1. Handle Input Diversity
    box = np.array(box)
    
    if box.ndim == 1 and len(box) == 4:
        # Standard format: [xtl, ytl, xbr, ybr]
        x_min, y_min, x_max, y_max = box
    else:
        # Polygon format: [[x1, y1], [x2, y2]...] or flattened [x1, y1, x2, y2...]
        # Reshape to (-1, 2) just in case it's flattened
        #coords = box.reshape(-1, 2)
        x_min, y_min = box.min(axis=0)
        x_max, y_max = box.max(axis=0)

    # 2. Calculate dimensions
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width / 2
    center_y = y_min + height / 2

    # 3. Apply Scaling
    new_width = width * scale_factor
    new_height = height * scale_factor

    # 4. Final Coordinates (Now using scalars, so max() works!)
    #i am giving x,y as img_shape in input
    img_w, img_h = img_shape
    
    dx_min= max(center_x - new_width / 2, 0) - (center_x - new_width / 2) #i compute how much is lost due to the cropping at the edge
    dy_min= max(center_y - new_height / 2, 0) - (center_y - new_height / 2)
    dx_max = (center_x + new_width / 2)-min(center_x + new_width / 2, img_w) 
    dy_max = (center_y + new_height / 2) - min(center_y + new_height / 2, img_h) 


    new_x_min = int(max(center_x - new_width / 2 - dx_max, 0)) #i resize nd add to the border the part lost on the other side. Then I crop at the border
    new_y_min = int(max(center_y - new_height / 2 - dy_max, 0))
    new_x_max = int(min(center_x + new_width / 2 + dx_min, img_w))
    new_y_max = int(min(center_y + new_height / 2 + dy_min, img_h))

    return (new_x_min, new_y_min, new_x_max, new_y_max)

'''
def get_center(coords):
    x_min, y_min, x_max, y_max = coords
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return center_x, center_y'''
#also deals with polygons (returns the center of mass of the points) and with flat lists of polygon points [x1, y1, x2, y2...]
def get_center(coords):
    # Ensure we are working with a numpy array
    coords = np.array(coords)
    
    # CASE 1: Standard [xtl, ytl, xbr, ybr] (1D array with 4 elements)
    if coords.ndim == 1 and len(coords) == 4:
        xtl, ytl, xbr, ybr = coords
        return (xtl + xbr) // 2, (ytl + ybr) // 2
    
    # CASE 2: Polygon or List of Points [[x1,y1], [x2,y2]...]
    # Reshape if it's a flat list of polygon points [x1, y1, x2, y2...]
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
        
    center_x = int(np.mean(coords[:, 0]))
    center_y = int(np.mean(coords[:, 1]))
    
    return center_x, center_y

def template_matching(f_roi, t_roi, coord, mode="cv2",threshold=0.7,shift_wr_tl=(0,0)):
    w, h = t_roi.shape[1], t_roi.shape[0]
    

    # Template matching
    res = cv2.matchTemplate(f_roi, t_roi, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) # positions are relative to top left of f_roi

    if max_val < threshold:
        # If no good match is found, return the original center
        center_x = coord[0] + (coord[2] - coord[0]) // 2
        center_y = coord[1] + (coord[3] - coord[1]) // 2
        return False,max_val,center_x, center_y
    # For TM_SQDIFF methods, the best match is min_loc; otherwise, max_loc
    top_left = max_loc
    # tìi correct the computed shift if the crop region was extended, thus shift_wr_tl is not (0,0)
    shift_x = top_left[0] - shift_wr_tl[0]
    shift_y = top_left[1] - shift_wr_tl[1]
    return True,max_val,shift_x, shift_y


def compute_misalignment(filled_png, rois, img_shape, pre_computed_template, scale_factor=2,
                         matching_threshold=0.7, pre_computed_rois=None,return_confidences=False, metric="matchTemplate",
                         rescale_x_y=None,**kwargs):
    if pre_computed_rois:
        pre_computed=True
    else:
        pre_computed=False
    mode = "cv2"

    shifts = []
    centers = []
    processed_rois=[]
    confidences=[]
    for i,coord in enumerate(rois):
        #print("-"*50)
        #print("before coords enlargement", coord)
        #print(img_shape)
        center_x, center_y = get_center(coord) #coordinates of the center of the bounding box in the image frame
        new_coord = enlarge_crop_coords(coord, scale_factor=scale_factor, img_shape=img_shape) #new coords are in the absolutre reference frame (image frame)
        #print("after coords enlargement", new_coord)
        new_center_x, new_center_y = get_center(new_coord) #coordinates of the center of the enlarged box in the image frame
        shift_wr_center = (new_center_x - center_x, new_center_y - center_y) #if the rescaled patch is not cropped 
        #we expect to find the template at w/2,h/2 in the referece frame of the enlarged patch; If it is cropped we expect to find it at -shift_wr_center
        shift_wr_tl = (coord[0]-new_coord[0], coord[1]-new_coord[1]) #coordinate di top-left corner of the original box in the reference frame of the enlarged patch 
        if metric == "orb":
            orb_parameters = kwargs.get('orb_parameters',{})
            orb_nfeatures = orb_parameters.get('orb_nfeatures',2000)
            orb_match_threshold = orb_parameters.get('orb_match_threshold',10)
            orb_top_n_matches = orb_parameters.get('orb_top_n_matches',50)
            orb_method_to_find_matches = orb_parameters.get('orb_method_to_find_matches','brute_force')
            orb_match_filtering_method = orb_parameters.get('orb_match_filtering_method',"best_n")
            lowe_threshold=orb_parameters.get("orb_lowe_threshold",0.7)
            orb_decision_procedure = orb_parameters.get("orb_decision_procedure",'simple')
            is_matched, n_good_matches, shift_x, shift_y, _, _ = orb_matching(filled_png, new_coord, pre_computed_template[i], shift_wr_tl,top_n_matches=orb_top_n_matches, 
                                                                              orb_nfeatures=orb_nfeatures, match_threshold=orb_match_threshold, 
                                                                              method_to_find_matches=orb_method_to_find_matches, 
                                                                              match_filtering_method=orb_match_filtering_method, 
                                                                              lowe_threshold=lowe_threshold, rescale_x_y=rescale_x_y, 
                                                                              decision_procedure=orb_decision_procedure)
            max_val = n_good_matches
        elif metric == "matchTemplate":
            t_roi = pre_computed_template[i]['full']
            if not pre_computed:
                f_roi = preprocess_alignment_roi(filled_png, new_coord, mode=mode, verbose=False)
            else:
                f_roi = pre_computed_rois[i]
            
            if rescale_x_y is not None:
                f_roi = resize_patch_asymmetric(f_roi, rescale_x_y[0], rescale_x_y[1])
            #print(f_roi.shape, t_roi.shape)
            #print("-"*50)
            is_matched,max_val,shift_x, shift_y = template_matching(f_roi, t_roi, coord, mode=mode,
                                                                    shift_wr_tl=shift_wr_tl,threshold=matching_threshold)
            #show f_roi and t_roi for debugging
            '''
            # 2. Determine the maximum dimensions needed to fit either image
            # This handles cases where the "resized" image might actually be larger (scale > 1)
            max_h = max(t_roi.shape[0], f_roi.shape[0])
            max_w = max(t_roi.shape[1], f_roi.shape[1])

            # 3. Create two identical canvases of the same max size
            # Use the same dtype as your patches (usually uint8 for images)
            canvas_resized = np.zeros((max_h, max_w), dtype=t_roi.dtype)
            canvas_original = np.zeros((max_h, max_w), dtype=t_roi.dtype)

            # 4. Place the images onto the canvases
            # Using [0:h, 0:w] ensures they start at the same top-left origin (0,0)
            h_res, w_res = f_roi.shape[:2]
            canvas_resized[:h_res, :w_res] = f_roi

            h_orig, w_orig = t_roi.shape[:2]
            canvas_original[:h_orig, :w_orig] = t_roi

            # 5. Plotting on the same scale
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(canvas_resized, cmap='gray')
            plt.title(f"Resized Patch: {w_res}x{h_res}\n(Scale X={rescale_x_y[0]:.2f}, Y={rescale_x_y[1]:.2f})")
            plt.axis('on') 

            plt.subplot(1, 2, 2)
            plt.imshow(canvas_original, cmap='gray')
            plt.title(f"Original T_ROI: {w_orig}x{h_orig}, {is_matched}")
            plt.axis('on')

            plt.tight_layout()
            plt.show()'''
            
        #if during matching the patch was rescaled i have to adjust the computed shift accordingly
        #consider that i am computing the shift in the scale of the template and i want the shifts in the scale of the image
        confidences.append(max_val)
        if is_matched: #only include regions for which you have a match
            if rescale_x_y is not None:
                shift_x = shift_x / rescale_x_y[0]
                shift_y = shift_y / rescale_x_y[1]
            shifts.append((shift_x, shift_y))
            centers.append((center_x, center_y))
        if metric == "matchTemplate":
            processed_rois.append(f_roi)
        else:
            processed_rois.append(None)
    if return_confidences:
        return shifts, centers,processed_rois,confidences
    return shifts, centers,processed_rois


def orb_matching(img=None,box=None,template_properties=None, image_kpts=None,template_kpts=None, 
                 compute_method="center_of_mass",
                 shift_wr_tl=(0,0), 
                 top_n_matches=50, orb_nfeatures=2000, match_threshold=10, decision_procedure='simple',
                 method_to_find_matches='brute_force',match_filtering_method="best_n",lowe_threshold=0.7, rescale_x_y=None):
    #by default i consider that i am comparing patches that are at teh same absolute position in the image (shift_wr_tl=(0,0)) 
    
    if image_kpts is not None and template_kpts is not None:
        kps_image, des_image = image_kpts
        kps_template, des_template = template_kpts
    else:
        # preprocess image
        preprocessed_patch = preprocess_roi(img, box, target_size=None)
        if rescale_x_y is not None:
            preprocessed_patch = resize_patch_asymmetric(preprocessed_patch, rescale_x_y[0], rescale_x_y[1])
        # Display the preprocessed patch for debugging or visualization
        '''plt.imshow(preprocessed_patch, cmap='gray')
        plt.title("Preprocessed Patch")
        plt.axis('off')
        plt.show()'''

        #save an image of the patch for debugging with time of processing in the name to avoid overwriting
        # 1. Generate Timestamp (YearMonthDay_HourMinuteSecond)
        '''timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 2. Generate a short Random Identifier (8 characters)
        # We use uuid4 and take the first 8 characters for brevity
        random_id = str(uuid.uuid4())[:8]
        cv2.imwrite(f"//vms-e34n-databr/2025-handwriting\\data\\test_censoring_pipeline\\debug_temporary\\debug_patch_{timestamp}_{random_id}.png", preprocessed_patch)'''

        if template_properties['orb_args']:
            orb_nfeatures=template_properties['orb_args']['orb_nfeatures']
            orb_edgeThreshold=template_properties['orb_args']['orb_edgeThreshold']
            orb_patchSize=template_properties['orb_args']['orb_patchSize']
            orb_fastThreshold=template_properties['orb_args']['orb_fastThreshold']

        #compute orb features for the patch
        pre_comp = extract_features_from_roi(preprocessed_patch,to_compute=['orb'],orb_nfeatures=orb_nfeatures, 
                                             orb_edgeThreshold=orb_edgeThreshold, orb_patchSize=orb_patchSize, orb_fastThreshold=orb_fastThreshold)
        kps_image , des_image = deserialize_keypoints(pre_comp['orb_kp']) , pre_comp['orb_des'] 

        kps_template, des_template = deserialize_keypoints(template_properties['orb_kp']), template_properties['orb_des']

    #warnings
    error=''
    if des_image is None :
        error += "des_image is empty, "
    if des_template is None:
        error += 'des_template is empty, '
    if error=='' and des_image.dtype != des_template.dtype:
        error += "Error: Descriptor type mismatch! {des_image.dtype} vs {des_template.dtype}, "
    if error=='' and des_image.shape[1] != des_template.shape[1]:
        error += "Error: Descriptor dimension mismatch! {des_image.shape[1]} vs {des_template.shape[1]}, "
    
    if error != '':
        return False, error, None, None, None, None 
        #if there is an error in the descriptors we consider that there is no match and we return None for the transformation parameters
    
    if method_to_find_matches == 'brute_force':
        # 2. Match features
        if match_filtering_method == "lowe_ratio":
            # For Lowe's ratio test, we need to use knnMatch to get the 2 best matches for each point
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck must be False for k>1
            matches = bf.knnMatch(des_image, des_template, k=2)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des_image, des_template), key=lambda x: x.distance)
    elif method_to_find_matches == 'knn':
        # 2. FLANN Matcher (faster than Brute Force for large point sets)
        #FLANN_INDEX_KDTREE = 1
        #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # Use LSH for ORB descriptors
        index_params = dict(
            algorithm = 6, # FLANN_INDEX_LSH
            table_number = 6,      # 12 is also a good choice
            key_size = 12,         # 20 is also a good choice
            multi_probe_level = 1  # 2 is better but slower
        )
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Use knnMatch to get the 2 best matches for each point
        matches = flann.knnMatch(des_image, des_template, k=2)

    if match_filtering_method == "best_n":
        # Use only the top 50 matches for stability
        good_matches = matches[:min(top_n_matches, len(matches))]
        n_good_matches = len(matches)
    elif match_filtering_method == "lowe_ratio":
        # 3. Apply Lowe's Ratio Test
        # This discards matches where the best and second-best are too similar
        good_matches = []
        if len(matches) > 0 :
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < lowe_threshold * n.distance:
                        good_matches.append(m)
        n_good_matches = len(good_matches)
    elif match_filtering_method == "distance_threshold":
        # Filter matches based on a distance threshold
        good_matches = [m for m in matches if m.distance < lowe_threshold]
        n_good_matches = len(good_matches)
    elif match_filtering_method == "all":
        good_matches = matches[:]
        n_good_matches = len(matches)

    if decision_procedure == "simple":
        is_matched = n_good_matches > match_threshold
    elif decision_procedure == "homography":
        is_matched, inliers_count, inlier_ratio = are_images_same_ORB(kps_image, kps_template, good_matches, 
                                         min_inliers=match_threshold, inlier_ratio_threshold=0.3)
        return is_matched, [n_good_matches,inliers_count,inlier_ratio],0,0,1,0
    elif decision_procedure == "geometric":
        if len(good_matches) < 4:
            #i stop immediately because i cannot compute the homography ffectively
            return False, [n_good_matches,'matches<4'], 0,0,1,0
        # Extract coordinates of matched points
        image_pts = np.float32([kps_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        template_pts = np.float32([kps_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 
        M, mask = cv2.findHomography(template_pts, image_pts, cv2.USAC_MAGSAC, 5.0)
        if M is None:
            is_matched = False
        transformation = {'reference': (0,0), 'scale_factor': M, 'shift_x': 'homography', 'shift_y': 'homography', 'angle_degrees': 'homography'}
        flag,error = is_geometry_valid(test_w=100,test_h=100, transformation=transformation, angle_tolerance=10,
                                           resize_factor_area=1)
        return flag, [n_good_matches,error],0,0,1,0

    if is_matched:
        # Extract coordinates of matched points
        image_pts = np.float32([kps_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        template_pts = np.float32([kps_template[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) 

        if compute_method == "affine":
            M, inliers = cv2.estimateAffinePartial2D(template_pts, image_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            x_coords_image = image_pts[:, 0, 0] 
            x_coords_template = template_pts[:, 0, 0] 

            
            if len(x_coords_image) > 0:
                max_x_dist_template = np.ptp(x_coords_template)  # ptp = "peak to peak" (max - min)
                max_x_dist_image = np.ptp(x_coords_image)
                #print(f"Max X distance in template: {max_x_dist_template}, Max X distance in image: {max_x_dist_image}, Shift {M[0, 2]}")

            # 4. Extract Info from 2x3 Matrix M
            shift_x = M[0, 2] - shift_wr_tl[0]
            shift_y = M[1, 2] - shift_wr_tl[1]
            
            # In a Partial Affine matrix:
            # s_cos = M[0,0]
            # s_sin = M[1,0]
            scale = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
            angle = math.atan2(M[1, 0], M[0, 0]) * (180 / math.pi)
        elif compute_method == "homography":
            M, mask = cv2.findHomography(template_pts, image_pts, cv2.USAC_MAGSAC, 5.0)
            if M is None:
                is_matched = False
            #cv2.USAC_MAGSAC
            #cv2.RANSAC
            return is_matched, n_good_matches,'homography','homography',M,'homography' #i return the M matrix in place of the scale
        elif compute_method == "center_of_mass":
            # 1. Calculate the 'Center of Mass' of keypoints in both coordinate systems
            # template_pts and image_pts are shape (-1, 1, 2)
            com_template = np.mean(template_pts, axis=0).flatten() # [avg_x, avg_y]
            com_image = np.mean(image_pts, axis=0).flatten()     # [avg_x, avg_y]

            # 2. Calculate the "Intuitive Shift"
            # This is how much the actual center of your features moved
            intuitive_shift_x = com_image[0] - com_template[0]
            intuitive_shift_y = com_image[1] - com_template[1]

            # 3. Apply your enlargement correction
            shift_x = intuitive_shift_x - shift_wr_tl[0]
            shift_y = intuitive_shift_y - shift_wr_tl[1]
            #print(f"Final Shift after correction: ({shift_x:.2f}, {shift_y:.2f})")
            scale = 1.0
            angle = 0.0

        return is_matched, n_good_matches,shift_x, shift_y, scale, angle
    else:
        return is_matched,n_good_matches, None, None, None, None  # Not enough matches to compute transformation

def are_images_same_ORB(kp1, kp2, good_matches, min_inliers=15, inlier_ratio_threshold=0.3):

    if len(good_matches) < min_inliers:
        #i stop immediately because i cannot compute the homography ffectively
        return False, 'matches<min', None

    # 4. Find Homography (The "Truth" Test)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC returns a mask where 1 = inlier, 0 = outlier
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if mask is None:
        return False,'mask_empty',None

    inliers_count = np.sum(mask)
    inlier_ratio = inliers_count / len(good_matches)

    # 5. Final Decision
    # We want a decent absolute number of points AND a high percentage of "valid" matches
    is_same = inliers_count >= min_inliers
    #is_same = inliers_count >= min_inliers and inlier_ratio >= inlier_ratio_threshold
    
    return is_same, inliers_count, inlier_ratio

def compute_distance(c1,c2):
    return np.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)

def compute_transformation(shifts, centers, selection="top_left"):
    '''You can select top_left or most_distant for the pair of rois to consider for computing alignement'''
    if len(shifts) < 2:
        return 1.0,0,0,0,(0,0)  # Not enough data to compute transformation
    max_dist = 0

    if selection == "most_distant":
        for i in range(len(shifts)):
            for j in range(i + 1, len(shifts)):
                c1 = centers[i]
                c2 = centers[j]
                dist = compute_distance(c1, c2)
                if dist > max_dist:
                    max_dist = dist
                    idx1, idx2 = i, j
    elif selection == "top_left":
        # Select the pair of ROIs whose centers are closest to the top-left corner (0,0)
        min_dist_to_top_left = float('inf')
        for i in range(len(shifts)):
            c1 = centers[i]
            dist_to_top_left = compute_distance(c1, (0, 0))
            if dist_to_top_left < min_dist_to_top_left:
                min_dist_to_top_left = dist_to_top_left
                idx1 = i
        # Find the second point that is farthest among the remaining points
        for j in range(len(shifts)):
            if j != idx1:
                c2 = centers[j]
                dist_between_points = compute_distance(centers[idx1], c2)
                if dist_between_points > max_dist:
                    max_dist = dist_between_points
                    idx2 = j
    elif selection == "origin":
        '''it doesn't make sense because if i take the shift from the first box the rotation is around that box'''
        # find the point that is closest to the origin to compute shift
        min_dist_to_top_left = float('inf')
        for i in range(len(shifts)):
            c1 = centers[i]
            dist_to_top_left = compute_distance(c1, (0, 0))
            if dist_to_top_left < min_dist_to_top_left:
                min_dist_to_top_left = dist_to_top_left
                idx1 = i
        # Find the point that is farthest from the origin (0,0) to compute rotation
        for j in range(len(shifts)):
            if j != idx1:
                c2 = centers[j]
                dist_between_points = compute_distance((0,0), c2)
                if dist_between_points > max_dist:
                    max_dist = dist_between_points
                    idx2 = j
    center_1=centers[idx1]
    center_2=centers[idx2]
    shift_1=shifts[idx1]
    shift_2=shifts[idx2]
    # Compute scale factor
    original_distance = compute_distance(center_1, center_2)
    shifted_center_1 = (center_1[0] + shift_1[0], center_1[1] + shift_1[1])
    shifted_center_2 = (center_2[0] + shift_2[0], center_2[1] + shift_2[1])
    shifted_distance = compute_distance(shifted_center_1, shifted_center_2)
    scale_factor = shifted_distance / original_distance if original_distance != 0 else 1.0
    # Compute shift
    shift_x = shifts[0][0]
    shift_y = shifts[0][1]
    # compute rotation angle
    delta_y = shifted_center_2[1] - shifted_center_1[1]
    delta_x = shifted_center_2[0] - shifted_center_1[0]
    shift_angle = np.arctan(delta_x / delta_y) - np.arctan((center_2[0] - center_1[0]) / (center_2[1] - center_1[1]))
    shift_angle_degrees = np.degrees(shift_angle)
    return scale_factor, shift_x, shift_y, shift_angle_degrees,center_1

def rotate_points_about_pivot(points, px, py, alpha_deg):
    a = np.deg2rad(alpha_deg)
    R = np.array([[np.cos(a), -np.sin(a)],
                  [np.sin(a),  np.cos(a)]], dtype=float)
    P = np.atleast_2d(points).astype(float)   # (N,2)
    rotated = (P - [px, py]) @ R.T + [px, py]
    return rotated

def apply_transformation(reference,coords, scale_factor, shift_x, shift_y, angle_degrees, inverse=False):
    #rotation is around reference box (upper left)
    if shift_x == 'homography':
        #scale factor is an homography transformation
        M = scale_factor
        pts = np.float32([
            [coords[0], coords[1]], 
            [coords[2], coords[1]], 
            [coords[2], coords[3]], 
            [coords[0], coords[3]]
        ]).reshape(-1, 1, 2)

        # 3. Apply the Homography transformation
        # This maps the points from Image 1's perspective to Image 2's perspective
        transformed_pts = cv2.perspectiveTransform(pts, M)
        return transformed_pts.reshape(-1, 2)  # Return as (4, 2) array of points
    else:
        if inverse:
            scale_factor = 1 / scale_factor
            shift_x = -shift_x
            shift_y = -shift_y
            angle_degrees = -angle_degrees
        angle_degrees = -angle_degrees
        x_min, y_min, x_max, y_max = coords
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        new_width = width * scale_factor
        new_height = height * scale_factor
        angle_radians = np.radians(angle_degrees)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        new_center_x = center_x + shift_x
        new_center_y = center_y + shift_y
        reference_x_min, reference_y_min = reference
        # Compute coordinates. Rotate the whole box around the reference point (sides may not be axis-aligned anymore)
        corners = np.array([
            [new_center_x - new_width/2, new_center_y - new_height/2],
            [new_center_x + new_width/2, new_center_y - new_height/2],
            [new_center_x + new_width/2, new_center_y + new_height/2],
            [new_center_x - new_width/2, new_center_y + new_height/2],
        ])
        corners_rot = rotate_points_about_pivot(corners, reference_x_min, reference_y_min, angle_degrees)

    return corners_rot #recall that now the box is not axis-aligned anymore corners_rot is a list of 4 (x,y) points

def plot_rois_on_image(img, rois, save_path, colors=None): 
    image=img.copy()

    h, w = image.shape[:2]

    # Create figure with size matching the image pixel dimensions
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    if colors is None:
        colors = ['red'] * len(rois)
    #print(colors)

    ax = plt.axes()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Remove axes for clean output
    ax.axis('off')

    for i, coord in enumerate(rois):
        x_min, y_min, x_max, y_max = coord
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.5,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(rect)

    # Save without extra borders
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_rois_on_image_stackable(image, rois, colors=None):
    """
    Draws ROIs directly onto a copy of the image and returns the modified image.
    """
    # Create a copy so we don't overwrite the original image
    output_image = image.copy()
    
    if colors is None:
        # Default to Red (BGR format)
        colors = [(0, 0, 255)] * len(rois)
    elif all(isinstance(c, str) for c in colors):
        # Quick conversion for basic string colors if needed
        color_map = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}
        colors = [color_map.get(c.lower(), (0, 0, 255)) for c in colors]

    for i, coord in enumerate(rois):
        x_min, y_min, x_max, y_max = map(int, coord)
        
        # cv2.rectangle(img, pt1, pt2, color, thickness)
        cv2.rectangle(
            output_image, 
            (x_min, y_min), 
            (x_max, y_max), 
            colors[i], 
            thickness=2
        )

    return output_image

def plot_rois_on_image_polygons(img, rois, save_path, colors=None):
    image=img.copy()

    h, w = image.shape[:2]

    # Maintain original image resolution
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes()

    if colors is None:
        colors = ['red'] * len(rois)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

    for i, coord in enumerate(rois):
        #print(coord)
        coord = box_to_polygon(coord)  # Ensure we have a proper polygon format
        pts = np.array(coord, dtype=float)  # expected shape: (4, 2)
        poly = Polygon(
            pts,
            closed=True,
            linewidth=0.5,
            edgecolor=colors[i],
            facecolor='none'
        )
        ax.add_patch(poly)

        # If labeling is needed, uncomment below
        # center_x, center_y = pts[:, 0].mean(), pts[:, 1].mean()
        # ax.text(center_x, center_y, str(i), color=colors[i], fontsize=12,
        #        ha='center', va='center')

    # Save as original resolution
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_both_rois_on_image(image, rois_1, rois_2, save_path,
                            color_1="red", color_2="green"):
    """
    Draw rectangular ROIs (rois_1) and polygon ROIs (rois_2)
    while preserving the original image resolution.
    """

    h, w = image.shape[:2]

    # Create a figure matching the image's resolution
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.axes()

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis("off")

    # --- Draw rectangular ROIs ---
    for coord in rois_1:
        x_min, y_min, x_max, y_max = coord
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=0.5,
            edgecolor=color_1,
            facecolor='none'
        )
        ax.add_patch(rect)

    # --- Draw polygon ROIs ---
    for coord in rois_2:
        pts = np.array(coord, dtype=float)  # expected shape (4,2)
        poly = Polygon(
            pts,
            closed=True,
            linewidth=0.5,
            edgecolor=color_2,
            facecolor='none'
        )
        ax.add_patch(poly)

    # Save image with original resolution
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def adjust_boundary_boxes(boxes, img_size_1, img_size_2, epsilon=2.0):
    """
    Adjusts box edges that coincide with img_size_1 boundaries 
    to match img_size_2 boundaries, without scaling internal coordinates.
    """
    w1, h1 = img_size_1
    w2, h2 = img_size_2
    
    # Convert to numpy array for vectorized operations
    boxes = np.array(boxes, dtype=float)
    
    # X-coordinates: Xtl (index 0) and Xbr (index 2)
    for i in [0, 2]:
        # If it was at/near the left edge (0), keep it at 0
        boxes[boxes[:, i] <= epsilon, i] = 0
        
        # If it was at/near the old right edge (w1), move it to the new right edge (w2)
        boxes[boxes[:, i] >= (w1 - epsilon), i] = w2

    # Y-coordinates: Ytl (index 1) and Ybr (index 3)
    for i in [1, 3]:
        # If it was at/near the top edge (0), keep it at 0
        boxes[boxes[:, i] <= epsilon, i] = 0
        
        # If it was at/near the old bottom edge (h1), move it to the new bottom edge (h2)
        boxes[boxes[:, i] >= (h1 - epsilon), i] = h2
        
    return boxes.tolist()
######### CHECK FOR ALIGNMENT using ROIs #########
# -----------------------------
# Decision logic per ROI
# -----------------------------
def roi_decision(f_roi,phash_hamm_thresh=8,
                 ncc_thresh=0.92,
                 edge_iou_thresh=0.75,
                 proj_ncc_thresh=0.90,t_roi=None,pre_computed_roi=None,
                 to_compute=['crc32','dct_phash', 'ncc','edge_iou','profile'],treshold_test=2,logger=None):
    """
    Returns True if the two ROIs are considered aligned/similar enough.
    Order of checks:
      0) Binary checksum (fast pass)
      1) DCT pHash (cv2-only)
      2) NCC
      3) Edge IoU
      4) Projection profiles (H & V)
    """
    tot_tests = len(to_compute)
    ok_tests = 0
    if pre_computed_roi:
        pre_computed=True
    else:
        pre_computed=False
    # --- 0) Binary checksum (Otsu binarize + CRC32)
    logger and logger.call_start(f'crc32')
    if 'crc32' in to_compute:
        if pre_computed:
            t_crc = pre_computed_roi['crc32']
        else:
            t_crc = binary_crc32(t_roi)
        f_crc = binary_crc32(f_roi)
        if t_crc == f_crc:
            ok_tests += 1
    logger and logger.call_end(f'crc32')

    # --- 1) DCT-based pHash (cv2.dct)
    logger and logger.call_start(f'dct')
    if 'dct_phash' in to_compute:
        if pre_computed:
            h1 = pre_computed_roi['dct_phash']#check that was precomputed with the same params
        else:
            h1 = dct_phash(t_roi, hash_size=8, dct_size=32) 
        h2 = dct_phash(f_roi, hash_size=8, dct_size=32)
        hdist = phash_hamming_distance(h1, h2)
        if hdist <= phash_hamm_thresh:
            ok_tests += 1
    logger and logger.call_end(f'dct')

    # --- 2) NCC on intensities
    logger and logger.call_start(f'ncc')
    if 'ncc' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        ncc_value,_,_ = ncc(t_roi, f_roi)
        if ncc_value >= ncc_thresh:
            ok_tests += 1
    logger and logger.call_end(f'ncc')

    # --- 3) Edge IoU
    logger and logger.call_start(f'edge_iou')
    if 'edge_iou' in to_compute:
        if pre_computed:
            t_roi = pre_computed_roi['full']
        if edge_iou(t_roi, f_roi) >= edge_iou_thresh:
            ok_tests += 1
    logger and logger.call_end(f'edge_iou')

    # --- 4) Projection profiles (horizontal & vertical)
    logger and logger.call_start(f'profiles')
    if 'profile' in to_compute:
        if pre_computed:
            th = pre_computed_roi['profile_h']
            tv = pre_computed_roi['profile_v']
        else:
            th = projection_profiles(t_roi, axis=1)  # horizontal
            tv = projection_profiles(t_roi, axis=0)  # vertical
        fh = projection_profiles(f_roi, axis=1)
        fv = projection_profiles(f_roi, axis=0) 
        # Require both directions to be a strong match (or relax to either if desired)
        if profile_ncc(th, fh) >= proj_ncc_thresh and profile_ncc(tv, fv) >= proj_ncc_thresh:
            ok_tests += 1
    logger and logger.call_end(f'profiles')
    if ok_tests >= treshold_test:
        return True
    return False

def roi_blank_decision(f_roi, n_black_thresh=0.1, return_features = False,
                       t_roi=None,pre_computed_roi=None,to_compute=['cc','n_black'],threshold_test=1):
    ok_tests = 0
    if pre_computed_roi:
        pre_computed=True
    else:
        pre_computed=False
    
    if 'n_black' in to_compute:
        if pre_computed:
            t_n_black = pre_computed_roi['n_black']
        else:
            t_n_black = count_black_pixels(t_roi)
        f_n_black = count_black_pixels(f_roi)
        
        black_diff_to_template = (f_n_black-t_n_black)/(t_n_black+1e-10)

        if np.abs(black_diff_to_template) <= n_black_thresh:
            ok_tests += 1
    else:
        black_diff_to_template = None
    if 'cc' in to_compute:
        if pre_computed:
            t_cc = pre_computed_roi['cc']
        else:
            t_cc = count_connected_components(t_roi)
        f_cc = count_connected_components(f_roi)
        cc_difference_to_template= f_cc-t_cc
        if cc_difference_to_template == 0:
            ok_tests += 1
    else:
        cc_difference_to_template=None
    tests_ok=False
    if ok_tests >= threshold_test:
        tests_ok=True
    if return_features:
        return tests_ok,black_diff_to_template,cc_difference_to_template
    else:
        return tests_ok
# -----------------------------
# Page-level voting
# -----------------------------
def page_vote(filled_png, rois, min_votes=3,
              template_png=None,pre_computed_rois=None,logger=None):
    
    logger and logger.call_start('page_vote',block=True)

    votes = 0
    total = 0
    mode="cv2"
    if pre_computed_rois:
        pre_computed = True
    else:
        pre_computed = False
    #rois is the list of coordinates of the regions
    for i,coord in enumerate(rois[:-1]):
        # scale ROI coords
        decision=False

        logger and logger.call_start(f'preprocess_roi_{i}')
        f_roi = preprocess_roi(filled_png, coord, mode=mode, verbose=False)
        logger and logger.call_end(f'preprocess_roi_{i}')

        if not pre_computed:
            t_roi = preprocess_roi(template_png, coord, mode=mode, verbose=False)
            decision = roi_decision(f_roi, t_roi=t_roi)
        else:
            pre_comp_roi = pre_computed_rois[i]
            logger and logger.call_start(f'roi_decision_{i}')
            decision = roi_decision(f_roi, pre_computed_roi=pre_comp_roi,logger=logger)
            logger and logger.call_end(f'roi_decision_{i}')
        total += 1
        if decision:
            votes += 1
        # early exit: impossible to reach min_votes
        if votes + (len(rois)-total) < min_votes:
            logger and logger.call_end('page_vote',block=True)
            return False
    coord = rois[-1]  # blank ROI is the last one
    logger and logger.call_start(f'blank_preprocess')
    f_roi = preprocess_blank_roi(filled_png, coord, mode=mode, verbose=False)
    logger and logger.call_end(f'blank_preprocess')
    if not pre_computed:
        t_roi = preprocess_blank_roi(template_png, coord, mode=mode, verbose=False)
        decision = roi_blank_decision(f_roi, t_roi=t_roi)
    else:
        pre_comp_roi = pre_computed_rois[-1]
        logger and logger.call_start(f'blank_decision')
        decision = roi_blank_decision(f_roi, pre_computed_roi=pre_comp_roi)
        logger and logger.call_end(f'blank_decision')
    if decision:
        votes += 1
    
    logger and logger.call_end('page_vote',block=True)

    return votes >= min_votes