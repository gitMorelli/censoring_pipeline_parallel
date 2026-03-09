
import json
import re

def extract_page_number(image_path):
    """
    Estrae il numero X dal nome del file che termina in page_X.png
    """
    match = re.search(r'page_(\d+)\.png$', image_path)
    if match:
        return int(match.group(1))
    return None

def normalize_attribute_list(data, key):
    """
    Normalizza l'attributo in una lista, anche se nel JSON appare come stringa singola.
    Gestisce anche il caso in cui la chiave sia assente.
    """
    val = data.get(key, [])
    if isinstance(val, str):
        return [val]
    return val

def get_attributes_by_page(json_data, target_page_number):
    """
    Restituisce una lista di regioni (con geometria, etichetta e sotto-attributo)
    solo per il numero di pagina specificato.
    """
    page_attributes = []

    # Cerca l'oggetto pagina corrispondente al numero richiesto
    target_page_data = None
    for page in json_data:
        if extract_page_number(page.get('image', '')) == target_page_number:
            target_page_data = page
            break
    
    # Se la pagina non viene trovata, restituisce una lista vuota
    if not target_page_data:
        return []

    # 1. Prepara le liste dei sotto-attributi (normalizzate)
    roi_types = normalize_attribute_list(target_page_data, 'roi_type')
    censor_types = normalize_attribute_list(target_page_data, 'censor_type')
    censor_close_types = normalize_attribute_list(target_page_data, 'close_type')

    # Contatori per tracciare l'indice corrente dei sotto-attributi
    roi_counter = 0
    censor_counter = 0
    censor_close_counter = 0

    # 2. Itera sulle regioni (bounding boxes) della pagina trovata
    labels = target_page_data.get('label', [])
    
    for region in labels:
        # Estrai geometria
        geometry = {
            'x': region.get('x'),
            'y': region.get('y'),
            'width': region.get('width'),
            'height': region.get('height')
        }

        # Estrai l'etichetta principale (roi o censor)
        rect_labels = region.get('rectanglelabels', [])
        primary_label = rect_labels[0] if rect_labels else None

        # 3. Estrai il sotto-attributo corretto basandoti sull'ordine
        sub_attribute = None
        
        if primary_label == 'roi':
            if roi_counter < len(roi_types):
                sub_attribute = roi_types[roi_counter]
                roi_counter += 1
        elif primary_label == 'censor':
            if censor_counter < len(censor_types):
                sub_attribute = censor_types[censor_counter]
                censor_counter += 1
        elif primary_label == 'censor-close':
            if censor_close_counter < len(censor_close_types):
                sub_attribute = censor_close_types[censor_close_counter]
                censor_close_counter += 1

        # Aggiungi alla lista dei risultati
        page_attributes.append({
            'geometry': geometry,
            'label': primary_label,
            'sub_attribute': sub_attribute
        })

    return page_attributes

def get_page_list(json_data):
    """
    Restituisce una lista ordinata dei numeri di pagina presenti nel JSON.
    """
    page_numbers = []
    for entry in json_data:
        image_path = entry.get('image', '')
        p_num = extract_page_number(image_path)
        if p_num is not None:
            page_numbers.append(p_num)
    return sorted(page_numbers)

def get_page_dimensions(json_data, target_page_number):
    """
    Retrieves the (width, height) of the specified page.
    Returns a tuple (width, height) or None if not found.
    """
    for entry in json_data:
        # Check if this entry corresponds to the target page
        if extract_page_number(entry.get('image', '')) == target_page_number:
            
            # The dimensions are stored inside the 'label' list items
            labels = entry.get('label', [])
            
            if labels:
                # We assume the page dimensions are the same for all labels on that page,
                # so we take them from the first one.
                first_label = labels[0]
                width = first_label.get('original_width')
                height = first_label.get('original_height')
                return width, height
            else:
                # Label list is empty, cannot determine dimensions from this schema
                return None
                
    return None

def get_censor_type(json_data, target_page_number):
    for entry in json_data:
        # Check if this entry corresponds to the target page
        if extract_page_number(entry.get('image', '')) == target_page_number:
            
            return entry.get('image_type', '') 
    return None

def get_box_coords_json(box,image_size):
    width=image_size[0]
    height=image_size[1]
    g=box['geometry']
    box_coords = g['x']/100.0*width,g['y']/100.0 * height,(g['x']+g['width'])/100.0 * width,(g['y']+g['height'])/100.0*height
    return box_coords
# --- Esempio di utilizzo ---

# Immaginiamo che 'data' sia il contenuto del file JSON caricato
# data = json.load(open('doc_5.json'))

'''
# Esempio: Estrai attributi solo per la Pagina 3
attributes_page_3 = get_attributes_by_page(data, 3) # Assumendo che 'data' sia definito

print(f"Trovate {len(attributes_page_3)} regioni nella pagina 3:")
for attr in attributes_page_3:
    print(attr)'''


def get_align_boxes(root,pre_computed,img_id):
    ''' given a root file for the questionnairre of interest it returns the list of the 
    bb regions thqt hqve roi qttribute qnd qlign sub qttribute'''
    roi_boxes = []
    pre_computed_rois = []
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['sub_attribute'] == "align":
            roi_boxes.append(box_coords)
            pre_computed_rois.append(pre_computed[i])
        i+=1
    return roi_boxes, pre_computed_rois

def get_ocr_boxes(root,pre_computed,img_id): 
    roi_boxes = []
    pre_computed_rois = []
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['sub_attribute'] == "ocr":
            roi_boxes.append(box_coords)
            pre_computed_rois.append(pre_computed[i])
        i+=1
    return roi_boxes, pre_computed_rois


def get_roi_boxes(root,pre_computed,img_id):
    roi_boxes = []
    pre_computed_rois = []
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['label'] == "roi" and (box['sub_attribute']=='standard' or box['sub_attribute']=='text'): #i should make a specific output for text boxes?
            roi_boxes.append(box_coords)
            pre_computed_rois.append(pre_computed[i])
        elif box['label'] == "roi" and box['sub_attribute']=="blank":
            blank_box=box_coords
            pre_computed_blank=pre_computed[i]
            #print("Found blank box")
        i+=1
    #print(i)
    roi_boxes.append(blank_box) # i put the blank box as the last one
    pre_computed_rois.append(pre_computed_blank)
    return roi_boxes, pre_computed_rois

def get_censor_boxes(root,img_id):
    roi_boxes = []
    partial_coverage=[]
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['label'] == "censor":
            roi_boxes.append(box_coords)
            if box['sub_attribute'] == "partial":
                partial_coverage.append(True)
            else:
                partial_coverage.append(False)
        i+=1
    return roi_boxes, partial_coverage

def get_censor_close_boxes(root,img_id):
    roi_boxes = []
    id_boxes = []
    
    bb_list=get_attributes_by_page(root, img_id)
    img_size = get_page_dimensions(root,img_id)

    i=0
    for box in bb_list:
        box_coords=get_box_coords_json(box,img_size)
        if box['label'] == "censor-close":# and (box['sub_attribute'] == "standard" or box['sub_attribute'] == "not_sure"):
            roi_boxes.append(box_coords)
        '''elif box['label'] == "censor-close" and box['sub_attribute'] == "identification":
            id_boxes.append(box_coords)'''
        i+=1
    return roi_boxes, id_boxes
