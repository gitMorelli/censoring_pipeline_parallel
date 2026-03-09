from pathlib import Path
from typing import List, Union
from PIL import Image
import os
import shutil
import json
import numpy as np
import cv2

def list_files_with_extension(
    folder: Union[str, Path],
    extension: Union[str, List[str], None] = None,
    recursive: bool = False,
) -> List[Path]:
    """ 
    Return file paths in `folder` that match `extension`.

    Args:
        folder: Directory to search (path string or Path).
        extension: File extension to match (with or without leading dot, e.g. "txt" or ".txt"),
                   a list of such extensions, or None/"*" / "all" to include all files.
        recursive: If True, search subdirectories recursively.

    Returns:
        List[Path]: Sorted list of matching file paths.

    Raises:
        ValueError: If `folder` does not exist or is not a directory.
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    # treat None, "*", or "all" as match-all
    if extension is None or (isinstance(extension, str) and extension in ("*", "all")):
        iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
        return sorted([p for p in iterator if p.is_file()])

    # handle list/tuple of extensions
    if isinstance(extension, (list, tuple)):
        exts = [(e if e.startswith(".") else f".{e}") for e in extension if isinstance(e, str)]
        results = set()
        for ext in exts:
            pattern = f"*{ext}"
            iterator = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
            for p in iterator:
                if p.is_file():
                    results.add(p)
        return sorted(results)

    # single extension string
    ext = extension if extension.startswith(".") else f".{extension}"
    pattern = f"*{ext}"
    iterator = folder_path.rglob(pattern) if recursive else folder_path.glob(pattern)
    return sorted([p for p in iterator if p.is_file()])

def get_basename(file_path: Union[str, Path], remove_extension: bool = False) -> str:
    """
    Return the base name (final path component) of `file_path`.

    Args:
        file_path: Path or path string to the file.
        remove_extension: If True, return the stem (name without suffix); otherwise return the full name.

    Returns:
        str: Base name or stem of the file path.
    """
    p = Path(file_path)
    return p.stem if remove_extension else p.name

def create_folder(folder: Union[str, Path], parents: bool = True, exist_ok: bool = True) -> Path:
    """
    Create the directory `folder` if it does not exist and return its Path.

    Args:
        folder: Directory path (string or Path).
        parents: If True, create parent directories as needed.
        exist_ok: Passed to Path.mkdir(); if False and the directory already exists an error is raised.

    Returns:
        Path: Path object for the created (or existing) directory.

    Raises:
        ValueError: If a file exists at `folder`.
        OSError: If the directory cannot be created.
    """
    p = Path(folder)
    if p.exists():
        if p.is_file():
            raise ValueError(f"Path exists and is a file: {folder}")
        return p
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p

def save_to_png_safe(input_path, output_path):
    with Image.open(input_path) as im:
        icc = im.info.get("icc_profile")
        im.save(
            output_path,
            format="PNG",
            optimize=False,
            icc_profile=icc,
        )
    return 0

def list_subfolders(folder: Union[str, Path], recursive: bool = False, include_hidden: bool = False) -> List[Path]:
    """
    Return directories contained in `folder`.

    Args:
        folder: Directory to list (path string or Path).
        recursive: If True, include subdirectories at all depths.
        include_hidden: If False, exclude entries whose name starts with a dot.

    Returns:
        List[Path]: Sorted list of subdirectory Paths.

    Raises:
        ValueError: If `folder` does not exist or is not a directory.
    """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    iterator = folder_path.rglob("*") if recursive else folder_path.iterdir()
    subdirs = [
        p for p in iterator
        if p.is_dir() and (include_hidden or not p.name.startswith("."))
    ]
    return sorted(subdirs)

def check_name_matching(annotation_file_names, template_folder_names, logger):
    ann_set = set(annotation_file_names)
    tpl_set = set(template_folder_names)

    if ann_set != tpl_set:
        missing = sorted(list(ann_set - tpl_set))
        extra = sorted(list(tpl_set - ann_set))
        logger.write("Annotation files and template folders do not match.")
        if missing:
            logger.write(f"Annotations without corresponding template folders: {missing}")
        if extra:
            logger.write(f"Template folders without corresponding annotations: {extra}")
        return 1

    logger.write(f"Annotation filenames and template folder names match ({len(ann_set)} items).")

def get_page_number(file_name: str) -> int:
    """
    Extract the page number from a filename formatted as 'page_XX.png'.

    Args:
        file_name: Filename string.

    Returns:
        int: Extracted page number.

    Raises:
        ValueError: If the filename does not match the expected format.
    """
    base_name = Path(file_name).stem
    try:
        page_number = int(base_name.split("_")[-1])
    except (IndexError, ValueError):
        raise ValueError(f"Filename does not contain a valid page number: {file_name}")
    return page_number

def sort_files_by_page_number(file_paths: List[Union[str, Path]]) -> List[Path]:
    """
    Sort a list of file paths based on the page number extracted from their filenames.

    Args:
        file_paths: List of file paths (strings or Paths)."""
    return sorted(file_paths, key=lambda p: get_page_number(str(p)))
    
def remove_folder(folder_path):
    """
    Delete a folder and all of its contents.

    Parameters:
        folder_path (str): Path to the folder to delete.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder removed: {folder_path}")
        except Exception as e:
            print(f"Error removing folder: {e}")
    else:
        print("Folder does not exist.")

def load_annotation_tree(logger,annotation_path):
    ''' this function takes the annotation path and returns the list of the annotation file paths and the list of their root names
    the files are sorted alphabetically (because the function "list_files_with_extension" does it)
    '''
    annotation_files = list_files_with_extension(annotation_path, "json", recursive=False)
    logger.write(f"Found {len(annotation_files)} annotation file(s) in {annotation_path}") 
    if not annotation_files:
        logger.write("No annotation files found. Exiting.")
        return 0

    #print(annotation_path)
    annotation_file_names = [get_basename(annotation_file, remove_extension=True) for annotation_file in annotation_files]

    return annotation_file_names, annotation_files

def load_templates_tree(logger,template_path,annotation_file_names=None):
    ''' given the folder in which the png templates are stored it returns the folder paths and their basenames (q_1,q_2, ...)
    also it checks (optionally) that they correspond to the annotation tree'''
    template_folders = list_subfolders(template_path, recursive=False)
    template_folder_names = [get_basename(p, remove_extension=False) for p in template_folders]
    logger.write(f"Template folder names: {template_folder_names}")

    if annotation_file_names:
        #check that names match
        if check_name_matching(annotation_file_names, template_folder_names, logger) == 1:
            logger.write("Mismatch between annotation files and template folders. Exiting.")
            return 1
        #check that they are sorted in the same way
        assert annotation_file_names == template_folder_names, "Annotation files and template folders are not in the same order."
    return template_folder_names, template_folders

def load_subjects_tree(logger, filled_path):
    ''' this function takes the path that collects all pngs to process and returns the list of the paths of the subfolders
    and of their base names (I expect the filled path to have N folders inside each containing the data of a different subject)
    the folder names are sorted alphabetically (because the function "list_files_with_extension" does it)

    The function also initialize the warning_map in which i save warning messages for the processed files for each subject, questionnaire and page
    '''
    filled_folders = list_subfolders(filled_path, recursive=False)
    filled_folder_names = [get_basename(p, remove_extension=False) for p in filled_folders]
    logger.debug("Filled folder names: %s", len(filled_folder_names))
    warning_map=[[] for _ in range(len(filled_folders))]
    return warning_map, filled_folder_names, filled_folders

#i can load all the pre-computed data at once to spare time; since i won't re-open the files every time (shouldn't be intensive on memory) 
def load_template_info(logger,annotation_files,annotation_file_names,annotation_path, 
                       security_check=True, selected_files = None):
    if selected_files == None:
        selected_files = [f"q_{i}" for i in range(1,13)]+["q_1v2"]
    annotation_roots=[]
    path_npy=os.path.join(annotation_path,"precomputed_features")
    npy_files=list_files_with_extension(path_npy, "npy", recursive=False)
    npy_file_names = [get_basename(npy_file, remove_extension=True) for npy_file in npy_files]

    # I match them with the annotation file names (will be a more complex function, in this test the names are the same)
    #check that names match
    if security_check:
        if check_name_matching(annotation_file_names, npy_file_names, logger) == 1:
            logger.error(f"Mismatch between annotation files and numpy files . Exiting.")
            return 1
        #check that they are sorted in the same way
        assert annotation_file_names == npy_file_names, "Annotation files and numpy files are not in the same order."

    #i load the data i will use (xml first)
    for i, annotation_file in enumerate(annotation_files):
        #_ ,root = load_xml(annotation_file)
        if annotation_file_names[i] in selected_files: #i return a template only if it is selectd (standard behavior is select all)
            with open(annotation_file, 'r') as f: root = json.load(f)
            annotation_roots.append(root)
    #then numpy arrays
    npy_data=[]
    for i, npy_file in enumerate(npy_files):
        if annotation_file_names[i] in selected_files:
            data_dict = np.load(npy_file, allow_pickle=True).item()
            npy_data.append(data_dict)
    return annotation_roots, npy_data


def serialize_keypoints(kp_list):
    # Extracts (x, y, size, angle, response, octave, class_id)
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kp_list]
def deserialize_keypoints(kp_data):
    return [cv2.KeyPoint(x=pt[0], y=pt[1], size=pt[2], angle=pt[3], 
                         response=pt[4], octave=pt[5], class_id=pt[6]) 
            for pt in kp_data]