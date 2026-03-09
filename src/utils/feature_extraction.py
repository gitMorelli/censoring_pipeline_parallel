from PIL import Image
from typing import Tuple
from time import perf_counter, time
import cv2
import numpy as np
from typing import Tuple, Union, Any
import zlib
from pathlib import Path
from src.utils.logging import FileWriter
from scipy.fftpack import dct

import pytesseract #for ocr
pytesseract.pytesseract.tesseract_cmd = r'//vms-e34n-databr/2025-handwriting\programs\tesseract\tesseract.exe'

from src.utils.file_utils import serialize_keypoints

ModeImage = Union[Image.Image, np.ndarray]

def _normalize_mode(mode: str) -> str:
    if mode is None:
        return "PIL"
    m = mode.lower()
    if m in ("cv2", "opencv"):
        return "cv2"
    return "PIL"

### load and show image ###################################################################
def load_image(image_path: str, mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Load and return an image using either PIL.Image (mode='PIL') or cv2 (mode='cv2').

    Returns PIL.Image.Image for mode='PIL' or numpy.ndarray (BGR) for mode='cv2'."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        img = Image.open(image_path)
        img.load()  # ✅ force full decode into memory
    else:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if verbose:
        _t1 = perf_counter()
        if mode == "PIL":
            print(f"load_image: mode=PIL path={image_path} size={img.size} mode={img.mode} time={( _t1 - _t0 ):0.6f}s")
        else:
            h, w = (img.shape[:2] if isinstance(img, np.ndarray) else (0, 0))
            ch = img.shape[2] if (isinstance(img, np.ndarray) and img.ndim == 3) else 1
            print(f"load_image: mode=cv2 path={image_path} size=({w},{h}) channels={ch} time={( _t1 - _t0 ):0.6f}s")

    return img

def show_image(img,mode='cv2'):
    if mode=='PIL':
        img.show()
    elif mode=='cv2':
        import matplotlib.pyplot as plt
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

### initial preprocessing ###
# function to crop patches from images
def crop_patch(img: ModeImage, box: Tuple[int, int, int, int], mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Crop and return a patch. box = (left, top, right, bottom)."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(img, Image.Image):
            raise TypeError("img must be a PIL.Image.Image for mode='PIL'")
        left, top, right, bottom = map(int, box)
        left = max(0, min(left, img.width))
        top = max(0, min(top, img.height))
        right = max(left, min(right, img.width))
        bottom = max(top, min(bottom, img.height))
        patch = img.crop((left, top, right, bottom)).copy()
        if verbose:
            _t1 = perf_counter()
            print(f"crop_patch: mode=PIL box=({left},{top},{right},{bottom}) size={patch.size} time={( _t1 - _t0 ):0.6f}s")
        return patch

    else:
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a numpy.ndarray for mode='cv2'")
        left, top, right, bottom = map(int, box)
        h, w = img.shape[:2]
        left = max(0, min(left, w))
        top = max(0, min(top, h))
        right = max(left, min(right, w))
        bottom = max(top, min(bottom, h))
        patch = img[top:bottom, left:right].copy()
        if verbose:
            _t1 = perf_counter()
            ph, pw = patch.shape[:2]
            print(f"crop_patch: mode=cv2 box=({left},{top},{right},{bottom}) size=({pw},{ph}) time={( _t1 - _t0 ):0.6f}s")
        return patch

### deal with polygons ###
def _order_quad(pts):
    """
    Ensure points are ordered as: top-left, top-right, bottom-right, bottom-left.
    Accepts shape (4,2) list/array of (x,y).
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("pts must be an array-like of shape (4, 2)")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def crop_quad_mask(image, pts):
    """
    Mask + crop to bounding box (no rectification).
    - image: BGR/GRY numpy array from cv2
    - pts: 4 vertices (x,y), any order
    Returns: cropped image (same perspective) tightly around the polygon.
    """
    h, w = image.shape[:2]
    quad = _order_quad(pts)

    # Create polygon mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)

    # Apply mask
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Crop to polygon's bounding box
    x_min = max(int(np.floor(quad[:,0].min())), 0)
    x_max = min(int(np.ceil (quad[:,0].max())), w)
    y_min = max(int(np.floor(quad[:,1].min())), 0)
    y_max = min(int(np.ceil (quad[:,1].max())), h)

    return masked[y_min:y_max, x_min:x_max]

def crop_quad_warp(image, pts, out_size=None):
    """
    Perspective-rectified crop of a 4-point polygon.
    - image: BGR/GRY numpy array from cv2
    - pts: 4 vertices (x,y), any order
    - out_size: (width, height) for output; if None, inferred from quad geometry
    Returns: rectified crop.
    """
    quad = _order_quad(pts)

    # Compute side lengths to infer a sensible output size when not provided
    (tl, tr, br, bl) = quad
    def dist(a, b): 
        return np.hypot(*(a - b))

    widthA = dist(br, bl)
    widthB = dist(tr, tl)
    heightA = dist(tr, br)
    heightB = dist(tl, bl)
    W = int(round(max(widthA, widthB)))
    H = int(round(max(heightA, heightB)))

    if out_size is not None:
        W, H = int(out_size[0]), int(out_size[1])

    dst = np.array([[0, 0],
                    [W - 1, 0],
                    [W - 1, H - 1],
                    [0, H - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(image, M, (W, H), flags=cv2.INTER_LINEAR)
    return warped


# check if crop region is a polygon or rectangle
def is_polygon(box: Any) -> bool:
    """Check if box is a polygon (list/array of 4 (x,y) points) or rectangle (4-tuple)."""
    if isinstance(box, (list, np.ndarray)):
        arr = np.array(box)
        return arr.ndim == 2 and arr.shape == (4, 2)
    return False

# convert to grayscale
def convert_to_grayscale(patch: ModeImage, mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Convert a patch to grayscale.
    Returns PIL.Image.Image (mode='L') for PIL, or numpy.ndarray single-channel for cv2."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        gray_patch = patch.convert("L")
        if verbose:
            _t1 = perf_counter()
            print(f"convert_to_grayscale: mode=PIL original_mode={patch.mode} time={( _t1 - _t0 ):0.6f}s")
        return gray_patch

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
        if patch.ndim == 2:
            gray = patch.copy()
        elif patch.ndim == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported ndarray shape for grayscale conversion")
        if verbose:
            _t1 = perf_counter()
            print(f"convert_to_grayscale: mode=cv2 shape={patch.shape} time={( _t1 - _t0 ):0.6f}s")
        return gray

# resize the patch to a fixed size
def resize_patch_to_fixed(patch: ModeImage, size: Tuple[int, int], mode: str = "cv2", verbose: bool = False):
    """Resize a patch to (width, height).
    Returns (resized_image, scale_x, scale_y)."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    target_w, target_h = map(int, size)

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        resized_patch = patch.resize((target_w, target_h), Image.Resampling.LANCZOS)
        orig_w, orig_h = patch.size
        scale_x = target_w / orig_w if orig_w != 0 else 0
        scale_y = target_h / orig_h if orig_h != 0 else 0
        if verbose:
            _t1 = perf_counter()
            print(f"resize_patch: mode=PIL original_size={patch.size} target_size={(target_w,target_h)} time={( _t1 - _t0 ):0.6f}s")
        return resized_patch, scale_x, scale_y

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
        orig_h, orig_w = patch.shape[:2]
        # Choose Lanczos-like interpolation if available
        interp = cv2.INTER_LANCZOS4 if hasattr(cv2, "INTER_LANCZOS4") else cv2.INTER_AREA
        resized = cv2.resize(patch, (target_w, target_h), interpolation=interp)
        scale_x = target_w / orig_w if orig_w != 0 else 0
        scale_y = target_h / orig_h if orig_h != 0 else 0
        if verbose:
            _t1 = perf_counter()
            print(f"resize_patch: mode=cv2 original_size=({orig_w,orig_h}) target_size={(target_w,target_h)} time={( _t1 - _t0 ):0.6f}s")
        return resized, scale_x, scale_y

#resize patch by scale factor scale (no target size)
def resize_patch(patch: ModeImage, scale: float, mode: str = "cv2", verbose: bool = False):
    """Resize a patch by scale factor.
    Returns (resized_image, scale_x, scale_y)."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        orig_w, orig_h = patch.size
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))
        resized_patch = patch.resize((target_w, target_h), Image.Resampling.LANCZOS)
        if verbose:
            _t1 = perf_counter()
            print(f"resize_patch: mode=PIL original_size={patch.size} scale={scale} time={( _t1 - _t0 ):0.6f}s")
        return resized_patch, scale, scale

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
        orig_h, orig_w = patch.shape[:2]
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))
        # Choose Lanczos-like interpolation if available
        interp = cv2.INTER_LANCZOS4 if hasattr(cv2, "INTER_LANCZOS4") else cv2.INTER_AREA
        resized = cv2.resize(patch, (target_w, target_h), interpolation=interp)
        if verbose:
            _t1 = perf_counter()
            print(f"resize_patch: mode=cv2 original_size=({orig_w,orig_h}) scale={scale} time={( _t1 - _t0 ):0.6f}s")
        return resized, scale, scale

def resize_patch_asymmetric(patch: ModeImage, scale_x: float, scale_y):
    if not isinstance(patch, np.ndarray):
        raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
    orig_h, orig_w = patch.shape[:2]
    target_w = int(round(orig_w * scale_x))
    target_h = int(round(orig_h * scale_y))
    # Choose Lanczos-like interpolation if available
    interp = cv2.INTER_LANCZOS4 if hasattr(cv2, "INTER_LANCZOS4") else cv2.INTER_AREA
    resized = cv2.resize(patch, (target_w, target_h), interpolation=interp)
    return resized

# binarize the patch
def binarize_patch(patch: ModeImage, threshold: int = 128, mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Binarize a patch using threshold. For PIL returns mode='1' image, for cv2 returns uint8 array 0/255."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        gray_patch = patch.convert("L")
        binary_patch = gray_patch.point(lambda p: 255 if p >= threshold else 0, mode='1')
        if verbose:
            _t1 = perf_counter()
            print(f"binarize_patch: mode=PIL threshold={threshold} time={( _t1 - _t0 ):0.6f}s")
        return binary_patch

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
        # ensure grayscale
        gray = patch if (patch.ndim == 2) else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        if verbose:
            _t1 = perf_counter()
            print(f"binarize_patch: mode=cv2 threshold={threshold} time={( _t1 - _t0 ):0.6f}s")
        return binary

def crop_borders(img, border_pct) :
    """
    Crop a fixed percentage from each border.
    border_pct is fraction of width/height cropped from each side (e.g., 0.04 = 4%).
    """
    if border_pct <= 0:
        return img

    h, w = img.shape[:2]
    top = int(h * border_pct)
    bottom = h - top
    left = int(w * border_pct)
    right = w - left
    img = img[top:bottom, left:right]

    return img

### Feature extraction functions ######################################################################
### Functions that assume grayscale images###
# extract the number of black pixels in a patch
def count_black_pixels(patch: ModeImage, threshold: int = 10, mode: str = "cv2", verbose: bool = False) -> int:
    """Count number of pixels with intensity < threshold. Works for PIL Image or cv2 ndarray."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        gray = patch.convert("L")
        black_pixels = sum(1 for pixel in gray.getdata() if pixel < threshold)
        if verbose:
            _t1 = perf_counter()
            print(f"count_black_pixels: mode=PIL black_pixels={black_pixels} time={( _t1 - _t0 ):0.6f}s")
        return black_pixels

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError(f"patch must be a numpy.ndarray for mode='cv2'; got {type(patch)}")
        gray = patch if (patch.ndim == 2) else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        black_pixels = int(np.count_nonzero(gray < threshold))
        if verbose:
            _t1 = perf_counter()
            print(f"count_black_pixels: mode=cv2 black_pixels={black_pixels} time={( _t1 - _t0 ):0.6f}s")
        return black_pixels

#count connected components in a binary patch
def count_connected_components(patch: ModeImage, mode: str = "cv2", verbose: bool = False) -> int:
    """Count number of connected components in a binary patch. Works for PIL Image or cv2 ndarray."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    if mode == "PIL":
        if not isinstance(patch, Image.Image):
            raise TypeError("patch must be a PIL.Image.Image for mode='PIL'")
        binary = patch.convert("1")
        bw_array = np.array(binary, dtype=np.uint8) * 255
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(bw_array, connectivity=8)
        num_components = num_labels - 1  # exclude background
        if verbose:
            _t1 = perf_counter()
            print(f"count_connected_components: mode=PIL components={num_components} time={( _t1 - _t0 ):0.6f}s")
        return num_components

    else:
        if not isinstance(patch, np.ndarray):
            raise TypeError("patch must be a numpy.ndarray for mode='cv2'")
        binary = patch if (patch.ndim == 2) else cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_components = num_labels - 1  # exclude background
        if verbose:
            _t1 = perf_counter()
            print(f"count_connected_components: mode=cv2 components={num_components} time={( _t1 - _t0 ):0.6f}s")
        return num_components

#perceptual hash
def dct_phash(img, hash_size=8, dct_size=32):
    """
    Perceptual hash using 2D DCT (cv2.dct). Returns a boolean array of shape (hash_size, hash_size).
    Steps:
      - resize to (dct_size, dct_size)
      - compute DCT
      - use top-left (hash_size x hash_size) excluding the DC bias for thresholding by median
    """
    small = cv2.resize(img, (dct_size, dct_size), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    dct = cv2.dct(small)

    # Use the top-left block
    dct_low = dct[:hash_size, :hash_size]
    # Median threshold (exclude the DC term to reduce global luminance bias)
    median = np.median(dct_low[1:, 1:])
    return (dct_low > median).astype(np.uint8)


def page_phash(
    image,
    hash_size: int = 8,
    highfreq_factor: int = 4,
    border_crop_pct: float = 0.0,
) -> np.ndarray:
    """
    Perceptual hash (pHash) using DCT.
    Returns a boolean array of length hash_size*hash_size.

    border_crop_pct: crops that % from each side before hashing (e.g., 0.04 = 4%).
    """
    """
    Perceptual hash (pHash) using OpenCV DCT.
    Expects a grayscale numpy array (cv2 image) as input.
    """
    img = image.copy()

    # 1. Handle Border Cropping
    if border_crop_pct > 0:
        img = crop_borders(img, border_crop_pct)

    # 2. Resize
    # Note: OpenCV uses (width, height) for size. 
    # PIL's LANCZOS is roughly equivalent to INTER_AREA or INTER_CUBIC.
    img_size = hash_size * highfreq_factor
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 3. Prepare for DCT (Must be float32 or float64)
    pixels = img.astype(np.float32)

    # 4. 2D Discrete Cosine Transform
    # OpenCV's cv2.dct operates on the whole 2D array at once
    dct_full = cv2.dct(pixels)

    # 5. Extract the low-frequency coefficients (top-left)
    dct_low = dct_full[:hash_size, :hash_size]

    # 6. Calculate median excluding the DC term (0,0)
    dct_flat = dct_low.flatten()
    # We exclude the first element because it represents the average color 
    # of the image and can skew the hash.
    med = np.median(dct_flat[1:])

    # 7. Generate bit array
    return dct_flat > med

def phash_hamming_distance(a_bool, b_bool):
    # a_bool, b_bool are uint8 arrays of 0/1
    return int(np.bitwise_xor(a_bool, b_bool).sum())

# normalized cross-correlation
def ncc(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = (a.std() * b.std()) + 1e-6
    return float((a*b).mean() / denom),a,b

# edge-based IoU
def edge_iou(a, b):
    e1 = cv2.Canny(a, 100, 200)
    e2 = cv2.Canny(b, 100, 200)
    inter = np.logical_and(e1>0, e2>0).sum()
    union = np.logical_or(e1>0, e2>0).sum() + 1e-6
    return inter / union

#checksum functions
def binarize_for_checksum(img, target=64):
    """
    Prepare a stable binary representation:
      - downscale to (target, target)
      - mild blur to stabilize noise
      - Otsu threshold to binary
    """
    small = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (3,3), 0)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw
def binary_crc32(img):
    bw = binarize_for_checksum(img, target=64)
    return zlib.crc32(bw.tobytes()) & 0xffffffff

# projection profiles
def projection_profiles(img, axis=0):
    """
    Sum of pixel intensities along rows (axis=1) or columns (axis=0).
    Normalize to zero-mean, unit-variance for NCC comparison.
    """
    # Optionally shrink to stabilize comparisons on large ROIs
    base = img
    if max(img.shape) > 256:
        scale = 256.0 / max(img.shape)
        base = cv2.resize(img, (int(round(img.shape[1]*scale)), int(round(img.shape[0]*scale))), interpolation=cv2.INTER_AREA)

    if axis == 0:   # vertical projection (sum over rows -> per-column)
        prof = base.sum(axis=0).astype(np.float32)
    else:           # horizontal projection (sum over columns -> per-row)
        prof = base.sum(axis=1).astype(np.float32)

    # normalize for NCC computation
    prof -= prof.mean()
    std = prof.std()
    if std < 1e-6:
        std = 1.0
    prof /= std
    return prof
def profile_ncc(p1, p2):
    m = min(len(p1), len(p2))
    if m == 0:
        return 0.0
    # Center-crop to same length if needed
    def center_crop(x, m):
        if len(x) == m: return x
        start = (len(x)-m)//2
        return x[start:start+m]
    a = center_crop(p1, m)
    b = center_crop(p2, m)
    return float((a*b).mean())

### aggregator functions ######################################################################
#patch preprocessing (crop, grayscale, resize)
def preprocess_roi(img: ModeImage, box, target_size=(128,128), mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Crop, convert to grayscale, and resize a patch from img given box and target_size."""
    if is_polygon(box):
        patch = crop_quad_warp(img, np.array(box))
    else:
        patch = crop_patch(img, box, mode=mode, verbose=verbose)
    gray_patch = convert_to_grayscale(patch, mode=mode, verbose=verbose)
    if target_size is not None:
        resized_patch, _, _ = resize_patch_to_fixed(gray_patch, target_size, mode=mode, verbose=verbose)
        return resized_patch
    else:
        return gray_patch
#function to extract features from crops (rois) (expects preprocessed images)
def extract_features_from_roi(patch: ModeImage, mode: str = "cv2", 
                              verbose: bool = False,to_compute=['crc32','dct_phash', 'ncc','edge_iou','profile'],**kwargs) -> Any:
    """Extract features from a grayscale patch. Returns a dict of features."""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    features = {}
    # Example features:
    if 'crc32' in to_compute:
        features['crc32']=binary_crc32(patch)
    if 'dct_phash' in to_compute:
        features['dct_phash']=dct_phash(patch, hash_size=8, dct_size=32)
    if 'ncc' in to_compute or 'edge_iou' in to_compute:
        features['full'] = patch #for ncc and edge iou
    if 'profile' in to_compute:
        features['profile_h'] = projection_profiles(patch, axis=1)
        features['profile_v'] = projection_profiles(patch, axis=0)
    if 'orb' in to_compute:
        nfeatures=kwargs.get('orb_nfeatures',500)
        edgeThreshold=kwargs.get('orb_edgeThreshold',5)
        patchSize=kwargs.get('orb_patchSize',5)
        fastThreshold=kwargs.get('orb_fastThreshold',5)
        orb = cv2.ORB_create(nfeatures=nfeatures,fastThreshold=fastThreshold,edgeThreshold=edgeThreshold, patchSize=patchSize) #the patch should be grayscaled (it is since i expect it also for phash)
        kp, des = orb.detectAndCompute(patch, None)
        features['orb_kp']=serialize_keypoints(kp)
        features['orb_des']=des
        features['orb_args'] = {
            'orb_nfeatures': nfeatures,
            'orb_edgeThreshold': edgeThreshold,
            'orb_patchSize': patchSize,
            'orb_fastThreshold': fastThreshold
        }

    if verbose:
        _t1 = perf_counter()
        print(f"extract_features_from_patch: mode={mode} time={( _t1 - _t0 ):0.6f}s")

    return features
#function to extract features from crops (blank)
def preprocess_blank_roi(img: ModeImage, box, mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Crop, convert to grayscale, and resize a blank patch from img given box and target_size."""
    if is_polygon(box):
        patch = crop_quad_warp(img, np.array(box))
    else:
        patch = crop_patch(img, box, mode=mode, verbose=verbose)
    gray_patch = convert_to_grayscale(patch, mode=mode, verbose=verbose)
    binary_patch = binarize_patch(gray_patch, threshold=128, mode=mode, verbose=verbose)
    return binary_patch
def extract_features_from_blank_roi(patch: ModeImage, mode: str = "cv2", 
                              verbose: bool = False,to_compute=['cc','n_black']) -> Any:
    """Extract features from a binary patch. Returns a dict of features."""
    mode = _normalize_mode(mode)
    if verbose: 
        _t0 = perf_counter()

    features = {}
    # Example features:
    if 'cc' in to_compute:
        features['cc']=count_connected_components(patch, mode=mode, verbose=verbose)
    if 'n_black' in to_compute:
        features['n_black']=count_black_pixels(patch, threshold=10, mode=mode, verbose=verbose)

    if verbose:
        _t1 = perf_counter()
        print(f"extract_features_from_blank_patch: mode={mode} time={( _t1 - _t0 ):0.6f}s")

    return features
#function to extract features from crops (alignment)
def preprocess_alignment_roi(img: ModeImage, box: Tuple[int, int, int, int], mode: str = "cv2", verbose: bool = False) -> ModeImage:
    """Crop and convert to grayscale a patch from img given box."""
    patch = crop_patch(img, box, mode=mode, verbose=verbose)
    gray_patch = convert_to_grayscale(patch, mode=mode, verbose=verbose)
    return gray_patch

def preprocess_page(img, mode="gray_only",extension='png',crop_mode='none',border_pct=None):
  # --- Image Preprocessing ---
    #i assume the image in input is in cv2 format, bgr
    original_image_bgr = img.copy()

    if crop_mode=='up':
        h, w, _ = original_image_bgr.shape
        top_third = original_image_bgr[: h // 3, :, :]
        original_image_bgr = top_third.copy()
    elif crop_mode == 'borders':
        original_image_bgr = crop_borders(original_image_bgr,border_pct)


    # 2. Convert to Grayscale
    gray_image = convert_to_grayscale(original_image_bgr, mode="cv2", verbose=False)

    if mode=="gray_only":
        return gray_image
    elif mode=="binarization":
        # 3. Binarization using Otsu's method
        # Otsu's binarization automatically finds the optimal threshold value.
        _, binarized_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized_image

    return 

def extract_features_from_page(patch: ModeImage, mode: str = "cv2", 
                              verbose: bool = False,to_compute=['page_phash'], **kwargs) -> Any:
    """Extract features from the preprocessed image"""
    mode = _normalize_mode(mode)
    if verbose:
        _t0 = perf_counter()

    features = {}
    # Example features:
    if 'page_phash' in to_compute:
        features['page_phash']=page_phash(patch, hash_size=8, border_crop_pct=kwargs.get('border_crop_pct',0))
        features['border_crop_pct']=kwargs.get('border_crop_pct',0) #I need to store the parameter to ensure the images to match will be processed in the same way
    if 'orb' in to_compute:
        nfeatures=kwargs.get('orb_nfeatures',2000)
        edgeThreshold=kwargs.get('orb_edgeThreshold',31)
        patchSize=kwargs.get('orb_patchSize',31)
        fastThreshold=kwargs.get('orb_fastThreshold',20)
        orb = cv2.ORB_create(nfeatures=nfeatures,fastThreshold=fastThreshold,edgeThreshold=edgeThreshold, patchSize=patchSize)
        kp, des = orb.detectAndCompute(patch, None)
        features['orb_kp']=serialize_keypoints(kp)
        features['orb_des']=des
        features['orb_args'] = {
            'orb_nfeatures': nfeatures,
            'orb_edgeThreshold': edgeThreshold,
            'orb_patchSize': patchSize,
            'orb_fastThreshold': fastThreshold
        }
    if verbose:
        _t1 = perf_counter()
        print(f"extract_features_from_patch: mode={mode} time={( _t1 - _t0 ):0.6f}s")

    return features

def preprocess_text_region(img,box,mode='cv2',aggressive=True,verbose=False): #now it is the same as the preprocess alignement region but i can tune it
    """Crop and convert to grayscale a patch from img given box."""
    patch = crop_patch(img, box, mode=mode, verbose=verbose)
    gray_patch = convert_to_grayscale(patch, mode=mode, verbose=verbose)
    if aggressive:
        # 2. Rescale (Zoom in) - Tesseract loves larger text
        gray_patch = cv2.resize(gray_patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 3. Apply Gaussian Blur to reduce noise
        gray_patch = cv2.GaussianBlur(gray_patch, (5, 5), 0)

        # 4. Apply Adaptive Thresholding (Better than standard Otsu for pages)
        gray_patch = cv2.adaptiveThreshold(gray_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    return gray_patch

def extract_features_from_text_region(patch: ModeImage, mode: str = "cv2", 
                            verbose: bool = False, lang: str = "fra", psm: int = 6) -> Any:
    '''psm=6 assumes a uniform block of text; other useful values: 3, 4, 7, 11.'''
    if verbose:
        _t0 = perf_counter()

    features = {}
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(patch, lang=lang, config=config)

    if verbose:
        print("Partial image text: ", text)

    features['text'] = text
    features['psm'] = psm

    if verbose:
        _t1 = perf_counter()
        print(f"extract_text_features_from_patch: mode={mode} time={( _t1 - _t0 ):0.6f}s")

    return features

######### Censoring function #########################################################################
# get a cv2 image and a set of ROIs, put the pixel black in those regions. SHould work both with polygons and rectangles


def fill_polygon_striped_relative(
    img: ModeImage, 
    pts: np.ndarray, 
    thickness_pct: float = 0.1, 
    spacing_mult: float = 0.5,
    color=(0, 0, 0)
):
    """
    Fills a polygon (ASSUMED TO BE A ROTATED RECTANGLE) with parallel stripes 
    that run parallel to the MOST VERTICAL AXIS of the rectangle.
    
    The function uses cv2.minAreaRect to calculate the rectangle's properties 
    and then draws rotated stripe segments using cv2.boxPoints.
    """
    
    # 1. Image and Channel Analysis
    is_color = len(img.shape) == 3
    if not is_color:
        if len(color) == 3:
            color = color[0] 
    
    # 2. Get Rotated Rectangle Properties
    # rect is a tuple: ((center_x, center_y), (width, height), angle)
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    
    w, h = size
    
    # Determine the MOST VERTICAL AXIS based on the angle.
    # The angle (A_w) is between the width (w) side and the x-axis, A_w in [-90, 0).
    # The angle of the height (h) side is A_h = A_w + 90.
    
    # If abs(angle) > 45, the w-side is closer to vertical (-90).
    # If abs(angle) <= 45, the h-side is closer to vertical (angle+90 is closer to 90).
    
    if abs(angle) <= 45:
        # Case 1: w-side is the most vertical. Stripes run parallel to w.
        vertical_dim = w
        horizontal_dim = h
        
        # Stripe Coverage Dimension is the dimension we step across (w)
        stripe_coverage_dim = vertical_dim
        # Stripe Length is the full length of the stripe (h)
        stripe_length = horizontal_dim
        
        # The rotation angle of the stripe segment is the angle of the vertical axis (w)
        final_angle = angle 
        
    else: # abs(angle) <= 45
        # Case 2: h-side is the most vertical. Stripes run parallel to h.
        vertical_dim = h
        horizontal_dim = w
        
        # Stripe Coverage Dimension is the dimension we step across (h)
        stripe_coverage_dim = vertical_dim
        # Stripe Length is the full length of the stripe (w)
        stripe_length = horizontal_dim
        
        # The rotation angle of the stripe segment is the angle of the vertical axis (h)
        final_angle = angle + 90
        
    # Apply the determined angle
    angle = final_angle 
        
    # 3. Calculate Stripe Parameters
    
    # The thickness is calculated relative to the coverage dimension
    thickness = max(1, int(stripe_coverage_dim * thickness_pct))
    gap = max(1, int(thickness * spacing_mult))
    step = thickness + gap

    # Total number of stripes needed to cover the dimension
    num_steps = int(np.ceil(stripe_coverage_dim / step))
    
    half_coverage_dim = stripe_coverage_dim / 2.0
    
    # 4. Draw Stripes by Filling Rotated Rectangles

    for i in range(num_steps):
        # Calculate the position along the coverage dimension's axis (from -half_coverage_dim to +half_coverage_dim)
        # Start position is -half_coverage_dim + offset to the center of the first stripe
        current_pos = -half_coverage_dim + i * step + thickness / 2.0
        
        # Calculate the transformation vector (dx, dy) for the offset from the main center
        # The rotation direction is determined by the final_angle
        rad = np.deg2rad(angle)
        dx = np.cos(rad) * current_pos
        dy = np.sin(rad) * current_pos
        
        # The center of the current stripe segment
        stripe_center = (center[0] + dx, center[1] + dy)
        
        # Stripe rectangle definition: (center, size, angle)
        # Size is (thickness, full length)
        stripe_rect = (
            stripe_center, 
            (thickness, stripe_length), 
            angle
        )
        
        # Get the four vertices of the rotated stripe rectangle
        stripe_pts = cv2.boxPoints(stripe_rect)
        stripe_pts = np.int32(stripe_pts)
        
        # Fill the sub-polygon (the stripe segment) directly on the image copy
        cv2.fillPoly(img, [stripe_pts], color=color)

def censor_image(img: ModeImage, roi_boxes, verbose: bool = False, partial_coverage=None,logger=None,**kwargs) -> ModeImage:
    """Censor (blacken) regions in img defined by roi_boxes (list of boxes)."""
    logger and logger.call_start(f'save_censored_image')
    if partial_coverage == None:
        partial_coverage=[False for i in range(len(roi_boxes))]
    if verbose:
        _t0 = perf_counter()

    logger and logger.call_start(f'copy_image')
    censored_img = img.copy()
    logger and logger.call_end(f'copy_image')

    for i,box in enumerate(roi_boxes):
        if is_polygon(box):
            pts = np.array(box, dtype=np.int32)
        else:
            left, top, right, bottom = map(int, box)
            pts = np.array([
                [left,  top],
                [right, top],
                [right, bottom],
                [left,  bottom]
            ], dtype=np.int32)
        if partial_coverage[i]==False:
            logger and logger.call_start(f'fill_region')
            cv2.fillPoly(censored_img, [pts], color=(0, 0, 0))
            logger and logger.call_end(f'fill_region')
        else:
            thickness_pct=kwargs.get('thickness_pct',0.1)
            spacing_mult=kwargs.get('spacing_mult',0.5)
            logger and logger.call_start(f'fill_striped_region')
            fill_polygon_striped_relative(
                censored_img, pts,
                thickness_pct=thickness_pct,
                spacing_mult=spacing_mult
            )
            logger and logger.call_end(f'fill_striped_region')

    if verbose:
        _t1 = perf_counter()
        print(f"censor_image: num_rois={len(roi_boxes)} time={( _t1 - _t0 ):0.6f}s")
    logger and logger.call_end(f'save_censored_image')
    return censored_img

 
def censor_image_with_boundary(img: ModeImage, roi_boxes, boundary_boxes, logger=None,
                               verbose: bool = False, partial_coverage=None,**kwargs) -> ModeImage:
    """Censor (blacken) regions in img defined by roi_boxes (list of boxes)"""
    logger and logger.call_start(f'save_censored_image')
    if partial_coverage == None:
        partial_coverage=[False for i in range(len(roi_boxes))]
    if verbose:
        _t0 = perf_counter()

    logger and logger.call_start(f'copy_image')
    censored_img = img.copy()
    logger and logger.call_end(f'copy_image')

    for i,box in enumerate(roi_boxes):
        #convert to polygon
        if is_polygon(box):
            pts = np.array(box, dtype=np.int32)
        else:
            left, top, right, bottom = map(int, box)
            pts = np.array([
                [left,  top],
                [right, top],
                [right, bottom],
                [left,  bottom]
            ], dtype=np.int32)

        #get boundary and create mask
        x1, y1, x2, y2 = np.round(boundary_boxes[i]).astype(int)
        # 3. Translate the rotated polygon to the AABB's local coordinate system
        # We subtract the AABB top-left corner (x1, y1)
        local_poly = pts - [x1, y1]

        # 4. Create a small mask the size of the AABB
        roi_h, roi_w = (y2 - y1), (x2 - x1)
        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        is_color = len(img.shape) == 3 #true if is rgb

        if partial_coverage[i]==False:
            logger and logger.call_start(f'fill_region')
            cv2.fillPoly(mask_roi, [local_poly.astype(np.int32)], 255)
            logger and logger.call_end(f'fill_region')
            img_roi = censored_img[y1:y2, x1:x2] #this actually modifies the censored_img since img_roi is a view
        else:
            #print("Sono entrato !")
            thickness_pct=kwargs.get('thickness_pct',0.1)
            spacing_mult=kwargs.get('spacing_mult',0.5)
            logger and logger.call_start(f'fill_striped_region')
            fill_polygon_striped_relative(
                mask_roi, local_poly.astype(np.int32),
                thickness_pct=thickness_pct,
                spacing_mult=spacing_mult,
                color = [255]
            )
            img_roi = censored_img[y1:y2, x1:x2] #this actually modifies the censored_img since img_roi is a view
            logger and logger.call_end(f'fill_striped_region')
        if not is_color:
            img_roi[mask_roi == 255] = 0
        else:
            img_roi[mask_roi == 255] = (0, 0, 0)
    if verbose:
        _t1 = perf_counter()
    logger and logger.call_end(f'save_censored_image')
    return censored_img