"""
pose_align_utils.py – Core utilities and helper functions for pose alignment
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import math
import time
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

# ─────────────────────────── Constants ────────────────────────────
NOSE, NECK, R_SH, L_SH, MIDHIP = 0, 1, 2, 5, 8
NUM_BODY_JOINTS = 18
FULL_TORSO = [NOSE, NECK, R_SH, L_SH, MIDHIP]
UPPER_TORSO = [NOSE, NECK, R_SH, L_SH]
TORSO = np.asarray(FULL_TORSO)

# Extra segments and joints we trust for robust estimates
HEAD_SEG = [(15, 16), (0, 15), (0, 16)]      # ear-to-ear, nose-ear
TORSO_SEG = [(R_SH, L_SH), (NECK, MIDHIP)]    # shoulder width, neck-hip
STABLE_SEGMENTS = HEAD_SEG + TORSO_SEG
ROBUST_JOINTS = np.asarray([NOSE, NECK, R_SH, L_SH, MIDHIP, 15, 16])

# OpenPose colour map (RGB 0-255) used by the mask extractor
OPENPOSE_COLOUR_MAP = {
    0:(255,0,0),1:(255,85,0),2:(255,170,0),3:(255,255,0),4:(170,255,0),5:(85,255,0),6:(0,255,0),
    7:(0,255,85),8:(0,255,170),9:(0,255,255),10:(0,170,255),11:(0,85,255),12:(0,0,255),13:(85,0,255),
    14:(170,0,255),15:(255,0,255),16:(255,0,170),17:(255,0,85)
}

# BODY-25 limb pairs for the viewer debug node
BODY_25_PAIRS = [(1,8),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(10,11),
                 (8,12),(12,13),(13,14),(1,0),(0,15),(15,17),(0,16),(16,18),
                 (14,19),(19,20),(14,21),(11,22),(22,23),(11,24)]

# Type aliases
Keypoints = np.ndarray

# ───────────────── Tensor ↔ OpenCV Convenience Functions ─────────────────────────────
def _to_chw(img: torch.Tensor) -> torch.Tensor:
    """Convert image tensor to CHW format"""
    if img.dim() == 4:             # (1,3,H,W) from ComfyUI
        img = img[0]
    return img if img.shape[0] == 3 else img.permute(2,0,1)

def torch_to_u8(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to uint8 BGR numpy array"""
    chw = _to_chw(t)                              # (3,H,W)
    rgb = chw.permute(1,2,0).cpu().numpy()
    return cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def u8_to_torch(bgr: np.ndarray) -> torch.Tensor:
    """Convert uint8 BGR numpy array to torch tensor"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return torch.from_numpy(rgb[None])            # (1,H,W,3) for ComfyUI

def torch_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert torch tensor to PIL Image for saving"""
    chw = _to_chw(tensor)
    rgb = chw.permute(1,2,0).cpu().numpy()
    rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8)

# ───────────────── Keypoint Processing Functions ─────────────────────────────
def _reshape(flat: List[float]) -> np.ndarray:
    """Reshape flat list of coordinates to Nx3 array"""
    return np.asarray(flat, np.float32).reshape(-1, 3)[:, :3]

def _coords_to_xy(arr: np.ndarray, thr: float = 0.15) -> Keypoints:
    """Convert coordinate array to xy keypoints with confidence thresholding"""
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    J = min(NUM_BODY_JOINTS, arr.shape[0])
    for j in range(J):
        x, y, c = arr[j]
        if c >= thr:
            kps[j] = (x, y)
    return kps

def dict_to_kps_single(p: Dict[str,Any], w: int, h: int) -> Keypoints:
    """Convert single pose dictionary to keypoints"""
    raw = _reshape(p["pose_keypoints_2d"] if "pose_keypoints_2d" in p else p["keypoints"])
    if raw[:, :2].max() <= 1.01:                   # 0-1 normalised coords
        raw[:, 0] *= w
        raw[:, 1] *= h
    return _coords_to_xy(raw)

def kps_from_pose_json(js: List[Dict[str,Any]]) -> List[Keypoints]:
    """Extract keypoints from pose JSON data"""
    if not (js and isinstance(js, list)):
        return []
    frame = js[0]
    w, h = frame.get("canvas_width",1), frame.get("canvas_height",1)
    if "people" in frame:      # OpenPose
        return [dict_to_kps_single(p,w,h) for p in frame["people"]]
    if "animals" in frame:     # AP-10K
        return [dict_to_kps_single({"pose_keypoints_2d":a[:NUM_BODY_JOINTS*3]}, w, h)
                for a in frame["animals"]]
    return []

def extract_kps_from_mask(img: np.ndarray, mask: Optional[np.ndarray]=None, tol: int = 10) -> Keypoints:
    """Extract keypoints from colored mask image"""
    if mask is None:
        mask = np.ones(img.shape[:2], bool)
    kps = np.full((NUM_BODY_JOINTS,2), np.nan, np.float32)
    for j,(r,g,b) in OPENPOSE_COLOUR_MAP.items():
        m = ((abs(img[...,2]-r)<tol)&(abs(img[...,1]-g)<tol)&
             (abs(img[...,0]-b)<tol)&mask)
        if m.any():
            ys,xs = np.nonzero(m)
            kps[j]=(xs.mean(),ys.mean())
    return kps

# ───────────────── Transformation and Alignment Functions ─────────────────────────────
def estimate_translation(kps_json: Keypoints, kps_img: Keypoints,
                         focus: np.ndarray = TORSO, min_pairs: int = 2) -> np.ndarray:
    """Estimate translation between two sets of keypoints"""
    vis = (~np.isnan(kps_json[:,0]) & ~np.isnan(kps_img[:,0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), focus))
    if vis.sum() < min_pairs:
        return np.zeros(2, np.float32)
    return np.nanmedian(kps_img[vis] - kps_json[vis], 0).astype(np.float32)

def correct_json_offset(kps_json: Keypoints, kps_img: Keypoints) -> Keypoints:
    """Correct JSON keypoints offset using image keypoints"""
    return kps_json + estimate_translation(kps_json, kps_img)

def robust_scale(src: Keypoints, dst: Keypoints) -> Optional[float]:
    """Compute robust scale estimate using stable segments"""
    ratios = []
    for i,j in STABLE_SEGMENTS:
        if not (np.isnan(src[i]).any() or np.isnan(src[j]).any() or
                np.isnan(dst[i]).any() or np.isnan(dst[j]).any()):
            d_src = np.linalg.norm(src[i]-src[j])
            d_dst = np.linalg.norm(dst[i]-dst[j])
            if d_src > 1e-3:
                ratios.append(d_dst / d_src)
    return np.median(ratios) if ratios else None

def robust_translation(src: Keypoints, dst: Keypoints,
                       s: float, R: np.ndarray,
                       joints: np.ndarray = ROBUST_JOINTS) -> np.ndarray:
    """Compute robust translation estimate"""
    deltas = []
    for j in joints:
        if j < NUM_BODY_JOINTS and not (np.isnan(src[j]).any() or np.isnan(dst[j]).any()):
            deltas.append(dst[j] - s*(R @ src[j]))
    return (np.median(deltas,0).astype(np.float32)
            if deltas else np.zeros(2,np.float32))

def refine_translation(src: Keypoints, dst: Keypoints,
                       s: float, R: np.ndarray, t0: np.ndarray,
                       max_iter: int = 3, tol: float = 0.05) -> np.ndarray:
    """Iteratively refine translation estimate"""
    vis = (~np.isnan(src[:,0]) & ~np.isnan(dst[:,0]) &
           np.isin(np.arange(NUM_BODY_JOINTS), TORSO))
    if vis.sum() < 2:
        return t0
    t = t0.copy()
    for _ in range(max_iter):
        warped = (s*(R @ src[vis].T)).T + t
        delta  = np.nanmedian(dst[vis]-warped,0).astype(np.float32)
        if np.linalg.norm(delta) < tol:
            break
        t += delta
    return t

def fit_pair(src: Keypoints, dst: Keypoints) -> Tuple[float,np.ndarray,np.ndarray,float]:
    """
    Fit similarity transformation from source to destination keypoints
    Returns: (scale, rotation_matrix, translation, mse_error)
    """
    vis = ~np.isnan(src[:,0]) & ~np.isnan(dst[:,0])
    if vis.sum() < 2:
        return 1., np.eye(2,dtype=np.float32), np.zeros(2,np.float32), float("inf")

    # 1) Scale estimation
    s = robust_scale(src, dst) or 1.0

    # 2) Rotation from Procrustes at that scale
    muX, muY = src[vis].mean(0), dst[vis].mean(0)
    Xc, Yc   = src[vis]-muX,  dst[vis]-muY
    U,_,Vt   = np.linalg.svd(Xc.T @ Yc)
    R        = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T

    # 3) Translation (robust, then iteratively polished)
    t  = robust_translation(src, dst, s, R)
    t  = refine_translation(src, dst, s, R, t)

    # 4) Residual
    recon = (s*(R @ src[vis].T)).T + t
    mse = float(np.mean(np.linalg.norm(recon-dst[vis],axis=1)**2))
    return s, R.astype(np.float32), t.astype(np.float32), mse

# ───────────────── Image Processing Utilities ─────────────────────────────
def two_largest(mask: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Find the two largest connected components in a mask"""
    n,lab = cv2.connectedComponents(mask.astype(np.uint8), 8)
    if n<=2:
        return [mask], [np.zeros_like(mask,bool)]
    areas=[(lab==i).sum() for i in range(1,n)]
    a,b  = np.argsort(areas)[-2:]
    return [(lab==a+1),(lab==b+1)]

# ───────────────── Affine Transformation Utilities ─────────────────────────────
def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to 0-360 degrees range"""
    return ((angle_deg % 360) + 360) % 360

def _build_affine(scale: float, angle_deg: float,
                  tx: float, ty: float,
                  cx: float, cy: float) -> np.ndarray:
    """Build affine transformation matrix from parameters"""
    # Normalize angle to prevent overflow issues
    angle_deg = normalize_angle(angle_deg)
    
    th = math.radians(angle_deg)
    R  = np.array([[math.cos(th), -math.sin(th)],
                   [math.sin(th),  math.cos(th)]], np.float32) * scale
    t  = np.array([tx, ty], np.float32) + np.array([cx, cy]) - R @ np.array([cx, cy])
    return np.hstack([R, t[:, None]])

def decompose_affine_matrix(matrix: np.ndarray, cx: float, cy: float) -> Tuple[float, float, float, float]:
    """
    Decompose an affine transformation matrix back into scale, rotation, and translation components.
    
    Args:
        matrix: 2x3 affine transformation matrix [R|t] where R is 2x2 rotation+scale, t is 2x1 translation
        cx, cy: Center of rotation used in the original transformation
        
    Returns:
        (scale, angle_deg, tx, ty) - Individual transformation components
    """
    # Extract rotation+scale matrix and translation vector
    R = matrix[:2, :2]  # 2x2 rotation+scale matrix
    t = matrix[:2, 2]   # 2x1 translation vector
    
    # Decompose scale and rotation from the 2x2 matrix
    # For a rotation+scale matrix: R = scale * [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    scale = np.sqrt(np.linalg.det(R))  # Determinant gives scale^2 for rotation matrix
    
    # Handle potential negative scales (reflections)
    if np.linalg.det(R) < 0:
        scale = -scale
    
    # Extract rotation matrix by normalizing out the scale
    R_normalized = R / scale
    
    # Extract angle from normalized rotation matrix
    # cos(θ) = R_normalized[0,0], sin(θ) = R_normalized[1,0]
    angle_rad = math.atan2(R_normalized[1, 0], R_normalized[0, 0])
    angle_deg = math.degrees(angle_rad)
    
    # Recover the original translation (tx, ty) by reversing the center compensation
    # In _build_affine: final_t = t + center - R @ center
    # So: original_t = final_t - center + R @ center
    center = np.array([cx, cy])
    original_t = t - center + R @ center
    
    tx, ty = original_t[0], original_t[1]
    
    # Normalize angle to 0-360 range
    angle_deg = normalize_angle(angle_deg)
    
    return float(scale), float(angle_deg), float(tx), float(ty)

# ───────────────── Data Storage for API Access ─────────────────────────────
# Global storage for transformation data (for API access)
_transform_data_cache = {}

def store_transform_data(node_id: str, data: Dict[str, Any]):
    """Store transformation data for API access - simplified to handle new structure"""
    global _transform_data_cache
    
    # Handle both old and new calling conventions
    if isinstance(data, dict) and 'matrices' in data:
        # New format: data is already structured
        matrices = data['matrices']
        offset_corrections = data.get('offsetCorrections', {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}})
        input_dimensions = data.get('inputDimensions', {})
    else:
        # Old format: data contains matrices directly, second param might be offset_corrections
        matrices = data
        offset_corrections = {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}
        input_dimensions = {}
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_matrices = {}
    for key, matrix in matrices.items():
        if matrix is not None:
            serializable_matrices[key] = matrix.tolist()
        else:
            serializable_matrices[key] = None
    
    _transform_data_cache[node_id] = {
        'timestamp': time.time(),
        'matrices': serializable_matrices,
        'offsetCorrections': offset_corrections,
        'inputDimensions': input_dimensions  # NEW: Store input image dimensions for bounding boxes
    }


def get_transform_data(node_id: str) -> Optional[Dict]:
    """Retrieve transformation data for API access"""
    global _transform_data_cache
    return _transform_data_cache.get(node_id)
