"""
pose_align_nodes.py – ComfyUI custom nodes for pose alignment
"""

from __future__ import annotations
import cv2
import numpy as np
import torch
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from nodes import PreviewImage
from aiohttp import web

# Import all utilities from our utils module
try:
    from pose_align_utils import (
        # Constants
        NUM_BODY_JOINTS, TORSO, ROBUST_JOINTS, STABLE_SEGMENTS, BODY_25_PAIRS, OPENPOSE_COLOUR_MAP,
        
        # Type aliases
        Keypoints,
        
        # Tensor conversion functions
        torch_to_u8, u8_to_torch, torch_to_pil,
        
        # Keypoint processing functions
        kps_from_pose_json, extract_kps_from_mask, estimate_translation, correct_json_offset,
        
        # Transformation functions
        robust_scale, robust_translation, refine_translation, fit_pair,
        
        # Image processing utilities
        two_largest,
        
        # Affine transformation utilities
        normalize_angle, _build_affine, decompose_affine_matrix,
        
        # Data storage functions
        store_transform_data, get_transform_data
    )
except ImportError as e:
    print(f"[PoseAlign] Import error: {e}")
    print("[PoseAlign] Falling back to direct imports...")
    # If relative import fails, try absolute import
    from pose_align_utils import (
        NUM_BODY_JOINTS, TORSO, ROBUST_JOINTS, STABLE_SEGMENTS, BODY_25_PAIRS, OPENPOSE_COLOUR_MAP,
        Keypoints, torch_to_u8, u8_to_torch, torch_to_pil, kps_from_pose_json, extract_kps_from_mask,
        estimate_translation, correct_json_offset, robust_scale, robust_translation, refine_translation,
        fit_pair, two_largest, normalize_angle, _build_affine, decompose_affine_matrix,
        store_transform_data, get_transform_data
    )

# ──────────────────────────── Enhanced Main Alignment Node ─────────────────────────
class PoseAlignTwoToOne(PreviewImage):
    CATEGORY = "AInseven"

    def __init__(self):
        # Initialize parent class
        super().__init__()

        # Add the required attribute for PreviewImage
        self.prefix_append = ""
        self.compress_level = 4

        # Your custom attributes
        self._MA: Optional[np.ndarray] = None
        self._MB: Optional[np.ndarray] = None
        self._out_size: Optional[Tuple[int, int]] = None
        self._offset_corrections: Dict[str, Dict[str, float]] = {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}

        # SIMPLE: Just store original input dimensions for bounding box calculation
        self._input_dimensions: Dict[str, Tuple[int, int]] = {}

    # ─────────────────────────── Node Inputs ──────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_pose_img": ("IMAGE",),
                "ref_pose_json": ("POSE_KEYPOINT",),
                "poseA_img": ("IMAGE",),
                "poseA_json": ("POSE_KEYPOINT",),
                # Manual-mode sliders for pose A
                "angle_deg_A": ("FLOAT", {"default": 0, "min": -720, "max": 720, "step": 0.1}),
                "scale_A": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 3.0, "step": 0.01}),
                "tx_A": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                "ty_A": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                # Manual-mode sliders for pose B
                "angle_deg_B": ("FLOAT", {"default": 0, "min": -720, "max": 720, "step": 0.1}),
                "scale_B": ("FLOAT", {"default": 1.0, "min": 0.20, "max": 3.0, "step": 0.01}),
                "tx_B": ("INT", {"default": 0, "min": -2048, "max": 2048}),
                "ty_B": ("INT", {"default": 0, "min": -2048, "max": 2048}),
            },
            "optional": {
                "poseB_img": ("IMAGE",),
                "poseB_json": ("POSE_KEYPOINT",),
                "debug": ("BOOLEAN", {"default": False})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned_poseA", "aligned_poseB", "combined_AB", "combine_all")
    FUNCTION = "align"

    # ────────────────── Helper Methods ────────────────────
    def _get_kps(self, img: np.ndarray, js: List[Dict[str, Any]], idx: int) -> Keypoints:
        """Helper to fetch key-points from JSON or extract from image mask"""
        people = kps_from_pose_json(js)
        return people[idx] if people and idx < len(people) else extract_kps_from_mask(img)

    def _save_preview_images(self, ref_pose_img, poseA_img, poseB_img=None, prompt=None, extra_pnginfo=None):
        """Save input images using ComfyUI's PreviewImage mechanism"""
        # Get tensors from the first batch
        ref_tensor = ref_pose_img[0:1]
        A_tensor = poseA_img[0:1]
        
        if poseB_img is not None:
            B_tensor = poseB_img[0:1]
            # Find the maximum height and width
            max_height = max(ref_tensor.shape[1], A_tensor.shape[1], B_tensor.shape[1])
            max_width = max(ref_tensor.shape[2], A_tensor.shape[2], B_tensor.shape[2])
        else:
            # Only ref and A
            max_height = max(ref_tensor.shape[1], A_tensor.shape[1])
            max_width = max(ref_tensor.shape[2], A_tensor.shape[2])

        # Resize all images to the same dimensions
        import torch.nn.functional as F

        def resize_tensor(tensor, target_h, target_w):
            # tensor shape: (B, H, W, C) in ComfyUI
            # Need to permute to (B, C, H, W) for F.interpolate
            tensor = tensor.permute(0, 3, 1, 2)
            resized = F.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # Permute back to (B, H, W, C)
            return resized.permute(0, 2, 3, 1)

        # Resize all images to the maximum dimensions
        ref_resized = resize_tensor(ref_tensor, max_height, max_width)
        A_resized = resize_tensor(A_tensor, max_height, max_width)
        
        if poseB_img is not None:
            B_resized = resize_tensor(B_tensor, max_height, max_width)
            # Stack images horizontally (along width dimension)
            combined = torch.cat([ref_resized, A_resized, B_resized], dim=2)  # dim=2 is width in (B,H,W,C)
        else:
            # Only ref and A
            combined = torch.cat([ref_resized, A_resized], dim=2)

        # Use PreviewImage's save_images method
        return self.save_images(combined, filename_prefix="pose_align_preview", prompt=prompt, extra_pnginfo=extra_pnginfo)

    def _update_widget_values(self, cx: float, cy: float, debug: bool = False):
        """
        Decompose cached transformation matrices and update widget values.
        This ensures the canvas widget can read the current transformation parameters.
        """
        if self._MA is not None:
            scale_A, angle_deg_A, tx_A, ty_A = decompose_affine_matrix(self._MA, cx, cy)
            
            # Update node properties (which the canvas reads)
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_A': scale_A,
                'angle_deg_A': angle_deg_A,
                'tx_A': int(round(tx_A)),
                'ty_A': int(round(ty_A))
            })
            
            if debug:
                print(f"[PoseAlign] Updated widget A: scale={scale_A:.3f}, angle={angle_deg_A:.1f}°, tx={tx_A:.1f}, ty={ty_A:.1f}")
        
        if self._MB is not None:
            scale_B, angle_deg_B, tx_B, ty_B = decompose_affine_matrix(self._MB, cx, cy)
            
            # Update node properties (which the canvas reads)
            self.properties = getattr(self, 'properties', {})
            self.properties.update({
                'scale_B': scale_B,
                'angle_deg_B': angle_deg_B,
                'tx_B': int(round(tx_B)),
                'ty_B': int(round(ty_B))
            })
            
            if debug:
                print(f"[PoseAlign] Updated widget B: scale={scale_B:.3f}, angle={angle_deg_B:.1f}°, tx={tx_B:.1f}, ty={ty_B:.1f}")

    def _store_transform_data_for_canvas(self, debug: bool = False, has_poseB: bool = True):
        """Store transformation data for canvas access via API"""
        node_id = str(id(self))  # Use object id as unique identifier

        # ENHANCED LOGGING: Check what we're actually storing
        print(f"[PoseAlign] Storing data for node_id: {node_id}")
        print(f"[PoseAlign] Input dimensions being stored: {self._input_dimensions}")
        print(f"[PoseAlign] Input dimensions type: {type(self._input_dimensions)}")
        print(f"[PoseAlign] Offset corrections: {self._offset_corrections}")
        print(f"[PoseAlign] Has poseB: {has_poseB}")

        # Only store B transformation if poseB actually exists
        matrices = {'A': self._MA}
        if has_poseB:
            matrices['B'] = self._MB
        else:
            matrices['B'] = None  # Explicitly mark as None for single pose mode

        data = {
            'matrices': matrices,
            'offsetCorrections': self._offset_corrections,
            'inputDimensions': self._input_dimensions,
            'singlePoseMode': not has_poseB  # Flag for UI to know this is single pose mode
        }

        store_transform_data(node_id, data)

        # VERIFY what was actually stored
        from pose_align_utils import get_transform_data
        stored_data = get_transform_data(node_id)
        if stored_data:
            print(f"[PoseAlign] Verification - stored inputDimensions: {stored_data.get('inputDimensions')}")
            print(f"[PoseAlign] Verification - stored inputDimensions type: {type(stored_data.get('inputDimensions'))}")
        else:
            print(f"[PoseAlign] ERROR: Failed to retrieve stored data for verification!")

        if debug:
            print(f"[PoseAlign] Full stored data: {stored_data}")

    # ───────────────────────── Main Alignment Function ──────────────────────────
    def align(self, ref_pose_img, ref_pose_json, poseA_img, poseA_json,
              poseB_img=None, poseB_json=None,
              debug=False,
              angle_deg_A=0.0, scale_A=1.0, tx_A=0, ty_A=0,
              angle_deg_B=0.0, scale_B=1.0, tx_B=0, ty_B=0,
              prompt=None, extra_pnginfo=None):

        # Store Python object ID in node properties for JavaScript access
        python_node_id = str(id(self))
        self.properties = getattr(self, 'properties', {})
        self.properties['python_node_id'] = python_node_id
        
        print(f"[PoseAlign] PYTHON NODE ID: {python_node_id}")
        print(f"[PoseAlign] Transform params - A: tx={tx_A}, ty={ty_A}, scale={scale_A}, angle={angle_deg_A}")
        
        # SIMPLE: Capture original input dimensions before any processing
        ref_np = torch_to_u8(ref_pose_img[0:1])
        A_np = torch_to_u8(poseA_img[0:1])
        
        if poseB_img is not None:
            B_np = torch_to_u8(poseB_img[0:1])
            self._input_dimensions = {
                'ref': (ref_np.shape[1], ref_np.shape[0]),  # (width, height)
                'A': (A_np.shape[1], A_np.shape[0]),
                'B': (B_np.shape[1], B_np.shape[0])
            }
        else:
            B_np = None
            # In single pose mode, don't include B dimensions at all
            self._input_dimensions = {
                'ref': (ref_np.shape[1], ref_np.shape[0]),  # (width, height)
                'A': (A_np.shape[1], A_np.shape[0])
                # No 'B' key - this should signal to UI that there's no poseB
            }
        
        # ENHANCED LOGGING: Show what dimensions we captured
        print(f"[PoseAlign] ===== CAPTURED INPUT DIMENSIONS =====")
        print(f"[PoseAlign] ref_np.shape: {ref_np.shape} -> stored as: {self._input_dimensions['ref']}")
        print(f"[PoseAlign] A_np.shape: {A_np.shape} -> stored as: {self._input_dimensions['A']}")
        if B_np is not None:
            print(f"[PoseAlign] B_np.shape: {B_np.shape} -> stored as: {self._input_dimensions['B']}")
        else:
            print(f"[PoseAlign] B_np: None (single pose mode)")
        print(f"[PoseAlign] ==========================================")
        
        if debug:
            print(f"[PoseAlign] Input dimensions: {self._input_dimensions}")

        # Save preview images
        ui_result = self._save_preview_images(ref_pose_img, poseA_img, poseB_img, prompt, extra_pnginfo)

        N = poseA_img.shape[0]
        h, w = ref_np.shape[:2]  # Use reference dimensions for output
        cx, cy = w / 2.0, h / 2.0
        self._out_size = (w, h)

        # Normalize angles
        angle_deg_A = normalize_angle(angle_deg_A)
        angle_deg_B = normalize_angle(angle_deg_B)

        # SIMPLIFIED: Always use manual transformations
        print(f"[PoseAlign] Building transformation from widget values")
        print(f"[PoseAlign] Widget values - A: tx={tx_A}, ty={ty_A}, scale={scale_A}, angle={angle_deg_A}")
        if poseB_img is not None:
            print(f"[PoseAlign] Widget values - B: tx={tx_B}, ty={ty_B}, scale={scale_B}, angle={angle_deg_B}")
        
        # Build affine matrix using reference center
        # This ensures the transformation matches what's shown on the canvas
        MA = _build_affine(scale_A, angle_deg_A, tx_A, ty_A, cx, cy).astype(np.float32)
        
        if poseB_img is not None:
            MB = _build_affine(scale_B, angle_deg_B, tx_B, ty_B, cx, cy).astype(np.float32)
        else:
            MB = np.eye(2, 3, dtype=np.float32)  # Identity matrix for no transformation
        
        # Update properties for JavaScript access
        self.properties.update({
            'scale_A': scale_A, 'angle_deg_A': angle_deg_A, 'tx_A': tx_A, 'ty_A': ty_A,
            'scale_B': scale_B, 'angle_deg_B': angle_deg_B, 'tx_B': tx_B, 'ty_B': ty_B
        })
        
        self._MA, self._MB = MA, MB
        
        # No offset corrections needed in manual-only mode
        self._offset_corrections = {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}

        # Store transformation data for canvas
        self._store_transform_data_for_canvas(debug, has_poseB=(poseB_img is not None))

        # Apply transforms to batch
        outA, outB, outC, outAll = [], [], [], []
        for i in range(N):
            A_np_i = torch_to_u8(poseA_img[i:i+1])
            
            # Transform A
            A_w = cv2.warpAffine(A_np_i, MA, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            if poseB_img is not None:
                # Transform B if provided
                B_np_i = torch_to_u8(poseB_img[i:i+1])
                B_w = cv2.warpAffine(B_np_i, MB, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                # Combine A and B
                combined = np.where(A_w > 0, A_w, B_w)
                combo_all = ref_np.copy()
                m = A_w > 0; combo_all[m] = A_w[m]
                m = B_w > 0; combo_all[m] = B_w[m]
            else:
                # Single pose mode - B is empty
                B_w = np.zeros_like(A_w)  # Empty image for B
                combined = A_w.copy()  # Combined is just A
                combo_all = ref_np.copy()
                m = A_w > 0; combo_all[m] = A_w[m]  # Overlay only A on reference

            outA.append(u8_to_torch(A_w))
            outB.append(u8_to_torch(B_w))
            outC.append(u8_to_torch(combined))
            outAll.append(u8_to_torch(combo_all))

        result = {"result": (
            torch.cat(outA, 0), 
            torch.cat(outB, 0), 
            torch.cat(outC, 0), 
            torch.cat(outAll, 0)
        )}
        
        if "ui" in ui_result:
            result["ui"] = ui_result["ui"]
            
        return result


# ───────────────────── Pose Viewer Debug Node ──────────────────
class PoseViewer:
    CATEGORY = "AInseven/Debug"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "image": ("IMAGE",),
                    "pose_json": ("POSE_KEYPOINT",),
                    "point_radius": ("INT", {"default": 8, "min": 1, "max": 50}),
                    "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20}),
                    "draw_limbs": ("BOOLEAN", {"default": True}),
                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "view"

    def view(self, image: torch.Tensor, pose_json: List[Dict[str, Any]],
             point_radius: int, line_thickness: int, draw_limbs: bool):
        """Visualize pose keypoints and limbs on image"""
        img_np = torch_to_u8(image)
        all_kps = kps_from_pose_json(pose_json)
        
        for person_kps in all_kps:
            color = tuple(map(int, np.random.randint(100, 256, 3)))
            
            if draw_limbs:
                for p1, p2 in BODY_25_PAIRS:
                    if p1 < len(person_kps) and p2 < len(person_kps):
                        pt1, pt2 = person_kps[p1], person_kps[p2]
                        if not np.isnan(pt1).any() and not np.isnan(pt2).any():
                            cv2.line(img_np, tuple(np.int32(pt1)), tuple(np.int32(pt2)),
                                     color, line_thickness)
            
            for pt in person_kps:
                if not np.isnan(pt).any():
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, color, -1)
                    cv2.circle(img_np, tuple(np.int32(pt)), point_radius, (0, 0, 0), 2)
        
        return (u8_to_torch(img_np),)


# ──────────────────────────── Node Registration ─────────────────────────
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
    "PoseViewer": PoseViewer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align Two To One (Fixed)",
    "PoseViewer": "Pose Viewer (Debug)"
}

