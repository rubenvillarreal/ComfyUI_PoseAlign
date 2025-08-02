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
                "assignment": (["auto", "A_to_first", "A_to_second"],),
                "manual": ("BOOLEAN", {"default": False}),
                "reset": ("BOOLEAN", {"default": False}),
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
                "debug": ("BOOLEAN", {"default": False}),
                "alignment_input": ("POSE_ALIGNMENT",)  # New optional input for alignment passthrough
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "POSE_ALIGNMENT")
    RETURN_NAMES = ("aligned_poseA", "aligned_poseB", "combined_AB", "combine_all", "alignment_output")
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
              assignment="auto", manual=False, reset=False, debug=False,
              angle_deg_A=0.0, scale_A=1.0, tx_A=0, ty_A=0,
              angle_deg_B=0.0, scale_B=1.0, tx_B=0, ty_B=0,
              alignment_input=None,
              prompt=None, extra_pnginfo=None):

        # Store Python object ID in node properties for JavaScript access
        python_node_id = str(id(self))
        self.properties = getattr(self, 'properties', {})
        self.properties['python_node_id'] = python_node_id
        
        print(f"[PoseAlign] PYTHON NODE ID: {python_node_id}")
        print(f"[PoseAlign] Stored in properties: {self.properties.get('python_node_id')}")
        
        # If alignment_input is provided and we're not in manual mode or resetting,
        # extract the transformation values to use as overrides
        if alignment_input is not None and not manual and not reset:
            # Get reference dimensions for decomposition
            ref_np = torch_to_u8(ref_pose_img[0:1])
            h, w = ref_np.shape[:2]
            cx = alignment_input.get('center_x', w / 2.0)
            cy = alignment_input.get('center_y', h / 2.0)
            
            # Extract and decompose matrix A if available
            ma = alignment_input.get('matrix_A')
            if ma is not None:
                matrix_a = np.array(ma, dtype=np.float32)
                scale_A, angle_deg_A, tx_A, ty_A = decompose_affine_matrix(matrix_a, cx, cy)
                # Convert to appropriate types for the function parameters
                angle_deg_A = float(angle_deg_A)
                scale_A = float(scale_A)
                tx_A = int(round(tx_A))
                ty_A = int(round(ty_A))
                
                if debug:
                    print(f"[PoseAlign] Overriding A params from alignment_input: scale={scale_A}, angle={angle_deg_A}, tx={tx_A}, ty={ty_A}")
            
            # Extract and decompose matrix B if available
            mb = alignment_input.get('matrix_B')
            if mb is not None and poseB_img is not None:
                matrix_b = np.array(mb, dtype=np.float32)
                scale_B, angle_deg_B, tx_B, ty_B = decompose_affine_matrix(matrix_b, cx, cy)
                # Convert to appropriate types for the function parameters
                angle_deg_B = float(angle_deg_B)
                scale_B = float(scale_B)
                tx_B = int(round(tx_B))
                ty_B = int(round(ty_B))
                
                if debug:
                    print(f"[PoseAlign] Overriding B params from alignment_input: scale={scale_B}, angle={angle_deg_B}, tx={tx_B}, ty={ty_B}")
            
            # Also load the offset corrections if provided
            if alignment_input.get('offset_corrections'):
                self._offset_corrections = alignment_input.get('offset_corrections')
            
            # Update properties so JavaScript can read the values
            self.properties.update({
                'scale_A': scale_A, 'angle_deg_A': angle_deg_A, 'tx_A': tx_A, 'ty_A': ty_A,
                'scale_B': scale_B, 'angle_deg_B': angle_deg_B, 'tx_B': tx_B, 'ty_B': ty_B
            })
        

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

        # Manual mode - always build from slider values (overrides any alignment_input)
        if manual:
            MA = _build_affine(scale_A, angle_deg_A, tx_A, ty_A, cx, cy).astype(np.float32)
            if poseB_img is not None:
                MB = _build_affine(scale_B, angle_deg_B, tx_B, ty_B, cx, cy).astype(np.float32)
            else:
                MB = np.eye(2, 3, dtype=np.float32)  # Identity matrix for no transformation
            
            self.properties.update({
                'scale_A': scale_A, 'angle_deg_A': angle_deg_A, 'tx_A': tx_A, 'ty_A': ty_A,
                'scale_B': scale_B, 'angle_deg_B': angle_deg_B, 'tx_B': tx_B, 'ty_B': ty_B
            })
            
            self._MA, self._MB = MA, MB
            
            if debug:
                print(f"[PoseAlign] Manual mode: Applied user adjustments")
            
        else:
            # Automatic mode logic
            # Only calculate fit if:
            # 1. Reset is requested
            # 2. We have no cached matrices AND no alignment_input
            # 3. We need matrix B but don't have it
            need_fit = reset or (self._MA is None and alignment_input is None) or (self._MB is None and poseB_img is not None and alignment_input is None)
            
            # If we have alignment_input but no cached matrices, use alignment_input
            if not need_fit and self._MA is None and alignment_input is not None:
                # Convert lists back to numpy arrays if needed
                ma = alignment_input.get('matrix_A')
                mb = alignment_input.get('matrix_B')
                self._MA = np.array(ma, dtype=np.float32) if ma is not None else None
                self._MB = np.array(mb, dtype=np.float32) if mb is not None else None
                self._offset_corrections = alignment_input.get('offset_corrections', {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}})
                
                # The widget values have already been overridden from alignment_input above
                if debug:
                    print(f"[PoseAlign] Auto mode: Using matrices from alignment_input")
            
            if need_fit:
                # Reference keypoints
                ref_people_raw = kps_from_pose_json(ref_pose_json)
                
                # Single pose mode - align A to one of the reference poses
                if poseB_img is None:
                    # Extract reference poses (can be 1 or 2) but don't make them manipulatable
                    if len(ref_people_raw) >= 2:
                        # Two reference poses available - extract both as targets
                        m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                        img_people = [extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)]
                        kR1 = correct_json_offset(ref_people_raw[0], img_people[0])
                        kR2 = correct_json_offset(ref_people_raw[1], img_people[1])
                    elif len(ref_people_raw) >= 1:
                        # One reference pose available
                        masks = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                        m1 = masks[0] if isinstance(masks, tuple) else masks
                        img_person = extract_kps_from_mask(ref_np, m1)
                        kR1 = correct_json_offset(ref_people_raw[0], img_person)
                        kR2 = None  # Only one reference
                    else:
                        # Extract from mask if no JSON
                        masks = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                        if isinstance(masks, tuple) and len(masks) >= 2:
                            m1, m2 = masks[0], masks[1]
                            kR1 = extract_kps_from_mask(ref_np, m1)
                            kR2 = extract_kps_from_mask(ref_np, m2)
                        else:
                            m1 = masks[0] if isinstance(masks, tuple) else masks
                            kR1 = extract_kps_from_mask(ref_np, m1)
                            kR2 = None
                    
                    # Get pose A keypoints
                    kA_json = self._get_kps(A_np, poseA_json, 0)
                    kA_img = extract_kps_from_mask(A_np)
                    
                    # Calculate offset correction for A only
                    offset_A = estimate_translation(kA_json, kA_img)
                    self._offset_corrections = {
                        'A': {'x': float(offset_A[0]), 'y': float(offset_A[1])},
                        'B': {'x': 0.0, 'y': 0.0}  # No B pose to correct
                    }
                    
                    kA = correct_json_offset(kA_json, kA_img)
                    
                    # Choose which reference pose to align to based on assignment or best fit
                    if kR2 is not None:
                        # Two reference poses - choose based on assignment or best error
                        sA1, RA1, tA1, eA1 = fit_pair(kA, kR1)
                        sA2, RA2, tA2, eA2 = fit_pair(kA, kR2)
                        
                        if assignment == "A_to_first":
                            sA, RA, tA = sA1, RA1, tA1
                        elif assignment == "A_to_second":
                            sA, RA, tA = sA2, RA2, tA2
                        else:  # auto - choose best fit
                            if eA1 <= eA2:
                                sA, RA, tA = sA1, RA1, tA1
                            else:
                                sA, RA, tA = sA2, RA2, tA2
                    else:
                        # Only one reference pose
                        sA, RA, tA, _ = fit_pair(kA, kR1)
                    
                    # Create transformation for A only
                    MA = np.hstack([sA * RA, tA[:, None]]).astype(np.float32)
                    # B has no transformation - it doesn't exist
                    MB = np.eye(2, 3, dtype=np.float32)
                    
                else:
                    # Two pose mode - original logic
                    if len(ref_people_raw) >= 2:
                        m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                        img_people = [extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)]
                        kR1 = correct_json_offset(ref_people_raw[0], img_people[0])
                        kR2 = correct_json_offset(ref_people_raw[1], img_people[1])
                    else:
                        m1, m2 = two_largest(cv2.cvtColor(ref_np, cv2.COLOR_BGR2GRAY) > 2)
                        kR1, kR2 = extract_kps_from_mask(ref_np, m1), extract_kps_from_mask(ref_np, m2)

                    # Pose A & B keypoints
                    kA_json = self._get_kps(A_np, poseA_json, 0)
                    kB_json = self._get_kps(B_np, poseB_json, 0)
                    kA_img = extract_kps_from_mask(A_np)
                    kB_img = extract_kps_from_mask(B_np)
                    
                    # Calculate offset corrections
                    offset_A = estimate_translation(kA_json, kA_img)
                    offset_B = estimate_translation(kB_json, kB_img)
                    self._offset_corrections = {
                        'A': {'x': float(offset_A[0]), 'y': float(offset_A[1])},
                        'B': {'x': float(offset_B[0]), 'y': float(offset_B[1])}
                    }
                    
                    kA = correct_json_offset(kA_json, kA_img)
                    kB = correct_json_offset(kB_json, kB_img)

                    # Similarity fits
                    sA1, RA1, tA1, eA1 = fit_pair(kA, kR1)
                    sB2, RB2, tB2, eB2 = fit_pair(kB, kR2)
                    sA2, RA2, tA2, eA2 = fit_pair(kA, kR2)
                    sB1, RB1, tB1, eB1 = fit_pair(kB, kR1)

                    pick = 0 if assignment == "A_to_first" else \
                           1 if assignment == "A_to_second" else \
                           (0 if eA1 + eB2 <= eA2 + eB1 else 1)

                    if pick == 0:
                        sA, RA, tA = sA1, RA1, tA1
                        sB, RB, tB = sB2, RB2, tB2
                    else:
                        sA, RA, tA = sA2, RA2, tA2
                        sB, RB, tB = sB1, RB1, tB1

                    MA = np.hstack([sA * RA, tA[:, None]]).astype(np.float32)
                    MB = np.hstack([sB * RB, tB[:, None]]).astype(np.float32)
                    
                self._MA, self._MB = MA, MB
                
            else:
                MA, MB = self._MA, self._MB

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

        # Create alignment output data
        # Convert numpy arrays to lists for serialization
        alignment_output = {
            'matrix_A': self._MA.tolist() if self._MA is not None else None,
            'matrix_B': self._MB.tolist() if self._MB is not None else None,
            'offset_corrections': self._offset_corrections,
            'center_x': cx,
            'center_y': cy,
            'output_size': self._out_size,
            'has_poseB': poseB_img is not None
        }
        
        result = {"result": (
            torch.cat(outA, 0), 
            torch.cat(outB, 0), 
            torch.cat(outC, 0), 
            torch.cat(outAll, 0),
            alignment_output  # Add alignment as 5th output
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

