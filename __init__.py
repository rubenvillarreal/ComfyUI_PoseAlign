"""ComfyUI PoseAlign Custom Node"""
import sys
import os

print("[ComfyUI_PoseAlign] Starting to load custom node...")

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"[ComfyUI_PoseAlign] Added to path: {current_dir}")

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Simple test node to verify loading
class SimpleTestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello from PoseAlign!"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "AInseven/Test"
    
    def process(self, text):
        return (f"Processed: {text}",)

# Register test node
NODE_CLASS_MAPPINGS["SimpleTestNode"] = SimpleTestNode
NODE_DISPLAY_NAME_MAPPINGS["SimpleTestNode"] = "Simple Test Node"

# Try to load the actual PoseAlign node
try:
    print("[ComfyUI_PoseAlign] Attempting to load pose_align_node module...")
    import pose_align_node
    print("[ComfyUI_PoseAlign] Successfully imported pose_align_node module")
    
    # Get the class
    PoseAlignTwoToOne = pose_align_node.PoseAlignTwoToOne
    print("[ComfyUI_PoseAlign] Got PoseAlignTwoToOne class")
    
    # Register it
    NODE_CLASS_MAPPINGS["PoseAlignTwoToOne"] = PoseAlignTwoToOne
    NODE_DISPLAY_NAME_MAPPINGS["PoseAlignTwoToOne"] = "Pose Align (2→1)"
    print("[ComfyUI_PoseAlign] Registered PoseAlignTwoToOne node")
    
except Exception as e:
    print(f"[ComfyUI_PoseAlign] ERROR loading pose_align_node: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Web directory
WEB_DIRECTORY = "./web"

print(f"[ComfyUI_PoseAlign] Final registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
print("[ComfyUI_PoseAlign] Loading complete!")

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]