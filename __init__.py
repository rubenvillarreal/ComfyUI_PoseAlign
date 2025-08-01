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

# ──────────────────────────── API ROUTES FOR CANVAS SYNC ─────────────────────────
try:
    from server import PromptServer
    from aiohttp import web
    import time
    
    # Import the data access functions
    from pose_align_utils import get_transform_data, _transform_data_cache
    
    @PromptServer.instance.routes.get("/AInseven/pose_align_data/{node_id}")
    async def api_get_pose_align_data(request):
        """API endpoint to get transformation data for canvas"""
        try:
            node_id = request.match_info.get('node_id')
            data = get_transform_data(node_id)
            
            if data is None:
                return web.json_response({
                    'error': 'No data found for node',
                    'timestamp': time.time(),
                    'matrices': {'A': None, 'B': None},
                    'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}},
                    'singlePoseMode': False
                }, status=200)
            
            return web.json_response(data)
        except Exception as e:
            print(f"[PoseAlign API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'timestamp': time.time(),
                'matrices': {'A': None, 'B': None},
                'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}},
                'singlePoseMode': False
            }, status=200)
    
    @PromptServer.instance.routes.get("/AInseven/debug_node_ids")
    async def api_debug_node_ids(request):
        """Debug endpoint to see all stored node IDs and their data"""
        try:
            debug_info = {
                'stored_node_ids': list(_transform_data_cache.keys()),
                'total_stored': len(_transform_data_cache),
                'cache_contents': {}
            }

            # Include summary data for debugging
            for node_id, data in _transform_data_cache.items():
                debug_info['cache_contents'][node_id] = {
                    'timestamp': data.get('timestamp'),
                    'singlePoseMode': data.get('singlePoseMode'),
                    'matrices_available': {
                        'A': data.get('matrices', {}).get('A') is not None,
                        'B': data.get('matrices', {}).get('B') is not None
                    },
                    'inputDimensions': data.get('inputDimensions')
                }

            return web.json_response(debug_info)

        except Exception as e:
            print(f"[PoseAlign Debug API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'stored_node_ids': [],
                'total_stored': 0
            })

    print("[PoseAlign] API routes registered successfully")

except ImportError as e:
    print(f"[PoseAlign] Could not import PromptServer: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")
except Exception as e:
    print(f"[PoseAlign] Error registering API routes: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")

print(f"[ComfyUI_PoseAlign] Final registered nodes: {list(NODE_CLASS_MAPPINGS.keys())}")
print("[ComfyUI_PoseAlign] Loading complete!")

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]