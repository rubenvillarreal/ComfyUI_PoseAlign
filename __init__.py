"""
__init__.py for the PoseAlignTwoToOne custom node package
"""
from aiohttp import web
import json
import time
import sys
import os

# Add current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"[PoseAlign] Loading from directory: {current_dir}")
print(f"[PoseAlign] Files in directory: {os.listdir(current_dir)}")

# Import the node classes from your files
try:
    # Try importing from pose_align_node.py (correct filename)
    from .pose_align_node import PoseAlignTwoToOne
    from .pose_align_utils import get_transform_data
    
    print("[PoseAlign] Successfully imported PoseAlignTwoToOne and get_transform_data")
    
    # Try to import PoseViewer if it exists
    try:
        from .pose_align_node import PoseViewer
        HAS_POSE_VIEWER = True
        print("[PoseAlign] Successfully imported PoseViewer")
    except ImportError:
        HAS_POSE_VIEWER = False
        print("[PoseAlign] PoseViewer not found, continuing without it")
        
except ImportError as e:
    print(f"[PoseAlign] Import error: {e}")
    print("[PoseAlign] Trying absolute imports...")
    
    try:
        # Try absolute imports as fallback
        import pose_align_node
        import pose_align_utils
        
        PoseAlignTwoToOne = pose_align_node.PoseAlignTwoToOne
        get_transform_data = pose_align_utils.get_transform_data
        
        try:
            PoseViewer = pose_align_node.PoseViewer
            HAS_POSE_VIEWER = True
        except:
            HAS_POSE_VIEWER = False
            
        print("[PoseAlign] Absolute imports successful")
        
    except ImportError as e2:
        print(f"[PoseAlign] Absolute import also failed: {e2}")
        print("[PoseAlign] Creating dummy classes to prevent ComfyUI from crashing")
        
        # Create dummy classes to prevent ComfyUI from crashing
        class PoseAlignTwoToOne:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "dummy": ("STRING", {"default": "Import failed - check console"}),
                    }
                }
            RETURN_TYPES = ("IMAGE",)
            FUNCTION = "dummy"
            CATEGORY = "AInseven/Error"
            
            def dummy(self, dummy=""): 
                import torch
                # Return a black image to indicate error
                return (torch.zeros(1, 64, 64, 3),)
        
        def get_transform_data(node_id):
            return None
        
        HAS_POSE_VIEWER = False

# Test the imported class
try:
    # Verify the class works
    test_instance = PoseAlignTwoToOne()
    input_types = PoseAlignTwoToOne.INPUT_TYPES()
    print(f"[PoseAlign] Node validation successful. Input types: {list(input_types.get('required', {}).keys())}")
except Exception as e:
    print(f"[PoseAlign] Node validation failed: {e}")

# Tell ComfyUI about the nodes in this package
NODE_CLASS_MAPPINGS = {
    "PoseAlignTwoToOne": PoseAlignTwoToOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseAlignTwoToOne": "Pose Align (2→1) Fixed",
}

# Add PoseViewer if available
if HAS_POSE_VIEWER:
    try:
        NODE_CLASS_MAPPINGS["PoseViewer"] = PoseViewer
        NODE_DISPLAY_NAME_MAPPINGS["PoseViewer"] = "Pose Viewer (Debug)"
        print("[PoseAlign] PoseViewer added to mappings")
    except Exception as e:
        print(f"[PoseAlign] Failed to add PoseViewer to mappings: {e}")

print(f"[PoseAlign] Final node mappings: {list(NODE_CLASS_MAPPINGS.keys())}")

# Tell ComfyUI that this node has a web directory to serve
WEB_DIRECTORY = "./web"

# Check if web directory exists
web_dir = os.path.join(current_dir, "web")
if os.path.exists(web_dir):
    print(f"[PoseAlign] Web directory found: {web_dir}")
    web_files = []
    for root, dirs, files in os.walk(web_dir):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), web_dir)
            web_files.append(rel_path)
    print(f"[PoseAlign] Web files: {web_files}")
else:
    print(f"[PoseAlign] Web directory not found: {web_dir}")

# ──────────────────────────── API ROUTES FOR CANVAS SYNC ─────────────────────────
# This is the CORRECT way to register routes in ComfyUI

try:
    from server import PromptServer
    
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
                    'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}
                }, status=200)  # Return 200 instead of 404 to avoid console errors
            
            return web.json_response(data)
        except Exception as e:
            print(f"[PoseAlign API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'timestamp': time.time(),
                'matrices': {'A': None, 'B': None},
                'offsetCorrections': {'A': {'x': 0, 'y': 0}, 'B': {'x': 0, 'y': 0}}
            }, status=200)  # Return 200 instead of 500 to avoid console errors
    
    print("[PoseAlign] API routes registered successfully")

    @PromptServer.instance.routes.get("/AInseven/debug_node_ids")
    async def api_debug_node_ids(request):
        """Debug endpoint to see all stored node IDs and their data"""
        try:
            # Import the cache from the utils module
            from .pose_align_utils import _transform_data_cache

            debug_info = {
                'stored_node_ids': list(_transform_data_cache.keys()),
                'total_stored': len(_transform_data_cache),
                'cache_contents': {}
            }

            # Include full data for debugging
            for node_id, data in _transform_data_cache.items():
                debug_info['cache_contents'][node_id] = {
                    'timestamp': data.get('timestamp'),
                    'matrices_available': {
                        'A': data.get('matrices', {}).get('A') is not None,
                        'B': data.get('matrices', {}).get('B') is not None
                    },
                    'offset_corrections': data.get('offsetCorrections'),
                    'input_dimensions': data.get('inputDimensions'),
                    'input_dimensions_type': type(data.get('inputDimensions')).__name__
                }

            return web.json_response(debug_info)

        except Exception as e:
            print(f"[PoseAlign Debug API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'stored_node_ids': [],
                'total_stored': 0
            })

    # Also add a route to get data by any ID pattern:
    @PromptServer.instance.routes.get("/AInseven/debug_find_data/{search_id}")
    async def api_debug_find_data(request):
        """Find data by searching for ID patterns"""
        try:
            search_id = request.match_info.get('search_id')
            from .pose_align_utils import _transform_data_cache
            results = {}

            # Search for exact match and partial matches
            for stored_id, data in _transform_data_cache.items():
                if (search_id == stored_id or
                    search_id in stored_id or
                    stored_id in search_id):
                    results[stored_id] = data

            return web.json_response({
                'search_id': search_id,
                'found_matches': len(results),
                'results': results
            })

        except Exception as e:
            print(f"[PoseAlign Debug API] Error: {e}")
            return web.json_response({
                'error': str(e),
                'search_id': search_id,
                'found_matches': 0,
                'results': {}
            })

    print("[PoseAlign] Debug API routes registered")

except ImportError as e:
    print(f"[PoseAlign] Could not import PromptServer: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")
except Exception as e:
    print(f"[PoseAlign] Error registering API routes: {e}")
    print("[PoseAlign] Canvas will fall back to widget-only mode")

# Export everything ComfyUI needs
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("[PoseAlign] __init__.py loading complete!")
