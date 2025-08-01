/* modules/image_manager.js - Handles image loading and processing */

export class ImageManager {
	constructor(node, canvasSizeCallback = null) {
		this.node = node;
		this.loadedImages = {
			ref: null,
			A: null,
			B: null
		};
		this.previewImage = null;
		this.refImageSize = { width: 512, height: 512 }; // Default size
		this.canvasSizeCallback = canvasSizeCallback; // Callback to update canvas size
	}

	// Get the reference image size (used for coordinate system)
	getReferenceImageSize() {
		return this.refImageSize;
	}

	// Extract individual poses from combined preview image
	async extractPosesFromPreview(previewImage, singlePoseMode = false) {
		if (!previewImage) return { ref: null, A: null, B: null };
		
		// In single pose mode, preview has 2 poses (ref, A), otherwise 3 (ref, A, B)
		const numPoses = singlePoseMode ? 2 : 3;
		const width = previewImage.width / numPoses;
		const height = previewImage.height;
		
		console.log(`[ImageManager] Extracting ${numPoses} poses from preview (${previewImage.width}x${previewImage.height})`);
		console.log(`[ImageManager] Individual pose size: ${width}x${height}`);
		
		// Update reference image size for coordinate system
		this.refImageSize = { width, height };
		
		// Notify canvas to update its size
		if (this.canvasSizeCallback) {
			this.canvasSizeCallback(width, height);
		}
		
		// Create canvases for each pose
		const poses = {};
		
		if (singlePoseMode) {
			// Single pose mode: preview has [ref, poseA]
			console.log(`[ImageManager] Single pose mode - extracting ref and poseA`);
			
			// Extract reference image (position 0)
			const refCanvas = document.createElement('canvas');
			refCanvas.width = width;
			refCanvas.height = height;
			const refCtx = refCanvas.getContext('2d');
			refCtx.drawImage(previewImage, 0, 0, width, height, 0, 0, width, height);
			poses['ref'] = await createImageBitmap(refCanvas);
			
			// Extract poseA image (position 1) 
			const aCanvas = document.createElement('canvas');
			aCanvas.width = width;
			aCanvas.height = height;
			const aCtx = aCanvas.getContext('2d');
			aCtx.drawImage(previewImage, width, 0, width, height, 0, 0, width, height);
			poses['A'] = await createImageBitmap(aCanvas);
			
			// No pose B in single pose mode
			poses['B'] = null;
			
		} else {
			// Dual pose mode: preview has [ref, poseA, poseB]
			console.log(`[ImageManager] Dual pose mode - extracting ref, poseA, and poseB`);
			
			const names = ['ref', 'A', 'B'];
			for (let i = 0; i < 3; i++) {
				const tempCanvas = document.createElement('canvas');
				tempCanvas.width = width;
				tempCanvas.height = height;
				const tempCtx = tempCanvas.getContext('2d');
				
				// Draw the portion of the preview image
				tempCtx.drawImage(previewImage, 
					i * width, 0, width, height,  // source
					0, 0, width, height            // destination
				);
				
				// Convert to image bitmap for use
				poses[names[i]] = await createImageBitmap(tempCanvas);
			}
		}
		
		return poses;
	}

	// Check if we're in single pose mode by looking for transform data
	async getSinglePoseMode(transformManager) {
		try {
			const transformData = await transformManager.getTransformCache();
			console.log("[ImageManager] Full transform data:", transformData);
			console.log("[ImageManager] singlePoseMode flag:", transformData.singlePoseMode);
			console.log("[ImageManager] inputDimensions:", transformData.inputDimensions);
			console.log("[ImageManager] matrices:", transformData.matrices);
			
			// Check multiple ways to detect single pose mode
			const hasSinglePoseFlag = transformData.singlePoseMode === true;
			const noBDimensions = !transformData.inputDimensions?.B;
			const noBMatrix = !transformData.matrices?.B;
			
			console.log("[ImageManager] Detection methods:");
			console.log("  - singlePoseMode flag:", hasSinglePoseFlag);
			console.log("  - No B dimensions:", noBDimensions);  
			console.log("  - No B matrix:", noBMatrix);
			
			// Use any of these methods to detect single pose mode
			const singleMode = hasSinglePoseFlag || noBDimensions || noBMatrix;
			console.log("[ImageManager] Final single pose mode decision:", singleMode);
			
			return singleMode;
		} catch (error) {
			console.log("[ImageManager] Could not get transform data, assuming dual pose mode:", error);
			return false;
		}
	}

	// Get images from node's execution results
	async getImagesFromNode(transformManager = null) {
		try {
			// Check if we have a preview image from the node's UI output
			if (this.node.imgs && this.node.imgs.length > 0 && this.node.imgs[0].src) {
				console.log("Found preview image in node.imgs:", this.node.imgs[0]);
				
				// Load the preview image
				const img = new Image();
				img.src = this.node.imgs[0].src;
				await new Promise((resolve, reject) => {
					img.onload = resolve;
					img.onerror = reject;
				});
				
				this.previewImage = img;
				
				// Check if we're in single pose mode
				const singlePoseMode = transformManager ? await this.getSinglePoseMode(transformManager) : false;
				console.log(`[ImageManager] Single pose mode: ${singlePoseMode}`);
				console.log(`[ImageManager] Preview image dimensions: ${img.width}x${img.height}`);
				
				// Extract individual poses from the combined preview
				const poses = await this.extractPosesFromPreview(img, singlePoseMode);
				console.log("Extracted poses from preview image, ref size:", this.refImageSize);
				
				// Update loaded images cache
				this.loadedImages = poses;
				return poses;
			}
			
			// Return cached images if no new preview available
			return this.loadedImages;
			
		} catch (error) {
			console.error("Error getting images from node:", error);
			return { ref: null, A: null, B: null };
		}
	}

	// Check if we have valid images loaded
	hasValidImages() {
		return !!(this.loadedImages.ref || this.loadedImages.A || this.loadedImages.B);
	}

	// Get the current loaded images
	getLoadedImages() {
		return this.loadedImages;
	}

	// Get debug information about loaded images
	getDebugInfo() {
		const nodeImgCount = this.node.imgs?.length || 0;
		const validCount = this.loadedImages ? 
			(this.loadedImages.ref ? 1 : 0) + (this.loadedImages.A ? 1 : 0) + (this.loadedImages.B ? 1 : 0) : 0;
		const previewStatus = this.previewImage ? "Preview loaded" : "No preview";
		
		return {
			nodeImgCount,
			validCount,
			previewStatus,
			refImageSize: this.refImageSize,
			hasValidImages: this.hasValidImages()
		};
	}
}
