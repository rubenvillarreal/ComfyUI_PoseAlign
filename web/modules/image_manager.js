/* modules/image_manager.js - Handles image loading and processing */

export class ImageManager {
	constructor(node) {
		this.node = node;
		this.loadedImages = {
			ref: null,
			A: null,
			B: null
		};
		this.previewImage = null;
		this.refImageSize = { width: 512, height: 512 }; // Default size
	}

	// Get the reference image size (used for coordinate system)
	getReferenceImageSize() {
		return this.refImageSize;
	}

	// Extract individual poses from combined preview image
	async extractPosesFromPreview(previewImage) {
		if (!previewImage) return { ref: null, A: null, B: null };
		
		// The preview image has all three poses stacked horizontally
		const width = previewImage.width / 3;
		const height = previewImage.height;
		
		// Update reference image size for coordinate system
		this.refImageSize = { width, height };
		
		// Create canvases for each pose
		const poses = {};
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
		
		return poses;
	}

	// Get images from node's execution results
	async getImagesFromNode() {
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
				
				// Extract individual poses from the combined preview
				const poses = await this.extractPosesFromPreview(img);
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
