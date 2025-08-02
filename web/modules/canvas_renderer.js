/* modules/canvas_renderer.js - Fixed dimension handling for poses A and B */

export class CanvasRenderer {
	constructor(canvas, ctx, imageManager, transformManager) {
		this.canvas = canvas;
		this.ctx = ctx;
		this.imageManager = imageManager;
		this.transformManager = transformManager;
		this.state = null;
	}

	setState(state) {
		this.state = state;
	}

	// Calculate proper coordinate system for drawing
	calculateCoordinateSystem() {
		const actualCanvasWidth = this.canvas.width;
		const actualCanvasHeight = this.canvas.height;
		
		const refSize = this.imageManager.getReferenceImageSize();
		const refW = refSize.width;
		const refH = refSize.height;
		
		const canvasAspect = actualCanvasWidth / actualCanvasHeight;
		const refAspect = refW / refH;
		
		let canvasScale, offsetX, offsetY;
		
		if (refAspect > canvasAspect) {
			canvasScale = actualCanvasWidth / refW * 0.9;
			offsetX = actualCanvasWidth * 0.05;
			offsetY = (actualCanvasHeight - refH * canvasScale) / 2;
		} else {
			canvasScale = actualCanvasHeight / refH * 0.9;
			offsetY = actualCanvasHeight * 0.05;
			offsetX = (actualCanvasWidth - refW * canvasScale) / 2;
		}
		
		return { canvasScale, offsetX, offsetY, refW, refH, actualCanvasWidth, actualCanvasHeight };
	}

	// Draw grid
	drawGrid(coordSys) {
		const { canvasScale, offsetX, offsetY, actualCanvasWidth, actualCanvasHeight } = coordSys;
		
		this.ctx.strokeStyle = "#333";
		this.ctx.lineWidth = 1;
		this.ctx.setLineDash([2, 4]);
		
		const gridStep = 32 * canvasScale;
		
		for (let i = offsetX; i < actualCanvasWidth; i += gridStep) {
			this.ctx.beginPath();
			this.ctx.moveTo(i, 0);
			this.ctx.lineTo(i, actualCanvasHeight);
			this.ctx.stroke();
		}
		
		for (let i = offsetY; i < actualCanvasHeight; i += gridStep) {
			this.ctx.beginPath();
			this.ctx.moveTo(0, i);
			this.ctx.lineTo(actualCanvasWidth, i);
			this.ctx.stroke();
		}
		
		this.ctx.setLineDash([]);
	}

	// Generate placeholder
	generatePoseVisualization(type, width = 384, height = 384) {
		const canvas = document.createElement('canvas');
		canvas.width = width;
		canvas.height = height;
		const ctx = canvas.getContext('2d');
		
		ctx.fillStyle = '#1a1a1a';
		ctx.fillRect(0, 0, width, height);
		
		ctx.strokeStyle = type === 'ref' ? '#666' : type === 'A' ? '#ff4a4a' : '#4a9eff';
		ctx.lineWidth = 3;
		ctx.lineCap = 'round';
		
		const centerX = width / 2;
		const centerY = height / 2;
		const scale = 50;
		
		// Simple stick figure
		ctx.beginPath();
		ctx.arc(centerX, centerY - scale, 15, 0, Math.PI * 2);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.moveTo(centerX, centerY - scale + 15);
		ctx.lineTo(centerX, centerY + scale);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.moveTo(centerX - scale * 0.8, centerY - scale * 0.3);
		ctx.lineTo(centerX + scale * 0.8, centerY - scale * 0.3);
		ctx.stroke();
		
		ctx.beginPath();
		ctx.moveTo(centerX, centerY + scale);
		ctx.lineTo(centerX - scale * 0.5, centerY + scale * 1.5);
		ctx.moveTo(centerX, centerY + scale);
		ctx.lineTo(centerX + scale * 0.5, centerY + scale * 1.5);
		ctx.stroke();
		
		ctx.fillStyle = ctx.strokeStyle;
		ctx.font = '12px monospace';
		ctx.textAlign = 'center';
		const label = type === 'ref' ? 'REF (placeholder)' : 
		             type === 'A' ? 'POSE A (placeholder)' : 'POSE B (placeholder)';
		ctx.fillText(label, centerX, height - 20);
		
		return canvas;
	}

	// CRITICAL FIX: Better dimension detection for poses A and B
	getPoseDimensions(img, keyPrefix, inputDims) {
		// Priority 1: Use API input dimensions if available
		if (inputDims && inputDims[keyPrefix]) {
			const [w, h] = inputDims[keyPrefix];
			console.log(`Using API dimensions for ${keyPrefix}: ${w}x${h}`);
			return { width: w, height: h };
		}
		
		// Priority 2: Use actual image dimensions
		if (img) {
			console.log(`Using image dimensions for ${keyPrefix}: ${img.width}x${img.height}`);
			return { width: img.width, height: img.height };
		}
		
		// Priority 3: Use reference dimensions as fallback
		const refSize = this.imageManager.getReferenceImageSize();
		console.log(`FALLBACK: Using reference dimensions for ${keyPrefix}: ${refSize.width}x${refSize.height}`);
		return refSize;
	}

	// FIXED: Draw pose layer with proper dimension handling
	drawPoseLayer(img, keyPrefix, color, isActive, coordSys, hasValidImages) {
		const { canvasScale, offsetX, offsetY, refW, refH } = coordSys;
		
		if (!img) {
			if (!hasValidImages) {
				const type = keyPrefix === "REF" ? "ref" : keyPrefix === "A" ? "A" : "B";
				img = this.generatePoseVisualization(type, refW, refH);
			} else {
				console.log(`No image for ${keyPrefix}`);
				return;
			}
		}

		this.ctx.setTransform(1, 0, 0, 1, 0, 0);

		if (keyPrefix === "REF") {
			// Reference image: draw at reference size and position
			this.ctx.globalAlpha = 0.8;
			this.ctx.drawImage(img, offsetX, offsetY, refW * canvasScale, refH * canvasScale);
			console.log(`Drew reference pose at: ${offsetX}, ${offsetY}, ${refW * canvasScale}x${refH * canvasScale}`);
		} else {
			// For poses A and B, get current transform values
			const transform = this.transformManager.getCurrentTransform(keyPrefix, refW, refH);
			const { tx, ty, scale, rotD, matrix } = transform;
			
			console.log(`Pose ${keyPrefix} transform:`, { tx, ty, scale, rotD, matrix });

			// CRITICAL FIX: Get the correct dimensions for this pose
			const inputDims = this.transformManager.getInputDimensions();
			const poseDims = this.getPoseDimensions(img, keyPrefix, inputDims);
			const drawW = poseDims.width;
			const drawH = poseDims.height;

			console.log(`Pose ${keyPrefix} will be drawn at dimensions: ${drawW}x${drawH}`);

			// Apply transformation matrix with proper scaling
			this.ctx.setTransform(
				matrix.a * canvasScale,    // Scale X transformation
				matrix.b * canvasScale,    // Scale Y shear
				matrix.c * canvasScale,    // Scale X shear  
				matrix.d * canvasScale,    // Scale Y transformation
				matrix.e * canvasScale + offsetX,  // Translate X to canvas coords
				matrix.f * canvasScale + offsetY   // Translate Y to canvas coords
			);

			// Set appropriate transparency
			if (keyPrefix === "A") {
				this.ctx.globalAlpha = 0.6; // More transparent so B shows through
			} else {
				this.ctx.globalAlpha = 0.8; // Less transparent but still allows ref to show
			}
			
			// CRITICAL FIX: Draw the image using the CORRECT dimensions for this pose
			this.ctx.drawImage(img, 0, 0, drawW, drawH);
			console.log(`Drew pose ${keyPrefix} at correct dimensions: ${drawW}x${drawH}`);
			
			// Reset transform to draw outline in canvas coordinates
			this.ctx.setTransform(1, 0, 0, 1, 0, 0);
			
			// Calculate transformed corners using the correct drawing dimensions
			const corners = [
				[0, 0], [drawW, 0], [drawW, drawH], [0, drawH]
			].map(([x, y]) => [
				(matrix.a * x + matrix.c * y + matrix.e) * canvasScale + offsetX,
				(matrix.b * x + matrix.d * y + matrix.f) * canvasScale + offsetY
			]);

			// Draw colored outline showing the transformed boundary
			this.ctx.globalAlpha = isActive ? 1.0 : 0.6;
			this.ctx.strokeStyle = color;
			this.ctx.lineWidth = isActive ? 3 : 2;
			this.ctx.setLineDash(isActive ? [] : [5, 5]);
			this.ctx.beginPath();
			this.ctx.moveTo(corners[0][0], corners[0][1]);
			for (let i = 1; i < corners.length; i++) {
				this.ctx.lineTo(corners[i][0], corners[i][1]);
			}
			this.ctx.closePath();
			this.ctx.stroke();
			this.ctx.setLineDash([]);
			
			console.log(`Drew bounding box for ${keyPrefix} using ${drawW}x${drawH} at corners:`, corners);
		}
	}

	// Draw UI overlay
	drawUIOverlay(coordSys, hasValidImages) {
		const { actualCanvasWidth, actualCanvasHeight } = coordSys;
		
		this.ctx.setTransform(1, 0, 0, 1, 0, 0);
		this.ctx.globalAlpha = 1.0;
		
		if (!hasValidImages) {
			this.ctx.fillStyle = "#888";
			this.ctx.font = "14px monospace";
			this.ctx.textAlign = "center";
			this.ctx.fillText("Run workflow to generate images", actualCanvasWidth/2, actualCanvasHeight/2 - 20);
			
			this.ctx.fillStyle = "#666";
			this.ctx.font = "12px monospace";
			this.ctx.fillText("Images will appear after node execution", actualCanvasWidth/2, actualCanvasHeight/2 + 5);
			
			this.ctx.font = "10px monospace";
			const nodeImgCount = this.imageManager.node.imgs?.length || 0;
			this.ctx.fillText(`node.imgs found: ${nodeImgCount}`, actualCanvasWidth/2, actualCanvasHeight/2 + 25);
		}
		
		// Active pose indicator
		this.ctx.fillStyle = this.state.which === "A" ? "#ff4a4a" : "#4a9eff";
		this.ctx.fillRect(10, 10, 20, 20);
		this.ctx.strokeStyle = "#fff";
		this.ctx.lineWidth = 2;
		this.ctx.strokeRect(10, 10, 20, 20);
		
		this.ctx.fillStyle = "#fff";
		this.ctx.font = "12px monospace";
		this.ctx.textAlign = "left";
		this.ctx.fillText(`Active: Pose ${this.state.which}`, 40, 25);
		
		// Instructions
		this.ctx.fillStyle = "#aaa";
		this.ctx.font = "10px monospace";
		this.ctx.fillText("Left-click: Move Pose A", 10, actualCanvasHeight - 70);
		this.ctx.fillText("Right-click: Move Pose B", 10, actualCanvasHeight - 55);
		this.ctx.fillText("Wheel: Scale | Shift+Wheel: Rotate", 10, actualCanvasHeight - 40);
		this.ctx.fillText("Arrow keys: Fine movement | R: Reset pose", 10, actualCanvasHeight - 25);
		
		// Enhanced debug info with dimension details
		const refSize = this.imageManager.getReferenceImageSize();
		const canvasScale = coordSys.canvasScale;
		const inputDims = this.transformManager.getInputDimensions();
		
		const transform = this.transformManager.getCurrentTransform(this.state.which, refSize.width, refSize.height);
		
		let dimInfo = `Ref: ${refSize.width}x${refSize.height}`;
		if (inputDims) {
			if (inputDims.A) dimInfo += ` | A: ${inputDims.A[0]}x${inputDims.A[1]}`;
			if (inputDims.B) dimInfo += ` | B: ${inputDims.B[0]}x${inputDims.B[1]}`;
		} else {
			dimInfo += ` | API dims: NULL`;
		}
		dimInfo += ` | Scale: ${canvasScale.toFixed(3)}`;
		
		// Safe handling of potentially undefined transform values
		const tx = (transform.tx !== undefined && !isNaN(transform.tx)) ? transform.tx.toFixed(1) : '0.0';
		const ty = (transform.ty !== undefined && !isNaN(transform.ty)) ? transform.ty.toFixed(1) : '0.0';
		const scale = (transform.scale !== undefined && !isNaN(transform.scale)) ? transform.scale.toFixed(2) : '1.00';
		const angle = (transform.rotD !== undefined && !isNaN(transform.rotD)) ? transform.rotD.toFixed(1) : '0.0';
		
		dimInfo += ` | ${this.state.which}: tx=${tx}, ty=${ty}, s=${scale}, r=${angle}°`;
		
		this.ctx.fillText(dimInfo, 10, actualCanvasHeight - 10);
	}

	// Main drawing function with enhanced debugging
	async draw() {
		try {
			console.log("=== DRAW FUNCTION START ===");
			
			const dataUpdated = await this.transformManager.updateFromNode();
			if (dataUpdated) {
				console.log("Transform data updated from Python node");
			}
			
			const images = await this.imageManager.getImagesFromNode(this.transformManager);
			const hasValidImages = !!(images.ref || images.A || images.B);
			const coordSys = this.calculateCoordinateSystem();
			const { actualCanvasWidth, actualCanvasHeight } = coordSys;
			
			// Enhanced debug logging with dimension info
			const inputDims = this.transformManager.getInputDimensions();
			const refSize = this.imageManager.getReferenceImageSize();
			
			const transformA = this.transformManager.getCurrentTransform('A', refSize.width, refSize.height);
			const transformB = this.transformManager.getCurrentTransform('B', refSize.width, refSize.height);
			
			console.log("CANVAS DRAW STATE:", { 
				hasValidImages,
				refImageSize: refSize,
				inputDimensions: inputDims,
				inputDimsStatus: inputDims ? "Available" : "NULL from API",
				canvasActualSize: { width: actualCanvasWidth, height: actualCanvasHeight },
				canvasScale: coordSys.canvasScale,
				activeTransforms: {
					A: { tx: transformA.tx, ty: transformA.ty, scale: transformA.scale, angle: transformA.rotD },
					B: { tx: transformB.tx, ty: transformB.ty, scale: transformB.scale, angle: transformB.rotD }
				},
				properties: this.transformManager.getAllProperties()
			});
			
			// Clear canvas
			this.ctx.setTransform(1, 0, 0, 1, 0, 0);
			this.ctx.clearRect(0, 0, actualCanvasWidth, actualCanvasHeight);
			
			// Background
			this.ctx.fillStyle = "#1a1a1a";
			this.ctx.fillRect(0, 0, actualCanvasWidth, actualCanvasHeight);
			
			// Draw layers in correct order (back to front)
			this.drawGrid(coordSys);
			this.drawPoseLayer(images.ref, "REF", "#666", false, coordSys, hasValidImages);
			
			// Only draw pose B if it exists (not in single pose mode)
			if (images.B) {
				this.drawPoseLayer(images.B, "B", "#4a9eff", this.state.which === "B", coordSys, hasValidImages);
			}
			
			this.drawPoseLayer(images.A, "A", "#ff4a4a", this.state.which === "A", coordSys, hasValidImages);
			this.drawUIOverlay(coordSys, hasValidImages);
			
			console.log("=== DRAW FUNCTION END ===");
			
		} catch (error) {
			console.error("Error in draw function:", error);
			
			this.ctx.setTransform(1, 0, 0, 1, 0, 0);
			this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
			this.ctx.fillStyle = "#1a1a1a";
			this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
			this.ctx.fillStyle = "#ff4444";
			this.ctx.font = "14px monospace";
			this.ctx.textAlign = "center";
			this.ctx.fillText("Error loading images", this.canvas.width/2, this.canvas.height/2);
			this.ctx.font = "10px monospace";
			this.ctx.fillText("Check console for details", this.canvas.width/2, this.canvas.height/2 + 20);
		}
	}

	// Get canvas coordinates from mouse event
	getCanvasCoordinates(e) {
		const rect = this.canvas.getBoundingClientRect();
		return {
			x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
			y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
		};
	}
}
