/* modules/interaction_handler.js - Fixed for immediate live updates */

export class InteractionHandler {
	constructor(canvas, transformManager, renderer) {
		this.canvas = canvas;
		this.transformManager = transformManager;
		this.renderer = renderer;
		this.state = null; // Will be set by main widget
		this.isInteracting = false; // Track if user is actively interacting
	}

	setState(state) {
		this.state = state;
	}

	// CRITICAL FIX: Force manual mode during interactions
	startInteraction() {
		this.isInteracting = true;
		this.transformManager.setManualModeOverride(true);
		console.log("[Interaction] Started - forcing manual mode");
	}

	endInteraction() {
		this.isInteracting = false;
		// Keep manual mode active for a short time after interaction ends
		setTimeout(() => {
			if (!this.isInteracting) {
				// Only reset if no new interaction started
				// User can manually toggle back if desired
				console.log("[Interaction] Ended - interaction complete");
			}
		}, 1000);
	}

	// Set up all event listeners
	setupEventListeners() {
		this.setupMouseEvents();
		this.setupKeyboardEvents();
	}

	// Set up mouse event listeners with immediate feedback
	setupMouseEvents() {
		// Prevent context menu
		this.canvas.addEventListener("contextmenu", e => e.preventDefault());
		
		// Mouse down - start dragging
		this.canvas.addEventListener("mousedown", (e) => {
			this.startInteraction(); // CRITICAL: Force manual mode
			
			this.state.dragging = true;
			this.state.which = e.button === 2 ? "B" : "A"; // Right click = B, Left click = A
			const coords = this.renderer.getCanvasCoordinates(e);
			this.state.lastX = coords.x; 
			this.state.lastY = coords.y;
			this.canvas.style.cursor = "grabbing";
			
			console.log(`[Interaction] Mouse down - pose ${this.state.which} active`);
			
			// IMMEDIATE redraw when starting drag
			this.renderer.draw();
		});
		
		// Mouse up - stop dragging
		this.canvas.addEventListener("mouseup", () => {
			this.state.dragging = false;
			this.canvas.style.cursor = "crosshair";
			
			console.log("[Interaction] Mouse up - dragging stopped");
			
			// IMMEDIATE redraw when ending drag
			this.renderer.draw();
			
			// End interaction after a delay
			setTimeout(() => this.endInteraction(), 100);
		});
		
		// Mouse leave - stop dragging and hovering
		this.canvas.addEventListener("mouseleave", () => {
			this.state.dragging = false;
			this.state.hovering = false;
			this.canvas.style.cursor = "crosshair";
			this.endInteraction();
			this.renderer.draw();
		});
		
		// Mouse enter - start hovering
		this.canvas.addEventListener("mouseenter", () => {
			this.state.hovering = true;
			this.renderer.draw();
		});
		
		// Mouse move - handle dragging with immediate updates
		this.canvas.addEventListener("mousemove", (e) => {
			if (!this.state.dragging) return;
			
			const coords = this.renderer.getCanvasCoordinates(e);
			const dx = coords.x - this.state.lastX;
			const dy = coords.y - this.state.lastY;
			this.state.lastX = coords.x; 
			this.state.lastY = coords.y;
			
			// Convert canvas pixel movement to reference image coordinate movement
			const coordSys = this.renderer.calculateCoordinateSystem();
			const { canvasScale } = coordSys;
			
			// Scale the movement to match the reference coordinate system
			const scaledDx = dx / canvasScale;
			const scaledDy = dy / canvasScale;
			
			const pose = this.state.which;
			const currentTx = this.transformManager.getProperty(`tx_${pose}`, 0);
			const currentTy = this.transformManager.getProperty(`ty_${pose}`, 0);
			
			// Update properties
			this.transformManager.setProperty(`tx_${pose}`, currentTx + scaledDx);
			this.transformManager.setProperty(`ty_${pose}`, currentTy + scaledDy);
			
			console.log(`[Interaction] Move ${pose}: tx=${(currentTx + scaledDx).toFixed(1)}, ty=${(currentTy + scaledDy).toFixed(1)}`);
			
			// CRITICAL: Immediate redraw with forced manual mode
			this.renderer.draw();
		});
		
		// Mouse wheel - handle scaling and rotation with immediate updates
		this.canvas.addEventListener("wheel", (e) => {
			e.preventDefault();
			
			this.startInteraction(); // Force manual mode for wheel events
			
			const pose = this.state.which;
			
			if (e.shiftKey) { 
				// Rotation with shift key
				const rotationStep = e.deltaY > 0 ? 5 : -5;
				const currentAngle = this.transformManager.getProperty(`angle_deg_${pose}`, 0);
				const newAngle = currentAngle + rotationStep;
				this.transformManager.setProperty(`angle_deg_${pose}`, newAngle);
				
				console.log(`[Interaction] Rotate ${pose}: ${newAngle.toFixed(1)}°`);
			} else { 
				// Scaling without shift key
				const currentScale = this.transformManager.getProperty(`scale_${pose}`, 1);
				const scaleStep = e.deltaY > 0 ? -0.05 : 0.05;
				const newScale = Math.max(0.1, currentScale + scaleStep);
				this.transformManager.setProperty(`scale_${pose}`, newScale);
				
				console.log(`[Interaction] Scale ${pose}: ${newScale.toFixed(2)}`);
			}
			
			// CRITICAL: Immediate redraw
			this.renderer.draw();
			
			// End interaction after wheel event
			setTimeout(() => this.endInteraction(), 500);
		});
	}

	// Set up keyboard event listeners
	setupKeyboardEvents() {
		this.canvas.addEventListener("keydown", (e) => {
			if (!this.state.hovering) return;
			
			// Start interaction for keyboard events
			this.startInteraction();
			
			// Calculate step size based on reference image coordinate system
			const coordSys = this.renderer.calculateCoordinateSystem();
			const { canvasScale } = coordSys;
			const step = (e.shiftKey ? 10 : 1) / canvasScale; // Convert to reference coordinates
			const pose = this.state.which;
			
			switch(e.key) {
				case 'ArrowLeft':
					e.preventDefault();
					this.adjustProperty(`tx_${pose}`, -step);
					break;
				case 'ArrowRight':
					e.preventDefault();
					this.adjustProperty(`tx_${pose}`, step);
					break;
				case 'ArrowUp':
					e.preventDefault();
					this.adjustProperty(`ty_${pose}`, -step);
					break;
				case 'ArrowDown':
					e.preventDefault();
					this.adjustProperty(`ty_${pose}`, step);
					break;
				case 'a':
				case 'A':
					e.preventDefault();
					this.state.which = "A";
					console.log("[Interaction] Switched to pose A");
					this.renderer.draw();
					break;
				case 'b':
				case 'B':
					e.preventDefault();
					this.state.which = "B";
					console.log("[Interaction] Switched to pose B");
					this.renderer.draw();
					break;
				case 'r':
				case 'R':
					e.preventDefault();
					this.resetCurrentPose();
					break;
			}
			
			// End interaction after keyboard event
			setTimeout(() => this.endInteraction(), 500);
		});
	}

	// Helper method to adjust a property by a delta value
	adjustProperty(propertyName, delta) {
		const currentValue = this.transformManager.getProperty(propertyName, 0);
		const newValue = currentValue + delta;
		this.transformManager.setProperty(propertyName, newValue);
		
		console.log(`[Interaction] Adjust ${propertyName}: ${currentValue.toFixed(2)} → ${newValue.toFixed(2)}`);
		
		this.renderer.draw();
	}

	// Reset the current active pose to default values
	resetCurrentPose() {
		const pose = this.state.which;
		this.transformManager.setProperty(`tx_${pose}`, 0);
		this.transformManager.setProperty(`ty_${pose}`, 0);
		this.transformManager.setProperty(`scale_${pose}`, 1.0);
		this.transformManager.setProperty(`angle_deg_${pose}`, 0);
		
		console.log(`[Interaction] Reset pose ${pose} to defaults`);
		
		this.renderer.draw();
	}
}
