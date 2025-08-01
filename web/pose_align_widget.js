/* pose_align_widget.js - Fixed node sizing and minimal UI */
import { app } from "../../scripts/app.js";
import { CanvasRenderer } from "./modules/canvas_renderer.js";
import { InteractionHandler } from "./modules/interaction_handler.js";
import { ImageManager } from "./modules/image_manager.js";
import { TransformManager } from "./modules/transform_manager.js";
import { UIComponents } from "./modules/ui_components.js";
import { LogUtils } from "./modules/utils.js";

app.registerExtension({
	name: "AInseven.PoseAlignCanvasWidget.Fixed",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Debug logging to see what nodes are being registered
		console.log("[PoseAlign Widget] Checking node:", nodeData.name, nodeData);
		
		// Exit if this is not the node we want to modify
		if (nodeData.name !== "PoseAlignTwoToOne") {
			console.log("[PoseAlign Widget] Skipping node:", nodeData.name);
			return;
		}

		console.log("[PoseAlign Widget] Found PoseAlignTwoToOne node, setting up widget");

		// Hijack the onNodeCreated method to add our custom widget
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			// Call the original onNodeCreated method
			onNodeCreated?.apply(this, arguments);

			const node = this;
			console.log("[PoseAlign Widget] Node created:", node);

			// FIXED: Set initial node size to accommodate canvas
			const CANVAS_SIZE = 512;
			const PADDING = 20; // Extra space for widgets
			
			// Set node size to properly contain the canvas with minimal UI space
			node.size = [CANVAS_SIZE + PADDING, CANVAS_SIZE + 30]; // Only 30px for tiny toggle
			
			// Create the main canvas element
			const canvas = document.createElement("canvas");
			
			// Set up canvas properties - FIXED sizing
			canvas.width = CANVAS_SIZE;
			canvas.height = CANVAS_SIZE;
			canvas.style.border = "2px solid #555";
			canvas.style.backgroundColor = "#1a1a1a";
			canvas.style.display = "block";
			canvas.style.cursor = "crosshair";
			canvas.style.borderRadius = "4px";
			
			// CRITICAL FIX: Make sure canvas fits within node bounds
			canvas.style.width = `${CANVAS_SIZE}px`;
			canvas.style.height = `${CANVAS_SIZE}px`;
			canvas.style.maxWidth = `${CANVAS_SIZE}px`;
			canvas.style.maxHeight = `${CANVAS_SIZE}px`;
			canvas.style.minWidth = `${CANVAS_SIZE}px`;
			canvas.style.minHeight = `${CANVAS_SIZE}px`;
			canvas.style.objectFit = "contain";
			canvas.style.flexShrink = "0";
			canvas.style.margin = "0 auto"; // Center the canvas
			canvas.tabIndex = 0; // Make focusable for keyboard events

			const ctx = canvas.getContext("2d");

			// Initialize managers
			const imageManager = new ImageManager(node);
			const transformManager = new TransformManager(node);
			const renderer = new CanvasRenderer(canvas, ctx, imageManager, transformManager);
			const interactionHandler = new InteractionHandler(canvas, transformManager, renderer);
			const uiComponents = new UIComponents(node, renderer, imageManager, transformManager);

			// Set up the interaction state
			const state = {
				dragging: false,
				which: "A", // Active pose (A or B)
				lastX: 0,
				lastY: 0,
				hovering: false
			};

			// Share state with components that need it
			interactionHandler.setState(state);
			renderer.setState(state);

			// Set up event listeners
			interactionHandler.setupEventListeners();

			// FIXED: Create a properly sized container for the canvas
			const canvasContainer = document.createElement("div");
			canvasContainer.style.display = "flex";
			canvasContainer.style.flexDirection = "column";
			canvasContainer.style.alignItems = "center";
			canvasContainer.style.width = "100%";
			canvasContainer.style.padding = "2px";
			canvasContainer.style.boxSizing = "border-box";
			canvasContainer.style.position = "relative"; // For absolute positioning of children
			canvasContainer.appendChild(canvas);

			// Add the canvas as a DOM widget to the node's body
			node.addDOMWidget("pose_canvas", "div", canvasContainer, {
				serialize: false,
				hideOnZoom: false,
			});

			// Add ONLY essential UI components (drastically reduced)
			uiComponents.createAllComponents();

			// Set up monitoring for property changes and transform updates
			transformManager.setupPropertyMonitoring(renderer);

			// FIXED: Ensure node doesn't auto-resize smaller than canvas
			const originalComputeSize = node.computeSize;
			node.computeSize = function() {
				const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 200];
				// Ensure minimum size to accommodate canvas and tiny UI
				return [
					Math.max(size[0], CANVAS_SIZE + PADDING),
					Math.max(size[1], CANVAS_SIZE + 30) // Only 30px for tiny toggle
				];
			};

			// Override node methods to handle execution and property changes
			const originalOnExecuted = node.onExecuted;
			node.onExecuted = function(message) {
				originalOnExecuted?.apply(this, arguments);
				LogUtils.logDebug("Node executed", message);
				
				if (message && message.images && message.images.length > 0) {
					LogUtils.logDebug("Found images in execution message", message.images);
					setTimeout(() => renderer.draw(), 100);
				} else {
					LogUtils.logDebug("No images in execution message");
					setTimeout(() => renderer.draw(), 500);
				}

				// Force a property check after execution
				setTimeout(async () => {
					const updated = await transformManager.updateFromNode();
					if (updated) {
						LogUtils.logDebug("Transform data updated after execution, redrawing");
						renderer.draw();
					}
				}, 1000);
			};

			const originalOnPropertyChanged = node.onPropertyChanged;
			node.onPropertyChanged = function(name, value) {
				originalOnPropertyChanged?.apply(this, arguments);
				
				const monitoredParams = ['tx_A', 'ty_A', 'scale_A', 'angle_deg_A', 
				                         'tx_B', 'ty_B', 'scale_B', 'angle_deg_B'];
				if (monitoredParams.includes(name)) {
					LogUtils.logDebug(`Property ${name} changed to ${value}`);
					renderer.draw();
				}
			};

			const onConnectionsChange = node.onConnectionsChange;
			node.onConnectionsChange = function() {
				onConnectionsChange?.apply(this, arguments);
				console.log("Connections changed");
				setTimeout(() => renderer.draw(), 100);
			};

			// Window resize handling
			const handleResize = () => {
				console.log("Window resized, redrawing canvas");
				setTimeout(() => renderer.draw(), 100);
			};
			window.addEventListener('resize', handleResize);
			
			// Clean up when node is removed
			const originalOnRemoved = node.onRemoved;
			node.onRemoved = function() {
				window.removeEventListener('resize', handleResize);
				transformManager.cleanup();
				originalOnRemoved?.apply(this, arguments);
			};

			// Initial draw
			setTimeout(() => renderer.draw(), 500);
		};
	}
});
