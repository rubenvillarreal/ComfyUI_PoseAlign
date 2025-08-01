/* modules/ui_components.js - Compact toggle positioned below canvas */
import { DOMUtils, LogUtils } from "./utils.js";

export class UIComponents {
	constructor(node, renderer, imageManager, transformManager) {
		this.node = node;
		this.renderer = renderer;
		this.imageManager = imageManager;
		this.transformManager = transformManager;
	}

	// Create very compact manual mode toggle button positioned below canvas
	createModeToggle() {
		const toggleContainer = document.createElement("div");
		toggleContainer.style.marginTop = "2px";
		toggleContainer.style.marginBottom = "2px";
		toggleContainer.style.padding = "2px 6px";
		toggleContainer.style.backgroundColor = "#2a2a2a";
		toggleContainer.style.borderRadius = "2px";
		toggleContainer.style.display = "flex";
		toggleContainer.style.alignItems = "center";
		toggleContainer.style.justifyContent = "center";
		toggleContainer.style.gap = "4px";
		toggleContainer.style.fontSize = "9px";
		toggleContainer.style.height = "16px"; // Very compact height
		toggleContainer.style.width = "auto";
		toggleContainer.style.maxWidth = "120px"; // Prevent it from getting too wide
		toggleContainer.style.flexShrink = "0";
		toggleContainer.style.position = "relative"; // Normal flow, not overlapping

		const label = document.createElement("label");
		label.style.color = "#fff";
		label.style.fontSize = "9px";
		label.style.fontFamily = "monospace";
		label.style.margin = "0";
		label.style.padding = "0";
		label.style.cursor = "pointer";
		label.textContent = "Live:";

		const toggle = document.createElement("input");
		toggle.type = "checkbox";
		toggle.checked = false;
		toggle.style.margin = "0";
		toggle.style.padding = "0";
		toggle.style.width = "12px";
		toggle.style.height = "12px";
		toggle.style.cursor = "pointer";

		const statusSpan = document.createElement("span");
		statusSpan.style.color = "#aaa";
		statusSpan.style.fontSize = "9px";
		statusSpan.style.fontFamily = "monospace";
		statusSpan.style.margin = "0";
		statusSpan.style.padding = "0";
		statusSpan.style.minWidth = "25px"; // Prevent text jumping

		const updateStatus = () => {
			const hasAutoData = this.transformManager.getTransformCache().matrices.A !== null;
			const isManualOverride = this.transformManager.shouldUseManualMode();
			
			if (isManualOverride) {
				statusSpan.textContent = "ON";
				statusSpan.style.color = "#00ff88";
			} else if (hasAutoData) {
				statusSpan.textContent = "AUTO";
				statusSpan.style.color = "#ffaa00";
			} else {
				statusSpan.textContent = "OFF";
				statusSpan.style.color = "#666";
			}
		};

		// Make the label clickable too
		label.addEventListener("click", () => {
			toggle.checked = !toggle.checked;
			toggle.dispatchEvent(new Event('change'));
		});

		toggle.addEventListener("change", () => {
			this.transformManager.setManualModeOverride(toggle.checked);
			updateStatus();
			this.renderer.draw();
		});

		// Update status periodically
		setInterval(updateStatus, 500);
		updateStatus();

		toggleContainer.appendChild(label);
		toggleContainer.appendChild(toggle);
		toggleContainer.appendChild(statusSpan);

		// Add as a widget positioned below the canvas
		this.node.addDOMWidget("mode_toggle", "div", toggleContainer, {
			serialize: false,
			hideOnZoom: false,
		});
	}

	// Create all UI components - Only the compact toggle
	createAllComponents() {
		this.createModeToggle(); // Only the essential compact toggle
	}
}
