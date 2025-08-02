/* modules/ui_components.js - Compact toggle positioned below canvas */
import { DOMUtils, LogUtils } from "./utils.js";

export class UIComponents {
	constructor(node, renderer, imageManager, transformManager) {
		this.node = node;
		this.renderer = renderer;
		this.imageManager = imageManager;
		this.transformManager = transformManager;
	}

	// Create all UI components - simplified for manual-only mode
	createAllComponents() {
		// No UI components needed - everything is manual now
		console.log("[UIComponents] Manual-only mode - no toggle needed");
	}
}
