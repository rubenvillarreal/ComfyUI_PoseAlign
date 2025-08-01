/* modules/utils.js - Utility functions and classes */

// DOM manipulation utilities
export class DOMUtils {
	static styleElement(element, styles) {
		Object.assign(element.style, styles);
	}

	static createElement(tag, styles = {}, attributes = {}) {
		const element = document.createElement(tag);
		this.styleElement(element, styles);
		Object.entries(attributes).forEach(([key, value]) => {
			element[key] = value;
		});
		return element;
	}
}

// Logging utilities
export class LogUtils {
	static logDebug(...args) {
		console.log("[PoseAlign Widget]", ...args);
	}

	static logError(...args) {
		console.error("[PoseAlign Widget]", ...args);
	}

	static logWarn(...args) {
		console.warn("[PoseAlign Widget]", ...args);
	}
}

// Math utilities
export class MathUtils {
	static clamp(value, min, max) {
		return Math.max(min, Math.min(max, value));
	}

	static degToRad(degrees) {
		return degrees * Math.PI / 180;
	}

	static radToDeg(radians) {
		return radians * 180 / Math.PI;
	}
}