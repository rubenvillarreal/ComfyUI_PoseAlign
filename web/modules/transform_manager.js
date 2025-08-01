/* modules/transform_manager.js - Fixed field names and ID matching */
import { api } from "../../../scripts/api.js";

export class TransformManager {
	constructor(node) {
		this.node = node;
		this.lastProperties = {};
		this.forceManualMode = false; // Force manual mode for live updates
		this.transformCache = {
			lastUpdate: 0,
			matrices: { A: null, B: null },
			offsetCorrections: { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } },
			inputDimensions: null
		};
		this.propertyMonitorInterval = null;
	}

	// Set manual mode override
	setManualModeOverride(enabled) {
		console.log(`[TransformManager] Manual mode override: ${this.forceManualMode} → ${enabled}`);
		this.forceManualMode = enabled;
	}

	// Check if we should use manual mode
	shouldUseManualMode() {
		return this.forceManualMode || !this.transformCache.matrices.A;
	}

	// Normalize angle to 0-360 degrees range
	normalizeAngle(angle) {
		return ((angle % 360) + 360) % 360;
	}

	// Set a property value on both the node and its widget
	setProperty(key, value) {
		if (key.includes('angle_deg_')) {
			value = this.normalizeAngle(value);
		}
		
		console.log(`[TransformManager] Setting ${key} = ${value}`);
		
		// Update node properties
		this.node.setProperty(key, value);
		
		// Update widget if it exists
		const widget = this.node.widgets?.find(w => w.name === key);
		if (widget && widget.value !== value) { 
			widget.value = value;
			if (widget.callback) {
				widget.callback(value);
			}
		}
	}

	// Get a property value safely with default
	getProperty(key, defaultValue = 0) {
		const value = this.node.properties && this.node.properties[key] !== undefined ? 
			this.node.properties[key] : defaultValue;
		return value;
	}

	// Get all current transformation properties
	getAllProperties() {
		return {
			tx_A: this.getProperty('tx_A'),
			ty_A: this.getProperty('ty_A'),
			scale_A: this.getProperty('scale_A', 1),
			angle_deg_A: this.getProperty('angle_deg_A'),
			tx_B: this.getProperty('tx_B'),
			ty_B: this.getProperty('ty_B'),
			scale_B: this.getProperty('scale_B', 1),
			angle_deg_B: this.getProperty('angle_deg_B')
		};
	}

	// Check if transformation properties have changed
	checkPropertiesChanged() {
		const currentProps = this.getAllProperties();
		
		const changed = Object.keys(currentProps).some(key => 
			this.lastProperties[key] !== currentProps[key]
		);

		if (changed) {
			this.lastProperties = { ...currentProps };
			console.log("[TransformManager] Properties changed:", currentProps);
			return true;
		}
		return false;
	}

	// Build affine transformation matrix exactly like the Python node
	buildAffineMatrix(scale, angleDeg, tx, ty, cx, cy) {
		const angleRad = angleDeg * Math.PI / 180;
		const cosA = Math.cos(angleRad);
		const sinA = Math.sin(angleRad);
		
		const R11 = cosA * scale;
		const R12 = -sinA * scale;
		const R21 = sinA * scale;
		const R22 = cosA * scale;
		
		const finalTx = tx + cx - (R11 * cx + R12 * cy);
		const finalTy = ty + cy - (R21 * cx + R22 * cy);
		
		return {
			a: R11, b: R21, c: R12, d: R22, e: finalTx, f: finalTy
		};
	}

	// Get input dimensions for bounding box calculation
	getInputDimensions() {
		return this.transformCache.inputDimensions;
	}

	// CRITICAL FIX: Always use manual mode when override is active
	getCurrentTransform(keyPrefix, refW, refH) {
		const isUsingManualMode = this.shouldUseManualMode();
		
		console.log(`[TransformManager] getCurrentTransform(${keyPrefix}): manual=${isUsingManualMode}, hasCache=${!!this.transformCache.matrices[keyPrefix]}`);
		
		// Get widget values
		const tx = this.getProperty(`tx_${keyPrefix}`, 0);
		const ty = this.getProperty(`ty_${keyPrefix}`, 0);
		const scale = this.getProperty(`scale_${keyPrefix}`, 1);
		const rotD = this.getProperty(`angle_deg_${keyPrefix}`, 0);
		
		// CRITICAL: When manual mode is active, ALWAYS use widget values
		if (isUsingManualMode) {
			console.log(`[TransformManager] Using MANUAL mode for ${keyPrefix}: tx=${tx}, ty=${ty}, scale=${scale}, angle=${rotD}`);
			
			// Apply offset corrections if available
			let finalTx = tx;
			let finalTy = ty;
			const offset = this.transformCache.offsetCorrections[keyPrefix];
			if (offset) {
				finalTx += offset.x;
				finalTy += offset.y;
				console.log(`[TransformManager] Applied offset correction: +${offset.x}, +${offset.y}`);
			}
			
			// Build the affine matrix for manual mode
			const cx = refW / 2.0;
			const cy = refH / 2.0;
			const matrix = this.buildAffineMatrix(scale, rotD, finalTx, finalTy, cx, cy);

			return { tx: finalTx, ty: finalTy, scale, rotD, matrix };
		}
		
		// Auto mode: use cached matrix data from Python
		if (this.transformCache.matrices[keyPrefix]) {
			const matrixArray = this.transformCache.matrices[keyPrefix];
			console.log(`[TransformManager] Using AUTO mode for ${keyPrefix}:`, matrixArray);
			
			// Parse the matrix array from Python
			let matrix;
			if (Array.isArray(matrixArray) && matrixArray.length === 2) {
				if (Array.isArray(matrixArray[0]) && matrixArray[0].length === 3) {
					matrix = {
						a: matrixArray[0][0], b: matrixArray[1][0], c: matrixArray[0][1],
						d: matrixArray[1][1], e: matrixArray[0][2], f: matrixArray[1][2]
					};
				} else {
					matrix = {
						a: matrixArray[0], b: matrixArray[1], c: matrixArray[2],
						d: matrixArray[3], e: matrixArray[4], f: matrixArray[5]
					};
				}
			} else {
				console.error(`[TransformManager] Invalid matrix format for ${keyPrefix}:`, matrixArray);
				matrix = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
			}
			
			// Decompose the matrix for display purposes
			const cx = refW / 2.0;
			const cy = refH / 2.0;
			const autoScale = Math.sqrt(matrix.a * matrix.a + matrix.b * matrix.b);
			const autoRotD = Math.atan2(matrix.b, matrix.a) * 180 / Math.PI;
			const autoTx = matrix.e;
			const autoTy = matrix.f;
			
			return { tx: autoTx, ty: autoTy, scale: autoScale, rotD: autoRotD, matrix };
		}
		
		// Fallback: no data available, use widget values
		console.log(`[TransformManager] FALLBACK mode for ${keyPrefix}: using widget values`);
		const cx = refW / 2.0;
		const cy = refH / 2.0;
		const matrix = this.buildAffineMatrix(scale, rotD, tx, ty, cx, cy);
		return { tx, ty, scale, rotD, matrix };
	}

	// ENHANCED: Try all available node IDs and fix field name mismatch
	async updateFromNode() {
		try {
			// Strategy 1: Try ComfyUI node ID first
			let nodeId = this.node.id;
			console.log(`[TransformManager] Trying ComfyUI node ID: ${nodeId}`);
			let data = await this.tryFetchData(nodeId);
			
			if (data) {
				console.log(`[TransformManager] SUCCESS with ComfyUI node ID: ${nodeId}`);
				return this.processSuccessfulData(data);
			}
			
			// Strategy 2: Try Python node ID from properties
			const pythonNodeId = this.node.properties?.python_node_id;
			if (pythonNodeId) {
				console.log(`[TransformManager] Trying Python node ID: ${pythonNodeId}`);
				data = await this.tryFetchData(pythonNodeId);
				if (data) {
					console.log(`[TransformManager] SUCCESS with Python node ID: ${pythonNodeId}`);
					return this.processSuccessfulData(data);
				}
			}
			
			// Strategy 3: Get latest available data
			try {
				console.log(`[TransformManager] Trying to find latest stored data...`);
				const debugResponse = await api.fetchApi('/AInseven/debug_node_ids');
				if (debugResponse.ok) {
					const debugData = await debugResponse.json();
					console.log(`[TransformManager] Available node IDs:`, debugData.stored_node_ids);
					
					if (debugData.stored_node_ids.length > 0) {
						// Find the most recent data entry
						let latestId = null;
						let latestTimestamp = 0;
						
						for (const storedId of debugData.stored_node_ids) {
							const cacheEntry = debugData.cache_contents[storedId];
							if (cacheEntry && cacheEntry.timestamp > latestTimestamp) {
								latestTimestamp = cacheEntry.timestamp;
								latestId = storedId;
							}
						}
						
						if (latestId) {
							console.log(`[TransformManager] Using latest stored ID: ${latestId}`);
							data = await this.tryFetchData(latestId);
							if (data) {
								console.log(`[TransformManager] SUCCESS with latest stored ID: ${latestId}`);
								return this.processSuccessfulData(data);
							}
						}
					}
				}
			} catch (debugError) {
				console.log(`[TransformManager] Debug endpoint failed:`, debugError);
			}
			
			console.log(`[TransformManager] No valid data found with any strategy`);
			return false;
			
		} catch (error) {
			console.log(`[TransformManager] API call failed:`, error);
			return false;
		}
	}

	// Helper method to try fetching data with a specific ID
	async tryFetchData(nodeId) {
		try {
			const response = await api.fetchApi(`/AInseven/pose_align_data/${nodeId}`);
			
			if (response.ok) {
				const data = await response.json();
				console.log(`[TransformManager] Raw API response for ${nodeId}:`, data);
				
				if (!data.error && (data.matrices || data.inputDimensions || data.input_dimensions)) {
					return data;
				} else {
					console.log(`[TransformManager] ID ${nodeId} returned error or empty data`);
				}
			} else {
				console.log(`[TransformManager] HTTP error for ID ${nodeId}:`, response.status);
			}
			return null;
		} catch (error) {
			console.log(`[TransformManager] Fetch failed for ID ${nodeId}:`, error.message);
			return null;
		}
	}

	// CRITICAL FIX: Handle both field name formats
	processSuccessfulData(data) {
		console.log(`[TransformManager] Processing successful data:`, data);
		
		// Handle field name mismatch: input_dimensions vs inputDimensions
		let inputDimensions = data.inputDimensions || data.input_dimensions || null;
		
		console.log(`[TransformManager] Input dimensions extracted:`, inputDimensions);
		console.log(`[TransformManager] Matrices from response:`, data.matrices);
		
		if (data.timestamp > this.transformCache.lastUpdate) {
			this.transformCache = {
				lastUpdate: data.timestamp,
				matrices: data.matrices || { A: null, B: null },
				offsetCorrections: data.offsetCorrections || data.offset_corrections || { A: { x: 0, y: 0 }, B: { x: 0, y: 0 } },
				inputDimensions: inputDimensions  // FIXED: Now correctly extracts dimensions
			};
			console.log(`[TransformManager] Updated transform cache:`, this.transformCache);
			return true;
		}
		return false;
	}

	// Get current transform cache
	getTransformCache() {
		return this.transformCache;
	}

	// Enhanced property monitoring with immediate response
	setupPropertyMonitoring(renderer) {
		const monitoredParams = ['tx_A', 'ty_A', 'scale_A', 'angle_deg_A', 
		                         'tx_B', 'ty_B', 'scale_B', 'angle_deg_B'];
		
		const setupWidgetMonitoring = () => {
			monitoredParams.forEach(paramName => {
				const widget = this.node.widgets?.find(w => w.name === paramName);
				if (widget) {
					const originalCallback = widget.callback;
					widget.callback = function(value) {
						if (originalCallback) {
							originalCallback.call(this, value);
						}
						console.log(`[Widget] ${paramName} changed to ${value}`);
						// Immediate redraw for widget changes
						setTimeout(() => renderer.draw(), 1);
					};
				}
			});
		};

		setTimeout(setupWidgetMonitoring, 100);

		// Very fast polling for property changes (for live updates)
		this.propertyMonitorInterval = setInterval(() => {
			const propsChanged = this.checkPropertiesChanged();
			
			if (propsChanged) {
				renderer.draw();
			}
		}, 50); // Very fast polling for responsiveness

		// Separate slower polling for API data
		setInterval(async () => {
			const dataUpdated = await this.updateFromNode();
			if (dataUpdated) {
				console.log("[TransformManager] Transform data updated from API");
				renderer.draw();
			}
		}, 2000); // Check for new API data every 2 seconds
	}

	// Clean up monitoring when node is removed
	cleanup() {
		if (this.propertyMonitorInterval) {
			clearInterval(this.propertyMonitorInterval);
			this.propertyMonitorInterval = null;
		}
	}
}
