/**
 * Camera sampling required before scene rendering begins.
 */
export interface CameraJitterRequirement {
	/** Jitter sequence shared by backend camera-uniform implementations. */
	readonly sequence: "halton-2-3";
	/** Normalized sub-pixel jitter amplitude. */
	readonly scale: number;
}

/**
 * Backend-agnostic work that must be applied while preparing a render frame.
 */
export interface FramePreparationRequirements {
	readonly cameraJitter?: CameraJitterRequirement;
}
