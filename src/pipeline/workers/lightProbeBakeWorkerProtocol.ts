import type { TextureColorSpace } from "../../core/Texture";

export type LightProbeBakeTextureData =
	| Uint8ClampedArray
	| Float32Array
	| Uint8Array;

export interface LightProbeBakeWorkerEnvMapPayload {
	width: number;
	height: number;
	colorSpace: TextureColorSpace;
	data: LightProbeBakeTextureData | null;
}

export interface LightProbeBakeWorkerTaskPayload {
	type: "prefilter-mip";
	envMap: LightProbeBakeWorkerEnvMapPayload;
	baseWidth: number;
	baseHeight: number;
	maxMipLevels: number;
	level: number;
}

export interface LightProbeBakeWorkerTaskResult {
	type: "prefilter-mip";
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}
