import type { TextureColorSpace } from "../../../core/Texture";

export type EnvironmentIBLBakeTextureData =
	| Uint8ClampedArray
	| Float32Array
	| Uint8Array;

export interface EnvironmentIBLBakeWorkerEnvMapPayload {
	width: number;
	height: number;
	colorSpace: TextureColorSpace;
	data: EnvironmentIBLBakeTextureData | null;
}

export interface EnvironmentIBLBakeWorkerTaskPayload {
	type: "prefilter-mip";
	envMap: EnvironmentIBLBakeWorkerEnvMapPayload;
	baseWidth: number;
	baseHeight: number;
	maxMipLevels: number;
	level: number;
}

export interface EnvironmentIBLBakeWorkerTaskResult {
	type: "prefilter-mip";
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}
