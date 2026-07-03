import type { TextureColorSpace } from "../../../core/Texture";

export type IBLPrefilterWorkerTextureData =
	| Uint8ClampedArray
	| Float32Array
	| Uint8Array;

export interface IBLPrefilterWorkerEnvMapPayload {
	width: number;
	height: number;
	colorSpace: TextureColorSpace;
	data: IBLPrefilterWorkerTextureData | null;
}

export interface IBLPrefilterWorkerTaskPayload {
	type: "prefilter-mip";
	envMap: IBLPrefilterWorkerEnvMapPayload;
	baseWidth: number;
	baseHeight: number;
	maxMipLevels: number;
	level: number;
}

export interface IBLPrefilterWorkerTaskResult {
	type: "prefilter-mip";
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}
