import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
} from "../types";
import { BufferUsage } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	WEBGPU_FRAME_UNIFORM_FLOATS,
	packFrameUniformData,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "./";
import type { PreparedScene } from "../../pipeline/types";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";

export class WebGPUFrameBindingCache {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _frameUniformBuffer: IRenderBuffer | null = null;
	private _cache = new Map<IRenderPipeline, IBindingGroup>();
	private _lastDirectionalAtlas: IRenderTexture | null = null;
	private _lastSpotAtlas: IRenderTexture | null = null;

	constructor(
		backend: WebGPUBackend,
		layouts: WebGPUPipelineLayouts,
		textureRegistry: WebGPUTextureRegistry,
		shadowAtlases: WebGPUShadowAtlasAllocator
	) {
		this._backend = backend;
		this._layouts = layouts;
		this._textureRegistry = textureRegistry;
		this._shadowAtlases = shadowAtlases;
	}

	public prepare(
		frame: PreparedScene,
		lightingState: WebGPULightingState,
		features: WebGPUFeatureState
	): void {
		const frameUniform = this._getFrameUniformBuffer();
		const frameData = packFrameUniformData({
			viewProjectionMatrix: frame.camera.viewProjectionMatrix,
			cameraPosition: frame.camera.position,
			ambientColor: lightingState.ambientColor,
			directionalLights: lightingState.directionalLights,
			directionalShadows: lightingState.directionalShadows,
			pointLights: lightingState.pointLights,
			spotLights: lightingState.spotLights,
			spotShadows: lightingState.spotShadows,
			enableLighting: features.enableLighting,
			enableGamma: features.enableGamma,
			enableShadows: features.enableShadows,
		});

		this._backend.writeBuffer(frameUniform, new Float32Array(frameData));

		const currentDirectional = this._shadowAtlases.directionalAtlas;
		const currentSpot = this._shadowAtlases.spotAtlas;

		if (
			this._lastDirectionalAtlas !== currentDirectional ||
			this._lastSpotAtlas !== currentSpot
		) {
			this._cache.clear();
			this._lastDirectionalAtlas = currentDirectional;
			this._lastSpotAtlas = currentSpot;
		}
	}

	public getBinding(pipeline: IRenderPipeline): IBindingGroup {
		let cached = this._cache.get(pipeline);
		if (!cached) {
			cached = this._backend.createBindingGroup({
				label: `FrameBinding_${pipeline.label ?? "scene"}`,
				layout: this._layouts.frameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameUniformBuffer() },
					{
						binding: 1,
						resource:
							this._shadowAtlases.directionalAtlas ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 2,
						resource:
							this._shadowAtlases.spotAtlas ??
							this._textureRegistry.getWhiteTexture(),
					},
				],
			});
			this._cache.set(pipeline, cached);
		}
		return cached;
	}

	private _getFrameUniformBuffer(): IRenderBuffer {
		if (!this._frameUniformBuffer) {
			this._frameUniformBuffer = this._backend.createBuffer({
				size: WEBGPU_FRAME_UNIFORM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameUniforms",
			});
		}
		return this._frameUniformBuffer;
	}
}
