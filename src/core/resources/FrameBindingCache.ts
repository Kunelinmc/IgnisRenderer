import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderPipeline,
} from "../ral/types";
import { BufferUsage } from "../ral/types";
import type { WebGPUBackend } from "../backend/WebGPUBackend";
import {
	WEBGPU_FRAME_UNIFORM_FLOATS,
	packFrameUniformData,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "../bridge/webgpu";
import type { PreparedScene } from "../pipeline/types";
import { TextureRegistry } from "./TextureRegistry";
import { ShadowAtlasAllocator } from "./ShadowAtlasAllocator";
import type { WebGPUPipelineLayouts } from "../backend/webgpu/WebGPUPipelineLayouts";

export class FrameBindingCache {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: TextureRegistry;
	private _shadowAtlases: ShadowAtlasAllocator;
	private _frameUniformBuffer: IRenderBuffer | null = null;
	private _cache = new Map<IRenderPipeline, IBindingGroup>();

	constructor(
		backend: WebGPUBackend,
		layouts: WebGPUPipelineLayouts,
		textureRegistry: TextureRegistry,
		shadowAtlases: ShadowAtlasAllocator
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
		this._cache.clear();
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
