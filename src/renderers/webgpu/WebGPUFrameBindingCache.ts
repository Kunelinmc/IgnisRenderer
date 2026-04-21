import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderTexture,
	ISampler,
} from "../types";
import { BufferUsage } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	WEBGPU_FRAME_UNIFORM_FLOATS,
	packFrameUniformData,
	type WebGPUEnvironmentState,
	type WebGPUFeatureState,
	type WebGPULightingState,
} from "./";
import type { PreparedScene } from "../../pipeline/types";
import {
	DEFAULT_FOG_OPTIONS,
	type FogOptions,
} from "../../pipeline/types";
import { CameraType } from "../../cameras/Camera";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import {
	computeHaltonJitterNDC,
	finiteOr,
} from "../../maths/screenSpaceMath";
import { TAA_JITTER_SEQUENCE_LENGTH } from "../constants";
import type { WebGPUSceneTargetMode } from "./WebGPUPipelineLibrary";
import { clamp } from "../../maths/Common";

export class WebGPUFrameBindingCache {
	private _backend: WebGPUBackend;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _frameUniformBuffer: IRenderBuffer | null = null;
	private _fogUniformBuffer: IRenderBuffer | null = null;
	private _fogUniformData: Float32Array<ArrayBuffer> = new Float32Array(8);
	private _sceneBinding: IBindingGroup | null = null;
	private _skyboxBinding: IBindingGroup | null = null;
	private _shadowAtlas: IRenderTexture | null = null;
	private _skyboxTexture: IRenderTexture | null = null;
	private _envSpecularTexture: IRenderTexture | null = null;
	private _skyboxSampler: ISampler | null = null;
	private _envSpecularSampler: ISampler | null = null;
	private _prevViewProjection:
		| PreparedScene["camera"]["viewProjectionMatrix"]
		| null = null;
	private _taaFrameIndex = 0;
	private _taaJitterCurrent: [number, number] = [0, 0];
	private _taaEnabledLastFrame = false;

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
		environmentState: WebGPUEnvironmentState,
		features: WebGPUFeatureState,
		renderWidth: number,
		renderHeight: number,
		sceneTargetMode: WebGPUSceneTargetMode
	): void {
		const viewElements = frame.camera.viewMatrix.elements;
		const isOrthographic = frame.camera.type === CameraType.Orthographic;
		const fovRad = (frame.camera.fov * Math.PI) / 180;
		const tanHalfFov = isOrthographic ? 0 : Math.tan(fovRad * 0.5);
		const aspect = frame.camera.aspectRatio || 1;
		const frameUniform = this._getFrameUniformBuffer();
		const prevViewProjection =
			this._prevViewProjection ?? frame.camera.viewProjectionMatrix;
		const taaJitter = this._computeTAAJitter(
			features,
			isOrthographic,
			renderWidth,
			renderHeight
		);
		const frameData = packFrameUniformData({
			viewProjectionMatrix: frame.camera.viewProjectionMatrix,
			prevViewProjectionMatrix: prevViewProjection,
			cameraPosition: frame.camera.getWorldPosition(),
			skyboxRight: [viewElements[0][0], viewElements[0][1], viewElements[0][2]],
			skyboxUp: [viewElements[1][0], viewElements[1][1], viewElements[1][2]],
			skyboxBackward: [
				viewElements[2][0],
				viewElements[2][1],
				viewElements[2][2],
			],
			skyboxTanHalfFov: tanHalfFov,
			skyboxAspect: aspect,
			skyboxIsOrthographic: isOrthographic,
			ambientColor: lightingState.ambientColor,
			shAmbientCoeffs: environmentState.shAmbientCoeffs,
			directionalLights: lightingState.directionalLights,
			directionalShadows: lightingState.directionalShadows,
			pointLights: lightingState.pointLights,
			spotLights: lightingState.spotLights,
			spotShadows: lightingState.spotShadows,
			reflectionProbeCount: environmentState.reflectionProbeCount,
			reflectionProbes: environmentState.reflectionProbes,
			enableLighting: features.enableLighting,
			enableGamma: features.enableGamma,
			enableShadows: features.enableShadows,
			enableClusteredLighting: features.enableClusteredLighting,
			encodeGammaInShader:
				features.enableGamma && sceneTargetMode === "single",
			enableSH: environmentState.enableSH,
			hasSHAmbient: environmentState.hasSHAmbient,
			hasSkybox: !!environmentState.skyboxTexture,
			skyboxIsLinear:
				!environmentState.skyboxTexture ||
				environmentState.skyboxTexture.colorSpace !== "sRGB",
			hasEnvSpecular: !!environmentState.envSpecularTexture,
			hasBRDFLUT: !!environmentState.brdfLUTTexture,
			envSpecularMaxMipLevel: environmentState.envSpecularMaxMipLevel,
			taaJitterCurrentPrev: taaJitter,
		});

		this._backend.writeBuffer(frameUniform, new Float32Array(frameData));
		this._backend.writeBuffer(
			this._getFogUniformBuffer(),
			this._packFogUniformData(features)
		);
		this._prevViewProjection = frame.camera.viewProjectionMatrix.clone();

		const currentShadowAtlas = this._shadowAtlases.atlas;
		const currentSkybox =
			environmentState.skyboxTexture ?
				this._textureRegistry.getTextureForSlot(
					environmentState.skyboxTexture,
					0
				)
			:	this._textureRegistry.getWhiteTexture();
		const currentSkyboxSampler =
			environmentState.skyboxTexture ?
				this._textureRegistry.getSamplerForTexture(
					environmentState.skyboxTexture
				)
			:	this._textureRegistry.getWhiteSampler();
		const currentEnvSpecular =
			environmentState.envSpecularTexture ?
				this._textureRegistry.getTextureForSlot(
					environmentState.envSpecularTexture,
					0
				)
			:	this._textureRegistry.getWhiteTexture();
		const currentEnvSpecularSampler =
			environmentState.envSpecularTexture ?
				this._textureRegistry.getSamplerForTexture(
					environmentState.envSpecularTexture
				)
			:	this._textureRegistry.getWhiteSampler();

		if (
			this._shadowAtlas !== currentShadowAtlas ||
			this._skyboxTexture !== currentSkybox ||
			this._envSpecularTexture !== currentEnvSpecular ||
			this._skyboxSampler !== currentSkyboxSampler ||
			this._envSpecularSampler !== currentEnvSpecularSampler
		) {
			this._destroyBindingGroup(this._sceneBinding);
			this._destroyBindingGroup(this._skyboxBinding);
			this._sceneBinding = null;
			this._skyboxBinding = null;
			this._shadowAtlas = currentShadowAtlas;
			this._skyboxTexture = currentSkybox;
			this._envSpecularTexture = currentEnvSpecular;
			this._skyboxSampler = currentSkyboxSampler;
			this._envSpecularSampler = currentEnvSpecularSampler;
		}
	}

	private _computeTAAJitter(
		features: WebGPUFeatureState,
		isOrthographic: boolean,
		renderWidth: number,
		renderHeight: number
	): [number, number, number, number] {
		if (
			!features.enableTAA ||
			isOrthographic ||
			renderWidth <= 0 ||
			renderHeight <= 0
		) {
			this._taaJitterCurrent = [0, 0];
			this._taaFrameIndex = 0;
			this._taaEnabledLastFrame = false;
			return [0, 0, 0, 0];
		}

		const prevJitter =
			this._taaEnabledLastFrame ? this._taaJitterCurrent : [0, 0];
		const jitterScale =
			(
				typeof features.taaOptions?.jitterScale === "number" &&
				Number.isFinite(features.taaOptions.jitterScale)
			) ?
				Math.max(0, features.taaOptions.jitterScale)
			:	1;
		const nextJitter = computeHaltonJitterNDC(
			this._taaFrameIndex,
			renderWidth,
			renderHeight,
			jitterScale
		);
		this._taaJitterCurrent = nextJitter;
		this._taaFrameIndex =
			(this._taaFrameIndex + 1) % TAA_JITTER_SEQUENCE_LENGTH;
		this._taaEnabledLastFrame = true;
		return [nextJitter[0], nextJitter[1], prevJitter[0], prevJitter[1]];
	}

	public getSceneBinding(): IBindingGroup {
		if (!this._sceneBinding) {
			this._sceneBinding = this._backend.createBindingGroup({
				label: "FrameBinding_scene",
				layout: this._layouts.sceneFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameUniformBuffer() },
					{
						binding: 1,
						resource:
							this._shadowAtlas ??
							this._shadowAtlases.ensureAtlasForTileSize(1),
					},
					{
						binding: 2,
						resource:
							this._envSpecularTexture ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 3,
						resource:
							this._envSpecularSampler ??
							this._textureRegistry.getWhiteSampler(),
					},
					{ binding: 4, resource: this._getFogUniformBuffer() },
				],
			});
		}

		return this._sceneBinding;
	}

	public getSkyboxBinding(): IBindingGroup {
		if (!this._skyboxBinding) {
			this._skyboxBinding = this._backend.createBindingGroup({
				label: "FrameBinding_skybox",
				layout: this._layouts.skyboxFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameUniformBuffer() },
					{
						binding: 1,
						resource:
							this._skyboxTexture ?? this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 2,
						resource:
							this._skyboxSampler ?? this._textureRegistry.getWhiteSampler(),
					},
				],
			});
		}

		return this._skyboxBinding;
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

	private _getFogUniformBuffer(): IRenderBuffer {
		if (!this._fogUniformBuffer) {
			this._fogUniformBuffer = this._backend.createBuffer({
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFogUniforms",
			});
		}
		return this._fogUniformBuffer;
	}

	private _packFogUniformData(
		features: WebGPUFeatureState
	): Float32Array<ArrayBuffer> {
		const source = features.fogOptions ?? DEFAULT_FOG_OPTIONS;
		const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
		const start = Math.max(
			0,
			finiteOr(source.start, DEFAULT_FOG_OPTIONS.start)
		);
		const end = Math.max(
			start + 1e-4,
			finiteOr(source.end, DEFAULT_FOG_OPTIONS.end)
		);
		const density = Math.max(
			0,
			finiteOr(source.density, DEFAULT_FOG_OPTIONS.density)
		);
		const sceneFogEnabled =
			features.enableFog &&
			(source.application ?? DEFAULT_FOG_OPTIONS.application) === "scene";
		const strength = sceneFogEnabled ?
			Math.max(0, finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength))
		:	0;
		const data = this._fogUniformData;
		data[0] = this._resolveFogMode(source.mode);
		data[1] = start;
		data[2] = end;
		data[3] = density;
		data[4] = clamp(finiteOr(color[0], DEFAULT_FOG_OPTIONS.color[0]), 0, 1);
		data[5] = clamp(finiteOr(color[1], DEFAULT_FOG_OPTIONS.color[1]), 0, 1);
		data[6] = clamp(finiteOr(color[2], DEFAULT_FOG_OPTIONS.color[2]), 0, 1);
		data[7] = strength;
		return data;
	}

	private _resolveFogMode(mode: FogOptions["mode"] | undefined): number {
		switch (mode) {
			case "exp":
				return 1;
			case "exp2":
				return 2;
			default:
				return 0;
		}
	}

	public destroy(): void {
		this._destroyBindingGroup(this._sceneBinding);
		this._destroyBindingGroup(this._skyboxBinding);
		this._sceneBinding = null;
		this._skyboxBinding = null;
		this._frameUniformBuffer?.destroy();
		this._frameUniformBuffer = null;
		this._fogUniformBuffer?.destroy();
		this._fogUniformBuffer = null;
		this._shadowAtlas = null;
		this._skyboxTexture = null;
		this._envSpecularTexture = null;
		this._skyboxSampler = null;
		this._envSpecularSampler = null;
		this._prevViewProjection = null;
		this._taaFrameIndex = 0;
		this._taaJitterCurrent = [0, 0];
		this._taaEnabledLastFrame = false;
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}
}
