import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderTexture,
	ISampler,
} from "../types";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	WEBGPU_SH_COEFFICIENT_COUNT,
	packFrameCameraUniformData,
	packFrameEnvironmentUniformData,
	packFrameLightUniformData,
	packFrameShadowUniformData,
	type WebGPUEnvironmentState,
	type WebGPUFeatureState,
	type WebGPUFrameUniformInput,
	type WebGPULightingState,
} from "./";
import type {
	FrameContext,
	ParticleRenderBatch,
	PreparedScene,
} from "../../pipeline/types";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import {
	DEFAULT_FOG_OPTIONS,
	FOG_PASS_ID,
	TEMPORAL_ANTI_ALIASING_PASS_ID,
	type FogOptions,
	type TAAOptions,
} from "../../postprocess";
import { CameraType } from "../../cameras/Camera";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type { WebGPUPagedShadowRuntime } from "./WebGPUPagedShadowRuntime";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import { finiteOr } from "../../maths/Misc";
import { TemporalJitterState } from "../cross/TemporalJitterState";
import type { WebGPUSceneTargetMode } from "./WebGPUPipelineLibrary";
import { clamp } from "../../maths/Common";
import type { Matrix4 } from "../../maths/Matrix4";
import type { ShadowMap } from "../../lights/shadows/ShadowMapping";
import {
	createParticleShadowVolumeGrid,
	hasParticleShadowCastingBatches,
	injectParticleBatchIntoShadowVolume,
} from "../../pipeline/ParticleShadowVolume";

const PARTICLE_SHADOW_VOLUME_MAX_SLICES = 4;
const PARTICLE_SHADOW_VOLUME_META_FLOATS = 24;
const PARTICLE_SHADOW_VOLUME_HEADER_FLOATS =
	PARTICLE_SHADOW_VOLUME_MAX_SLICES * PARTICLE_SHADOW_VOLUME_META_FLOATS;
const PARTICLE_SHADOW_VOLUME_DENSITY_FLOATS = 64 * 64 * 32;
const PARTICLE_SHADOW_VOLUME_FALLBACK_FLOATS =
	PARTICLE_SHADOW_VOLUME_HEADER_FLOATS + 1;

interface PreparedSceneEnvironmentLike {
	backgroundEnabled: boolean;
	lightingEnabled: boolean;
	backgroundTexture: unknown;
	iblTexture: unknown;
	backgroundStrength: number;
	backgroundTintLinear: { r: number; g: number; b: number };
	backgroundExposure: number;
}

export type WebGPUTemporalStateMode = "advance" | "reuse" | "disabled";

export interface WebGPUFrameBindingPrepareOptions {
	readonly temporalStateMode?: WebGPUTemporalStateMode;
	readonly temporalHistoryReset?: boolean;
}

export class WebGPUFrameBindingCache {
	private _backend: WebGPUDeviceResourceHost;
	private _resourceManager: WebGPUResourceManager;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _pagedShadowRuntime: WebGPUPagedShadowRuntime;
	private _frameCameraUniformBuffer: IRenderBuffer | null = null;
	private _frameLightUniformBuffer: IRenderBuffer | null = null;
	private _frameShadowUniformBuffer: IRenderBuffer | null = null;
	private _frameEnvironmentUniformBuffer: IRenderBuffer | null = null;
	private _fogUniformBuffer: IRenderBuffer | null = null;
	private _environmentBackgroundParamsBuffer: IRenderBuffer | null = null;
	private _environmentBackgroundParamsData: Float32Array<ArrayBuffer> =
		new Float32Array(4);
	private _particleShadowVolumeBuffer: IRenderBuffer | null = null;
	private _particleShadowVolumeBufferSize = 0;
	private _fogUniformData: Float32Array<ArrayBuffer> = new Float32Array(8);
	private _sceneBinding: IBindingGroup | null = null;
	private _decalFrameBinding: IBindingGroup | null = null;
	private _environmentBinding: IBindingGroup | null = null;
	private _shadowAtlas: IRenderTexture | null = null;
	private _shadowTransmittanceAtlas: IRenderTexture | null = null;
	private _environmentTexture: IRenderTexture | null = null;
	private _envSpecularTexture: IRenderTexture | null = null;
	private _envSpecularFallbackTexture: IRenderTexture | null = null;
	private _brdfLUTTexture: IRenderTexture | null = null;
	private _irradianceProbeGridTexture: IRenderTexture | null = null;
	private _pagedShadowPageTableTexture: IRenderTexture | null = null;
	private _pagedShadowPhysicalDepthAtlas: IRenderTexture | null = null;
	private _ownedIrradianceProbeGridTexture: IRenderTexture | null = null;
	private _irradianceProbeGridTextureRevision = -1;
	private _irradianceProbeGridTextureCellCount = 0;
	private _irradianceProbeGridTextureGridId: string | null = null;
	private _irradianceProbeGridTextureData = new Float32Array(
		WEBGPU_SH_COEFFICIENT_COUNT * 4
	);
	private _environmentSampler: ISampler | null = null;
	private _envSpecularSampler: ISampler | null = null;
	private _envSpecularFallbackSampler: ISampler | null = null;
	private _shadowComparisonSampler: ISampler | null = null;
	private _prevViewProjection:
		| PreparedScene["camera"]["viewProjectionMatrix"]
		| null = null;
	private _currentFramePrevViewProjection:
		| PreparedScene["camera"]["viewProjectionMatrix"]
		| null = null;
	private _temporalJitterState = new TemporalJitterState();
	private _temporalJitterCurrentPrev: [number, number, number, number] = [0, 0, 0, 0];

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		layouts: WebGPUPipelineLayouts,
		textureRegistry: WebGPUTextureRegistry,
		shadowAtlases: WebGPUShadowAtlasAllocator,
		pagedShadowRuntime: WebGPUPagedShadowRuntime
	) {
		this._backend = backend;
		this._resourceManager = resourceManager;
		this._layouts = layouts;
		this._textureRegistry = textureRegistry;
		this._shadowAtlases = shadowAtlases;
		this._pagedShadowRuntime = pagedShadowRuntime;
	}

	public prepare(
		frame: PreparedScene,
		lightingState: WebGPULightingState,
		environmentState: WebGPUEnvironmentState,
		features: WebGPUFeatureState,
		renderWidth: number,
		renderHeight: number,
		sceneTargetMode: WebGPUSceneTargetMode,
		options: WebGPUFrameBindingPrepareOptions = {}
	): void {
		const viewElements = frame.camera.viewMatrix.elements;
		const isOrthographic = frame.camera.type === CameraType.Orthographic;
		const fovRad = (frame.camera.fov * Math.PI) / 180;
		let environmentProjectionX = isOrthographic ? 0 : Math.tan(fovRad * 0.5);
		const aspect = frame.camera.aspectRatio || 1;
		let environmentProjectionY = aspect;
		if (isOrthographic) {
			const orthographicCamera = frame.camera as unknown as {
				getBounds?: () => {
					left: number;
					right: number;
					bottom: number;
					top: number;
				};
				size?: number;
			};
			const bounds =
				typeof orthographicCamera.getBounds === "function" ?
					orthographicCamera.getBounds()
				:	{
						left: -((orthographicCamera.size ?? 100) * aspect) * 0.5,
						right: ((orthographicCamera.size ?? 100) * aspect) * 0.5,
						bottom: -((orthographicCamera.size ?? 100) * 0.5),
						top: (orthographicCamera.size ?? 100) * 0.5,
					};
			environmentProjectionX = Math.max(
				1e-6,
				Math.abs(bounds.right - bounds.left) * 0.5
			);
			environmentProjectionY = Math.max(
				1e-6,
				Math.abs(bounds.top - bounds.bottom) * 0.5
			);
		}
		const temporalStateMode = options.temporalStateMode ?? "advance";
		const prevViewProjection = this._resolvePreviousViewProjection(
			frame,
			temporalStateMode,
			options.temporalHistoryReset === true
		);
		const temporalJitter = this._computeTemporalJitter(
			features,
			isOrthographic,
			renderWidth,
			renderHeight,
			temporalStateMode,
			options.temporalHistoryReset === true
		);
		const frameUniformInput: WebGPUFrameUniformInput = {
			viewProjectionMatrix: frame.camera.viewProjectionMatrix,
			prevViewProjectionMatrix: prevViewProjection,
			cameraPosition: frame.camera.getWorldPosition(),
			environmentRight: [viewElements[0][0], viewElements[0][1], viewElements[0][2]],
			environmentUp: [viewElements[1][0], viewElements[1][1], viewElements[1][2]],
			environmentBackward: [
				viewElements[2][0],
				viewElements[2][1],
				viewElements[2][2],
			],
			environmentTanHalfFov: environmentProjectionX,
			environmentAspect: environmentProjectionY,
			environmentIsOrthographic: isOrthographic,
			ambientColor: lightingState.ambientColor,
			shAmbientCoeffs: environmentState.shAmbientCoeffs,
			localLightProbeCount: environmentState.localLightProbeCount,
			localLightProbes: environmentState.localLightProbes,
			irradianceProbeGrid: environmentState.irradianceProbeGrid,
			directionalLights: lightingState.directionalLights,
			directionalShadows: lightingState.directionalShadows,
			pointLights: lightingState.pointLights,
			spotLights: lightingState.spotLights,
			spotShadows: lightingState.spotShadows,
			areaLights: lightingState.areaLights,
			reflectionProbeCount: environmentState.reflectionProbeCount,
			reflectionProbes: environmentState.reflectionProbes,
			enableLighting: features.enableLighting,
			enableShadows: features.enableShadows,
			enableClusteredLighting: features.enableClusteredLighting,
			enableSH: environmentState.enableSH,
			hasSHAmbient: environmentState.hasSHAmbient,
			environmentIsLinear:
				!environmentState.environmentTexture ||
				environmentState.environmentTexture.colorSpace !== "sRGB",
			hasEnvSpecular: !!environmentState.envSpecularTexture,
			hasEnvSpecularFallback: !!environmentState.envSpecularFallbackTexture,
			hasBRDFLUT: !!environmentState.brdfLUTTexture,
			envSpecularMaxMipLevel: environmentState.envSpecularMaxMipLevel,
			envSpecularFallbackMaxMipLevel:
				environmentState.envSpecularFallbackMaxMipLevel,
			taaJitterCurrentPrev: temporalJitter,
		};

		this._backend.writeBuffer(
			this._getFrameCameraUniformBuffer(),
			packFrameCameraUniformData(frameUniformInput) as Float32Array<ArrayBuffer>
		);
		this._backend.writeBuffer(
			this._getFrameLightUniformBuffer(),
			packFrameLightUniformData(frameUniformInput) as Float32Array<ArrayBuffer>
		);
		this._backend.writeBuffer(
			this._getFrameShadowUniformBuffer(),
			packFrameShadowUniformData(frameUniformInput) as Float32Array<ArrayBuffer>
		);
		this._backend.writeBuffer(
			this._getFrameEnvironmentUniformBuffer(),
			packFrameEnvironmentUniformData(frameUniformInput) as Float32Array<ArrayBuffer>
		);
		this._backend.writeBuffer(
			this._getFogUniformBuffer(),
			this._packFogUniformData(features)
		);
		this._backend.writeBuffer(
			this._getEnvironmentBackgroundParamsBuffer(),
			this._packEnvironmentBackgroundParams(frame)
		);
		if (temporalStateMode === "advance") {
			this._prevViewProjection = frame.camera.viewProjectionMatrix.clone();
		}
		this._writeParticleShadowVolumeData(
			new Float32Array(PARTICLE_SHADOW_VOLUME_FALLBACK_FLOATS)
		);

		const currentShadowAtlas = this._shadowAtlases.atlas;
		const currentShadowTransmittanceAtlas =
			this._shadowAtlases.transmittanceAtlas;
		const currentEnvironment =
			environmentState.environmentTexture ?
				this._textureRegistry.getTextureForSlot(
					environmentState.environmentTexture,
					0
				)
			:	this._textureRegistry.getWhiteTexture();
		const currentEnvironmentSampler =
			environmentState.environmentTexture ?
				this._textureRegistry.getSamplerForTexture(
					environmentState.environmentTexture
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
		const currentEnvSpecularFallback =
			environmentState.envSpecularFallbackTexture ?
				this._textureRegistry.getTextureForSlot(
					environmentState.envSpecularFallbackTexture,
					0
				)
			:	this._textureRegistry.getWhiteTexture();
		const currentEnvSpecularFallbackSampler =
			environmentState.envSpecularFallbackTexture ?
				this._textureRegistry.getSamplerForTexture(
					environmentState.envSpecularFallbackTexture
				)
			:	this._textureRegistry.getWhiteSampler();
		const currentBRDFLUT =
			environmentState.brdfLUTTexture ?
				this._textureRegistry.getTextureForSlot(
					environmentState.brdfLUTTexture,
					0
				)
			:	this._textureRegistry.getWhiteTexture();
		const currentIrradianceProbeGrid =
			this._getIrradianceProbeGridTexture(environmentState);
		const pagedShadowResources = this._pagedShadowRuntime.getSamplingResources();
		const currentPagedShadowPageTableTexture = pagedShadowResources.pageTableTexture;
		const currentPagedShadowPhysicalDepthAtlas =
			pagedShadowResources.physicalDepthAtlas;

		if (
			this._shadowAtlas !== currentShadowAtlas ||
			this._shadowTransmittanceAtlas !== currentShadowTransmittanceAtlas ||
			this._environmentTexture !== currentEnvironment ||
			this._envSpecularTexture !== currentEnvSpecular ||
			this._envSpecularFallbackTexture !== currentEnvSpecularFallback ||
			this._brdfLUTTexture !== currentBRDFLUT ||
			this._irradianceProbeGridTexture !== currentIrradianceProbeGrid ||
			this._pagedShadowPageTableTexture !== currentPagedShadowPageTableTexture ||
			this._pagedShadowPhysicalDepthAtlas !== currentPagedShadowPhysicalDepthAtlas ||
			this._environmentSampler !== currentEnvironmentSampler ||
			this._envSpecularSampler !== currentEnvSpecularSampler ||
			this._envSpecularFallbackSampler !== currentEnvSpecularFallbackSampler
		) {
			this._destroyBindingGroup(this._sceneBinding);
			this._destroyBindingGroup(this._environmentBinding);
			this._sceneBinding = null;
			this._environmentBinding = null;
			this._shadowAtlas = currentShadowAtlas;
			this._shadowTransmittanceAtlas = currentShadowTransmittanceAtlas;
			this._environmentTexture = currentEnvironment;
			this._envSpecularTexture = currentEnvSpecular;
			this._envSpecularFallbackTexture = currentEnvSpecularFallback;
			this._brdfLUTTexture = currentBRDFLUT;
			this._irradianceProbeGridTexture = currentIrradianceProbeGrid;
			this._pagedShadowPageTableTexture = currentPagedShadowPageTableTexture;
			this._pagedShadowPhysicalDepthAtlas =
				currentPagedShadowPhysicalDepthAtlas;
			this._environmentSampler = currentEnvironmentSampler;
			this._envSpecularSampler = currentEnvSpecularSampler;
			this._envSpecularFallbackSampler = currentEnvSpecularFallbackSampler;
		}
	}

	private _getIrradianceProbeGridTexture(
		environmentState: WebGPUEnvironmentState
	): IRenderTexture {
		const grid = environmentState.irradianceProbeGrid;
		if (!grid || grid.cellCount <= 0) {
			return this._textureRegistry.getWhiteTexture();
		}

		const width = WEBGPU_SH_COEFFICIENT_COUNT;
		const height = Math.max(1, Math.floor(grid.cellCount));
		if (
			!this._ownedIrradianceProbeGridTexture ||
			this._ownedIrradianceProbeGridTexture.width !== width ||
			this._ownedIrradianceProbeGridTexture.height !== height
		) {
			this._ownedIrradianceProbeGridTexture?.destroy();
			this._ownedIrradianceProbeGridTexture = this._backend.createTexture({
				width,
				height,
				format: TextureFormat.RGBA32Float,
				usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
				label: `WebGPUIrradianceProbeGridSH_${height}`,
			});
			this._irradianceProbeGridTextureRevision = -1;
			this._irradianceProbeGridTextureCellCount = 0;
			this._irradianceProbeGridTextureGridId = null;
		}

		if (
			this._irradianceProbeGridTextureRevision !== grid.textureRevision ||
			this._irradianceProbeGridTextureCellCount !== height ||
			this._irradianceProbeGridTextureGridId !== grid.id
		) {
			const data = this._packIrradianceProbeGridTextureData(grid);
			this._resourceManager.writeTexture(
				this._ownedIrradianceProbeGridTexture,
				data as Float32Array<ArrayBuffer>,
				{ bytesPerRow: width * 4 * 4, rowsPerImage: height },
				{ width, height, depthOrArrayLayers: 1 }
			);
			this._irradianceProbeGridTextureRevision = grid.textureRevision;
			this._irradianceProbeGridTextureCellCount = height;
			this._irradianceProbeGridTextureGridId = grid.id;
		}

		return this._ownedIrradianceProbeGridTexture;
	}

	private _packIrradianceProbeGridTextureData(
		grid: NonNullable<WebGPUEnvironmentState["irradianceProbeGrid"]>
	): Float32Array {
		const width = WEBGPU_SH_COEFFICIENT_COUNT;
		const height = Math.max(1, Math.floor(grid.cellCount));
		const requiredLength = width * height * 4;
		if (this._irradianceProbeGridTextureData.length !== requiredLength) {
			this._irradianceProbeGridTextureData = new Float32Array(requiredLength);
		}
		const data = this._irradianceProbeGridTextureData;
		data.fill(0);
		for (let cellIndex = 0; cellIndex < height; cellIndex++) {
			const valid = grid.validMask[cellIndex] ? 1 : 0;
			const cellSH = grid.sh[cellIndex];
			for (let coeffIndex = 0; coeffIndex < width; coeffIndex++) {
				const coeff = cellSH?.[coeffIndex];
				const base = (cellIndex * width + coeffIndex) * 4;
				data[base] = finiteOr(coeff?.r, 0);
				data[base + 1] = finiteOr(coeff?.g, 0);
				data[base + 2] = finiteOr(coeff?.b, 0);
				data[base + 3] = valid;
			}
		}
		return data;
	}

	private _computeTemporalJitter(
		features: WebGPUFeatureState,
		isOrthographic: boolean,
		renderWidth: number,
		renderHeight: number,
		temporalStateMode: WebGPUTemporalStateMode,
		temporalHistoryReset: boolean
	): [number, number, number, number] {
		if (temporalStateMode === "disabled") {
			return [0, 0, 0, 0];
		}
		if (temporalStateMode === "reuse") {
			return this._temporalJitterCurrentPrev;
		}

		const jitter = this._temporalJitterState.next({
			enabled: features.postProcess.isEnabled(TEMPORAL_ANTI_ALIASING_PASS_ID),
			isOrthographic,
			width: renderWidth,
			height: renderHeight,
			jitterScale: features.postProcess
				.getOptions<TAAOptions>(TEMPORAL_ANTI_ALIASING_PASS_ID)
				?.jitterScale,
			reset: temporalHistoryReset,
		});
		this._temporalJitterCurrentPrev = jitter;
		return jitter;
	}

	private _resolvePreviousViewProjection(
		frame: PreparedScene,
		temporalStateMode: WebGPUTemporalStateMode,
		temporalHistoryReset: boolean
	): PreparedScene["camera"]["viewProjectionMatrix"] {
		if (temporalStateMode === "disabled" || temporalHistoryReset) {
			if (temporalStateMode === "advance") {
				this._currentFramePrevViewProjection =
					frame.camera.viewProjectionMatrix.clone();
			}
			return frame.camera.viewProjectionMatrix;
		}
		if (temporalStateMode === "reuse") {
			return (
				this._currentFramePrevViewProjection ??
				this._prevViewProjection ??
				frame.camera.viewProjectionMatrix
			);
		}

		const prevViewProjection =
			this._prevViewProjection ?? frame.camera.viewProjectionMatrix;
		this._currentFramePrevViewProjection = prevViewProjection.clone();
		return prevViewProjection;
	}

	public getSceneBinding(): IBindingGroup {
		if (!this._sceneBinding) {
			this._sceneBinding = this._backend.createBindingGroup({
				label: "FrameBinding_scene",
				layout: this._layouts.sceneFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameCameraUniformBuffer() },
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
					{
						binding: 4,
						resource:
							this._envSpecularFallbackTexture ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 5,
						resource:
							this._envSpecularFallbackSampler ??
							this._textureRegistry.getWhiteSampler(),
					},
					{ binding: 6, resource: this._getFogUniformBuffer() },
					{ binding: 7, resource: this._getParticleShadowVolumeBuffer() },
					{
						binding: 8,
						resource:
							this._shadowTransmittanceAtlas ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 9,
						resource:
							this._brdfLUTTexture ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 10,
						resource:
							this._irradianceProbeGridTexture ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 11,
						resource:
							this._pagedShadowPageTableTexture ??
							this._pagedShadowRuntime.getSamplingResources().pageTableTexture,
					},
					{
						binding: 12,
						resource:
							this._pagedShadowPhysicalDepthAtlas ??
							this._pagedShadowRuntime.getSamplingResources().physicalDepthAtlas,
					},
					{
						binding: 13,
						resource: this._getShadowComparisonSampler(),
					},
					{ binding: 14, resource: this._getFrameLightUniformBuffer() },
					{ binding: 15, resource: this._getFrameShadowUniformBuffer() },
					{
						binding: 16,
						resource: this._getFrameEnvironmentUniformBuffer(),
					},
				],
			});
		}

		return this._sceneBinding;
	}

	private _getShadowComparisonSampler(): ISampler {
		if (!this._shadowComparisonSampler) {
			this._shadowComparisonSampler = this._backend.createSampler({
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				compare: "less-equal",
				label: "WebGPUShadowComparisonSampler",
			});
		}

		return this._shadowComparisonSampler;
	}

	/**
	 * Returns a frame binding group for decal shaders that only require frame uniforms.
	 *
	 * @returns A binding group compatible with `WebGPUDecalFrameBindGroupLayout`.
	 * @sideEffects Lazily creates the binding group while reusing the shared frame uniform buffer.
	 */
	public getDecalFrameBinding(): IBindingGroup {
		if (!this._decalFrameBinding) {
			this._decalFrameBinding = this._backend.createBindingGroup({
				label: "FrameBinding_decal",
				layout: this._layouts.decalFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameCameraUniformBuffer() },
				],
			});
		}

		return this._decalFrameBinding;
	}

	public getEnvironmentBinding(): IBindingGroup {
		if (!this._environmentBinding) {
			this._environmentBinding = this._backend.createBindingGroup({
				label: "FrameBinding_environment",
				layout: this._layouts.environmentFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameCameraUniformBuffer() },
					{
						binding: 1,
						resource:
							this._environmentTexture ?? this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 2,
						resource:
							this._environmentSampler ?? this._textureRegistry.getWhiteSampler(),
					},
					{
						binding: 3,
						resource: this._getEnvironmentBackgroundParamsBuffer(),
					},
				],
			});
		}

		return this._environmentBinding;
	}

	private _getFrameCameraUniformBuffer(): IRenderBuffer {
		if (!this._frameCameraUniformBuffer) {
			this._frameCameraUniformBuffer = this._backend.createBuffer({
				size: WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameCameraUniforms",
			});
		}
		return this._frameCameraUniformBuffer;
	}

	private _getFrameLightUniformBuffer(): IRenderBuffer {
		if (!this._frameLightUniformBuffer) {
			this._frameLightUniformBuffer = this._backend.createBuffer({
				size: WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameLightUniforms",
			});
		}
		return this._frameLightUniformBuffer;
	}

	private _getFrameShadowUniformBuffer(): IRenderBuffer {
		if (!this._frameShadowUniformBuffer) {
			this._frameShadowUniformBuffer = this._backend.createBuffer({
				size: WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameShadowUniforms",
			});
		}
		return this._frameShadowUniformBuffer;
	}

	private _getFrameEnvironmentUniformBuffer(): IRenderBuffer {
		if (!this._frameEnvironmentUniformBuffer) {
			this._frameEnvironmentUniformBuffer = this._backend.createBuffer({
				size: WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUFrameEnvironmentUniforms",
			});
		}
		return this._frameEnvironmentUniformBuffer;
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

	private _getEnvironmentBackgroundParamsBuffer(): IRenderBuffer {
		if (!this._environmentBackgroundParamsBuffer) {
			this._environmentBackgroundParamsBuffer = this._backend.createBuffer({
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: "WebGPUEnvironmentBackgroundParams",
			});
		}
		return this._environmentBackgroundParamsBuffer;
	}

	public updateParticleShadowVolumes(
		context: FrameContext,
		lightingState: WebGPULightingState
	): void {
		const data = this._packParticleShadowVolumeData(context, lightingState);
		this._writeParticleShadowVolumeData(data);
	}

	private _getParticleShadowVolumeBuffer(requiredByteSize = 0): IRenderBuffer {
		const byteSize = Math.max(
			PARTICLE_SHADOW_VOLUME_FALLBACK_FLOATS * 4,
			Math.ceil(requiredByteSize / 4) * 4
		);
		if (
			!this._particleShadowVolumeBuffer ||
			this._particleShadowVolumeBufferSize < byteSize
		) {
			this._particleShadowVolumeBuffer?.destroy();
			this._particleShadowVolumeBuffer = this._backend.createBuffer({
				size: byteSize,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: "WebGPUParticleShadowVolumeBuffer",
			});
			this._particleShadowVolumeBufferSize = byteSize;
			this._destroyBindingGroup(this._sceneBinding);
			this._sceneBinding = null;
		}
		return this._particleShadowVolumeBuffer;
	}

	private _writeParticleShadowVolumeData(data: Float32Array): void {
		const buffer = this._getParticleShadowVolumeBuffer(data.byteLength);
		this._backend.writeBuffer(buffer, data as Float32Array<ArrayBuffer>);
	}

	private _packParticleShadowVolumeData(
		context: FrameContext,
		lightingState: WebGPULightingState
	): Float32Array {
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| readonly ParticleRenderBatch[]
			| undefined;
		const directionalShadow = lightingState.directionalShadows[0];
		if (
			!context.features.enableShadows ||
			!directionalShadow?.enabled ||
			!hasParticleShadowCastingBatches(batches)
		) {
			return new Float32Array(PARTICLE_SHADOW_VOLUME_FALLBACK_FLOATS);
		}

		const cascadeCount =
			directionalShadow.strategyType === "csm" ?
				Math.max(1, Math.min(4, directionalShadow.cascadeCount | 0))
			:	1;
		const matrices = directionalShadow.strategyType === "csm" ?
			directionalShadow.cascadeViewProjectionMatrices
		:	[directionalShadow.viewProjectionMatrix];
		const densityOffsetStart = PARTICLE_SHADOW_VOLUME_HEADER_FLOATS;
		const data = new Float32Array(
			densityOffsetStart +
				PARTICLE_SHADOW_VOLUME_DENSITY_FLOATS *
					PARTICLE_SHADOW_VOLUME_MAX_SLICES
		);

		for (
			let sliceIndex = 0;
			sliceIndex < Math.min(PARTICLE_SHADOW_VOLUME_MAX_SLICES, cascadeCount);
			sliceIndex++
		) {
			const matrix = matrices[sliceIndex];
			if (!matrix) {
				continue;
			}
			const grid = createParticleShadowVolumeGrid();
			const shadowMap = {
				viewProjectionMatrix: matrix,
			} as ShadowMap;
			for (const batch of batches ?? []) {
				injectParticleBatchIntoShadowVolume(grid, shadowMap, batch);
			}
			if (!grid.active) {
				continue;
			}

			const metaOffset = sliceIndex * PARTICLE_SHADOW_VOLUME_META_FLOATS;
			writeParticleShadowVolumeMatrix(data, metaOffset, matrix);
			const densityOffset =
				densityOffsetStart +
				sliceIndex * PARTICLE_SHADOW_VOLUME_DENSITY_FLOATS;
			data[metaOffset + 16] = 1;
			data[metaOffset + 17] = grid.resolution.width;
			data[metaOffset + 18] = grid.resolution.height;
			data[metaOffset + 19] = grid.resolution.depth;
			data[metaOffset + 20] = densityOffset;
			data.set(grid.density, densityOffset);
		}

		return data;
	}

	private _packFogUniformData(
		features: WebGPUFeatureState
	): Float32Array<ArrayBuffer> {
		const source =
			features.postProcess.getOptions<FogOptions>(FOG_PASS_ID) ??
			DEFAULT_FOG_OPTIONS;
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
			features.postProcess.isEnabled(FOG_PASS_ID) &&
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

	private _packEnvironmentBackgroundParams(
		frame: PreparedScene
	): Float32Array<ArrayBuffer> {
		const environment = resolvePreparedSceneEnvironment(frame);
		const data = this._environmentBackgroundParamsData;
		data[0] = clamp(environment.backgroundTintLinear.r, 0, 1);
		data[1] = clamp(environment.backgroundTintLinear.g, 0, 1);
		data[2] = clamp(environment.backgroundTintLinear.b, 0, 1);
		data[3] =
			Math.max(1e-6, environment.backgroundExposure) *
			Math.max(0, environment.backgroundStrength);
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
		this._destroyBindingGroup(this._decalFrameBinding);
		this._destroyBindingGroup(this._environmentBinding);
		this._sceneBinding = null;
		this._decalFrameBinding = null;
		this._environmentBinding = null;
		this._frameCameraUniformBuffer?.destroy();
		this._frameCameraUniformBuffer = null;
		this._frameLightUniformBuffer?.destroy();
		this._frameLightUniformBuffer = null;
		this._frameShadowUniformBuffer?.destroy();
		this._frameShadowUniformBuffer = null;
		this._frameEnvironmentUniformBuffer?.destroy();
		this._frameEnvironmentUniformBuffer = null;
		this._fogUniformBuffer?.destroy();
		this._fogUniformBuffer = null;
		this._environmentBackgroundParamsBuffer?.destroy();
		this._environmentBackgroundParamsBuffer = null;
		this._particleShadowVolumeBuffer?.destroy();
		this._particleShadowVolumeBuffer = null;
		this._particleShadowVolumeBufferSize = 0;
		this._ownedIrradianceProbeGridTexture?.destroy();
		this._ownedIrradianceProbeGridTexture = null;
		this._irradianceProbeGridTexture = null;
		this._irradianceProbeGridTextureRevision = -1;
		this._irradianceProbeGridTextureCellCount = 0;
		this._irradianceProbeGridTextureGridId = null;
		this._shadowAtlas = null;
		this._shadowTransmittanceAtlas = null;
		this._environmentTexture = null;
		this._envSpecularTexture = null;
		this._envSpecularFallbackTexture = null;
		this._brdfLUTTexture = null;
		this._destroySampler(this._shadowComparisonSampler);
		this._shadowComparisonSampler = null;
		this._environmentSampler = null;
		this._envSpecularSampler = null;
		this._envSpecularFallbackSampler = null;
		this._prevViewProjection = null;
		this._currentFramePrevViewProjection = null;
		this._temporalJitterState.reset();
		this._temporalJitterCurrentPrev = [0, 0, 0, 0];
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _destroySampler(sampler: ISampler | null): void {
		const destroyFn = (sampler as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(sampler);
		}
	}
}

function resolvePreparedSceneEnvironment(
	scene: PreparedScene
): PreparedSceneEnvironmentLike {
	const rawEnvironment = (scene as { environment?: unknown }).environment;
	if (!rawEnvironment || typeof rawEnvironment !== "object") {
		return {
			backgroundEnabled: true,
			lightingEnabled: true,
			backgroundTexture: null,
			iblTexture: null,
			backgroundStrength: 1,
			backgroundTintLinear: { r: 1, g: 1, b: 1 },
			backgroundExposure: 1,
		};
	}
	const environment = rawEnvironment as Partial<PreparedSceneEnvironmentLike>;
	return {
		backgroundEnabled: environment.backgroundEnabled ?? true,
		lightingEnabled: environment.lightingEnabled ?? true,
		backgroundTexture: environment.backgroundTexture ?? null,
		iblTexture: environment.iblTexture ?? null,
		backgroundStrength:
			typeof environment.backgroundStrength === "number" ?
				environment.backgroundStrength
			:	1,
		backgroundTintLinear: {
			r:
				typeof environment.backgroundTintLinear?.r === "number" ?
					environment.backgroundTintLinear.r
				:	1,
			g:
				typeof environment.backgroundTintLinear?.g === "number" ?
					environment.backgroundTintLinear.g
				:	1,
			b:
				typeof environment.backgroundTintLinear?.b === "number" ?
					environment.backgroundTintLinear.b
				:	1,
		},
		backgroundExposure:
			typeof environment.backgroundExposure === "number" ?
				environment.backgroundExposure
			:	1,
	};
}

function writeParticleShadowVolumeMatrix(
	target: Float32Array,
	offset: number,
	matrix: Matrix4
): void {
	const rows = matrix.elements;
	for (let column = 0; column < 4; column++) {
		for (let row = 0; row < 4; row++) {
			target[offset + column * 4 + row] = rows[row]?.[column] ?? 0;
		}
	}
}
