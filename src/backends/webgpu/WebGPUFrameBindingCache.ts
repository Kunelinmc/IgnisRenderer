import {
	WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT,
	WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT,
	WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT,
} from "./bufferLayouts";
import type { Vec4Tuple } from "../../maths/Vector4";
import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderTexture,
	ISampler,
} from "../types";
import { TextureFormat } from "../../core/TextureFormat";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
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
	type WebGPUFrameCameraUniformInput,
	type WebGPUFrameLightUniformInput,
	type WebGPUFrameShadowUniformInput,
	type WebGPUFrameEnvironmentUniformInput,
	type WebGPULightingState,
} from "./";
import type {
	FrameContext,
	PreparedScene,
} from "../../pipeline/types";
import type { FramePreparationRequirements } from "../../pipeline/FrameRequirements";
import {
	DEFAULT_FOG_OPTIONS,
	FOG_PASS_ID,
	type FogOptions,
} from "../../postprocess";
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type { WebGPUShadowRuntime } from "./WebGPUShadowRuntime";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import { finiteOr } from "../../maths/Misc";
import {
	TemporalFrameState,
	type TemporalFrameSnapshot,
} from "../cross/TemporalFrameState";
import type { WebGPUSceneTargetMode } from "./WebGPUScenePassDescriptors";
import { clamp } from "../../maths/Common";
import type { Matrix4 } from "../../maths/Matrix4";

function validateCameraMatrix(matrix: Matrix4, label: string): void {
	const elements = matrix?.elements;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			const value = elements?.[row]?.[column];
			if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
				throw new Error(`WebGPU camera ${label} matrix must contain finite f32 values.`);
			}
		}
	}
}

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
	readonly frameRequirements?: FramePreparationRequirements;
}

export class WebGPUFrameBindingCache {
	private _backend: WebGPUDeviceResourceHost;
	private _resourceManager: WebGPUResourceManager;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowRuntime: WebGPUShadowRuntime;
	private _frameCameraUniformBuffer: IRenderBuffer | null = null;
	private _frameLightUniformBuffer: IRenderBuffer | null = null;
	private _frameShadowUniformBuffer: IRenderBuffer | null = null;
	private _frameEnvironmentUniformBuffer: IRenderBuffer | null = null;
	private _fogUniformBuffer: IRenderBuffer | null = null;
	private _environmentBackgroundParamsBuffer: IRenderBuffer | null = null;
	private readonly _environmentBackgroundParamsData = new Float32Array(4);
	private readonly _cameraWriter = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.createWriter();
	private readonly _lightWriter = WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.createWriter();
	private readonly _shadowWriter = WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT.createWriter();
	private readonly _environmentWriter = WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT.createWriter();
	private _cameraInput: WebGPUFrameCameraUniformInput | null = null;
	private readonly _lightInput: WebGPUFrameLightUniformInput = {
		ambientColor: [0, 0, 0], directionalLights: [], pointLights: [], spotLights: [],
		areaLights: [], enableLighting: false, enableClusteredLighting: false,
	};
	private readonly _shadowInput: WebGPUFrameShadowUniformInput = {
		directionalShadows: [], spotShadows: [], enableShadows: false,
	};
	private readonly _environmentInput: WebGPUFrameEnvironmentUniformInput = {
		shAmbientCoeffs: null, localLightProbeCount: 0, localLightProbes: [],
		irradianceProbeGrid: null, reflectionProbeCount: 0, reflectionProbes: [],
		shAmbientEnabled: false, hasEnvSpecular: false,
		hasBRDFLUT: false, envSpecularMaxMipLevel: 0,
	};
	private _fogUniformData: Float32Array<ArrayBuffer> = new Float32Array(8);
	private _sceneBinding: IBindingGroup | null = null;
	private _particleFrameBinding: IBindingGroup | null = null;
	private _depthFrameBinding: IBindingGroup | null = null;
	private _decalFrameBinding: IBindingGroup | null = null;
	private _environmentBinding: IBindingGroup | null = null;
	private _shadowAtlas: IRenderTexture | null = null;
	private _shadowTransmittanceAtlas: IRenderTexture | null = null;
	private _environmentTexture: IRenderTexture | null = null;
	private _envSpecularTexture: IRenderTexture | null = null;
	private _brdfLUTTexture: IRenderTexture | null = null;
	private _irradianceProbeGridTexture: IRenderTexture | null = null;
	private _ownedIrradianceProbeGridTexture: IRenderTexture | null = null;
	private _irradianceProbeGridTextureRevision = -1;
	private _irradianceProbeGridTextureCellCount = 0;
	private _irradianceProbeGridTextureGridId: string | null = null;
	private _irradianceProbeGridTextureData = new Float32Array(WEBGPU_SH_COEFFICIENT_COUNT * 4);
	private _environmentSampler: ISampler | null = null;
	private _envSpecularSampler: ISampler | null = null;
	private _shadowComparisonSampler: ISampler | null = null;
	private readonly _temporalFrameState = new TemporalFrameState();
	private _currentTemporalSnapshot: TemporalFrameSnapshot | null = null;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		layouts: WebGPUPipelineLayouts,
		textureRegistry: WebGPUTextureRegistry,
		shadowRuntime: WebGPUShadowRuntime,
	) {
		this._backend = backend;
		this._resourceManager = resourceManager;
		this._layouts = layouts;
		this._textureRegistry = textureRegistry;
		this._shadowRuntime = shadowRuntime;
	}

	public prepare(
		frame: PreparedScene,
		lightingState: WebGPULightingState,
		environmentState: WebGPUEnvironmentState,
		features: WebGPUFeatureState,
		renderWidth: number,
		renderHeight: number,
		sceneTargetMode: WebGPUSceneTargetMode,
		options: WebGPUFrameBindingPrepareOptions = {},
	): void {
		// Validation must finish before temporal history begins or GPU data is written.
		const camera = frame.camera;
		validateCameraMatrix(camera.viewMatrix, "view");
		validateCameraMatrix(camera.projectionMatrix, "projection");
		validateCameraMatrix(camera.viewProjectionMatrix, "view-projection");
		const inverseViewMatrix = camera.inverseViewMatrix;
		const inverseProjectionMatrix = camera.inverseProjectionMatrix;
		validateCameraMatrix(inverseViewMatrix, "view inverse");
		validateCameraMatrix(inverseProjectionMatrix, "projection inverse");
		const temporalStateMode = options.temporalStateMode ?? "advance";
		const temporal = this._resolveTemporalFrameUniforms(
			frame,
			options.frameRequirements ?? {},
			renderWidth,
			renderHeight,
			temporalStateMode,
			options.temporalHistoryReset === true,
		);
		const cameraInput = this._cameraInput ??= {
			viewProjectionMatrix: camera.viewProjectionMatrix,
			prevViewProjectionMatrix: temporal.previousViewProjection,
			inverseViewMatrix,
			inverseProjectionMatrix,
			jitterNdc: [0, 0], prevJitterNdc: [0, 0],
		};
		cameraInput.viewProjectionMatrix = frame.camera.viewProjectionMatrix;
		cameraInput.prevViewProjectionMatrix = temporal.previousViewProjection;
		cameraInput.inverseViewMatrix = inverseViewMatrix;
		cameraInput.inverseProjectionMatrix = inverseProjectionMatrix;
		const jitter = cameraInput.jitterNdc as [number, number];
		const previousJitter = cameraInput.prevJitterNdc as [number, number];
		jitter[0] = temporal.jitterCurrentPrev[0];
		jitter[1] = temporal.jitterCurrentPrev[1];
		previousJitter[0] = temporal.jitterCurrentPrev[2];
		previousJitter[1] = temporal.jitterCurrentPrev[3];
		const lightInput = this._lightInput;
		lightInput.ambientColor = lightingState.ambientColor;
		lightInput.directionalLights = lightingState.directionalLights;
		lightInput.pointLights = lightingState.pointLights;
		lightInput.spotLights = lightingState.spotLights;
		lightInput.areaLights = lightingState.areaLights;
		lightInput.enableLighting = features.enableLighting;
		lightInput.enableClusteredLighting = features.enableClusteredLighting;
		const shadowInput = this._shadowInput;
		shadowInput.directionalShadows = lightingState.directionalShadows;
		shadowInput.spotShadows = lightingState.spotShadows;
		shadowInput.enableShadows = features.enableShadows;
		const environmentInput = this._environmentInput;
		environmentInput.shAmbientCoeffs = environmentState.shAmbientCoeffs;
		environmentInput.localLightProbeCount = environmentState.localLightProbeCount;
		environmentInput.localLightProbes = environmentState.localLightProbes;
		environmentInput.irradianceProbeGrid = environmentState.irradianceProbeGrid;
		environmentInput.reflectionProbeCount = environmentState.reflectionProbeCount;
		environmentInput.reflectionProbes = environmentState.reflectionProbes;
		environmentInput.shAmbientEnabled = environmentState.enableSH && environmentState.hasSHAmbient;
		environmentInput.hasEnvSpecular = !!environmentState.envSpecularTexture;
		environmentInput.hasBRDFLUT = !!environmentState.brdfLUTTexture;
		environmentInput.envSpecularMaxMipLevel = environmentState.envSpecularMaxMipLevel;

		// Pack all CPU data before issuing writes; each view scope owns its writers.
		const cameraData = packFrameCameraUniformData(cameraInput, this._cameraWriter);
		const lightData = packFrameLightUniformData(lightInput, this._lightWriter);
		const shadowData = packFrameShadowUniformData(shadowInput, this._shadowWriter);
		const environmentData = packFrameEnvironmentUniformData(environmentInput, this._environmentWriter);
		this._backend.writeBuffer(this._getFrameCameraUniformBuffer(), cameraData);
		this._backend.writeBuffer(this._getFrameLightUniformBuffer(), lightData);
		this._backend.writeBuffer(this._getFrameShadowUniformBuffer(), shadowData);
		this._backend.writeBuffer(this._getFrameEnvironmentUniformBuffer(), environmentData);
		this._backend.writeBuffer(this._getFogUniformBuffer(), this._packFogUniformData(features));
		this._backend.writeBuffer(
			this._getEnvironmentBackgroundParamsBuffer(),
			this._packEnvironmentBackgroundParams(frame),
		);

		const currentShadowAtlas = this._shadowRuntime.atlas;
		const currentShadowTransmittanceAtlas = this._shadowRuntime.transmittanceAtlas;
		const currentEnvironment = environmentState.environmentTexture
			? this._textureRegistry.getTextureForSlot(environmentState.environmentTexture, 0)
			: this._textureRegistry.getWhiteTexture();
		const currentEnvironmentSampler = environmentState.environmentTexture
			? this._textureRegistry.getSamplerForTexture(environmentState.environmentTexture)
			: this._textureRegistry.getWhiteSampler();
		const currentEnvSpecular = environmentState.envSpecularTexture
			? this._textureRegistry.getTextureForSlot(environmentState.envSpecularTexture, 0)
			: this._textureRegistry.getWhiteTexture();
		const currentEnvSpecularSampler = environmentState.envSpecularTexture
			? this._textureRegistry.getSamplerForTexture(environmentState.envSpecularTexture)
			: this._textureRegistry.getWhiteSampler();
		const currentBRDFLUT = environmentState.brdfLUTTexture
			? this._textureRegistry.getTextureForSlot(environmentState.brdfLUTTexture, 0)
			: this._textureRegistry.getWhiteTexture();
		const currentIrradianceProbeGrid = this._getIrradianceProbeGridTexture(environmentState);

		if (
			this._shadowAtlas !== currentShadowAtlas ||
			this._shadowTransmittanceAtlas !== currentShadowTransmittanceAtlas ||
			this._environmentTexture !== currentEnvironment ||
			this._envSpecularTexture !== currentEnvSpecular ||
			this._brdfLUTTexture !== currentBRDFLUT ||
			this._irradianceProbeGridTexture !== currentIrradianceProbeGrid ||
			this._environmentSampler !== currentEnvironmentSampler ||
			this._envSpecularSampler !== currentEnvSpecularSampler
		) {
			this._destroyBindingGroup(this._sceneBinding);
			this._destroyBindingGroup(this._environmentBinding);
			this._sceneBinding = null;
			this._environmentBinding = null;
			this._shadowAtlas = currentShadowAtlas;
			this._shadowTransmittanceAtlas = currentShadowTransmittanceAtlas;
			this._environmentTexture = currentEnvironment;
			this._envSpecularTexture = currentEnvSpecular;
			this._brdfLUTTexture = currentBRDFLUT;
			this._irradianceProbeGridTexture = currentIrradianceProbeGrid;
			this._environmentSampler = currentEnvironmentSampler;
			this._envSpecularSampler = currentEnvSpecularSampler;
		}
	}

	private _getIrradianceProbeGridTexture(
		environmentState: WebGPUEnvironmentState,
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
				{ width, height, depthOrArrayLayers: 1 },
			);
			this._irradianceProbeGridTextureRevision = grid.textureRevision;
			this._irradianceProbeGridTextureCellCount = height;
			this._irradianceProbeGridTextureGridId = grid.id;
		}

		return this._ownedIrradianceProbeGridTexture;
	}

	private _packIrradianceProbeGridTextureData(
		grid: NonNullable<WebGPUEnvironmentState["irradianceProbeGrid"]>,
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

	private _resolveTemporalFrameUniforms(
		frame: PreparedScene,
		frameRequirements: FramePreparationRequirements,
		renderWidth: number,
		renderHeight: number,
		temporalStateMode: WebGPUTemporalStateMode,
		temporalHistoryReset: boolean,
	): {
		previousViewProjection: PreparedScene["camera"]["viewProjectionMatrix"];
		jitterCurrentPrev: Vec4Tuple;
	} {
		if (temporalStateMode === "disabled") {
			return {
				previousViewProjection: frame.camera.viewProjectionMatrix,
				jitterCurrentPrev: [0, 0, 0, 0],
			};
		}
		if (temporalStateMode === "reuse") {
			const snapshot = this._currentTemporalSnapshot;
			return {
				previousViewProjection:
					snapshot?.previousViewProjection ?? frame.camera.viewProjectionMatrix,
				jitterCurrentPrev: snapshot
					? [
							snapshot.jitterCurrentPrev[0],
							snapshot.jitterCurrentPrev[1],
							snapshot.jitterCurrentPrev[2],
							snapshot.jitterCurrentPrev[3],
						]
					: [0, 0, 0, 0],
			};
		}
		if (this._currentTemporalSnapshot) {
			throw new Error("WebGPU temporal frame state advanced more than once in one frame.");
		}
		const snapshot = this._temporalFrameState.beginFrame({
			camera: frame.camera,
			width: renderWidth,
			height: renderHeight,
			frameRequirements,
			reset: temporalHistoryReset,
		});
		this._currentTemporalSnapshot = snapshot;
		return {
			previousViewProjection:
				snapshot.previousViewProjection ?? frame.camera.viewProjectionMatrix,
			jitterCurrentPrev: [
				snapshot.jitterCurrentPrev[0],
				snapshot.jitterCurrentPrev[1],
				snapshot.jitterCurrentPrev[2],
				snapshot.jitterCurrentPrev[3],
			],
		};
	}

	public commitTemporalFrame(): void {
		this._temporalFrameState.commitFrame();
		this._currentTemporalSnapshot = null;
	}

	public abortTemporalFrame(): void {
		this._temporalFrameState.abortFrame();
		this._currentTemporalSnapshot = null;
	}

	/** @internal Owned by the WebGPU frame service lifecycle. */
	public resetTemporalState(): void {
		this._temporalFrameState.reset();
		this._currentTemporalSnapshot = null;
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
							this._shadowAtlas ?? this._shadowRuntime.ensureAtlasForTileSize(1),
					},
					{
						binding: 2,
						resource:
							this._envSpecularTexture ?? this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 3,
						resource:
							this._envSpecularSampler ?? this._textureRegistry.getWhiteSampler(),
					},
					{ binding: 4, resource: this._getFogUniformBuffer() },
					{
						binding: 6,
						resource:
							this._shadowTransmittanceAtlas ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 7,
						resource: this._brdfLUTTexture ?? this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 8,
						resource:
							this._irradianceProbeGridTexture ??
							this._textureRegistry.getWhiteTexture(),
					},
					{
						binding: 9,
						resource: this._getShadowComparisonSampler(),
					},
					{ binding: 10, resource: this._getFrameLightUniformBuffer() },
					{ binding: 11, resource: this._getFrameShadowUniformBuffer() },
					{
						binding: 12,
						resource: this._getFrameEnvironmentUniformBuffer(),
					},
				],
			});
		}

		return this._sceneBinding;
	}

	/**
	 * Returns camera and fog bindings for billboard particles.
	 *
	 * @internal Owned by WebGPU frame scopes; applications should use `Renderer.renderFrame()`.
	 * @returns The cached particle frame group, sharing this scope's camera and fog buffers.
	 * @sideEffects Lazily creates the binding group.
	 */
	public getParticleFrameBinding(): IBindingGroup {
		if (!this._particleFrameBinding) {
			this._particleFrameBinding = this._backend.createBindingGroup({
				label: "WebGPUParticleFrameBinding",
				layout: this._layouts.particleFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameCameraUniformBuffer() },
					{ binding: 4, resource: this._getFogUniformBuffer() },
				],
			});
		}
		return this._particleFrameBinding;
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
	/** @internal Camera-only Early-Z bindings owned by frame resources; use Renderer. */
	public getDepthFrameBinding(): IBindingGroup {
		if (!this._depthFrameBinding) {
			this._depthFrameBinding = this._backend.createBindingGroup({
				label: "WebGPUDepthFrameBinding",
				layout: this._layouts.depthFrameBindGroupLayout,
				entries: [{ binding: 0, resource: this._getFrameCameraUniformBuffer() }],
			});
		}
		return this._depthFrameBinding;
	}

	public getDecalFrameBinding(): IBindingGroup {
		if (!this._decalFrameBinding) {
			this._decalFrameBinding = this._backend.createBindingGroup({
				label: "FrameBinding_decal",
				layout: this._layouts.decalFrameBindGroupLayout,
				entries: [{ binding: 0, resource: this._getFrameCameraUniformBuffer() }],
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

	private _packFogUniformData(features: WebGPUFeatureState): Float32Array<ArrayBuffer> {
		const source =
			features.postProcess.getOptions<FogOptions>(FOG_PASS_ID) ?? DEFAULT_FOG_OPTIONS;
		const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
		const start = Math.max(0, finiteOr(source.start, DEFAULT_FOG_OPTIONS.start));
		const end = Math.max(start + 1e-4, finiteOr(source.end, DEFAULT_FOG_OPTIONS.end));
		const density = Math.max(0, finiteOr(source.density, DEFAULT_FOG_OPTIONS.density));
		const sceneFogEnabled =
			features.postProcess.isEnabled(FOG_PASS_ID) &&
			(source.application ?? DEFAULT_FOG_OPTIONS.application) === "scene";
		const strength = sceneFogEnabled
			? Math.max(0, finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength))
			: 0;
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
		frame: PreparedScene,
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
		this._destroyBindingGroup(this._particleFrameBinding);
		this._destroyBindingGroup(this._depthFrameBinding);
		this._destroyBindingGroup(this._decalFrameBinding);
		this._destroyBindingGroup(this._environmentBinding);
		this._sceneBinding = null;
		this._particleFrameBinding = null;
		this._depthFrameBinding = null;
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
		this._brdfLUTTexture = null;
		this._destroySampler(this._shadowComparisonSampler);
		this._shadowComparisonSampler = null;
		this._environmentSampler = null;
		this._envSpecularSampler = null;
		this.resetTemporalState();
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
