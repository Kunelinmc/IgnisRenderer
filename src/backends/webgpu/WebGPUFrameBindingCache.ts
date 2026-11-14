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
import {
	AddressMode,
	BufferUsage,
	FilterMode,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
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
import { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type { WebGPUShadowRuntime } from "./WebGPUShadowRuntime";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
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
	private readonly _resources: WebGPUResourceManager;
	private _layouts: WebGPUPipelineLayouts;
	private _textureRegistry: WebGPUTextureRegistry;
	private _shadowRuntime: WebGPUShadowRuntime;
	private _frameCameraUniformBuffer: IRenderBuffer | null = null;
	private _frameLightUniformBuffer: IRenderBuffer | null = null;
	private _frameShadowUniformBuffer: IRenderBuffer | null = null;
	private _frameEnvironmentUniformBuffer: IRenderBuffer | null = null;
	private _environmentBackgroundParamsBuffer: IRenderBuffer | null = null;
	private readonly _environmentBackgroundParamsData = new Float32Array(4);
	private readonly _cameraWriter = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.createWriter();
	private readonly _lightWriter = WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.createWriter();
	private readonly _shadowWriter = WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT.createWriter();
	private readonly _environmentWriter = WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT.createWriter();
	private _cameraInput: WebGPUFrameCameraUniformInput | null = null;
	private readonly _lightInput: WebGPUFrameLightUniformInput = {
		ambientColor: [0, 0, 0], directionalLights: [], pointLights: [], spotLights: [],
		areaLights: [], enableClusteredLighting: false,
	};
	private readonly _shadowInput: WebGPUFrameShadowUniformInput = {
		directionalShadows: [], spotShadows: [], enableShadows: false,
	};
	private readonly _environmentInput: WebGPUFrameEnvironmentUniformInput = {
		shAmbientCoeffs: null, localLightProbeCount: 0, localLightProbes: [],
		reflectionProbeCount: 0, reflectionProbes: [],
		shAmbientEnabled: false, hasEnvSpecular: false,
		hasBRDFLUT: false, envSpecularMaxMipLevel: 0,
	};
	private _sceneBinding: IBindingGroup | null = null;
	private _particleFrameBinding: IBindingGroup | null = null;
	private _depthFrameBinding: IBindingGroup | null = null;
	private _environmentBinding: IBindingGroup | null = null;
	private _shadowAtlas: IRenderTexture | null = null;
	private _shadowTransmittanceAtlas: IRenderTexture | null = null;
	private _environmentTexture: IRenderTexture | null = null;
	private _envSpecularTexture: IRenderTexture | null = null;
	private _brdfLUTTexture: IRenderTexture | null = null;
	private _environmentSampler: ISampler | null = null;
	private _envSpecularSampler: ISampler | null = null;
	private _shadowComparisonSampler: ISampler | null = null;
	private readonly _temporalFrameState = new TemporalFrameState();
	private _currentTemporalSnapshot: TemporalFrameSnapshot | null = null;

	constructor(
		backend: WebGPUDeviceResourceHost,
		layouts: WebGPUPipelineLayouts,
		textureRegistry: WebGPUTextureRegistry,
		shadowRuntime: WebGPUShadowRuntime,
	) {
		this._backend = backend;
		this._resources = backend.resourceManager;
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
		lightInput.enableClusteredLighting = features.enableClusteredLighting;
		const shadowInput = this._shadowInput;
		shadowInput.directionalShadows = lightingState.directionalShadows;
		shadowInput.spotShadows = lightingState.spotShadows;
		shadowInput.enableShadows = features.enableShadows;
		const environmentInput = this._environmentInput;
		environmentInput.shAmbientCoeffs = environmentState.shAmbientCoeffs;
		environmentInput.localLightProbeCount = environmentState.localLightProbeCount;
		environmentInput.localLightProbes = environmentState.localLightProbes;
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

		if (
			this._shadowAtlas !== currentShadowAtlas ||
			this._shadowTransmittanceAtlas !== currentShadowTransmittanceAtlas ||
			this._environmentTexture !== currentEnvironment ||
			this._envSpecularTexture !== currentEnvSpecular ||
			this._brdfLUTTexture !== currentBRDFLUT ||
			this._environmentSampler !== currentEnvironmentSampler ||
			this._envSpecularSampler !== currentEnvSpecularSampler
		) {
			this._sceneBinding?.destroy();
			this._environmentBinding?.destroy();
			this._sceneBinding = null;
			this._environmentBinding = null;
			this._shadowAtlas = currentShadowAtlas;
			this._shadowTransmittanceAtlas = currentShadowTransmittanceAtlas;
			this._environmentTexture = currentEnvironment;
			this._envSpecularTexture = currentEnvSpecular;
			this._brdfLUTTexture = currentBRDFLUT;
			this._environmentSampler = currentEnvironmentSampler;
			this._envSpecularSampler = currentEnvSpecularSampler;
		}
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
	 * Returns camera bindings for billboard particles.
	 *
	 * @internal Owned by WebGPU frame scopes; applications should use `Renderer.renderFrame()`.
	 * @returns The cached particle frame group, sharing this scope's camera buffer.
	 * @sideEffects Lazily creates the binding group.
	 */
	public getParticleFrameBinding(): IBindingGroup {
		if (!this._particleFrameBinding) {
			this._particleFrameBinding = this._backend.createBindingGroup({
				label: "WebGPUParticleFrameBinding",
				layout: this._layouts.particleFrameBindGroupLayout,
				entries: [
					{ binding: 0, resource: this._getFrameCameraUniformBuffer() },
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

	public destroy(): void {
		this._sceneBinding?.destroy();
		this._particleFrameBinding?.destroy();
		this._depthFrameBinding?.destroy();
		this._environmentBinding?.destroy();
		this._sceneBinding = null;
		this._particleFrameBinding = null;
		this._depthFrameBinding = null;
		this._environmentBinding = null;
		this._frameCameraUniformBuffer?.destroy();
		this._frameCameraUniformBuffer = null;
		this._frameLightUniformBuffer?.destroy();
		this._frameLightUniformBuffer = null;
		this._frameShadowUniformBuffer?.destroy();
		this._frameShadowUniformBuffer = null;
		this._frameEnvironmentUniformBuffer?.destroy();
		this._frameEnvironmentUniformBuffer = null;
		this._environmentBackgroundParamsBuffer?.destroy();
		this._environmentBackgroundParamsBuffer = null;
		this._shadowAtlas = null;
		this._shadowTransmittanceAtlas = null;
		this._environmentTexture = null;
		this._envSpecularTexture = null;
		this._brdfLUTTexture = null;
		this._shadowComparisonSampler?.destroy();
		this._shadowComparisonSampler = null;
		this._environmentSampler = null;
		this._envSpecularSampler = null;
		this.resetTemporalState();
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
