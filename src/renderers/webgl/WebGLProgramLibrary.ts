import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	createWebGLSceneUniforms,
	type WebGLSceneUniforms,
} from "./WebGLSceneProgramUniforms";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import {
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	createInlineShaderSourceMap,
	parseWebGLShaderInfoLog,
	ShaderCompileError,
	type ShaderCompilerMessage,
	type ShaderProcessResult,
	type ShaderSourceSegmentMap,
	type ShaderRuntime,
} from "../../shaders/runtime";
import {
	type WebGLShaderPart,
	ShaderSource,
	type WebGLSceneLightLimits,
} from "../../shaders/ShaderSource";
import { Logger } from "../../foundation/Logger";

export interface WebGLShadowDepthProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
	};
}

export interface WebGLShadowTransmittanceProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
		transmittance: WebGLUniformLocation | null;
	};
}

export interface WebGLCopyProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLOITResolveProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		oitAccumMap: WebGLUniformLocation | null;
		oitRevealMap: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAORawProgram {
	program: WebGLProgram;
	uniforms: {
		normalMap: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
		gtao: WebGLUniformLocation | null;
		blurProj: WebGLUniformLocation | null;
		pass: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		basisRight: WebGLUniformLocation | null;
		basisUp: WebGLUniformLocation | null;
		basisBackward: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAOBlurProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
		blurProj: WebGLUniformLocation | null;
		pass: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAOCombineProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		aoMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
	};
}

export interface WebGLTAAProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		historyMap: WebGLUniformLocation | null;
		motionMap: WebGLUniformLocation | null;
		motionHistory: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		historyWeight: WebGLUniformLocation | null;
		depthThreshold: WebGLUniformLocation | null;
		motionFactor: WebGLUniformLocation | null;
		varianceClampGamma: WebGLUniformLocation | null;
		sharpen: WebGLUniformLocation | null;
		historyValid: WebGLUniformLocation | null;
	};
}

export interface WebGLSSRProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		normalMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLVolumetricProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLFogProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		fogParams0: WebGLUniformLocation | null;
		fogParams1: WebGLUniformLocation | null;
	};
}


export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: WebGLSceneUniforms;
}

export interface WebGLEnvironmentProgram {
	program: WebGLProgram;
	uniforms: {
		environmentMap: WebGLUniformLocation | null;
		environmentBasisRight: WebGLUniformLocation | null;
		environmentBasisUp: WebGLUniformLocation | null;
		environmentBasisBackward: WebGLUniformLocation | null;
		environmentIsOrthographic: WebGLUniformLocation | null;
		environmentMapIsLinear: WebGLUniformLocation | null;
		environmentBackgroundTint: WebGLUniformLocation | null;
		environmentBackgroundExposure: WebGLUniformLocation | null;
		environmentBackgroundStrength: WebGLUniformLocation | null;
	};
}

export interface WebGLPresentProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		applyGamma: WebGLUniformLocation | null;
	};
}

export interface WebGLParticleProgram {
	program: WebGLProgram;
	uniforms: {
		viewProjection: WebGLUniformLocation | null;
		basisRight: WebGLUniformLocation | null;
		basisUp: WebGLUniformLocation | null;
		particleMap: WebGLUniformLocation | null;
		uvTransformA: WebGLUniformLocation | null;
		uvTransformB: WebGLUniformLocation | null;
		mapIsLinear: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		fogParams0: WebGLUniformLocation | null;
		fogParams1: WebGLUniformLocation | null;
		oitPassMode: WebGLUniformLocation | null;
	};
}

export interface WebGLFXAAProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLToneMappingProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLColorFilterProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		filterParams0: WebGLUniformLocation | null;
		filterParams1: WebGLUniformLocation | null;
	};
}

export interface WebGLInteractionOutlineProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		outlineColor: WebGLUniformLocation | null;
		outlineParams: WebGLUniformLocation | null;
		viewportSize: WebGLUniformLocation | null;
		circleCount: WebGLUniformLocation | null;
		circles: WebGLUniformLocation | null;
	};
}

export interface WebGLBloomProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		bloomParams: WebGLUniformLocation | null;
	};
}

export interface WebGLMotionBlurProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		motionParams: WebGLUniformLocation | null;
		centerWeight: WebGLUniformLocation | null;
	};
}

export interface WebGLDOFProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		focusParams: WebGLUniformLocation | null;
		dofParams: WebGLUniformLocation | null;
		chromaticAberration: WebGLUniformLocation | null;
	};
}

interface ShaderCompileMetadata {
	sourceMap?: ShaderSourceSegmentMap | null;
	variantKey?: string;
	materialId?: string;
	sourceKind?: "custom-material" | "unknown";
}

interface WebGLParallelShaderCompileExtension {
	readonly COMPLETION_STATUS_KHR: number;
}

export type WebGLProgramCompileState = "idle" | "pending" | "ready" | "failed";

export interface WebGLProgramLibraryOptions {
	validatePrograms?: boolean;
	onProgramCompilePending?: () => void;
}

export interface WebGLProgramWarmupHandle {
	readonly label: string;
	isComplete(): boolean;
	finalize(): void;
}

interface WebGLPendingShaderCompile {
	readonly shader: WebGLShader;
	readonly stage: "vertex" | "fragment";
	readonly label: string;
	readonly sourceKind: "custom-material" | "unknown";
	readonly variantKey?: string;
	readonly materialId?: string;
	readonly code: string;
	readonly sourceMap: ShaderSourceSegmentMap | null;
}

interface WebGLPendingProgramCompile {
	readonly label: string;
	readonly vertex: WebGLPendingShaderCompile;
	readonly fragment: WebGLPendingShaderCompile;
	readonly program: WebGLProgram;
	readonly startedFrame: number;
	status: "pending" | "ready" | "failed";
	finalized: boolean;
	error: unknown;
}

type WebGLProgramWarn = (key: string, message: string) => void;

const WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_TRANSMITTANCE = 17;
const WEBGL_MIN_TEXTURE_UNITS_FOR_IRRADIANCE_PROBE_GRID = 17;
const WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_AND_IRRADIANCE_PROBE_GRID = 18;
const WEBGL_FALLBACK_READY_FRAME_DELAY = 2;
const WEBGL_FALLBACK_FINALIZE_BUDGET_PER_FRAME = 1;

function resolveMaxFragmentTextureUnits(
	gl: WebGL2RenderingContext
): number {
	const textureUnitParameter = gl.MAX_TEXTURE_IMAGE_UNITS;
	if (
		!Number.isFinite(textureUnitParameter) ||
		typeof gl.getParameter !== "function"
	) {
		return 0;
	}
	try {
		const textureUnits = gl.getParameter(textureUnitParameter);
		return Number.isFinite(textureUnits) ?
				Math.max(0, Math.floor(textureUnits))
			:	0;
	} catch {
		return 0;
	}
}

function supportsShadowTransmittanceSampler(textureUnits: number): boolean {
	return textureUnits >= WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_TRANSMITTANCE;
}

function supportsIrradianceProbeGridSampler(
	textureUnits: number,
	shadowTransmittanceEnabled: boolean
): boolean {
	const requiredTextureUnits =
		shadowTransmittanceEnabled ?
			WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_AND_IRRADIANCE_PROBE_GRID
		:	WEBGL_MIN_TEXTURE_UNITS_FOR_IRRADIANCE_PROBE_GRID;
	return textureUnits >= requiredTextureUnits;
}

export class WebGLProgramLibrary {
	private _gl: WebGL2RenderingContext;
	private _shaderRuntime: ShaderRuntime | null;
	private _shaderCompileStage: ShaderBackendCompileStage | null;
	private _enableShadowTransmittanceSampling: boolean;
	private _enableIrradianceProbeGridSampling: boolean;
	private _parallelShaderCompile: WebGLParallelShaderCompileExtension | null;
	private _validatePrograms: boolean;
	private _onProgramCompilePending: (() => void) | null = null;
	private _warnCallback: WebGLProgramWarn | null = null;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _pendingProgramCompiles = new Map<string, WebGLPendingProgramCompile>();
	private _precompiledPrograms = new Map<string, WebGLProgram>();
	private _warmupHandleLog: WebGLProgramWarmupHandle[] = [];
	private _compileFrameIndex = 0;
	private _fallbackFinalizesThisFrame = 0;
	private _lastPendingNotificationFrame = -1;
	private _sceneProgram: WebGLSceneProgram | null = null;
	private _sceneProgramDirectiveTag: string = "";
	private _customScenePrograms = new Map<string, WebGLSceneProgram>();
	private _sceneDepthPrepassProgram: WebGLSceneProgram | null = null;
	private _sceneDepthPrepassProgramDirectiveTag: string = "";
	private _customSceneDepthPrepassPrograms = new Map<string, WebGLSceneProgram>();
	private _missingDepthPrepassShaderMaterialWarnings = new Set<number>();
	private _environmentProgram: WebGLEnvironmentProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _toneMappingProgram: WebGLToneMappingProgram | null = null;
	private _colorFilterProgram: WebGLColorFilterProgram | null = null;
	private _fxaaProgram: WebGLFXAAProgram | null = null;
	private _interactionOutlineProgram: WebGLInteractionOutlineProgram | null = null;
	private _bloomProgram: WebGLBloomProgram | null = null;
	private _motionBlurProgram: WebGLMotionBlurProgram | null = null;
	private _dofProgram: WebGLDOFProgram | null = null;

	private _shadowDepthProgram: WebGLShadowDepthProgram | null = null;
	private _shadowTransmittanceProgram: WebGLShadowTransmittanceProgram | null = null;
	private _copyProgram: WebGLCopyProgram | null = null;
	private _oitResolveProgram: WebGLOITResolveProgram | null = null;
	private _ssaoRawProgram: WebGLSSAORawProgram | null = null;
	private _ssaoBlurProgram: WebGLSSAOBlurProgram | null = null;
	private _ssaoCombineProgram: WebGLSSAOCombineProgram | null = null;
	private _taaProgram: WebGLTAAProgram | null = null;
	private _ssrProgram: WebGLSSRProgram | null = null;
	private _volumetricProgram: WebGLVolumetricProgram | null = null;
	private _fogProgram: WebGLFogProgram | null = null;

	constructor(
		gl: WebGL2RenderingContext,
		warn: WebGLProgramWarn,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options?: WebGLProgramLibraryOptions,
	);
	constructor(
		gl: WebGL2RenderingContext,
		warn: WebGLProgramWarn,
		options?: WebGLProgramLibraryOptions,
	);
	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options?: WebGLProgramLibraryOptions,
	);
	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntimeOrWarn?: ShaderRuntime | WebGLProgramWarn,
		shaderCompileStageOrRuntime?:
			| ShaderBackendCompileStage
			| ShaderRuntime
			| WebGLProgramLibraryOptions,
		shaderCompileStageMaybe?:
			| ShaderBackendCompileStage
			| WebGLProgramLibraryOptions,
		optionsMaybe?: WebGLProgramLibraryOptions,
	) {
		this._gl = gl;
		const maxFragmentTextureUnits = resolveMaxFragmentTextureUnits(gl);
		this._enableShadowTransmittanceSampling =
			supportsShadowTransmittanceSampler(maxFragmentTextureUnits);
		this._enableIrradianceProbeGridSampling =
			supportsIrradianceProbeGridSampler(
				maxFragmentTextureUnits,
				this._enableShadowTransmittanceSampling
			);
		this._parallelShaderCompile = resolveParallelShaderCompileExtension(gl);
		let shaderRuntime: ShaderRuntime | null = null;
		let shaderCompileStage: ShaderBackendCompileStage | null = null;
		let options: WebGLProgramLibraryOptions | null = null;
		if (typeof shaderRuntimeOrWarn === "function") {
			this._warnCallback = shaderRuntimeOrWarn;
			shaderRuntime =
				isShaderRuntime(shaderCompileStageOrRuntime) ?
					shaderCompileStageOrRuntime
				:	null;
			shaderCompileStage =
				shaderCompileStageMaybe instanceof ShaderBackendCompileStage ?
					shaderCompileStageMaybe
				:	null;
			options =
				isWebGLProgramLibraryOptions(shaderCompileStageOrRuntime) ?
					shaderCompileStageOrRuntime
				: isWebGLProgramLibraryOptions(shaderCompileStageMaybe) ?
					shaderCompileStageMaybe
				:	optionsMaybe ?? null;
		} else {
			shaderRuntime = shaderRuntimeOrWarn ?? null;
			shaderCompileStage =
				shaderCompileStageOrRuntime instanceof ShaderBackendCompileStage ?
					shaderCompileStageOrRuntime
				:	null;
			options =
				isWebGLProgramLibraryOptions(shaderCompileStageMaybe) ?
					shaderCompileStageMaybe
				:	null;
		}
		this._shaderRuntime = shaderRuntime;
		this._shaderCompileStage = shaderCompileStage;
		this._validatePrograms = options?.validatePrograms === true;
		this._onProgramCompilePending = options?.onProgramCompilePending ?? null;
		if (!this._shaderCompileStage && this._shaderRuntime) {
			this._shaderCompileStage = new ShaderBackendCompileStage({
				backend: "webgl",
				runtime: this._shaderRuntime,
				profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
				mode: this._shaderRuntime.getMode(),
			});
		}
		if (this._shaderRuntime) {
			this._disposeShaderRuntimeListener = this._shaderRuntime.onDidChange(() =>
				this._invalidateProgramCachesForShaderRuntime(),
			);
		}
	}

	/**
	 * Advances the internal frame tick used to budget fallback shader finalization.
	 *
	 * @returns Nothing.
	 * @sideEffects Resets the non-parallel compile fallback finalize budget.
	 */
	public beginFrame(): void {
		this._compileFrameIndex++;
		this._fallbackFinalizesThisFrame = 0;
	}

	/**
	 * Reports whether the built-in scene shader variant includes irradiance probe
	 * grid uniforms and the grid SH sampler.
	 *
	 * @returns True when the active WebGL device exposes enough fragment texture
	 * units for the optional grid sampler in the scene shader.
	 * @sideEffects None.
	 */
	public supportsIrradianceProbeGridSampling(): boolean {
		return this._enableIrradianceProbeGridSampling;
	}

	/**
	 * Returns the raw compile state for an internal WebGL program label.
	 *
	 * @param label Program label used by the program library.
	 * @returns Current compile state, or `"idle"` when no compile is pending.
	 * @sideEffects None.
	 */
	public getProgramCompileState(label: string): WebGLProgramCompileState {
		const pending = this._pendingProgramCompiles.get(label);
		if (pending) {
			return pending.status;
		}
		if (this._precompiledPrograms.has(label)) {
			return "ready";
		}
		return "idle";
	}

	public getSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single"
	): WebGLSceneProgram {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneProgram();
		}

		const custom = this._getShaderMaterialSceneProgram(material, mode);
		return custom ?? this._getBuiltinSceneProgram();
	}

	/**
	 * Returns the backend-owned WebGL depth pre-pass program for a material.
	 *
	 * @internal WebGL frame execution hook. Prefer configuring
	 * `WebGLBackendOptions.enableEarlyZPrepass` over calling this directly.
	 *
	 * @param material Optional material whose custom depth pre-pass shader should
	 * be resolved.
	 * @param mode Scene target mode used to match the color-pass variant.
	 * @returns A compiled depth pre-pass program, or `null` when the material does
	 * not opt into the depth pre-pass contract.
	 * @sideEffects May compile and cache a WebGL program.
	 */
	public getSceneDepthPrepassProgram(
		material?: Material,
		mode: ShaderTargetMode = "single"
	): WebGLSceneProgram | null {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneDepthPrepassProgram();
		}

		return this._getShaderMaterialDepthPrepassProgram(material, mode);
	}

	public markWarmupHandles(): number {
		return this._warmupHandleLog.length;
	}

	public collectWarmupHandlesSince(mark: number): WebGLProgramWarmupHandle[] {
		const start = Math.max(0, Math.min(mark, this._warmupHandleLog.length));
		const handles = this._warmupHandleLog.slice(start);
		this._warmupHandleLog.length = start;
		return handles;
	}

	public warmupSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single"
	): WebGLProgramWarmupHandle {
		if (!(material instanceof ShaderMaterial)) {
			return this._warmupBuiltinSceneProgram();
		}

		const custom = this._warmupShaderMaterialSceneProgram(material, mode);
		return custom ?? this._warmupBuiltinSceneProgram();
	}

	/**
	 * Queues or completes warmup for the backend-owned WebGL depth pre-pass
	 * program.
	 *
	 * @internal WebGL warmup planner hook.
	 *
	 * @param material Optional material whose custom depth pre-pass shader should
	 * be warmed.
	 * @param mode Scene target mode used to match the color-pass variant.
	 * @returns A warmup handle, or `null` when the material has no depth pre-pass
	 * source.
	 * @sideEffects May start asynchronous WebGL shader compilation.
	 */
	public warmupSceneDepthPrepassProgram(
		material?: Material,
		mode: ShaderTargetMode = "single"
	): WebGLProgramWarmupHandle | null {
		if (!(material instanceof ShaderMaterial)) {
			return this._warmupBuiltinSceneDepthPrepassProgram();
		}

		return this._warmupShaderMaterialDepthPrepassProgram(material, mode);
	}

	public warmupEnvironmentProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLEnvironmentProgram",
			() => this._environmentProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("environmentVertex"),
					this._shaderSource("environmentFragment"),
					"WebGLEnvironmentProgram",
				),
			() => {
				this.getEnvironmentProgram();
			},
		);
	}

	public warmupPresentProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLPresentProgram",
			() => this._presentProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("presentFragment"),
					"WebGLPresentProgram",
				),
			() => {
				this.getPresentProgram();
			},
		);
	}

	public warmupParticleProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLParticleProgram",
			() => this._particleProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("particleVertex"),
					this._shaderSource("particleFragment"),
					"WebGLParticleProgram",
				),
			() => {
				this.getParticleProgram();
			},
		);
	}

	public warmupFXAAProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLFXAAProgram",
			() => this._fxaaProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("fxaaFragment"),
					"WebGLFXAAProgram",
				),
			() => {
				this.getFXAAProgram();
			},
		);
	}

	public warmupToneMappingProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLToneMappingProgram",
			() => this._toneMappingProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("toneMappingFragment"),
					"WebGLToneMappingProgram",
				),
			() => {
				this.getToneMappingProgram();
			},
		);
	}

	public warmupInteractionOutlineProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLInteractionOutlineProgram",
			() => this._interactionOutlineProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("interactionOutlineFragment"),
					"WebGLInteractionOutlineProgram",
				),
			() => {
				this.getInteractionOutlineProgram();
			},
		);
	}

	public warmupColorFilterProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLColorFilterProgram",
			() => this._colorFilterProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("colorFilterFragment"),
					"WebGLColorFilterProgram",
				),
			() => {
				this.getColorFilterProgram();
			},
		);
	}

	public warmupBloomProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLBloomProgram",
			() => this._bloomProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("bloomFragment"),
					"WebGLBloomProgram",
				),
			() => {
				this.getBloomProgram();
			},
		);
	}

	public warmupMotionBlurProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLMotionBlurProgram",
			() => this._motionBlurProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("motionBlurFragment"),
					"WebGLMotionBlurProgram",
				),
			() => {
				this.getMotionBlurProgram();
			},
		);
	}

	public warmupDOFProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLDOFProgram",
			() => this._dofProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("dofFragment"),
					"WebGLDOFProgram",
				),
			() => {
				this.getDOFProgram();
			},
		);
	}

	public warmupShadowDepthProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLShadowDepthProgram",
			() => this._shadowDepthProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("shadowDepthVertex"),
					this._shaderSource("shadowDepthFragment"),
					"WebGLShadowDepthProgram",
				),
			() => {
				this.getShadowDepthProgram();
			},
		);
	}

	public warmupShadowTransmittanceProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLShadowTransmittanceProgram",
			() => this._shadowTransmittanceProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("shadowDepthVertex"),
					this._shaderSource("shadowTransmittanceFragment"),
					"WebGLShadowTransmittanceProgram",
				),
			() => {
				this.getShadowTransmittanceProgram();
			},
		);
	}

	public warmupCopyProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLCopyProgram",
			() => this._copyProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("copyFragment"),
					"WebGLCopyProgram",
				),
			() => {
				this.getCopyProgram();
			},
		);
	}

	public warmupOITResolveProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLOITResolveProgram",
			() => this._oitResolveProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("oitResolveFragment"),
					"WebGLOITResolveProgram",
				),
			() => {
				this.getOITResolveProgram();
			},
		);
	}

	public warmupSSAORawProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLSSAORawProgram",
			() => this._ssaoRawProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("ssaoRawFragment"),
					"WebGLSSAORawProgram",
				),
			() => {
				this.getSSAORawProgram();
			},
		);
	}

	public warmupSSAOBlurProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLSSAOBlurProgram",
			() => this._ssaoBlurProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("ssaoBlurFragment"),
					"WebGLSSAOBlurProgram",
				),
			() => {
				this.getSSAOBlurProgram();
			},
		);
	}

	public warmupSSAOCombineProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLSSAOCombineProgram",
			() => this._ssaoCombineProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("ssaoCombineFragment"),
					"WebGLSSAOCombineProgram",
				),
			() => {
				this.getSSAOCombineProgram();
			},
		);
	}

	public warmupTAAProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLTAAProgram",
			() => this._taaProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("taaFragment"),
					"WebGLTAAProgram",
				),
			() => {
				this.getTAAProgram();
			},
		);
	}

	public warmupSSRProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLSSRProgram",
			() => this._ssrProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("postProcessStubFragment"),
					"WebGLSSRProgram",
				),
			() => {
				this.getSSRProgram();
			},
		);
	}

	public warmupVolumetricProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLVolumetricProgram",
			() => this._volumetricProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("postProcessStubFragment"),
					"WebGLVolumetricProgram",
				),
			() => {
				this.getVolumetricProgram();
			},
		);
	}

	public warmupFogProgram(): WebGLProgramWarmupHandle {
		return this._warmupProgram(
			"WebGLFogProgram",
			() => this._fogProgram,
			() =>
				this._beginProgramCompile(
					this._shaderSource("presentVertex"),
					this._shaderSource("fogFragment"),
					"WebGLFogProgram",
				),
			() => {
				this.getFogProgram();
			},
		);
	}

	private _warmupBuiltinSceneProgram(): WebGLProgramWarmupHandle {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		if (this._sceneProgram && this._sceneProgramDirectiveTag !== directiveTag) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		const limits = this._getSceneLightLimits();
		const sceneShaderSource = ShaderSource.get("webgl.scene.raw", {
			limits,
		});
		const sceneCompositeSource = ShaderSource.get("webgl.scene.composite", {
			limits,
		});
		return this._warmupProgram(
			"WebGLSceneProgram",
			() =>
				this._sceneProgram && this._sceneProgramDirectiveTag === directiveTag ?
					this._sceneProgram
				:	null,
			() =>
				this._beginProgramCompile(
					sceneShaderSource.vertex,
					sceneShaderSource.fragment,
					"WebGLSceneProgram",
					{
						sourceMap: sceneCompositeSource.vertex.sourceMap,
						sourceKind: "unknown",
					},
					{
						sourceMap: sceneCompositeSource.fragment.sourceMap,
						sourceKind: "unknown",
					},
				),
			() => {
				this._getBuiltinSceneProgram();
			},
		);
	}

	private _warmupShaderMaterialSceneProgram(
		material: ShaderMaterial,
		mode: ShaderTargetMode
	): WebGLProgramWarmupHandle | null {
		const initialDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? "none";
		const shaderKey = this._createShaderMaterialCacheKey(
			material,
			mode,
			initialDirectiveTag,
		);
		const cached = this._customScenePrograms.get(shaderKey);
		if (cached) {
			return this._recordWarmupHandle(
				this._createCompletedWarmupHandle(
					`WebGLShaderMaterialProgram_${shaderKey}`,
				),
			);
		}

		let source: { vertexCode: string; fragmentCode: string };
		try {
			source = material.resolveWebGLProgram(mode, {
				enableRuntimeInjects: this._supportsRuntimeInjects(),
			});
		} catch (error) {
			const key = `webgl-shader-material-missing-source-${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} has no WebGL GLSL source; ` +
				`using built-in scene shader. ${String(error)}`;
			this._warn(key, message);
			return null;
		}

		const label = `WebGLShaderMaterialProgram_${shaderKey}`;
		return this._warmupProgram(
			label,
			() => this._customScenePrograms.get(shaderKey) ?? null,
			() =>
				this._beginProgramCompile(
					source.vertexCode,
					source.fragmentCode,
					label,
					{
						sourceMap: createInlineShaderSourceMap(
							source.vertexCode,
							`<shader-material:${shaderKey}:vertex>`,
							"source",
						),
						variantKey: shaderKey,
						materialId: String(material.shaderId),
						sourceKind: "custom-material",
					},
					{
						sourceMap: createInlineShaderSourceMap(
							source.fragmentCode,
							`<shader-material:${shaderKey}:fragment>`,
							"source",
						),
						variantKey: shaderKey,
						materialId: String(material.shaderId),
						sourceKind: "custom-material",
					},
				),
			() => {
				this._getShaderMaterialSceneProgram(material, mode);
			},
			(error) => {
				if (!this._isWarnMode()) {
					throw error;
				}
				const key = `webgl-shader-material-compile-failed-${material.shaderId}`;
				const message =
					`ShaderMaterial ${material.name} custom WebGL shader compile failed; ` +
					`using built-in scene shader. ${String(error)}`;
				this._warn(key, message);
				this._getBuiltinSceneProgram();
			},
		);
	}

	private _warmupBuiltinSceneDepthPrepassProgram(): WebGLProgramWarmupHandle {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		if (
			this._sceneDepthPrepassProgram &&
			this._sceneDepthPrepassProgramDirectiveTag !== directiveTag
		) {
			this._gl.deleteProgram(this._sceneDepthPrepassProgram.program);
			this._sceneDepthPrepassProgram = null;
		}
		const vertexSource = this._shaderSource("sceneDepthPrepassVertex");
		const fragmentSource = this._shaderSource("sceneDepthPrepassFragment");
		const vertexComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassVertex.composite"
		);
		const fragmentComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassFragment.composite"
		);
		return this._warmupProgram(
			"WebGLSceneDepthPrepassProgram",
			() =>
				this._sceneDepthPrepassProgram &&
				this._sceneDepthPrepassProgramDirectiveTag === directiveTag ?
					this._sceneDepthPrepassProgram
				:	null,
			() =>
				this._beginProgramCompile(
					vertexSource,
					fragmentSource,
					"WebGLSceneDepthPrepassProgram",
					{
						sourceMap: vertexComposite.sourceMap,
						sourceKind: "unknown",
					},
					{
						sourceMap: fragmentComposite.sourceMap,
						sourceKind: "unknown",
					},
				),
			() => {
				this._getBuiltinSceneDepthPrepassProgram();
			},
		);
	}

	private _warmupShaderMaterialDepthPrepassProgram(
		material: ShaderMaterial,
		mode: ShaderTargetMode
	): WebGLProgramWarmupHandle | null {
		const initialDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? "none";
		const shaderKey = this._createShaderMaterialDepthPrepassCacheKey(
			material,
			mode,
			initialDirectiveTag,
		);
		const cached = this._customSceneDepthPrepassPrograms.get(shaderKey);
		if (cached) {
			return this._recordWarmupHandle(
				this._createCompletedWarmupHandle(
					`WebGLShaderMaterialDepthPrepassProgram_${shaderKey}`,
				),
			);
		}

		const source = material.resolveWebGLDepthPrepassProgram(mode, {
			enableRuntimeInjects: this._supportsRuntimeInjects(),
		});
		if (!source) {
			this._warnMissingShaderMaterialDepthPrepassSource(material);
			return null;
		}

		const label = `WebGLShaderMaterialDepthPrepassProgram_${shaderKey}`;
		return this._warmupProgram(
			label,
			() => this._customSceneDepthPrepassPrograms.get(shaderKey) ?? null,
			() =>
				this._beginProgramCompile(
					source.vertexCode,
					source.fragmentCode,
					label,
					{
						sourceMap: createInlineShaderSourceMap(
							source.vertexCode,
							`<shader-material:${shaderKey}:vertex-depth>`,
							"source",
						),
						variantKey: shaderKey,
						materialId: String(material.shaderId),
						sourceKind: "custom-material",
					},
					{
						sourceMap: createInlineShaderSourceMap(
							source.fragmentCode,
							`<shader-material:${shaderKey}:fragment-depth>`,
							"source",
						),
						variantKey: shaderKey,
						materialId: String(material.shaderId),
						sourceKind: "custom-material",
					},
				),
			() => {
				this._getShaderMaterialDepthPrepassProgram(material, mode);
			},
			(error) => {
				if (!this._isWarnMode()) {
					throw error;
				}
				const key =
					`webgl-shader-material-depth-prepass-compile-failed-` +
					`${material.shaderId}`;
				const message =
					`ShaderMaterial ${material.name} custom WebGL depth prepass ` +
					`shader compile failed; skipping Early Z prepass for that ` +
					`material. ${String(error)}`;
				this._warn(key, message);
			},
		);
	}

	private _getSceneLightLimits(): WebGLSceneLightLimits {
		return {
			maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
			maxPointLights: MAX_POINT_LIGHTS,
			maxSpotLights: MAX_SPOT_LIGHTS,
			enableShadowTransmittance: this._enableShadowTransmittanceSampling,
			enableIrradianceProbeGrid: this._enableIrradianceProbeGridSampling,
		};
	}

	private _createShaderMaterialCacheKey(
		material: ShaderMaterial,
		mode: ShaderTargetMode,
		directiveTag: string,
	): string {
		return (
			`${material.getWebGLCacheKey()}` +
			`|mode:${mode}` +
			`|runtime:${this._shaderRuntime?.revision ?? 0}` +
			`|directive:${directiveTag}`
		);
	}

	private _createShaderMaterialDepthPrepassCacheKey(
		material: ShaderMaterial,
		mode: ShaderTargetMode,
		directiveTag: string,
	): string {
		return `depth|${this._createShaderMaterialCacheKey(material, mode, directiveTag)}`;
	}

	private _getBuiltinSceneProgram(): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		if (this._sceneProgram && this._sceneProgramDirectiveTag === directiveTag) {
			return this._sceneProgram;
		}
		if (this._sceneProgram && this._sceneProgramDirectiveTag !== directiveTag) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		if (!this._sceneProgram) {
			const limits = this._getSceneLightLimits();
			const sceneShaderSource = ShaderSource.get("webgl.scene.raw", {
				limits,
			});
			const sceneCompositeSource = ShaderSource.get("webgl.scene.composite", {
				limits,
			});
			this._sceneProgram = this._createSceneProgram(
				sceneShaderSource.vertex,
				sceneShaderSource.fragment,
				"WebGLSceneProgram",
				{
					sourceMap: sceneCompositeSource.vertex.sourceMap,
					sourceKind: "unknown",
				},
				{
					sourceMap: sceneCompositeSource.fragment.sourceMap,
					sourceKind: "unknown",
				},
			);
		}
		this._sceneProgramDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? directiveTag;
		return this._sceneProgram;
	}

	private _getBuiltinSceneDepthPrepassProgram(): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		if (
			this._sceneDepthPrepassProgram &&
			this._sceneDepthPrepassProgramDirectiveTag === directiveTag
		) {
			return this._sceneDepthPrepassProgram;
		}
		if (
			this._sceneDepthPrepassProgram &&
			this._sceneDepthPrepassProgramDirectiveTag !== directiveTag
		) {
			this._gl.deleteProgram(this._sceneDepthPrepassProgram.program);
			this._sceneDepthPrepassProgram = null;
		}
		if (!this._sceneDepthPrepassProgram) {
			const vertexComposite = ShaderSource.get(
				"webgl.part.sceneDepthPrepassVertex.composite"
			);
			const fragmentComposite = ShaderSource.get(
				"webgl.part.sceneDepthPrepassFragment.composite"
			);
			this._sceneDepthPrepassProgram = this._createSceneProgram(
				this._shaderSource("sceneDepthPrepassVertex"),
				this._shaderSource("sceneDepthPrepassFragment"),
				"WebGLSceneDepthPrepassProgram",
				{
					sourceMap: vertexComposite.sourceMap,
					sourceKind: "unknown",
				},
				{
					sourceMap: fragmentComposite.sourceMap,
					sourceKind: "unknown",
				},
			);
		}
		this._sceneDepthPrepassProgramDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? directiveTag;
		return this._sceneDepthPrepassProgram;
	}

	private _getShaderMaterialSceneProgram(
		material: ShaderMaterial,
		mode: ShaderTargetMode
	): WebGLSceneProgram | null {
		const initialDirectiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "none";
		const shaderKey = this._createShaderMaterialCacheKey(
			material,
			mode,
			initialDirectiveTag,
		);
		const cached = this._customScenePrograms.get(shaderKey);
		if (cached) {
			return cached;
		}

		let source: { vertexCode: string; fragmentCode: string };
		let customSamplerUniforms: string[] = [];
		let customUniforms: string[] = [];
		try {
			source = material.resolveWebGLProgram(mode, {
				enableRuntimeInjects: this._supportsRuntimeInjects(),
			});
			customSamplerUniforms = this._collectCustomSamplerUniforms(material);
			customUniforms = this._collectCustomUniforms(material);
		} catch (error) {
			const key = `webgl-shader-material-missing-source-${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} has no WebGL GLSL source; ` +
				`using built-in scene shader. ${String(error)}`;
			this._warn(key, message);
			return null;
		}

		let sceneProgram: WebGLSceneProgram;
		try {
			sceneProgram = this._createSceneProgram(
				source.vertexCode,
				source.fragmentCode,
				`WebGLShaderMaterialProgram_${shaderKey}`,
				{
					sourceMap: createInlineShaderSourceMap(
						source.vertexCode,
						`<shader-material:${shaderKey}:vertex>`,
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				{
					sourceMap: createInlineShaderSourceMap(
						source.fragmentCode,
						`<shader-material:${shaderKey}:fragment>`,
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				customSamplerUniforms,
				customUniforms,
			);
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			const key = `webgl-shader-material-compile-failed-${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} custom WebGL shader compile failed; ` +
				`using built-in scene shader. ${String(error)}`;
			this._warn(key, message);
			return null;
		}
		const finalDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? initialDirectiveTag;
		const finalShaderKey = this._createShaderMaterialCacheKey(
			material,
			mode,
			finalDirectiveTag,
		);
		const existingFinal = this._customScenePrograms.get(finalShaderKey);
		if (existingFinal) {
			this._gl.deleteProgram(sceneProgram.program);
			return existingFinal;
		}
		this._customScenePrograms.set(finalShaderKey, sceneProgram);
		if (finalShaderKey !== shaderKey) {
			this._customScenePrograms.delete(shaderKey);
		}
		return sceneProgram;
	}

	private _getShaderMaterialDepthPrepassProgram(
		material: ShaderMaterial,
		mode: ShaderTargetMode
	): WebGLSceneProgram | null {
		const initialDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? "none";
		const shaderKey = this._createShaderMaterialDepthPrepassCacheKey(
			material,
			mode,
			initialDirectiveTag,
		);
		const cached = this._customSceneDepthPrepassPrograms.get(shaderKey);
		if (cached) {
			return cached;
		}

		const source = material.resolveWebGLDepthPrepassProgram(mode, {
			enableRuntimeInjects: this._supportsRuntimeInjects(),
		});
		if (!source) {
			this._warnMissingShaderMaterialDepthPrepassSource(material);
			return null;
		}

		let sceneProgram: WebGLSceneProgram;
		try {
			sceneProgram = this._createSceneProgram(
				source.vertexCode,
				source.fragmentCode,
				`WebGLShaderMaterialDepthPrepassProgram_${shaderKey}`,
				{
					sourceMap: createInlineShaderSourceMap(
						source.vertexCode,
						`<shader-material:${shaderKey}:vertex-depth>`,
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				{
					sourceMap: createInlineShaderSourceMap(
						source.fragmentCode,
						`<shader-material:${shaderKey}:fragment-depth>`,
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				this._collectCustomSamplerUniforms(material),
				this._collectCustomUniforms(material),
			);
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			const key =
				`webgl-shader-material-depth-prepass-compile-failed-` +
				`${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} custom WebGL depth prepass ` +
				`shader compile failed; skipping Early Z prepass for that material. ` +
				`${String(error)}`;
			this._warn(key, message);
			return null;
		}
		const finalDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? initialDirectiveTag;
		const finalShaderKey = this._createShaderMaterialDepthPrepassCacheKey(
			material,
			mode,
			finalDirectiveTag,
		);
		const existingFinal =
			this._customSceneDepthPrepassPrograms.get(finalShaderKey);
		if (existingFinal) {
			this._gl.deleteProgram(sceneProgram.program);
			return existingFinal;
		}
		this._customSceneDepthPrepassPrograms.set(finalShaderKey, sceneProgram);
		if (finalShaderKey !== shaderKey) {
			this._customSceneDepthPrepassPrograms.delete(shaderKey);
		}
		return sceneProgram;
	}

	private _warnMissingShaderMaterialDepthPrepassSource(
		material: ShaderMaterial
	): void {
		if (material.alphaMode !== AlphaMode.Mask) {
			return;
		}
		if (this._missingDepthPrepassShaderMaterialWarnings.has(material.shaderId)) {
			return;
		}
		this._missingDepthPrepassShaderMaterialWarnings.add(material.shaderId);
		const key =
			`webgl-shader-material-depth-prepass-missing-source-` +
			`${material.shaderId}`;
		const message =
			`ShaderMaterial ${material.name} uses AlphaMode.Mask but has no ` +
			`WebGL fragment-depth chunk; skipping Early Z prepass for that ` +
			`material.`;
		this._warn(key, message);
	}

	private _createSceneProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
		customSamplerUniforms: string[] = [],
		customUniforms: string[] = [],
	): WebGLSceneProgram {
		const program = this._createProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata,
		);
		return {
			program,
			uniforms: createWebGLSceneUniforms(
				this._gl,
				program,
				customSamplerUniforms,
				customUniforms
			),
		};
	}

	public getEnvironmentProgram(): WebGLEnvironmentProgram {
		if (this._environmentProgram) {
			return this._environmentProgram;
		}
		const program = this._createProgram(
			this._shaderSource("environmentVertex"),
			this._shaderSource("environmentFragment"),
			"WebGLEnvironmentProgram",
		);
		this._environmentProgram = {
			program,
			uniforms: {
				environmentMap: this._gl.getUniformLocation(program, "uEnvironmentMap"),
				environmentBasisRight: this._gl.getUniformLocation(program, "uEnvironmentBasisRight"),
				environmentBasisUp: this._gl.getUniformLocation(program, "uEnvironmentBasisUp"),
				environmentBasisBackward: this._gl.getUniformLocation(program, "uEnvironmentBasisBackward"),
				environmentIsOrthographic: this._gl.getUniformLocation(program, "uEnvironmentIsOrthographic"),
				environmentMapIsLinear: this._gl.getUniformLocation(program, "uEnvironmentMapIsLinear"),
				environmentBackgroundTint: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundTint"
				),
				environmentBackgroundExposure: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundExposure"
				),
				environmentBackgroundStrength: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundStrength"
				),
			},
		};
		return this._environmentProgram;
	}

	public getPresentProgram(): WebGLPresentProgram {
		if (this._presentProgram) {
			return this._presentProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("presentFragment"),
			"WebGLPresentProgram",
		);
		this._presentProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				applyGamma: this._gl.getUniformLocation(program, "uApplyGamma"),
			},
		};
		return this._presentProgram;
	}

	/**
	 * Attempts to resolve the present program without blocking on shader status.
	 *
	 * @returns The cached/finalized present program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetPresentProgram(): WebGLPresentProgram | null {
		if (this._presentProgram) {
			return this._presentProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("presentFragment"),
			"WebGLPresentProgram",
		);
		if (!program) {
			return null;
		}
		this._presentProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				applyGamma: this._gl.getUniformLocation(program, "uApplyGamma"),
			},
		};
		return this._presentProgram;
	}

	public getParticleProgram(): WebGLParticleProgram {
		if (this._particleProgram) {
			return this._particleProgram;
		}
		const program = this._createProgram(
			this._shaderSource("particleVertex"),
			this._shaderSource("particleFragment"),
			"WebGLParticleProgram",
		);
		this._particleProgram = {
			program,
			uniforms: {
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				particleMap: this._gl.getUniformLocation(program, "uParticleMap"),
				uvTransformA: this._gl.getUniformLocation(program, "uUvTransformA"),
				uvTransformB: this._gl.getUniformLocation(program, "uUvTransformB"),
				mapIsLinear: this._gl.getUniformLocation(program, "uMapIsLinear"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				fogParams0: this._gl.getUniformLocation(program, "uFogParams0"),
				fogParams1: this._gl.getUniformLocation(program, "uFogParams1"),
				oitPassMode: this._gl.getUniformLocation(program, "uOITPassMode"),
			},
		};
		return this._particleProgram;
	}

	public getFXAAProgram(): WebGLFXAAProgram {
		if (this._fxaaProgram) {
			return this._fxaaProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fxaaFragment"),
			"WebGLFXAAProgram",
		);
		this._fxaaProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._fxaaProgram;
	}

	/**
	 * Attempts to resolve the FXAA program without blocking on shader status.
	 *
	 * @returns The cached/finalized FXAA program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetFXAAProgram(): WebGLFXAAProgram | null {
		if (this._fxaaProgram) {
			return this._fxaaProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fxaaFragment"),
			"WebGLFXAAProgram",
		);
		if (!program) {
			return null;
		}
		this._fxaaProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._fxaaProgram;
	}

	public getToneMappingProgram(): WebGLToneMappingProgram {
		if (this._toneMappingProgram) {
			return this._toneMappingProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("toneMappingFragment"),
			"WebGLToneMappingProgram",
		);
		this._toneMappingProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._toneMappingProgram;
	}

	/**
	 * Attempts to resolve the tone mapping program without blocking on shader status.
	 *
	 * @returns The cached/finalized tone mapping program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetToneMappingProgram(): WebGLToneMappingProgram | null {
		if (this._toneMappingProgram) {
			return this._toneMappingProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("toneMappingFragment"),
			"WebGLToneMappingProgram",
		);
		if (!program) {
			return null;
		}
		this._toneMappingProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._toneMappingProgram;
	}

	public getInteractionOutlineProgram(): WebGLInteractionOutlineProgram {
		if (this._interactionOutlineProgram) {
			return this._interactionOutlineProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("interactionOutlineFragment"),
			"WebGLInteractionOutlineProgram",
		);
		this._interactionOutlineProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				outlineColor: this._gl.getUniformLocation(program, "uOutlineColor"),
				outlineParams: this._gl.getUniformLocation(program, "uOutlineParams"),
				viewportSize: this._gl.getUniformLocation(program, "uViewportSize"),
				circleCount: this._gl.getUniformLocation(program, "uCircleCount"),
				circles: this._gl.getUniformLocation(program, "uCircles[0]"),
			},
		};
		return this._interactionOutlineProgram;
	}

	/**
	 * Attempts to resolve the interaction outline program without blocking.
	 *
	 * @returns The cached/finalized outline program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetInteractionOutlineProgram(): WebGLInteractionOutlineProgram | null {
		if (this._interactionOutlineProgram) {
			return this._interactionOutlineProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("interactionOutlineFragment"),
			"WebGLInteractionOutlineProgram",
		);
		if (!program) {
			return null;
		}
		this._interactionOutlineProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				outlineColor: this._gl.getUniformLocation(program, "uOutlineColor"),
				outlineParams: this._gl.getUniformLocation(program, "uOutlineParams"),
				viewportSize: this._gl.getUniformLocation(program, "uViewportSize"),
				circleCount: this._gl.getUniformLocation(program, "uCircleCount"),
				circles: this._gl.getUniformLocation(program, "uCircles[0]"),
			},
		};
		return this._interactionOutlineProgram;
	}

	public getColorFilterProgram(): WebGLColorFilterProgram {
		if (this._colorFilterProgram) {
			return this._colorFilterProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("colorFilterFragment"),
			"WebGLColorFilterProgram",
		);
		this._colorFilterProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				filterParams0: this._gl.getUniformLocation(program, "uFilterParams0"),
				filterParams1: this._gl.getUniformLocation(program, "uFilterParams1"),
			},
		};
		return this._colorFilterProgram;
	}

	/**
	 * Attempts to resolve the color filter program without blocking on shader status.
	 *
	 * @returns The cached/finalized color filter program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetColorFilterProgram(): WebGLColorFilterProgram | null {
		if (this._colorFilterProgram) {
			return this._colorFilterProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("colorFilterFragment"),
			"WebGLColorFilterProgram",
		);
		if (!program) {
			return null;
		}
		this._colorFilterProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				filterParams0: this._gl.getUniformLocation(program, "uFilterParams0"),
				filterParams1: this._gl.getUniformLocation(program, "uFilterParams1"),
			},
		};
		return this._colorFilterProgram;
	}

	public getBloomProgram(): WebGLBloomProgram {
		if (this._bloomProgram) {
			return this._bloomProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("bloomFragment"),
			"WebGLBloomProgram",
		);
		this._bloomProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				bloomParams: this._gl.getUniformLocation(program, "uBloomParams"),
			},
		};
		return this._bloomProgram;
	}

	/**
	 * Attempts to resolve the bloom program without blocking on shader status.
	 *
	 * @returns The cached/finalized bloom program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetBloomProgram(): WebGLBloomProgram | null {
		if (this._bloomProgram) {
			return this._bloomProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("bloomFragment"),
			"WebGLBloomProgram",
		);
		if (!program) {
			return null;
		}
		this._bloomProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				bloomParams: this._gl.getUniformLocation(program, "uBloomParams"),
			},
		};
		return this._bloomProgram;
	}

	public getMotionBlurProgram(): WebGLMotionBlurProgram {
		if (this._motionBlurProgram) {
			return this._motionBlurProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("motionBlurFragment"),
			"WebGLMotionBlurProgram",
		);
		this._motionBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				motionParams: this._gl.getUniformLocation(program, "uMotionParams"),
				centerWeight: this._gl.getUniformLocation(program, "uCenterWeight"),
			},
		};
		return this._motionBlurProgram;
	}

	/**
	 * Attempts to resolve the motion blur program without blocking on shader status.
	 *
	 * @returns The cached/finalized motion blur program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetMotionBlurProgram(): WebGLMotionBlurProgram | null {
		if (this._motionBlurProgram) {
			return this._motionBlurProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("motionBlurFragment"),
			"WebGLMotionBlurProgram",
		);
		if (!program) {
			return null;
		}
		this._motionBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				motionParams: this._gl.getUniformLocation(program, "uMotionParams"),
				centerWeight: this._gl.getUniformLocation(program, "uCenterWeight"),
			},
		};
		return this._motionBlurProgram;
	}

	public getDOFProgram(): WebGLDOFProgram {
		if (this._dofProgram) {
			return this._dofProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("dofFragment"),
			"WebGLDOFProgram",
		);
		this._dofProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				focusParams: this._gl.getUniformLocation(program, "uFocusParams"),
				dofParams: this._gl.getUniformLocation(program, "uDOFParams"),
				chromaticAberration: this._gl.getUniformLocation(program, "uChromaticAberration"),
			},
		};
		return this._dofProgram;
	}

	/**
	 * Attempts to resolve the depth of field program without blocking on shader status.
	 *
	 * @returns The cached/finalized DoF program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetDOFProgram(): WebGLDOFProgram | null {
		if (this._dofProgram) {
			return this._dofProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("dofFragment"),
			"WebGLDOFProgram",
		);
		if (!program) {
			return null;
		}
		this._dofProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				focusParams: this._gl.getUniformLocation(program, "uFocusParams"),
				dofParams: this._gl.getUniformLocation(program, "uDOFParams"),
				chromaticAberration: this._gl.getUniformLocation(program, "uChromaticAberration"),
			},
		};
		return this._dofProgram;
	}

	public getShadowDepthProgram(): WebGLShadowDepthProgram {
		if (this._shadowDepthProgram) {
			return this._shadowDepthProgram;
		}
		const program = this._createProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowDepthFragment"),
			"WebGLShadowDepthProgram",
		);
		this._shadowDepthProgram = {
			program,
			uniforms: {
				mvp: this._gl.getUniformLocation(program, "uMvp"),
			},
		};
		return this._shadowDepthProgram;
	}

	public getShadowTransmittanceProgram(): WebGLShadowTransmittanceProgram {
		if (this._shadowTransmittanceProgram) {
			return this._shadowTransmittanceProgram;
		}
		const program = this._createProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowTransmittanceFragment"),
			"WebGLShadowTransmittanceProgram",
		);
		this._shadowTransmittanceProgram = {
			program,
			uniforms: {
				mvp: this._gl.getUniformLocation(program, "uMvp"),
				transmittance: this._gl.getUniformLocation(program, "uTransmittance"),
			},
		};
		return this._shadowTransmittanceProgram;
	}

	public getCopyProgram(): WebGLCopyProgram {
		if (this._copyProgram) {
			return this._copyProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("copyFragment"),
			"WebGLCopyProgram",
		);
		this._copyProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._copyProgram;
	}

	public getOITResolveProgram(): WebGLOITResolveProgram {
		if (this._oitResolveProgram) {
			return this._oitResolveProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("oitResolveFragment"),
			"WebGLOITResolveProgram",
		);
		this._oitResolveProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				oitAccumMap: this._gl.getUniformLocation(program, "uOITAccumMap"),
				oitRevealMap: this._gl.getUniformLocation(program, "uOITRevealMap"),
			},
		};
		return this._oitResolveProgram;
	}

	public getSSAORawProgram(): WebGLSSAORawProgram {
		if (this._ssaoRawProgram) {
			return this._ssaoRawProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoRawFragment"),
			"WebGLSSAORawProgram",
		);
		this._ssaoRawProgram = {
			program,
			uniforms: {
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				gtao: this._gl.getUniformLocation(program, "uGTAO"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				basisBackward: this._gl.getUniformLocation(program, "uBasisBackward"),
			},
		};
		return this._ssaoRawProgram;
	}

	/**
	 * Attempts to resolve the SSAO raw program without blocking on shader status.
	 *
	 * @returns The cached/finalized SSAO raw program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetSSAORawProgram(): WebGLSSAORawProgram | null {
		if (this._ssaoRawProgram) {
			return this._ssaoRawProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoRawFragment"),
			"WebGLSSAORawProgram",
		);
		if (!program) {
			return null;
		}
		this._ssaoRawProgram = {
			program,
			uniforms: {
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				gtao: this._gl.getUniformLocation(program, "uGTAO"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				basisBackward: this._gl.getUniformLocation(program, "uBasisBackward"),
			},
		};
		return this._ssaoRawProgram;
	}

	public getSSAOBlurProgram(): WebGLSSAOBlurProgram {
		if (this._ssaoBlurProgram) {
			return this._ssaoBlurProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoBlurFragment"),
			"WebGLSSAOBlurProgram",
		);
		this._ssaoBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
			},
		};
		return this._ssaoBlurProgram;
	}

	/**
	 * Attempts to resolve the SSAO blur program without blocking on shader status.
	 *
	 * @returns The cached/finalized SSAO blur program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetSSAOBlurProgram(): WebGLSSAOBlurProgram | null {
		if (this._ssaoBlurProgram) {
			return this._ssaoBlurProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoBlurFragment"),
			"WebGLSSAOBlurProgram",
		);
		if (!program) {
			return null;
		}
		this._ssaoBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
			},
		};
		return this._ssaoBlurProgram;
	}

	public getSSAOCombineProgram(): WebGLSSAOCombineProgram {
		if (this._ssaoCombineProgram) {
			return this._ssaoCombineProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoCombineFragment"),
			"WebGLSSAOCombineProgram",
		);
		this._ssaoCombineProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				aoMap: this._gl.getUniformLocation(program, "uAoMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
			},
		};
		return this._ssaoCombineProgram;
	}

	/**
	 * Attempts to resolve the SSAO combine program without blocking on shader status.
	 *
	 * @returns The cached/finalized SSAO combine program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetSSAOCombineProgram(): WebGLSSAOCombineProgram | null {
		if (this._ssaoCombineProgram) {
			return this._ssaoCombineProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoCombineFragment"),
			"WebGLSSAOCombineProgram",
		);
		if (!program) {
			return null;
		}
		this._ssaoCombineProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				aoMap: this._gl.getUniformLocation(program, "uAoMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
			},
		};
		return this._ssaoCombineProgram;
	}

	public getTAAProgram(): WebGLTAAProgram {
		if (this._taaProgram) {
			return this._taaProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("taaFragment"),
			"WebGLTAAProgram",
		);
		this._taaProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				historyMap: this._gl.getUniformLocation(program, "uHistoryMap"),
				motionMap: this._gl.getUniformLocation(program, "uMotionMap"),
				motionHistory: this._gl.getUniformLocation(program, "uMotionHistory"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				historyWeight: this._gl.getUniformLocation(program, "uHistoryWeight"),
				depthThreshold: this._gl.getUniformLocation(program, "uDepthThreshold"),
				motionFactor: this._gl.getUniformLocation(program, "uMotionFactor"),
				varianceClampGamma: this._gl.getUniformLocation(program, "uVarianceClampGamma"),
				sharpen: this._gl.getUniformLocation(program, "uSharpen"),
				historyValid: this._gl.getUniformLocation(program, "uHistoryValid"),
			},
		};
		return this._taaProgram;
	}

	/**
	 * Attempts to resolve the TAA program without blocking on shader status.
	 *
	 * @returns The cached/finalized TAA program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetTAAProgram(): WebGLTAAProgram | null {
		if (this._taaProgram) {
			return this._taaProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("taaFragment"),
			"WebGLTAAProgram",
		);
		if (!program) {
			return null;
		}
		this._taaProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				historyMap: this._gl.getUniformLocation(program, "uHistoryMap"),
				motionMap: this._gl.getUniformLocation(program, "uMotionMap"),
				motionHistory: this._gl.getUniformLocation(program, "uMotionHistory"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				historyWeight: this._gl.getUniformLocation(program, "uHistoryWeight"),
				depthThreshold: this._gl.getUniformLocation(program, "uDepthThreshold"),
				motionFactor: this._gl.getUniformLocation(program, "uMotionFactor"),
				varianceClampGamma: this._gl.getUniformLocation(program, "uVarianceClampGamma"),
				sharpen: this._gl.getUniformLocation(program, "uSharpen"),
				historyValid: this._gl.getUniformLocation(program, "uHistoryValid"),
			},
		};
		return this._taaProgram;
	}

	public getSSRProgram(): WebGLSSRProgram {
		if (this._ssrProgram) {
			return this._ssrProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("postProcessStubFragment"),
			"WebGLSSRProgram",
		);
		this._ssrProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._ssrProgram;
	}

	public getVolumetricProgram(): WebGLVolumetricProgram {
		if (this._volumetricProgram) {
			return this._volumetricProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("postProcessStubFragment"),
			"WebGLVolumetricProgram",
		);
		this._volumetricProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._volumetricProgram;
	}

	public getFogProgram(): WebGLFogProgram {
		if (this._fogProgram) {
			return this._fogProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fogFragment"),
			"WebGLFogProgram",
		);
		this._fogProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				fogParams0: this._gl.getUniformLocation(program, "uFogParams0"),
				fogParams1: this._gl.getUniformLocation(program, "uFogParams1"),
			},
		};
		return this._fogProgram;
	}

	/**
	 * Attempts to resolve the fog program without blocking on shader status.
	 *
	 * @returns The cached/finalized fog program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetFogProgram(): WebGLFogProgram | null {
		if (this._fogProgram) {
			return this._fogProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fogFragment"),
			"WebGLFogProgram",
		);
		if (!program) {
			return null;
		}
		this._fogProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				fogParams0: this._gl.getUniformLocation(program, "uFogParams0"),
				fogParams1: this._gl.getUniformLocation(program, "uFogParams1"),
			},
		};
		return this._fogProgram;
	}

	private _shaderSource(part: WebGLShaderPart): string {
		return ShaderSource.get(`webgl.part.${part}.raw`);
	}

	public destroy(): void {
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this._disposePrograms();
	}

	private _createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgram {
		const precompiled = this._precompiledPrograms.get(label);
		if (precompiled) {
			this._precompiledPrograms.delete(label);
			return precompiled;
		}
		const pending = this._pendingProgramCompiles.get(label);
		if (pending) {
			return this._finalizeProgramCompile(pending);
		}
		return this._finalizeProgramCompile(
			this._beginProgramCompile(
				vertexSource,
				fragmentSource,
				label,
				vertexMetadata,
				fragmentMetadata,
			),
		);
	}

	private _tryCreateProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgram | null {
		const precompiled = this._precompiledPrograms.get(label);
		if (precompiled) {
			this._precompiledPrograms.delete(label);
			return precompiled;
		}
		const pending =
			this._pendingProgramCompiles.get(label) ??
			this._beginProgramCompile(
				vertexSource,
				fragmentSource,
				label,
				vertexMetadata,
				fragmentMetadata,
			);
		const program = this._tryFinalizeProgramCompile(pending);
		if (!program) {
			this._notifyProgramCompilePending();
		}
		return program;
	}

	private _warmupProgram(
		label: string,
		getCached: () => { program: WebGLProgram } | null,
		beginCompile: () => WebGLPendingProgramCompile,
		finalizeReadyProgram: () => void,
		handleCompileError?: (error: unknown) => void,
	): WebGLProgramWarmupHandle {
		if (getCached()) {
			return this._recordWarmupHandle(
				this._createCompletedWarmupHandle(label),
			);
		}
		const pending =
			this._pendingProgramCompiles.get(label) ?? beginCompile();
		return this._recordWarmupHandle(
			this._createPendingWarmupHandle(
				label,
				pending,
				finalizeReadyProgram,
				handleCompileError,
			),
		);
	}

	private _createCompletedWarmupHandle(label: string): WebGLProgramWarmupHandle {
		return {
			label,
			isComplete: () => true,
			finalize: () => {},
		};
	}

	private _createPendingWarmupHandle(
		label: string,
		pending: WebGLPendingProgramCompile,
		finalizeReadyProgram: () => void,
		handleCompileError?: (error: unknown) => void,
	): WebGLProgramWarmupHandle {
		return {
			label,
			isComplete: () => this._isProgramCompileComplete(pending),
			finalize: () => {
				if (pending.status === "failed") {
					if (handleCompileError) {
						handleCompileError(pending.error);
						return;
					}
					throw pending.error;
				}
				if (pending.status !== "ready") {
					try {
						const program = this._finalizeProgramCompile(pending);
						this._precompiledPrograms.set(label, program);
					} catch (error) {
						if (handleCompileError) {
							handleCompileError(error);
							return;
						}
						throw error;
					}
				}
				finalizeReadyProgram();
			},
		};
	}

	private _recordWarmupHandle(
		handle: WebGLProgramWarmupHandle
	): WebGLProgramWarmupHandle {
		this._warmupHandleLog.push(handle);
		return handle;
	}

	private _notifyProgramCompilePending(): void {
		if (
			this._lastPendingNotificationFrame === this._compileFrameIndex ||
			!this._onProgramCompilePending
		) {
			return;
		}
		this._lastPendingNotificationFrame = this._compileFrameIndex;
		this._onProgramCompilePending();
	}

	private _beginProgramCompile(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLPendingProgramCompile {
		const gl = this._gl;
		let vertexShader: WebGLPendingShaderCompile | null = null;
		let fragmentShader: WebGLPendingShaderCompile | null = null;
		let program: WebGLProgram | null = null;
		try {
			vertexShader = this._beginShaderCompile(
				gl.VERTEX_SHADER,
				vertexSource,
				`${label}:vertex`,
				vertexMetadata,
			);
			fragmentShader = this._beginShaderCompile(
				gl.FRAGMENT_SHADER,
				fragmentSource,
				`${label}:fragment`,
				fragmentMetadata,
			);
			program = gl.createProgram();
			if (!program) {
				throw new Error(`Failed to create WebGL program (${label})`);
			}

			gl.attachShader(program, vertexShader.shader);
			gl.attachShader(program, fragmentShader.shader);
			gl.linkProgram(program);
			const pending = {
				label,
				vertex: vertexShader,
				fragment: fragmentShader,
				program,
				startedFrame: this._compileFrameIndex,
				status: "pending" as const,
				finalized: false,
				error: null,
			};
			this._pendingProgramCompiles.set(label, pending);
			return pending;
		} catch (error) {
			if (program) {
				gl.deleteProgram(program);
			}
			if (vertexShader) {
				gl.deleteShader(vertexShader.shader);
			}
			if (fragmentShader) {
				gl.deleteShader(fragmentShader.shader);
			}
			this._pendingProgramCompiles.delete(label);
			throw error;
		}
	}

	private _tryFinalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): WebGLProgram | null {
		if (pending.status === "ready") {
			return pending.program;
		}
		if (pending.status === "failed") {
			throw pending.error;
		}
		if (!this._canFinalizeProgramCompile(pending)) {
			return null;
		}
		return this._finalizeProgramCompile(pending);
	}

	private _finalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): WebGLProgram {
		if (pending.status === "ready") {
			return pending.program;
		}
		if (pending.status === "failed") {
			throw pending.error;
		}
		const gl = this._gl;
		try {
			this._finalizeShaderCompile(pending.vertex);
			this._finalizeShaderCompile(pending.fragment);
			const linked = !!gl.getProgramParameter(
				pending.program,
				gl.LINK_STATUS,
			);
			if (!linked) {
				const log = gl.getProgramInfoLog(pending.program) || "No program link log";
				const messages = parseWebGLShaderInfoLog(log);
				throw new ShaderCompileError({
					backend: "webgl",
					language: "glsl",
					stage: "unknown",
					label: pending.label,
					sourceKind:
						pending.vertex.sourceKind ??
						pending.fragment.sourceKind ??
						"unknown",
					variantKey:
						pending.vertex.variantKey ?? pending.fragment.variantKey,
					materialId:
						pending.vertex.materialId ?? pending.fragment.materialId,
					code: `${pending.vertex.code}\n\n${pending.fragment.code}`,
					sourceMap: null,
					messages:
						messages.length > 0 ? messages : [this._toCompilerMessage(log)],
					rawLog: log,
				});
			}
			this._validateProgramIfRequested(pending.program, pending.label);
			pending.finalized = true;
			pending.status = "ready";
			return pending.program;
		} catch (error) {
			pending.finalized = true;
			pending.status = "failed";
			pending.error = error;
			gl.deleteProgram(pending.program);
			throw error;
		} finally {
			gl.deleteShader(pending.vertex.shader);
			gl.deleteShader(pending.fragment.shader);
			this._pendingProgramCompiles.delete(pending.label);
		}
	}

	private _canFinalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): boolean {
		if (this._parallelShaderCompile) {
			return this._isProgramCompileComplete(pending);
		}
		const frameAge = this._compileFrameIndex - pending.startedFrame;
		if (frameAge < WEBGL_FALLBACK_READY_FRAME_DELAY) {
			return false;
		}
		if (
			this._fallbackFinalizesThisFrame >=
			WEBGL_FALLBACK_FINALIZE_BUDGET_PER_FRAME
		) {
			return false;
		}
		this._fallbackFinalizesThisFrame++;
		return true;
	}

	private _isProgramCompileComplete(
		pending: WebGLPendingProgramCompile
	): boolean {
		if (pending.status !== "pending" || !this._parallelShaderCompile) {
			return true;
		}
		return !!this._gl.getProgramParameter(
			pending.program,
			this._parallelShaderCompile.COMPLETION_STATUS_KHR,
		);
	}

	private _validateProgramIfRequested(
		program: WebGLProgram,
		label: string
	): void {
		if (!this._validatePrograms) {
			return;
		}
		const gl = this._gl;
		gl.validateProgram(program);
		const validateStatus = gl.getProgramParameter(program, gl.VALIDATE_STATUS);
		if (validateStatus === false) {
			const key = `webgl-program-validate-${label}`;
			const message =
				`WebGL program validation reported issues (${label}): ` +
				`${gl.getProgramInfoLog(program) || "no log"}`;
			this._warn(key, message);
		}
	}

	private _beginShaderCompile(
		type: number,
		source: string,
		label: string,
		metadata?: ShaderCompileMetadata,
	): WebGLPendingShaderCompile {
		const stage = type === this._gl.VERTEX_SHADER ? "vertex" : "fragment";
		const sourceKind =
			metadata?.sourceKind ??
			(label.startsWith("WebGLShaderMaterialProgram_") ? "custom-material" : "unknown");
		const processed = this._processShaderSource(
			source,
			stage,
			sourceKind,
			label,
			metadata?.sourceMap,
		);
		if (processed.hasErrors) {
			this._reportShaderRuntimeDiagnostics(label, processed);
		}

		const gl = this._gl;
		const shader = gl.createShader(type);
		if (!shader) {
			throw new Error(`Failed to create WebGL shader (${label})`);
		}
		gl.shaderSource(shader, processed.code);
		gl.compileShader(shader);
		return {
			shader,
			stage,
			label,
			sourceKind,
			variantKey: metadata?.variantKey,
			materialId: metadata?.materialId,
			code: processed.code,
			sourceMap: processed.sourceMap,
		};
	}

	private _finalizeShaderCompile(shader: WebGLPendingShaderCompile): void {
		const compiled = !!this._gl.getShaderParameter(
			shader.shader,
			this._gl.COMPILE_STATUS,
		);
		if (!compiled) {
			const log = this._gl.getShaderInfoLog(shader.shader) || "No shader compile log";
			const parsed = parseWebGLShaderInfoLog(log);
			throw new ShaderCompileError({
				backend: "webgl",
				language: "glsl",
				stage: shader.stage,
				label: shader.label,
				sourceKind: shader.sourceKind,
				variantKey: shader.variantKey,
				materialId: shader.materialId,
				code: shader.code,
				sourceMap: shader.sourceMap,
				messages: parsed.length > 0 ? parsed : [this._toCompilerMessage(log)],
				rawLog: log,
			});
		}
	}

	private _isWarnMode(): boolean {
		return this._shaderRuntime?.getMode() === "warn";
	}

	private _supportsRuntimeInjects(): boolean {
		return this._shaderCompileStage !== null || this._shaderRuntime !== null;
	}

	private _collectCustomSamplerUniforms(material: ShaderMaterial): string[] {
		const uniforms = new Set<string>();
		for (const binding of material.getTextureBindings()) {
			if (binding.webglUniform.trim().length <= 0) {
				continue;
			}
			uniforms.add(binding.webglUniform);
		}
		return [...uniforms];
	}

	private _collectCustomUniforms(material: ShaderMaterial): string[] {
		const uniforms = new Set<string>();
		for (const binding of material.getUniformBindings()) {
			if (binding.webglUniform.trim().length <= 0) {
				continue;
			}
			uniforms.add(binding.webglUniform);
		}
		return [...uniforms];
	}

	private _processShaderSource(
		source: string,
		stage: "vertex" | "fragment",
		sourceKind: "custom-material" | "unknown",
		label: string,
		sourceMap?: ShaderSourceSegmentMap | null,
	): ShaderProcessResult {
		const directiveSourcePath = sourceMap?.segments[0]?.sourcePath ?? label ?? "<webgl-shader>";
		if (this._shaderCompileStage) {
			return this._shaderCompileStage.compile({
				code: source,
				language: "glsl",
				stage,
				entryPoint: "main",
				label,
				sourceKind,
				sourceMap: sourceMap ?? null,
				directiveSourcePath,
			});
		}
		if (!this._shaderRuntime) {
			const effectiveSourceMap =
				sourceMap ?? createInlineShaderSourceMap(source, label, "source");
			return {
				code: source,
				sourceMap: effectiveSourceMap,
				composite: {
					code: source,
					sourceMap: effectiveSourceMap,
				},
				diagnostics: [],
				hasErrors: false,
				fromCache: false,
			};
		}
		return this._shaderRuntime.process({
			code: source,
			language: "glsl",
			stage,
			entryPoint: "main",
			label,
			sourceKind,
			sourceMap: sourceMap ?? null,
			directiveSourcePath,
		});
	}

	private _reportShaderRuntimeDiagnostics(label: string, result: ShaderProcessResult): void {
		for (const diagnostic of result.diagnostics) {
			const key =
				`webgl-shader-runtime-${diagnostic.severity}-` + `${diagnostic.code}-${label}`;
			const message =
				`WebGL shader runtime ${diagnostic.severity} [${label}] ` +
				`${diagnostic.code}: ${diagnostic.message}`;
			this._warn(key, message);
		}
	}

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLProgramLibrary",
			onceKey: key,
		});
	}

	private _toCompilerMessage(log: string): ShaderCompilerMessage {
		return {
			type: "error",
			message: log,
			raw: log,
		};
	}

	private _invalidateProgramCachesForShaderRuntime(): void {
		this._disposePrograms();
	}

	private _disposePrograms(): void {
		for (const pending of this._pendingProgramCompiles.values()) {
			this._gl.deleteShader(pending.vertex.shader);
			this._gl.deleteShader(pending.fragment.shader);
			this._gl.deleteProgram(pending.program);
		}
		this._pendingProgramCompiles.clear();
		for (const program of this._precompiledPrograms.values()) {
			this._gl.deleteProgram(program);
		}
		this._precompiledPrograms.clear();
		this._warmupHandleLog.length = 0;
		if (this._sceneProgram) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		this._sceneProgramDirectiveTag = "";
		for (const sceneProgram of this._customScenePrograms.values()) {
			this._gl.deleteProgram(sceneProgram.program);
		}
		this._customScenePrograms.clear();
		if (this._sceneDepthPrepassProgram) {
			this._gl.deleteProgram(this._sceneDepthPrepassProgram.program);
			this._sceneDepthPrepassProgram = null;
		}
		this._sceneDepthPrepassProgramDirectiveTag = "";
		for (const sceneProgram of this._customSceneDepthPrepassPrograms.values()) {
			this._gl.deleteProgram(sceneProgram.program);
		}
		this._customSceneDepthPrepassPrograms.clear();
		this._missingDepthPrepassShaderMaterialWarnings.clear();
		if (this._environmentProgram) {
			this._gl.deleteProgram(this._environmentProgram.program);
			this._environmentProgram = null;
		}
		if (this._presentProgram) {
			this._gl.deleteProgram(this._presentProgram.program);
			this._presentProgram = null;
		}
		if (this._particleProgram) {
			this._gl.deleteProgram(this._particleProgram.program);
			this._particleProgram = null;
		}
		if (this._fxaaProgram) {
			this._gl.deleteProgram(this._fxaaProgram.program);
			this._fxaaProgram = null;
		}
		if (this._toneMappingProgram) {
			this._gl.deleteProgram(this._toneMappingProgram.program);
			this._toneMappingProgram = null;
		}
		if (this._colorFilterProgram) {
			this._gl.deleteProgram(this._colorFilterProgram.program);
			this._colorFilterProgram = null;
		}
		if (this._interactionOutlineProgram) {
			this._gl.deleteProgram(this._interactionOutlineProgram.program);
			this._interactionOutlineProgram = null;
		}
		if (this._bloomProgram) {
			this._gl.deleteProgram(this._bloomProgram.program);
			this._bloomProgram = null;
		}
		if (this._motionBlurProgram) {
			this._gl.deleteProgram(this._motionBlurProgram.program);
			this._motionBlurProgram = null;
		}
		if (this._dofProgram) {
			this._gl.deleteProgram(this._dofProgram.program);
			this._dofProgram = null;
		}
		if (this._shadowDepthProgram) {
			this._gl.deleteProgram(this._shadowDepthProgram.program);
			this._shadowDepthProgram = null;
		}
		if (this._shadowTransmittanceProgram) {
			this._gl.deleteProgram(this._shadowTransmittanceProgram.program);
			this._shadowTransmittanceProgram = null;
		}
		if (this._copyProgram) {
			this._gl.deleteProgram(this._copyProgram.program);
			this._copyProgram = null;
		}
		if (this._oitResolveProgram) {
			this._gl.deleteProgram(this._oitResolveProgram.program);
			this._oitResolveProgram = null;
		}
		if (this._ssaoRawProgram) {
			this._gl.deleteProgram(this._ssaoRawProgram.program);
			this._ssaoRawProgram = null;
		}
		if (this._ssaoBlurProgram) {
			this._gl.deleteProgram(this._ssaoBlurProgram.program);
			this._ssaoBlurProgram = null;
		}
		if (this._ssaoCombineProgram) {
			this._gl.deleteProgram(this._ssaoCombineProgram.program);
			this._ssaoCombineProgram = null;
		}
		if (this._taaProgram) {
			this._gl.deleteProgram(this._taaProgram.program);
			this._taaProgram = null;
		}
		if (this._ssrProgram) {
			this._gl.deleteProgram(this._ssrProgram.program);
			this._ssrProgram = null;
		}
		if (this._volumetricProgram) {
			this._gl.deleteProgram(this._volumetricProgram.program);
			this._volumetricProgram = null;
		}
		if (this._fogProgram) {
			this._gl.deleteProgram(this._fogProgram.program);
			this._fogProgram = null;
		}
	}
}

function isShaderRuntime(value: unknown): value is ShaderRuntime {
	return (
		typeof value === "object" &&
		value !== null &&
		"process" in value &&
		typeof (value as { process?: unknown }).process === "function" &&
		"onDidChange" in value &&
		typeof (value as { onDidChange?: unknown }).onDidChange === "function" &&
		"getMode" in value &&
		typeof (value as { getMode?: unknown }).getMode === "function"
	);
}

function isWebGLProgramLibraryOptions(
	value: unknown
): value is WebGLProgramLibraryOptions {
	return (
		typeof value === "object" &&
		value !== null &&
		("validatePrograms" in value || "onProgramCompilePending" in value)
	);
}

function resolveParallelShaderCompileExtension(
	gl: WebGL2RenderingContext
): WebGLParallelShaderCompileExtension | null {
	if (typeof gl.getExtension !== "function") {
		return null;
	}
	try {
		const extension = gl.getExtension("KHR_parallel_shader_compile");
		if (
			extension &&
			typeof (extension as { COMPLETION_STATUS_KHR?: unknown })
				.COMPLETION_STATUS_KHR === "number"
		) {
			return extension as WebGLParallelShaderCompileExtension;
		}
	} catch {
		return null;
	}
	return null;
}
