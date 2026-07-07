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
	type ShaderRuntime,
} from "../../shaders/runtime";
import {
	type WebGLShaderPart,
	ShaderSource,
	type WebGLSceneLightLimits,
} from "../../shaders/ShaderSource";
import { Logger } from "../../foundation/Logger";
import {
	WebGLProgramCompiler,
	type WebGLProgramCompileState,
	type WebGLProgramSlot,
	type WebGLProgramWarmupHandle,
	type WebGLShaderCompileMetadata,
} from "./WebGLProgramCompiler";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	normalizeWebGLSceneDepthVariantDescriptor,
	normalizeWebGLSceneVariantDescriptor,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";

export type {
	WebGLProgramCompileState,
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";

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

export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: WebGLSceneUniforms;
	targetMode?: ShaderTargetMode;
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

type ShaderCompileMetadata = WebGLShaderCompileMetadata;

export interface WebGLProgramLibraryOptions {
	validatePrograms?: boolean;
	onProgramCompilePending?: () => void;
	/** @internal Shared compiler owned by the WebGL frame executor. */
	compiler?: WebGLProgramCompiler;
}

interface WebGLProgramCompileRequest {
	readonly vertexSource: string;
	readonly fragmentSource: string;
	readonly label: string;
	readonly vertexMetadata?: ShaderCompileMetadata;
	readonly fragmentMetadata?: ShaderCompileMetadata;
}

interface WebGLBuiltinProgramCacheEntry {
	program: WebGLSceneProgram;
	directiveTag: string;
}

type WebGLProgramWarn = (key: string, message: string) => void;

const WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_TRANSMITTANCE = 17;
const WEBGL_MIN_TEXTURE_UNITS_FOR_IRRADIANCE_PROBE_GRID = 17;
const WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_AND_IRRADIANCE_PROBE_GRID = 18;

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
	private _compiler: WebGLProgramCompiler;
	private _ownsCompiler: boolean;
	private _disposeCompilerInvalidationListener: (() => void) | null = null;
	private _enableShadowTransmittanceSampling: boolean;
	private _enableIrradianceProbeGridSampling: boolean;
	private _warnCallback: WebGLProgramWarn | null = null;
	private _builtinScenePrograms = new Map<string, WebGLBuiltinProgramCacheEntry>();
	private _customScenePrograms = new Map<string, WebGLSceneProgram>();
	private _builtinSceneDepthPrepassPrograms =
		new Map<string, WebGLBuiltinProgramCacheEntry>();
	private _customSceneDepthPrepassPrograms = new Map<string, WebGLSceneProgram>();
	private _missingDepthPrepassShaderMaterialWarnings = new Set<number>();
	private _environmentProgram: WebGLEnvironmentProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _shadowDepthProgram: WebGLShadowDepthProgram | null = null;
	private _shadowTransmittanceProgram: WebGLShadowTransmittanceProgram | null = null;
	private _copyProgram: WebGLCopyProgram | null = null;
	private _oitResolveProgram: WebGLOITResolveProgram | null = null;

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
		if (!this._shaderCompileStage && this._shaderRuntime) {
			this._shaderCompileStage = new ShaderBackendCompileStage({
				backend: "webgl",
				runtime: this._shaderRuntime,
				profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
				mode: this._shaderRuntime.getMode(),
			});
		}
		this._ownsCompiler = !options?.compiler;
		this._compiler =
			options?.compiler ??
			new WebGLProgramCompiler(
				gl,
				this._shaderRuntime ?? undefined,
				this._shaderCompileStage ?? undefined,
				{
					validatePrograms: options?.validatePrograms === true,
					onProgramCompilePending: options?.onProgramCompilePending,
					warn: this._warnCallback ?? undefined,
				}
			);
		this._disposeCompilerInvalidationListener = this._compiler.onDidInvalidate(
			() => this._disposePrograms()
		);
	}

	/**
	 * Advances the internal frame tick used to budget fallback shader finalization.
	 *
	 * @returns Nothing.
	 * @sideEffects Resets the non-parallel compile fallback finalize budget.
	 */
	public beginFrame(): void {
		this._compiler.beginFrame();
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
		return this._compiler.getCompileState(label);
	}

	public getSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneVariantDescriptor
	): WebGLSceneProgram {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneProgram(variant);
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
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneDepthVariantDescriptor
	): WebGLSceneProgram | null {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneDepthPrepassProgram(variant);
		}

		return this._getShaderMaterialDepthPrepassProgram(material, mode);
	}

	public markWarmupHandles(): number {
		return this._compiler.markWarmupHandles();
	}

	public collectWarmupHandlesSince(mark: number): WebGLProgramWarmupHandle[] {
		return this._compiler.collectWarmupHandlesSince(mark);
	}

	public warmupSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneVariantDescriptor
	): WebGLProgramWarmupHandle {
		if (!(material instanceof ShaderMaterial)) {
			return this._warmupBuiltinSceneProgram(variant);
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
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneDepthVariantDescriptor
	): WebGLProgramWarmupHandle | null {
		if (!(material instanceof ShaderMaterial)) {
			return this._warmupBuiltinSceneDepthPrepassProgram(variant);
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

	private _warmupBuiltinSceneProgram(
		variant?: WebGLSceneVariantDescriptor
	): WebGLProgramWarmupHandle {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		const normalizedVariant = normalizeWebGLSceneVariantDescriptor(variant);
		const cacheKey = this._createBuiltinSceneProgramCacheKey(
			normalizedVariant,
			directiveTag
		);
		const label = this._createBuiltinSceneProgramLabel(normalizedVariant);
		const cached = this._builtinScenePrograms.get(cacheKey);
		const limits = this._getSceneLightLimits();
		const sceneShaderSource = ShaderSource.get("webgl.scene.raw", {
			limits,
			variant: normalizedVariant,
		});
		const sceneCompositeSource = ShaderSource.get("webgl.scene.composite", {
			limits,
			variant: normalizedVariant,
		});
		return this._warmupProgram(
			label,
			() => cached?.program ?? null,
			() =>
				this._beginProgramCompile(
					sceneShaderSource.vertex,
					sceneShaderSource.fragment,
					label,
					{
						sourceMap: sceneCompositeSource.vertex.sourceMap,
						variantKey: getWebGLSceneVariantKey(normalizedVariant),
						sourceKind: "builtin-scene",
					},
					{
						sourceMap: sceneCompositeSource.fragment.sourceMap,
						variantKey: getWebGLSceneVariantKey(normalizedVariant),
						sourceKind: "builtin-scene",
					},
				),
			() => {
				this._getBuiltinSceneProgram(normalizedVariant);
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
			return this._compiler.createCompletedWarmupHandle(
				`WebGLShaderMaterialProgram_${shaderKey}`
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

	private _warmupBuiltinSceneDepthPrepassProgram(
		variant?: WebGLSceneDepthVariantDescriptor
	): WebGLProgramWarmupHandle {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		const normalizedVariant =
			normalizeWebGLSceneDepthVariantDescriptor(variant);
		const cacheKey = this._createBuiltinSceneDepthProgramCacheKey(
			normalizedVariant,
			directiveTag
		);
		const label = this._createBuiltinSceneDepthProgramLabel(normalizedVariant);
		const cached = this._builtinSceneDepthPrepassPrograms.get(cacheKey);
		const { vertexSource, fragmentSource } =
			this._getBuiltinSceneDepthPrepassSources(normalizedVariant);
		const vertexComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassVertex.composite"
		);
		const fragmentComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassFragment.composite"
		);
		return this._warmupProgram(
			label,
			() => cached?.program ?? null,
			() =>
				this._beginProgramCompile(
					vertexSource,
					fragmentSource,
					label,
					{
						sourceMap: vertexComposite.sourceMap,
						variantKey: getWebGLSceneDepthVariantKey(normalizedVariant),
						sourceKind: "builtin-scene",
					},
					{
						sourceMap: fragmentComposite.sourceMap,
						variantKey: getWebGLSceneDepthVariantKey(normalizedVariant),
						sourceKind: "builtin-scene",
					},
				),
			() => {
				this._getBuiltinSceneDepthPrepassProgram(normalizedVariant);
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
			return this._compiler.createCompletedWarmupHandle(
				`WebGLShaderMaterialDepthPrepassProgram_${shaderKey}`
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

	private _createBuiltinSceneProgramCacheKey(
		variant: WebGLSceneVariantDescriptor,
		directiveTag: string
	): string {
		const limits = this._getSceneLightLimits();
		return (
			`${getWebGLSceneVariantKey(variant)}` +
			`|limits:${limits.maxDirectionalLights},` +
			`${limits.maxPointLights},${limits.maxSpotLights}` +
			`|shadowSampler:${limits.enableShadowTransmittance ? 1 : 0}` +
			`|gridSampler:${limits.enableIrradianceProbeGrid ? 1 : 0}` +
			`|runtime:${this._shaderRuntime?.revision ?? 0}` +
			`|directive:${directiveTag}`
		);
	}

	private _createBuiltinSceneDepthProgramCacheKey(
		variant: WebGLSceneDepthVariantDescriptor,
		directiveTag: string
	): string {
		return (
			`${getWebGLSceneDepthVariantKey(variant)}` +
			`|runtime:${this._shaderRuntime?.revision ?? 0}` +
			`|directive:${directiveTag}`
		);
	}

	private _hasPreparedBuiltinSceneSources(
		limits: WebGLSceneLightLimits,
		variant: WebGLSceneVariantDescriptor
	): boolean {
		const params = { limits, variant };
		return (
			ShaderSource.has("webgl.scene.raw", params) &&
			ShaderSource.has("webgl.scene.composite", params)
		);
	}

	private _createBuiltinSceneProgramLabel(
		variant: WebGLSceneVariantDescriptor
	): string {
		return `WebGLSceneProgram_${getWebGLSceneVariantKey(variant)}`;
	}

	private _createBuiltinSceneDepthProgramLabel(
		variant: WebGLSceneDepthVariantDescriptor
	): string {
		return `WebGLSceneDepthPrepassProgram_${getWebGLSceneDepthVariantKey(variant)}`;
	}

	private _getBuiltinSceneDepthPrepassSources(
		variant: WebGLSceneDepthVariantDescriptor
	): { vertexSource: string; fragmentSource: string } {
		const defines = [
			`#define WEBGL_DEPTH_ALPHA_MASK ${variant.alphaMask ? 1 : 0}`,
			`#define WEBGL_DEPTH_BASE_MAP ${variant.baseMap ? 1 : 0}`,
		].join("\n");
		return {
			vertexSource: this._shaderSource("sceneDepthPrepassVertex"),
			fragmentSource:
				`${defines}\n` + this._shaderSource("sceneDepthPrepassFragment"),
		};
	}

	private _getBuiltinSceneProgram(
		variant?: WebGLSceneVariantDescriptor
	): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		const limits = this._getSceneLightLimits();
		let normalizedVariant = normalizeWebGLSceneVariantDescriptor(variant);
		if (!this._hasPreparedBuiltinSceneSources(limits, normalizedVariant)) {
			normalizedVariant = normalizeWebGLSceneVariantDescriptor();
		}
		const cacheKey = this._createBuiltinSceneProgramCacheKey(
			normalizedVariant,
			directiveTag
		);
		const cached = this._builtinScenePrograms.get(cacheKey);
		if (cached) {
			return cached.program;
		}
		const sceneShaderSource = ShaderSource.get("webgl.scene.raw", {
			limits,
			variant: normalizedVariant,
		});
		const sceneCompositeSource = ShaderSource.get("webgl.scene.composite", {
			limits,
			variant: normalizedVariant,
		});
		const variantKey = getWebGLSceneVariantKey(normalizedVariant);
		const sceneProgram = this._createSceneProgram(
			sceneShaderSource.vertex,
			sceneShaderSource.fragment,
			this._createBuiltinSceneProgramLabel(normalizedVariant),
			{
				sourceMap: sceneCompositeSource.vertex.sourceMap,
				variantKey,
				sourceKind: "builtin-scene",
			},
			{
				sourceMap: sceneCompositeSource.fragment.sourceMap,
				variantKey,
				sourceKind: "builtin-scene",
			},
		);
		sceneProgram.targetMode = normalizedVariant.output;
		this._builtinScenePrograms.set(cacheKey, {
			program: sceneProgram,
			directiveTag,
		});
		return sceneProgram;
	}

	private _getBuiltinSceneDepthPrepassProgram(
		variant?: WebGLSceneDepthVariantDescriptor
	): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		const normalizedVariant =
			normalizeWebGLSceneDepthVariantDescriptor(variant);
		const cacheKey = this._createBuiltinSceneDepthProgramCacheKey(
			normalizedVariant,
			directiveTag
		);
		const cached = this._builtinSceneDepthPrepassPrograms.get(cacheKey);
		if (cached) {
			return cached.program;
		}
		const vertexComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassVertex.composite"
		);
		const fragmentComposite = ShaderSource.get(
			"webgl.part.sceneDepthPrepassFragment.composite"
		);
		const { vertexSource, fragmentSource } =
			this._getBuiltinSceneDepthPrepassSources(normalizedVariant);
		const variantKey = getWebGLSceneDepthVariantKey(normalizedVariant);
		const sceneProgram = this._createSceneProgram(
			vertexSource,
			fragmentSource,
			this._createBuiltinSceneDepthProgramLabel(normalizedVariant),
			{
				sourceMap: vertexComposite.sourceMap,
				variantKey,
				sourceKind: "builtin-scene",
			},
			{
				sourceMap: fragmentComposite.sourceMap,
				variantKey,
				sourceKind: "builtin-scene",
			},
		);
		this._builtinSceneDepthPrepassPrograms.set(cacheKey, {
			program: sceneProgram,
			directiveTag,
		});
		return sceneProgram;
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
			const hasMrtChunk = material.chunks.some(
				(chunk) => chunk.stage === "fragment" && chunk.mode === "mrt"
			);
			sceneProgram.targetMode = mode === "mrt" && !hasMrtChunk ? "single" : mode;
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

	/**
	 * Attempts to resolve the environment program without blocking on shader status.
	 *
	 * @returns The cached/finalized environment program, or `null` while compiling.
	 * @sideEffects May enqueue program compilation or finalize a ready program.
	 */
	public tryGetEnvironmentProgram(): WebGLEnvironmentProgram | null {
		if (this._environmentProgram) {
			return this._environmentProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("environmentVertex"),
			this._shaderSource("environmentFragment"),
			"WebGLEnvironmentProgram",
		);
		if (!program) {
			return null;
		}
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

	public tryGetParticleProgram(): WebGLParticleProgram | null {
		if (this._particleProgram) {
			return this._particleProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("particleVertex"),
			this._shaderSource("particleFragment"),
			"WebGLParticleProgram",
		);
		if (!program) {
			return null;
		}
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

	public tryGetShadowDepthProgram(): WebGLShadowDepthProgram | null {
		if (this._shadowDepthProgram) {
			return this._shadowDepthProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowDepthFragment"),
			"WebGLShadowDepthProgram",
		);
		if (!program) {
			return null;
		}
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

	public tryGetShadowTransmittanceProgram():
		WebGLShadowTransmittanceProgram | null {
		if (this._shadowTransmittanceProgram) {
			return this._shadowTransmittanceProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowTransmittanceFragment"),
			"WebGLShadowTransmittanceProgram",
		);
		if (!program) {
			return null;
		}
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

	public tryGetCopyProgram(): WebGLCopyProgram | null {
		if (this._copyProgram) {
			return this._copyProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("copyFragment"),
			"WebGLCopyProgram",
		);
		if (!program) {
			return null;
		}
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

	public tryGetOITResolveProgram(): WebGLOITResolveProgram | null {
		if (this._oitResolveProgram) {
			return this._oitResolveProgram;
		}
		const program = this._tryCreateProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("oitResolveFragment"),
			"WebGLOITResolveProgram",
		);
		if (!program) {
			return null;
		}
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

	private _shaderSource(part: WebGLShaderPart): string {
		return ShaderSource.get(`webgl.part.${part}.raw`);
	}

	public destroy(): void {
		this._disposeCompilerInvalidationListener?.();
		this._disposeCompilerInvalidationListener = null;
		this._disposePrograms();
		if (this._ownsCompiler) {
			this._compiler.destroy();
		}
	}

	private _createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgram {
		return this._compiler.createProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata
		);
	}

	private _tryCreateProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgram | null {
		return this._compiler.tryCreateProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata
		);
	}

	private _warmupProgram(
		label: string,
		getCached: () => { program: WebGLProgram } | null,
		beginCompile: () => WebGLProgramCompileRequest,
		finalizeReadyProgram: () => void,
		handleCompileError?: (error: unknown) => void,
	): WebGLProgramWarmupHandle {
		if (getCached()) {
			return this._compiler.createCompletedWarmupHandle(label);
		}
		const request = beginCompile();
		return this._compiler.warmupProgram(
			label,
			request.vertexSource,
			request.fragmentSource,
			finalizeReadyProgram,
			request.vertexMetadata,
			request.fragmentMetadata,
			handleCompileError
		);
	}

	private _beginProgramCompile(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgramCompileRequest {
		return {
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata,
		};
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

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLProgramLibrary",
			onceKey: key,
		});
	}

	private _disposePrograms(): void {
		for (const entry of this._builtinScenePrograms.values()) {
			this._gl.deleteProgram(entry.program.program);
		}
		this._builtinScenePrograms.clear();
		for (const sceneProgram of this._customScenePrograms.values()) {
			this._gl.deleteProgram(sceneProgram.program);
		}
		this._customScenePrograms.clear();
		for (const entry of this._builtinSceneDepthPrepassPrograms.values()) {
			this._gl.deleteProgram(entry.program.program);
		}
		this._builtinSceneDepthPrepassPrograms.clear();
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
		(
			"validatePrograms" in value ||
			"onProgramCompilePending" in value ||
			"compiler" in value
		)
	);
}
