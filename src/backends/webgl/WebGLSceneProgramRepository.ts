import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	WebGLCapabilityError,
	WebGLProgramPreparationError,
} from "../../foundation/Error";
import { createWebGLSceneUniforms } from "./WebGLSceneProgramUniforms";
import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import {
	createInlineShaderSourceMap,
	type ShaderBackendCompileStage,
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
	type WebGLProgramWarmupHandle,
	type WebGLShaderCompileMetadata,
} from "./WebGLProgramCompiler";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	createWebGLShaderMaterialFallbackVariant,
	normalizeWebGLSceneDepthVariantDescriptor,
	normalizeWebGLSceneVariantDescriptor,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import {
	createWebGLSceneSamplerLayout,
} from "./WebGLSceneSamplerLayout";
import type { WebGLSceneProgram } from "./WebGLSceneProgram";

export type { WebGLSceneProgram } from "./WebGLSceneProgram";

export type {
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";

type ShaderCompileMetadata = WebGLShaderCompileMetadata;

export interface WebGLSceneProgramRepositoryOptions {
	readonly compiler: WebGLProgramCompiler;
	readonly shaderRuntime?: ShaderRuntime;
	readonly shaderCompileStage?: ShaderBackendCompileStage;
	readonly warn?: WebGLProgramWarn;
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

function resolveMaxFragmentTextureUnits(
	gl: WebGL2RenderingContext
): number {
	const textureUnitParameter = gl.MAX_TEXTURE_IMAGE_UNITS;
	if (
		!Number.isFinite(textureUnitParameter) ||
		typeof gl.getParameter !== "function"
	) {
		return 32;
	}
	try {
		const textureUnits = gl.getParameter(textureUnitParameter);
		return Number.isFinite(textureUnits) && textureUnits > 0 ?
				Math.max(0, Math.floor(textureUnits))
			:	32;
	} catch {
		return 32;
	}
}

export class WebGLSceneProgramRepository {
	private _gl: WebGL2RenderingContext;
	private _shaderRuntime: ShaderRuntime | null;
	private _shaderCompileStage: ShaderBackendCompileStage | null;
	private _compiler: WebGLProgramCompiler;
	private _disposeCompilerInvalidationListener: (() => void) | null = null;
	private _warnCallback: WebGLProgramWarn | null = null;
	private _builtinScenePrograms = new Map<string, WebGLBuiltinProgramCacheEntry>();
	private _customScenePrograms = new Map<string, WebGLSceneProgram>();
	private _builtinSceneDepthPrepassPrograms =
		new Map<string, WebGLBuiltinProgramCacheEntry>();
	private _customSceneDepthPrepassPrograms = new Map<string, WebGLSceneProgram>();
	private _missingDepthPrepassShaderMaterialWarnings = new Set<number>();

	public constructor(options: WebGLSceneProgramRepositoryOptions) {
		this._compiler = options.compiler;
		this._gl = options.compiler.context;
		this._shaderRuntime = options.shaderRuntime ?? null;
		this._shaderCompileStage = options.shaderCompileStage ?? null;
		this._warnCallback = options.warn ?? null;
		this._disposeCompilerInvalidationListener = this._compiler.onDidInvalidate(
			() => this._disposePrograms()
		);
	}

	/**
	 * Reports whether the built-in scene shader variant includes irradiance probe
	 * grid uniforms and the grid SH sampler.
	 *
	 * @returns True when the active WebGL device exposes enough fragment texture
	 * units for the optional grid sampler in the scene shader.
	 * @sideEffects None.
	 */
	public getSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneVariantDescriptor
	): WebGLSceneProgram {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneProgram(variant);
		}

		const custom = this._getShaderMaterialSceneProgram(material, mode);
		return custom ?? this._getBuiltinSceneProgram(
			createWebGLShaderMaterialFallbackVariant(mode, variant ? {
				skinProfile: variant.skinProfile,
				morphSemanticMask: variant.morphSemanticMask,
			} : undefined),
		);
	}

	/** @internal Prepares exact built-in variants before synchronous frame draws. */
	public async prepareBuiltinSceneVariants(
		variants: Iterable<WebGLSceneVariantDescriptor>
	): Promise<void> {
		const limits = this._getSceneLightLimits();
		await ShaderSource.prepareMany(
			Array.from(variants).flatMap((variant) => [
				{ key: "webgl.scene.raw" as const, params: { limits, variant } },
				{ key: "webgl.scene.composite" as const, params: { limits, variant } },
			]),
		);
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

	public warmupSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single",
		variant?: WebGLSceneVariantDescriptor
	): WebGLProgramWarmupHandle {
		if (!(material instanceof ShaderMaterial)) {
			return this._warmupBuiltinSceneProgram(variant);
		}

		const custom = this._warmupShaderMaterialSceneProgram(material, mode);
		return custom ?? this._warmupBuiltinSceneProgram(
			createWebGLShaderMaterialFallbackVariant(mode),
		);
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
				if (error instanceof WebGLCapabilityError) throw error;
				if (!this._isWarnMode()) {
					throw error;
				}
				const key = `webgl-shader-material-compile-failed-${material.shaderId}`;
				const message =
					`ShaderMaterial ${material.name} custom WebGL shader compile failed; ` +
					`using built-in scene shader. ${String(error)}`;
				this._warn(key, message);
				this._getBuiltinSceneProgram(
					createWebGLShaderMaterialFallbackVariant(mode),
				);
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
				if (error instanceof WebGLCapabilityError) throw error;
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
		const influences =
			variant.skinProfile === "skin8" ? 8
			: variant.skinProfile === "skin4" ? 4
			: 0;
		const animationDefines = [
			`#define IGNIS_WEBGL_DEFORMATION_ACTIVE ${influences > 0 || variant.morphPosition ? 1 : 0}`,
			`#define IGNIS_WEBGL_SKIN_INFLUENCES ${influences}`,
		].join("\n");
		return {
			vertexSource: this._shaderSource("sceneDepthPrepassVertex").replace(
				"__IGNIS_WEBGL_ANIMATION_DEFINES__",
				animationDefines,
			),
			fragmentSource:
				`${defines}\n` + this._shaderSource("sceneDepthPrepassFragment"),
		};
	}

	private _getBuiltinSceneProgram(
		variant?: WebGLSceneVariantDescriptor
	): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		const limits = this._getSceneLightLimits();
		const normalizedVariant = normalizeWebGLSceneVariantDescriptor(variant);
		if (!this._hasPreparedBuiltinSceneSources(limits, normalizedVariant)) {
			throw new WebGLProgramPreparationError(
				"scene",
				getWebGLSceneVariantKey(normalizedVariant),
			);
		}
		const cacheKey = this._createBuiltinSceneProgramCacheKey(
			normalizedVariant,
			directiveTag
		);
		const cached = this._builtinScenePrograms.get(cacheKey);
		if (cached) {
			return cached.program;
		}
		const sourceParams = { limits, variant: normalizedVariant };
		const sceneShaderSource = ShaderSource.get("webgl.scene.raw", sourceParams);
		const sceneCompositeSource = ShaderSource.get(
			"webgl.scene.composite",
			sourceParams,
		);
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
			[],
			[],
			normalizedVariant,
		);
		sceneProgram.targetMode = normalizedVariant.output;
		sceneProgram.colorOutputCount =
			normalizedVariant.output === "single" ? 1
			: normalizedVariant.materialGBuffer ? 5
			: 3;
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
			normalizedVariant.baseMap ? ["uBaseMap"] : [],
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
			sceneProgram.colorOutputCount =
				sceneProgram.targetMode === "single" ? 1 : 3;
		} catch (error) {
			if (error instanceof WebGLCapabilityError) throw error;
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
			if (error instanceof WebGLCapabilityError) throw error;
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
		variant?: WebGLSceneVariantDescriptor,
	): WebGLSceneProgram {
		const program = this._createProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata,
		);
		try {
			return {
				program,
				uniforms: createWebGLSceneUniforms(
					this._gl,
					program,
					customSamplerUniforms,
					customUniforms
				),
				samplerLayout: createWebGLSceneSamplerLayout(
					resolveMaxFragmentTextureUnits(this._gl),
					variant,
					customSamplerUniforms,
				),
			};
		} catch (error) {
			this._gl.deleteProgram(program);
			throw error;
		}
	}

	private _shaderSource(part: WebGLShaderPart): string {
		return ShaderSource.get(`webgl.part.${part}.raw`);
	}

	public destroy(): void {
		this._disposeCompilerInvalidationListener?.();
		this._disposeCompilerInvalidationListener = null;
		this._disposePrograms();
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
			scope: "WebGLSceneProgramRepository",
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
	}
}
