import { Platform } from "../foundation/Platform";
import { embeddedSyncShaderSources } from "./generated/embeddedSyncShaderSources";
import {
	composeCompositeShaderSources,
	createInlineCompositeShaderSource,
	type CompositeShaderSource,
	type ShaderSourceSegmentKind,
} from "./runtime";

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;

function createBrowserShaderSources(): ImportMetaGlobLoaderMap {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return {
			...import.meta.glob<string>(
				["./webgpu/**/*.wgsl", "!./webgpu/utility/mipmapBlit.wgsl"],
				{
					query: "?raw",
					import: "default",
				}
			),
			...import.meta.glob<string>("./webgl/parts/*.glsl", {
				query: "?raw",
				import: "default",
			}),
		};
	} catch {
		return {};
	}
}

const browserShaderSources: ImportMetaGlobLoaderMap = createBrowserShaderSources();

type NodeFsModule = {
	readFile: (
		path: string | URL,
		options?: string | { encoding?: string }
	) => Promise<string>;
};

type NodeFsSyncModule = {
	readFileSync: (path: string | URL, encoding: "utf8") => string;
};

export type WebGPUSceneShaderPart =
	| "lightData"
	| "constants"
	| "definitions"
	| "utils"
	| "vertexStage"
	| "fragmentPrelude"
	| "fragmentPhong"
	| "fragmentPbrSetup"
	| "fragmentPbrDirectional"
	| "fragmentPbrPoint"
	| "fragmentPbrSpot"
	| "fragmentPbrArea"
	| "fragmentPbrAmbient"
	| "fragmentGBuffer"
	| "fragmentSingleTarget";

export type WebGPUPostProcessShaderPart =
	| "ssao"
	| "ssgi"
	| "taa"
	| "hiz"
	| "ssr"
	| "screenSpaceRefractions"
	| "volumetric"
	| "fog"
	| "motionBlur"
	| "dof"
	| "bloomDownsample"
	| "bloomBlurH"
	| "bloomBlurV"
	| "bloomUpsample"
	| "bloomComposite"
	| "toneMapping"
	| "colorFilter"
	| "interactionOutline"
	| "fxaa"
	| "copy"
	| "sobelNormal";

export type WebGPUShadowShaderPart = "depth";

export type WebGPUUtilityShaderPart =
	| "planarReflectionComposite"
	| "present"
	| "depthDirtyClear"
	| "decal"
	| "oitResolve"
	| "occlusionCulling"
	| "mipmapBlit";

export type WebGLShaderPart =
	| "sceneVertex"
	| "sceneFragment"
	| "sceneDepthPrepassVertex"
	| "sceneDepthPrepassFragment"
	| "environmentVertex"
	| "environmentFragment"
	| "presentVertex"
	| "presentFragment"
	| "particleVertex"
	| "particleFragment"
	| "shadowDepthVertex"
	| "shadowDepthFragment"
	| "shadowTransmittanceFragment"
	| "copyFragment"
	| "oitResolveFragment"
	| "postProcessStubFragment"
	| "toneMappingFragment"
	| "colorFilterFragment"
	| "fxaaFragment"
	| "bloomFragment"
	| "interactionOutlineFragment"
	| "motionBlurFragment"
	| "fogFragment"
	| "dofFragment"
	| "taaFragment"
	| "ssaoRawFragment"
	| "ssaoBlurFragment"
	| "ssaoCombineFragment";

export const WEBGPU_SCENE_SHADER_PARTS: readonly WebGPUSceneShaderPart[] = [
	"lightData",
	"constants",
	"definitions",
	"utils",
	"vertexStage",
	"fragmentPrelude",
	"fragmentPhong",
	"fragmentPbrSetup",
	"fragmentPbrDirectional",
	"fragmentPbrPoint",
	"fragmentPbrSpot",
	"fragmentPbrArea",
	"fragmentPbrAmbient",
	"fragmentGBuffer",
	"fragmentSingleTarget",
];

export const WEBGL_SHADER_PARTS: readonly WebGLShaderPart[] = [
	"sceneVertex",
	"sceneFragment",
	"sceneDepthPrepassVertex",
	"sceneDepthPrepassFragment",
	"environmentVertex",
	"environmentFragment",
	"presentVertex",
	"presentFragment",
	"particleVertex",
	"particleFragment",
	"shadowDepthVertex",
	"shadowDepthFragment",
	"shadowTransmittanceFragment",
	"copyFragment",
	"oitResolveFragment",
	"postProcessStubFragment",
	"toneMappingFragment",
	"colorFilterFragment",
	"fxaaFragment",
	"bloomFragment",
	"interactionOutlineFragment",
	"motionBlurFragment",
	"fogFragment",
	"dofFragment",
	"taaFragment",
	"ssaoRawFragment",
	"ssaoBlurFragment",
	"ssaoCombineFragment",
];

export const WEBGL_PIPELINE_SHADER_PARTS: readonly WebGLShaderPart[] =
	WEBGL_SHADER_PARTS.filter(
		(part) => part !== "sceneVertex" && part !== "sceneFragment"
	);

export interface WebGLSceneLightLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
	/**
	 * Enables the optional transparent-shadow transmittance sampler. Leave disabled
	 * on devices that only expose the WebGL2 minimum of 16 fragment texture units.
	 */
	enableShadowTransmittance?: boolean;
	/**
	 * Enables the optional irradiance probe grid sampler. Leave disabled on
	 * devices whose fragment texture unit budget cannot fit the extra sampler.
	 */
	enableIrradianceProbeGrid?: boolean;
}

export interface WebGLSceneShaderSource {
	vertex: string;
	fragment: string;
}

export interface WebGLSceneCompositeShaderSource {
	vertex: CompositeShaderSource;
	fragment: CompositeShaderSource;
}

type WebGPURawFixedShaderKey =
	| "webgpu.scene.raw"
	| "webgpu.environment.raw"
	| "webgpu.deferredLighting.raw"
	| "webgpu.particle.raw"
	| "webgpu.particleSimulation.raw"
	| "webgpu.environmentIblPrefilter.raw";

type WebGPUCompositeFixedShaderKey =
	| "webgpu.scene.composite"
	| "webgpu.environment.composite"
	| "webgpu.deferredLighting.composite"
	| "webgpu.particle.composite"
	| "webgpu.clusteredLightingCull.composite";

type WebGPUScenePartKey =
	| `webgpu.scene.part.${WebGPUSceneShaderPart}.raw`
	| `webgpu.scene.part.${WebGPUSceneShaderPart}.composite`;

type WebGPUPostProcessKey =
	| `webgpu.postprocess.${WebGPUPostProcessShaderPart}.raw`
	| `webgpu.postprocess.${WebGPUPostProcessShaderPart}.composite`;

type WebGPUShadowKey =
	| `webgpu.shadow.${WebGPUShadowShaderPart}.raw`
	| `webgpu.shadow.${WebGPUShadowShaderPart}.composite`;

type WebGPUUtilityKey =
	| `webgpu.utility.${WebGPUUtilityShaderPart}.raw`
	| `webgpu.utility.${WebGPUUtilityShaderPart}.composite`;

type WebGLPartKey =
	| `webgl.part.${WebGLShaderPart}.raw`
	| `webgl.part.${WebGLShaderPart}.composite`;

export type ShaderSourceKey =
	| WebGPURawFixedShaderKey
	| WebGPUCompositeFixedShaderKey
	| WebGPUScenePartKey
	| WebGPUPostProcessKey
	| WebGPUShadowKey
	| WebGPUUtilityKey
	| WebGLPartKey
	| "webgl.scene.raw"
	| "webgl.scene.composite";

export type ShaderSourceSyncKey = "webgpu.utility.mipmapBlit.raw";

export type ShaderSourceParams<K extends ShaderSourceKey> =
	K extends "webgl.scene.raw" | "webgl.scene.composite" ?
		{ limits: WebGLSceneLightLimits }
	:	undefined;

export type ShaderSourceResult<K extends ShaderSourceKey> =
	K extends "webgl.scene.raw" ? WebGLSceneShaderSource
	: K extends "webgl.scene.composite" ? WebGLSceneCompositeShaderSource
	: K extends `${string}.composite` ? CompositeShaderSource
	: string;

export interface ShaderSourceRequest<K extends ShaderSourceKey = ShaderSourceKey> {
	key: K;
	params?: ShaderSourceParams<K>;
}

export interface ShaderSourceCacheBucketStats {
	hits: number;
	misses: number;
	size: number;
}

export interface ShaderSourceCacheStats {
	rawFiles: ShaderSourceCacheBucketStats;
	fileComposites: ShaderSourceCacheBucketStats;
	results: ShaderSourceCacheBucketStats;
	prepared: ShaderSourceCacheBucketStats;
	inFlight: number;
}

export interface ShaderSourceFileDescriptor {
	scope: "webgpu" | "webgl";
	key: string;
	path: string;
	segmentKind?: ShaderSourceSegmentKind;
}

interface ShaderFileDescriptor extends ShaderSourceFileDescriptor {}

export type ShaderSourceLoader = (
	descriptor: ShaderSourceFileDescriptor
) => Promise<string>;

export type ShaderSourceSyncLoader = (
	descriptor: ShaderSourceFileDescriptor
) => string | undefined;

export interface ShaderSourceConfiguration {
	/**
	 * Custom asynchronous loader for built-in shader files.
	 */
	loader?: ShaderSourceLoader;
	/**
	 * Optional synchronous loader used only by `ShaderSource.getSync()`.
	 */
	syncLoader?: ShaderSourceSyncLoader;
	/**
	 * When true, custom loader failures are not allowed to fall back to built-ins.
	 */
	preferCustomLoader?: boolean;
}

interface MutableCacheBucketStats {
	hits: number;
	misses: number;
}

type AnyShaderSourceResult =
	| string
	| CompositeShaderSource
	| WebGLSceneShaderSource
	| WebGLSceneCompositeShaderSource;

const WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA =
	new Set<WebGPUPostProcessShaderPart>(["ssr", "volumetric"]);

const webgpuSceneShaderFiles: Record<WebGPUSceneShaderPart, string> = {
	lightData: "./webgpu/common/lightData.wgsl",
	constants: "./webgpu/common/constants.wgsl",
	definitions: "./webgpu/common/definitions.wgsl",
	utils: "./webgpu/common/utils.wgsl",
	vertexStage: "./webgpu/scene/vertexStage.wgsl",
	fragmentPrelude: "./webgpu/scene/fragmentPrelude.wgsl",
	fragmentPhong: "./webgpu/scene/fragmentPhong.wgsl",
	fragmentPbrSetup: "./webgpu/scene/fragmentPbrSetup.wgsl",
	fragmentPbrDirectional: "./webgpu/scene/fragmentPbrDirectional.wgsl",
	fragmentPbrPoint: "./webgpu/scene/fragmentPbrPoint.wgsl",
	fragmentPbrSpot: "./webgpu/scene/fragmentPbrSpot.wgsl",
	fragmentPbrArea: "./webgpu/scene/fragmentPbrArea.wgsl",
	fragmentPbrAmbient: "./webgpu/scene/fragmentPbrAmbient.wgsl",
	fragmentGBuffer: "./webgpu/scene/fragmentGBuffer.wgsl",
	fragmentSingleTarget: "./webgpu/scene/fragmentSingleTarget.wgsl",
};

const webgpuPostProcessShaderFiles: Record<WebGPUPostProcessShaderPart, string> = {
	ssao: "./webgpu/postprocess/ssao.wgsl",
	ssgi: "./webgpu/postprocess/ssgi.wgsl",
	taa: "./webgpu/postprocess/taa.wgsl",
	hiz: "./webgpu/postprocess/hiz.wgsl",
	ssr: "./webgpu/postprocess/ssr.wgsl",
	screenSpaceRefractions: "./webgpu/postprocess/screenSpaceRefractions.wgsl",
	volumetric: "./webgpu/postprocess/volumetric.wgsl",
	fog: "./webgpu/postprocess/fog.wgsl",
	motionBlur: "./webgpu/postprocess/motionBlur.wgsl",
	dof: "./webgpu/postprocess/dof.wgsl",
	bloomDownsample: "./webgpu/postprocess/bloomDownsample.wgsl",
	bloomBlurH: "./webgpu/postprocess/bloomBlurH.wgsl",
	bloomBlurV: "./webgpu/postprocess/bloomBlurV.wgsl",
	bloomUpsample: "./webgpu/postprocess/bloomUpsample.wgsl",
	bloomComposite: "./webgpu/postprocess/bloomComposite.wgsl",
	toneMapping: "./webgpu/postprocess/toneMapping.wgsl",
	colorFilter: "./webgpu/postprocess/colorFilter.wgsl",
	interactionOutline: "./webgpu/postprocess/interactionOutline.wgsl",
	fxaa: "./webgpu/postprocess/fxaa.wgsl",
	copy: "./webgpu/postprocess/copy.wgsl",
	sobelNormal: "./webgpu/postprocess/sobelNormal.wgsl",
};

const webgpuShadowShaderFiles: Record<WebGPUShadowShaderPart, string> = {
	depth: "./webgpu/shadow/depth.wgsl",
};

const webgpuUtilityShaderFiles: Record<WebGPUUtilityShaderPart, string> = {
	planarReflectionComposite: "./webgpu/utility/planarReflectionComposite.wgsl",
	present: "./webgpu/utility/present.wgsl",
	depthDirtyClear: "./webgpu/utility/depthDirtyClear.wgsl",
	decal: "./webgpu/scene/decal.wgsl",
	oitResolve: "./webgpu/utility/oitResolve.wgsl",
	occlusionCulling: "./webgpu/utility/occlusionCulling.wgsl",
	mipmapBlit: "./webgpu/utility/mipmapBlit.wgsl",
};

const webglShaderFiles: Record<WebGLShaderPart, string> = {
	sceneVertex: "./webgl/parts/sceneVertex.glsl",
	sceneFragment: "./webgl/parts/sceneFragment.glsl",
	sceneDepthPrepassVertex: "./webgl/parts/sceneDepthPrepassVertex.glsl",
	sceneDepthPrepassFragment: "./webgl/parts/sceneDepthPrepassFragment.glsl",
	environmentVertex: "./webgl/parts/environmentVertex.glsl",
	environmentFragment: "./webgl/parts/environmentFragment.glsl",
	presentVertex: "./webgl/parts/presentVertex.glsl",
	presentFragment: "./webgl/parts/presentFragment.glsl",
	particleVertex: "./webgl/parts/particleVertex.glsl",
	particleFragment: "./webgl/parts/particleFragment.glsl",
	shadowDepthVertex: "./webgl/parts/shadowDepthVertex.glsl",
	shadowDepthFragment: "./webgl/parts/shadowDepthFragment.glsl",
	shadowTransmittanceFragment: "./webgl/parts/shadowTransmittanceFragment.glsl",
	copyFragment: "./webgl/parts/copyFragment.glsl",
	oitResolveFragment: "./webgl/parts/oitResolveFragment.glsl",
	postProcessStubFragment: "./webgl/parts/postProcessStubFragment.glsl",
	toneMappingFragment: "./webgl/parts/toneMappingFragment.glsl",
	colorFilterFragment: "./webgl/parts/colorFilterFragment.glsl",
	fxaaFragment: "./webgl/parts/fxaaFragment.glsl",
	bloomFragment: "./webgl/parts/bloomFragment.glsl",
	interactionOutlineFragment: "./webgl/parts/interactionOutlineFragment.glsl",
	motionBlurFragment: "./webgl/parts/motionBlurFragment.glsl",
	fogFragment: "./webgl/parts/fogFragment.glsl",
	dofFragment: "./webgl/parts/dofFragment.glsl",
	taaFragment: "./webgl/parts/taaFragment.glsl",
	ssaoRawFragment: "./webgl/parts/ssaoRawFragment.glsl",
	ssaoBlurFragment: "./webgl/parts/ssaoBlurFragment.glsl",
	ssaoCombineFragment: "./webgl/parts/ssaoCombineFragment.glsl",
};

const webglInternalShaderFiles = {
	diffuseProbeFallbackFragment: "./webgl/parts/diffuseProbeFallbackFragment.glsl",
	irradianceProbeGridFragment: "./webgl/parts/irradianceProbeGridFragment.glsl",
} as const;

const syncShaderFiles: Record<ShaderSourceSyncKey, ShaderFileDescriptor> = {
	"webgpu.utility.mipmapBlit.raw": {
		scope: "webgpu",
		key: "webgpu.utility.mipmapBlit",
		path: "./webgpu/utility/mipmapBlit.wgsl",
	},
};

function createBrowserSyncShaderSources(): Record<string, string> {
	if (Platform.isNodeRuntime()) {
		return {};
	}
	try {
		return import.meta.glob<string>("./webgpu/utility/mipmapBlit.wgsl", {
			query: "?raw",
			import: "default",
			eager: true,
		});
	} catch {
		return {};
	}
}

const browserSyncShaderSources: Record<string, string> =
	createBrowserSyncShaderSources();

const emptyCacheStats = (): MutableCacheBucketStats => ({
	hits: 0,
	misses: 0,
});

function cloneCompositeSource(
	composite: CompositeShaderSource
): CompositeShaderSource {
	return {
		code: composite.code,
		sourceMap: {
			schemaVersion: composite.sourceMap.schemaVersion,
			lineCount: composite.sourceMap.lineCount,
			segments: composite.sourceMap.segments.map((segment) => ({ ...segment })),
		},
	};
}

function cloneResult<T extends AnyShaderSourceResult>(result: T): T {
	if (typeof result === "string") {
		return result;
	}
	if (
		typeof result === "object" &&
		result !== null &&
		"code" in result &&
		"sourceMap" in result
	) {
		return cloneCompositeSource(result) as T;
	}
	const value = result as WebGLSceneShaderSource | WebGLSceneCompositeShaderSource;
	if (
		typeof value.vertex === "object" &&
		value.vertex !== null &&
		typeof value.fragment === "object" &&
		value.fragment !== null
	) {
		return {
			vertex: cloneCompositeSource(value.vertex),
			fragment: cloneCompositeSource(value.fragment),
		} as T;
	}
	return {
		vertex: value.vertex,
		fragment: value.fragment,
	} as T;
}

function firstSourcePath(
	composite: CompositeShaderSource,
	fallback: string
): string {
	return composite.sourceMap.segments[0]?.sourcePath ?? fallback;
}

function composeShaderParts(
	parts: readonly CompositeShaderSource[],
	fallbackSourcePath: string
): CompositeShaderSource {
	return composeCompositeShaderSources(
		parts.map((part) => ({
			code: part.code,
			sourceMap: part.sourceMap,
			sourcePath: firstSourcePath(part, fallbackSourcePath),
			kind: "template",
		})),
		"\n\n",
		"template"
	);
}

interface WebGLSceneOptionalBlocks {
	diffuseProbeFallbackFragment: string;
	irradianceProbeGridFragment: string;
}

const WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS = [
	"uniform int uIrradianceProbeGridEnabled;",
	"uniform vec4 uIrradianceProbeGridWorldToGridRow0;",
	"uniform vec4 uIrradianceProbeGridWorldToGridRow1;",
	"uniform vec4 uIrradianceProbeGridWorldToGridRow2;",
	"uniform vec4 uIrradianceProbeGridDataA;",
	"uniform vec4 uIrradianceProbeGridDataB;",
	"uniform sampler2D uIrradianceProbeGridCoeffs;",
	"uniform vec2 uIrradianceProbeGridCoeffsSize;",
].join("\n");

function replaceOptionalDefines(
	source: string,
	limits: WebGLSceneLightLimits,
	optionalBlocks: WebGLSceneOptionalBlocks
): string {
	const shadowTransmittanceEnabled = !!limits.enableShadowTransmittance;
	const irradianceProbeGridEnabled = !!limits.enableIrradianceProbeGrid;
	return source
		.replaceAll(
			"__WEBGL_SHADOW_TRANSMITTANCE_DEFINE__",
			shadowTransmittanceEnabled ?
				"#define WEBGL_SHADOW_TRANSMITTANCE 1"
			:	""
		)
		.replaceAll(
			"__WEBGL_SHADOW_TRANSMITTANCE_UNIFORMS__",
			shadowTransmittanceEnabled ?
				[
					"uniform sampler2D uShadowTransmittanceAtlas;",
					"uniform int uShadowTransmittanceAtlasAvailable;",
				].join("\n")
			:	""
		)
		.replaceAll(
			"__WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS__",
			irradianceProbeGridEnabled ? WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS : ""
		)
		.replaceAll(
			"__WEBGL_IRRADIANCE_PROBE_GRID_FUNCTIONS__",
			irradianceProbeGridEnabled ?
				optionalBlocks.irradianceProbeGridFragment
			:	optionalBlocks.diffuseProbeFallbackFragment
		);
}

function normalizeWebGLSceneLimits(
	params: unknown
): WebGLSceneLightLimits {
	const limits = (params as { limits?: WebGLSceneLightLimits } | undefined)?.limits;
	if (!limits || typeof limits !== "object") {
		throw new Error(
			"ShaderSource webgl.scene keys require params.limits. " +
				"Pass { limits: { maxDirectionalLights, maxPointLights, maxSpotLights } }."
		);
	}
	return {
		maxDirectionalLights: Math.max(
			0,
			Math.floor(limits.maxDirectionalLights ?? 0)
		),
		maxPointLights: Math.max(0, Math.floor(limits.maxPointLights ?? 0)),
		maxSpotLights: Math.max(0, Math.floor(limits.maxSpotLights ?? 0)),
		enableShadowTransmittance: !!limits.enableShadowTransmittance,
		enableIrradianceProbeGrid: !!limits.enableIrradianceProbeGrid,
	};
}

function isCompositeKey(key: ShaderSourceKey): boolean {
	return key.endsWith(".composite");
}

function shaderSourceScope(key: string): "webgpu" | "webgl" {
	return key.startsWith("webgl.") ? "webgl" : "webgpu";
}

/**
 * Loads built-in shader source assets and caches raw, composite, and assembled
 * variants. Runtime shader processing and backend GPU resource caches remain
 * owned by their dedicated systems.
 */
export class ShaderSource {
	private static _rawFileCache = new Map<string, Promise<string>>();
	private static _fileCompositeCache = new Map<
		string,
		Promise<CompositeShaderSource>
	>();
	private static _resultCache = new Map<string, Promise<AnyShaderSourceResult>>();
	private static _inFlightResultKeys = new Set<string>();
	private static _preparedCache = new Map<string, AnyShaderSourceResult>();
	private static _configuration: ShaderSourceConfiguration = {};
	private static _rawFileStats = emptyCacheStats();
	private static _fileCompositeStats = emptyCacheStats();
	private static _resultStats = emptyCacheStats();
	private static _preparedStats = emptyCacheStats();

	/**
	 * Configures custom built-in shader source loading.
	 *
	 * @param configuration Loader callbacks and fallback policy.
	 * @returns Nothing.
	 * @constraints Custom loaders apply globally until `resetConfiguration()`.
	 * @sideEffects Clears shader source caches so future reads use the new loader.
	 */
	public static configure(configuration: ShaderSourceConfiguration): void {
		this._configuration = { ...configuration };
		this.clearCache();
	}

	/**
	 * Restores default built-in shader source loading.
	 *
	 * @returns Nothing.
	 * @sideEffects Clears shader source caches and removes custom loaders.
	 */
	public static resetConfiguration(): void {
		this._configuration = {};
		this.clearCache();
	}

	/**
	 * Asynchronously loads a built-in shader source and stores the assembled
	 * result in the prepared cache.
	 *
	 * @param key Built-in shader source key.
	 * @param params Required key-specific parameters, such as WebGL light limits.
	 * @returns A cloned source result matching `key`.
	 * @throws If the key is unsupported or a required source file cannot be read.
	 * @sideEffects Populates raw-file, composite, result, and prepared caches.
	 */
	public static async load<K extends ShaderSourceKey>(
		key: K,
		params?: ShaderSourceParams<K>
	): Promise<ShaderSourceResult<K>> {
		const cacheKey = this._buildResultCacheKey(key, params);
		const prepared = this._preparedCache.get(cacheKey);
		if (prepared) {
			this._preparedStats.hits++;
			return cloneResult(prepared) as ShaderSourceResult<K>;
		}
		this._preparedStats.misses++;

		const result = await this._getResultPromise(key, params, cacheKey);
		this._preparedCache.set(cacheKey, cloneResult(result));
		return cloneResult(result) as ShaderSourceResult<K>;
	}

	/**
	 * Synchronously loads a built-in shader source that is explicitly approved
	 * for sync use by low-level renderer paths.
	 *
	 * @param key Shader source key that supports synchronous loading.
	 * @returns A cloned source result matching `key`.
	 * @throws If the key is unsupported or a required source file cannot be read.
	 * @sideEffects Populates raw-file, result, and prepared caches.
	 */
	public static getSync<K extends ShaderSourceSyncKey>(
		key: K
	): ShaderSourceResult<K> {
		const cacheKey = this._buildResultCacheKey(key, undefined);
		const prepared = this._preparedCache.get(cacheKey);
		if (prepared) {
			this._preparedStats.hits++;
			return cloneResult(prepared) as ShaderSourceResult<K>;
		}
		this._preparedStats.misses++;
		this._resultStats.misses++;

		const descriptor = syncShaderFiles[key];
		const result = this._loadFileRawSync(descriptor);
		const cloned = cloneResult(result);
		this._resultCache.set(cacheKey, Promise.resolve(cloned));
		this._preparedCache.set(cacheKey, cloned);
		return cloneResult(cloned) as ShaderSourceResult<K>;
	}

	/**
	 * Preloads a built-in shader source so `ShaderSource.get()` can return it
	 * synchronously later.
	 *
	 * @param key Built-in shader source key.
	 * @param params Required key-specific parameters, such as WebGL light limits.
	 * @returns A promise that resolves after the source is prepared.
	 * @throws If loading fails for the requested key.
	 * @sideEffects Populates the same caches as `ShaderSource.load()`.
	 */
	public static async prepare<K extends ShaderSourceKey>(
		key: K,
		params?: ShaderSourceParams<K>
	): Promise<void> {
		await this.load(key, params);
	}

	/**
	 * Preloads multiple built-in shader sources concurrently.
	 *
	 * @param requests Source keys and optional key-specific parameters to prepare.
	 * @returns A promise that resolves after every source is prepared.
	 * @throws If any requested source cannot be loaded.
	 * @sideEffects Populates the same caches as `ShaderSource.load()`.
	 */
	public static async prepareMany(
		requests: readonly ShaderSourceRequest[]
	): Promise<void> {
		await Promise.all(
			requests.map((request) =>
				this.prepare(
					request.key,
					request.params as ShaderSourceParams<typeof request.key>
				)
			)
		);
	}

	/**
	 * Returns a previously prepared built-in shader source without asynchronous
	 * file or bundle loading.
	 *
	 * @param key Built-in shader source key.
	 * @param params Required key-specific parameters, such as WebGL light limits.
	 * @returns A cloned source result matching `key`.
	 * @throws If the key and params have not been prepared.
	 * @sideEffects Updates prepared-cache hit and miss counters.
	 */
	public static get<K extends ShaderSourceKey>(
		key: K,
		params?: ShaderSourceParams<K>
	): ShaderSourceResult<K> {
		const cacheKey = this._buildResultCacheKey(key, params);
		const prepared = this._preparedCache.get(cacheKey);
		if (!prepared) {
			this._preparedStats.misses++;
			throw new Error(
				`ShaderSource "${key}" is not prepared. ` +
					"Call ShaderSource.prepare()/prepareMany() before sync get()."
			);
		}
		this._preparedStats.hits++;
		return cloneResult(prepared) as ShaderSourceResult<K>;
	}

	/**
	 * Checks whether a built-in shader source is available to `ShaderSource.get()`
	 * without additional asynchronous work.
	 *
	 * @param key Built-in shader source key.
	 * @param params Required key-specific parameters, such as WebGL light limits.
	 * @returns True when the prepared cache contains the requested source.
	 * @throws If required key-specific params are missing or invalid.
	 * @sideEffects Does not load sources or mutate cache entries.
	 */
	public static has<K extends ShaderSourceKey>(
		key: K,
		params?: ShaderSourceParams<K>
	): boolean {
		return this._preparedCache.has(this._buildResultCacheKey(key, params));
	}

	/**
	 * Clears shader source caches for a backend scope.
	 *
	 * @param scope Cache scope to clear. Defaults to all shader source caches.
	 * @sideEffects Removes cached source promises and prepared source results.
	 */
	public static clearCache(scope: "all" | "webgpu" | "webgl" = "all"): void {
		this._deleteByScope(this._rawFileCache, scope);
		this._deleteByScope(this._fileCompositeCache, scope);
		this._deleteByScope(this._resultCache, scope);
		this._deleteByScope(this._preparedCache, scope);
		this._deleteSetByScope(this._inFlightResultKeys, scope);
		if (scope === "all") {
			this._rawFileStats = emptyCacheStats();
			this._fileCompositeStats = emptyCacheStats();
			this._resultStats = emptyCacheStats();
			this._preparedStats = emptyCacheStats();
		}
	}

	/**
	 * Reports shader source cache counters for diagnostics.
	 *
	 * @returns Hit, miss, size, and in-flight counts for source cache buckets.
	 * @sideEffects Does not mutate cache entries.
	 */
	public static getCacheStats(): ShaderSourceCacheStats {
		return {
			rawFiles: {
				...this._rawFileStats,
				size: this._rawFileCache.size,
			},
			fileComposites: {
				...this._fileCompositeStats,
				size: this._fileCompositeCache.size,
			},
			results: {
				...this._resultStats,
				size: this._resultCache.size,
			},
			prepared: {
				...this._preparedStats,
				size: this._preparedCache.size,
			},
			inFlight: this._inFlightResultKeys.size,
		};
	}

	private static _getResultPromise<K extends ShaderSourceKey>(
		key: K,
		params: ShaderSourceParams<K> | undefined,
		cacheKey: string
	): Promise<AnyShaderSourceResult> {
		const cached = this._resultCache.get(cacheKey);
		if (cached) {
			this._resultStats.hits++;
			return cached;
		}
		this._resultStats.misses++;
		this._inFlightResultKeys.add(cacheKey);
		let result: Promise<AnyShaderSourceResult>;
		result = this._loadUncached(key, params)
			.catch((error) => {
				if (this._resultCache.get(cacheKey) === result) {
					this._resultCache.delete(cacheKey);
				}
				throw error;
			})
			.finally(() => {
				this._inFlightResultKeys.delete(cacheKey);
			});
		this._resultCache.set(cacheKey, result);
		return result;
	}

	private static async _loadUncached<K extends ShaderSourceKey>(
		key: K,
		params: ShaderSourceParams<K> | undefined
	): Promise<AnyShaderSourceResult> {
		switch (key) {
			case "webgpu.scene.raw":
				return (await this._loadWebGPUSceneComposite()).code;
			case "webgpu.scene.composite":
				return this._loadWebGPUSceneComposite();
			case "webgpu.environment.raw":
				return (await this._loadWithSharedLightData(
					"webgpu.environment.composite",
					this._loadFileComposite({
						scope: "webgpu",
						key: "webgpu.environment",
						path: "./webgpu/environment/background.wgsl",
					})
				)).code;
			case "webgpu.environment.composite":
				return this._loadWithSharedLightData(
					"webgpu.environment.composite",
					this._loadFileComposite({
						scope: "webgpu",
						key: "webgpu.environment",
						path: "./webgpu/environment/background.wgsl",
					})
				);
			case "webgpu.deferredLighting.raw":
				return (await this._loadDeferredLightingComposite()).code;
			case "webgpu.deferredLighting.composite":
				return this._loadDeferredLightingComposite();
			case "webgpu.particle.raw":
				return (await this._loadWithSharedLightData(
					"webgpu.particle.composite",
					this._loadFileComposite({
						scope: "webgpu",
						key: "webgpu.particle",
						path: "./webgpu/particles/render.wgsl",
					})
				)).code;
			case "webgpu.particle.composite":
				return this._loadWithSharedLightData(
					"webgpu.particle.composite",
					this._loadFileComposite({
						scope: "webgpu",
						key: "webgpu.particle",
						path: "./webgpu/particles/render.wgsl",
					})
				);
			case "webgpu.particleSimulation.raw":
				return this._loadFileRaw({
					scope: "webgpu",
					key: "webgpu.particleSimulation",
					path: "./webgpu/particles/simulation.wgsl",
				});
			case "webgpu.clusteredLightingCull.composite":
				return this._loadWithSharedLightData(
					"webgpu.clusteredLightingCull.composite",
					this._loadFileComposite({
						scope: "webgpu",
						key: "webgpu.clusteredLightingCull",
						path: "./webgpu/lighting/clusteredLightingCull.wgsl",
					})
				);
			case "webgpu.environmentIblPrefilter.raw":
				return this._loadFileRaw({
					scope: "webgpu",
					key: "webgpu.environmentIblPrefilter",
					path: "./webgpu/environment/iblPrefilter.wgsl",
				});
			case "webgl.scene.raw":
				return this._loadWebGLSceneRaw(params);
			case "webgl.scene.composite":
				return this._loadWebGLSceneComposite(params);
			default:
				break;
		}

		if (key.startsWith("webgpu.scene.part.")) {
			return this._loadWebGPUScenePart(key as WebGPUScenePartKey);
		}
		if (key.startsWith("webgpu.postprocess.")) {
			return this._loadWebGPUPostProcessPart(key as WebGPUPostProcessKey);
		}
		if (key.startsWith("webgpu.shadow.")) {
			return this._loadWebGPUShadowPart(key as WebGPUShadowKey);
		}
		if (key.startsWith("webgpu.utility.")) {
			return this._loadWebGPUUtilityPart(key as WebGPUUtilityKey);
		}
		if (key.startsWith("webgl.part.")) {
			return this._loadWebGLPart(key as WebGLPartKey);
		}
		throw new Error(`Unsupported ShaderSource key "${String(key)}".`);
	}

	private static async _loadWebGPUSceneComposite():
		Promise<CompositeShaderSource> {
		const parts = await Promise.all(
			WEBGPU_SCENE_SHADER_PARTS.map((part) =>
				this._loadWebGPUScenePartComposite(part)
			)
		);
		return composeShaderParts(parts, "<webgpu-scene-part>");
	}

	private static async _loadDeferredLightingComposite():
		Promise<CompositeShaderSource> {
		const parts = await Promise.all([
			this._loadWebGPUScenePartComposite("lightData"),
			this._loadWebGPUScenePartComposite("constants"),
			this._loadWebGPUScenePartComposite("definitions"),
			this._loadWebGPUScenePartComposite("utils"),
			this._loadFileComposite({
				scope: "webgpu",
				key: "webgpu.deferredLighting",
				path: "./webgpu/lighting/deferredLighting.wgsl",
			}),
		]);
		return composeShaderParts(parts, "<webgpu-deferred-lighting-part>");
	}

	private static async _loadWithSharedLightData(
		key: string,
		shaderPromise: Promise<CompositeShaderSource>
	): Promise<CompositeShaderSource> {
		const cacheKey = `webgpu:shared:${key}`;
		const cached = this._fileCompositeCache.get(cacheKey);
		if (cached) {
			this._fileCompositeStats.hits++;
			return cached;
		}
		this._fileCompositeStats.misses++;
		const composed = Promise.all([
			this._loadWebGPUScenePartComposite("lightData"),
			shaderPromise,
		]).then(([lightData, shader]) =>
			composeShaderParts([lightData, shader], "<webgpu-shared-light-data-part>")
		);
		this._fileCompositeCache.set(cacheKey, composed);
		return composed;
	}

	private static _loadWebGPUScenePart(
		key: WebGPUScenePartKey
	): Promise<string | CompositeShaderSource> {
		const parsed = this._parsePartKey<WebGPUSceneShaderPart>(
			key,
			"webgpu.scene.part."
		);
		return parsed.composite ?
				this._loadWebGPUScenePartComposite(parsed.part)
			:	this._loadFileRaw({
					scope: "webgpu",
					key: `webgpu.scene.part.${parsed.part}`,
					path: webgpuSceneShaderFiles[parsed.part],
				});
	}

	private static _loadWebGPUScenePartComposite(
		part: WebGPUSceneShaderPart
	): Promise<CompositeShaderSource> {
		return this._loadFileComposite({
			scope: "webgpu",
			key: `webgpu.scene.part.${part}`,
			path: webgpuSceneShaderFiles[part],
		});
	}

	private static async _loadWebGPUPostProcessPart(
		key: WebGPUPostProcessKey
	): Promise<string | CompositeShaderSource> {
		const parsed = this._parsePartKey<WebGPUPostProcessShaderPart>(
			key,
			"webgpu.postprocess."
		);
		const descriptor: ShaderFileDescriptor = {
			scope: "webgpu",
			key: `webgpu.postprocess.${parsed.part}`,
			path: webgpuPostProcessShaderFiles[parsed.part],
		};
		if (WEBGPU_POST_PROCESS_PARTS_USING_SHARED_LIGHT_DATA.has(parsed.part)) {
			const composite = await this._loadWithSharedLightData(
				`webgpu.postprocess.${parsed.part}.composite`,
				this._loadFileComposite(descriptor)
			);
			return parsed.composite ? composite : composite.code;
		}
		return parsed.composite ?
				this._loadFileComposite(descriptor)
			:	this._loadFileRaw(descriptor);
	}

	private static _loadWebGPUShadowPart(
		key: WebGPUShadowKey
	): Promise<string | CompositeShaderSource> {
		const parsed = this._parsePartKey<WebGPUShadowShaderPart>(
			key,
			"webgpu.shadow."
		);
		const descriptor: ShaderFileDescriptor = {
			scope: "webgpu",
			key: `webgpu.shadow.${parsed.part}`,
			path: webgpuShadowShaderFiles[parsed.part],
		};
		return parsed.composite ?
				this._loadFileComposite(descriptor)
			:	this._loadFileRaw(descriptor);
	}

	private static _loadWebGPUUtilityPart(
		key: WebGPUUtilityKey
	): Promise<string | CompositeShaderSource> {
		const parsed = this._parsePartKey<WebGPUUtilityShaderPart>(
			key,
			"webgpu.utility."
		);
		const descriptor: ShaderFileDescriptor = {
			scope: "webgpu",
			key: `webgpu.utility.${parsed.part}`,
			path: webgpuUtilityShaderFiles[parsed.part],
		};
		return parsed.composite ?
				this._loadFileComposite(descriptor)
			:	this._loadFileRaw(descriptor);
	}

	private static _loadWebGLPart(
		key: WebGLPartKey
	): Promise<string | CompositeShaderSource> {
		const parsed = this._parsePartKey<WebGLShaderPart>(key, "webgl.part.");
		const descriptor: ShaderFileDescriptor = {
			scope: "webgl",
			key: `webgl.part.${parsed.part}`,
			path: webglShaderFiles[parsed.part],
		};
		return parsed.composite ?
				this._loadFileComposite(descriptor)
			:	this._loadFileRaw(descriptor);
	}

	private static async _loadWebGLSceneRaw<K extends ShaderSourceKey>(
		params: ShaderSourceParams<K> | undefined
	): Promise<WebGLSceneShaderSource> {
		const limits = normalizeWebGLSceneLimits(params);
		const [
			vertex,
			fragmentTemplate,
			diffuseProbeFallbackFragment,
			irradianceProbeGridFragment,
		] = await Promise.all([
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.part.sceneVertex",
				path: webglShaderFiles.sceneVertex,
			}),
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.part.sceneFragment",
				path: webglShaderFiles.sceneFragment,
			}),
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.internal.diffuseProbeFallbackFragment",
				path: webglInternalShaderFiles.diffuseProbeFallbackFragment,
			}),
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.internal.irradianceProbeGridFragment",
				path: webglInternalShaderFiles.irradianceProbeGridFragment,
			}),
		]);
		return {
			vertex,
			fragment: this._buildWebGLSceneFragment(fragmentTemplate, limits, {
				diffuseProbeFallbackFragment,
				irradianceProbeGridFragment,
			}),
		};
	}

	private static async _loadWebGLSceneComposite<K extends ShaderSourceKey>(
		params: ShaderSourceParams<K> | undefined
	): Promise<WebGLSceneCompositeShaderSource> {
		const limits = normalizeWebGLSceneLimits(params);
		const [
			vertex,
			fragmentTemplate,
			diffuseProbeFallbackFragment,
			irradianceProbeGridFragment,
		] = await Promise.all([
			this._loadFileComposite({
				scope: "webgl",
				key: "webgl.part.sceneVertex",
				path: webglShaderFiles.sceneVertex,
			}),
			this._loadFileComposite({
				scope: "webgl",
				key: "webgl.part.sceneFragment",
				path: webglShaderFiles.sceneFragment,
			}),
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.internal.diffuseProbeFallbackFragment",
				path: webglInternalShaderFiles.diffuseProbeFallbackFragment,
			}),
			this._loadFileRaw({
				scope: "webgl",
				key: "webgl.internal.irradianceProbeGridFragment",
				path: webglInternalShaderFiles.irradianceProbeGridFragment,
			}),
		]);
		const fragment = this._buildWebGLSceneFragment(
			fragmentTemplate.code,
			limits,
			{
				diffuseProbeFallbackFragment,
				irradianceProbeGridFragment,
			}
		);
		return {
			vertex,
			fragment: createInlineCompositeShaderSource(
				fragment,
				firstSourcePath(fragmentTemplate, webglShaderFiles.sceneFragment),
				"template"
			),
		};
	}

	private static _buildWebGLSceneFragment(
		template: string,
		limits: WebGLSceneLightLimits,
		optionalBlocks: WebGLSceneOptionalBlocks
	): string {
		return replaceOptionalDefines(template, limits, optionalBlocks);
	}

	private static _parsePartKey<T extends string>(
		key: string,
		prefix: string
	): { part: T; composite: boolean } {
		const body = key.slice(prefix.length);
		if (body.endsWith(".composite")) {
			return {
				part: body.slice(0, -".composite".length) as T,
				composite: true,
			};
		}
		if (body.endsWith(".raw")) {
			return {
				part: body.slice(0, -".raw".length) as T,
				composite: false,
			};
		}
		throw new Error(`Invalid ShaderSource part key "${key}".`);
	}

	private static _loadFileRaw(
		descriptor: ShaderFileDescriptor
	): Promise<string> {
		const cacheKey = this._buildFileCacheKey(descriptor);
		const cached = this._rawFileCache.get(cacheKey);
		if (cached) {
			this._rawFileStats.hits++;
			return cached;
		}
		this._rawFileStats.misses++;
		const source = this._readShaderSourceFile(descriptor);
		this._rawFileCache.set(cacheKey, source);
		return source;
	}

	private static _loadFileRawSync(
		descriptor: ShaderFileDescriptor
	): string {
		const cacheKey = this._buildFileCacheKey(descriptor);
		const cached = this._rawFileCache.get(cacheKey);
		if (cached) {
			this._rawFileStats.hits++;
		} else {
			this._rawFileStats.misses++;
		}

		const source = this._readShaderSourceFileSync(descriptor);
		this._rawFileCache.set(cacheKey, Promise.resolve(source));
		return source;
	}

	private static _loadFileComposite(
		descriptor: ShaderFileDescriptor
	): Promise<CompositeShaderSource> {
		const cacheKey = this._buildFileCacheKey(descriptor);
		const cached = this._fileCompositeCache.get(cacheKey);
		if (cached) {
			this._fileCompositeStats.hits++;
			return cached;
		}
		this._fileCompositeStats.misses++;
		const composite = this._loadFileRaw(descriptor).then((code) =>
			createInlineCompositeShaderSource(
				code,
				descriptor.path,
				descriptor.segmentKind ?? "source"
			)
		);
		this._fileCompositeCache.set(cacheKey, composite);
		return composite;
	}

	private static async _readShaderSourceFile(
		descriptor: ShaderFileDescriptor
	): Promise<string> {
		const custom = this._configuration.loader;
		if (custom) {
			try {
				return await custom(this._cloneDescriptor(descriptor));
			} catch (error) {
				if (this._configuration.preferCustomLoader) {
					throw error;
				}
			}
		}

		if (Platform.isNodeRuntime()) {
			try {
				return await this._readShaderSourceFileFromNode(descriptor);
			} catch (error) {
				const embedded = await this._readEmbeddedShaderSource(descriptor);
				if (embedded !== undefined) {
					return embedded;
				}
				throw error;
			}
		}

		const bundled = await this._readBrowserBundledShaderSource(descriptor);
		if (bundled !== undefined) {
			return bundled;
		}
		const embedded = await this._readEmbeddedShaderSource(descriptor);
		if (embedded !== undefined) {
			return embedded;
		}
		throw new Error(`Shader source not bundled: ${descriptor.path}`);
	}

	private static async _readShaderSourceFileFromNode(
		descriptor: ShaderFileDescriptor
	): Promise<string> {
		const fsSpecifier = ["node", "fs/promises"].join(":");
		const fsModule = (await import(/* @vite-ignore */ fsSpecifier)) as NodeFsModule;
		return fsModule.readFile(new URL(descriptor.path, import.meta.url), "utf8");
	}

	private static _readShaderSourceFileSync(
		descriptor: ShaderFileDescriptor
	): string {
		const custom = this._configuration.syncLoader;
		if (custom) {
			const source = custom(this._cloneDescriptor(descriptor));
			if (typeof source === "string") {
				return source;
			}
			if (this._configuration.preferCustomLoader) {
				throw new Error(
					`Custom ShaderSource syncLoader did not provide "${descriptor.path}".`
				);
			}
		} else if (this._configuration.loader && this._configuration.preferCustomLoader) {
			throw new Error(
				`ShaderSource.getSync("${descriptor.key}.raw") requires ` +
					"a syncLoader when preferCustomLoader is enabled."
			);
		}

		const bundled = browserSyncShaderSources[descriptor.path];
		if (typeof bundled === "string") {
			return bundled;
		}
		const embedded = this._readEmbeddedShaderSourceSync(descriptor);
		if (embedded !== undefined) {
			return embedded;
		}

		if (!Platform.isNodeRuntime()) {
			throw new Error(
				`Shader source "${descriptor.path}" is not available for sync browser loading.`
			);
		}
		return this._readShaderSourceFileSyncFromNode(descriptor);
	}

	private static _readShaderSourceFileSyncFromNode(
		descriptor: ShaderFileDescriptor
	): string {
		const nodeProcess = (
			globalThis as typeof globalThis & {
				process?: {
					getBuiltinModule?: (specifier: string) => unknown;
				};
			}
		).process;
		const fsModule = nodeProcess?.getBuiltinModule?.("fs") as
			| NodeFsSyncModule
			| undefined;
		if (!fsModule) {
			throw new Error("Node fs module is unavailable for sync shader loading.");
		}
		return fsModule.readFileSync(
			new URL(descriptor.path, import.meta.url),
			"utf8"
		);
	}

	private static async _readBrowserBundledShaderSource(
		descriptor: ShaderFileDescriptor
	): Promise<string | undefined> {
		const syncSource = browserSyncShaderSources[descriptor.path];
		if (typeof syncSource === "string") {
			return syncSource;
		}
		const loader = browserShaderSources[descriptor.path];
		if (!loader) {
			return undefined;
		}
		return loader();
	}

	private static async _readEmbeddedShaderSource(
		descriptor: ShaderFileDescriptor
	): Promise<string | undefined> {
		const { embeddedShaderSources } = await import(
			"./generated/embeddedShaderSources"
		);
		return embeddedShaderSources[descriptor.path];
	}

	private static _readEmbeddedShaderSourceSync(
		descriptor: ShaderFileDescriptor
	): string | undefined {
		return embeddedSyncShaderSources[descriptor.path];
	}

	private static _cloneDescriptor(
		descriptor: ShaderFileDescriptor
	): ShaderSourceFileDescriptor {
		return { ...descriptor };
	}

	private static _buildFileCacheKey(descriptor: ShaderFileDescriptor): string {
		return `${descriptor.scope}:file:${descriptor.key}:${descriptor.path}`;
	}

	private static _buildResultCacheKey<K extends ShaderSourceKey>(
		key: K,
		params: ShaderSourceParams<K> | undefined
	): string {
		if (key === "webgl.scene.raw" || key === "webgl.scene.composite") {
			const limits = normalizeWebGLSceneLimits(params);
			return (
				`webgl:result:${key}` +
				`|dir:${limits.maxDirectionalLights}` +
				`|point:${limits.maxPointLights}` +
				`|spot:${limits.maxSpotLights}` +
				`|shadow:${limits.enableShadowTransmittance ? 1 : 0}` +
				`|grid:${limits.enableIrradianceProbeGrid ? 1 : 0}`
			);
		}
		return `${shaderSourceScope(key)}:result:${key}`;
	}

	private static _deleteByScope<T>(
		cache: Map<string, T>,
		scope: "all" | "webgpu" | "webgl"
	): void {
		if (scope === "all") {
			cache.clear();
			return;
		}
		for (const key of cache.keys()) {
			if (key.startsWith(`${scope}:`)) {
				cache.delete(key);
				continue;
			}
			if (isCompositeKey(key as ShaderSourceKey) && key.includes(scope)) {
				cache.delete(key);
			}
		}
	}

	private static _deleteSetByScope(
		cache: Set<string>,
		scope: "all" | "webgpu" | "webgl"
	): void {
		if (scope === "all") {
			cache.clear();
			return;
		}
		for (const key of cache.keys()) {
			if (key.startsWith(`${scope}:`)) {
				cache.delete(key);
			}
		}
	}
}
