import { Platform } from "../foundation/Platform";
import { embeddedSyncShaderSources } from "./generated/embeddedSyncShaderSources";
import {
	buildShaderSourceArtifact,
	cloneShaderSourceArtifact,
	resolveShaderManifestRequest,
	type ShaderBackendManifest,
	type ShaderModuleSourceArtifact,
	type ShaderProgramSourceArtifact,
	type ShaderSourceArtifact,
} from "./ShaderManifest";
import { createInlineCompositeShaderSource } from "./runtime";
import {
	WEBGL_SHADER_MANIFEST,
	type WebGLShaderPart,
} from "./webgl/sources";
import type {
	WebGLSceneDepthVariantDescriptor,
	WebGLSceneVariantDescriptor,
} from "./webgl/sceneVariants";
import {
	WEBGPU_SHADER_MANIFEST,
	type WebGPUPostProcessShaderPart,
	type WebGPUSceneShaderPart,
	type WebGPUShadowShaderPart,
	type WebGPUUtilityShaderPart,
} from "./webgpu/sources";
import type { ShaderSourceSegmentKind } from "./runtime/types";

export type { WebGLShaderPart } from "./webgl/sources";
export type {
	ShaderModuleSourceArtifact,
	ShaderProgramSourceArtifact,
	ShaderSourceArtifact,
} from "./ShaderManifest";
export type {
	WebGPUPostProcessShaderPart,
	WebGPUSceneShaderPart,
	WebGPUShadowShaderPart,
	WebGPUUtilityShaderPart,
} from "./webgpu/sources";

type ImportMetaGlobLoaderMap = Record<string, () => Promise<string>>;
const browserShaderSources: ImportMetaGlobLoaderMap = createBrowserShaderSources();
const browserSyncShaderSources: Record<string, string> = createBrowserSyncShaderSources();

type NodeFsModule = {
	readFile: (path: string | URL, options?: string | { encoding?: string }) => Promise<string>;
};
type NodeFsSyncModule = {
	readFileSync: (path: string | URL, encoding: "utf8") => string;
};

export interface WebGLSceneLightLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
}

export interface WebGLDeformationSpecialization {
	skinProfile: "static" | "skin4" | "skin8";
	morphPosition: boolean;
}

type WebGPUFixedShaderKey =
	| "webgpu.scene"
	| "webgpu.environment"
	| "webgpu.deferredLighting"
	| "webgpu.particle"
	| "webgpu.particleSimulation"
	| "webgpu.clusteredLightingCull"
	| "webgpu.iblPrefilter";
type WebGPUScenePartKey = `webgpu.scene.part.${WebGPUSceneShaderPart}`;
type WebGPUPostProcessKey = `webgpu.postprocess.${WebGPUPostProcessShaderPart}`;
type WebGPUShadowKey = `webgpu.shadow.${WebGPUShadowShaderPart}`;
type WebGPUUtilityKey = `webgpu.utility.${WebGPUUtilityShaderPart}`;
type WebGPUDirectiveKey = `webgpu.directive.${"constants" | "srgb" | "fog" | "lumaWeights" | "lumaCommon"}`;
type WebGLPartKey = `webgl.part.${WebGLShaderPart}`;
type WebGLDirectiveKey = `webgl.directive.${"animation" | "constants" | "srgb" | "fog" | "lumaWeights" | "lumaCommon"}`;

export type ShaderSourceKey =
	| WebGPUFixedShaderKey
	| WebGPUScenePartKey
	| WebGPUPostProcessKey
	| WebGPUShadowKey
	| WebGPUUtilityKey
	| WebGPUDirectiveKey
	| WebGLPartKey
	| WebGLDirectiveKey
	| "webgl.scene"
	| "webgl.scene.depth"
	| "webgl.shadow.depth"
	| "webgl.shadow.transmittance";

export type ShaderSourceSyncKey = "webgpu.utility.mipmapBlit";
export type ShaderSourceParams<K extends ShaderSourceKey> =
	K extends "webgl.scene" ? { specialization?: WebGLSceneVariantDescriptor }
	: K extends "webgl.scene.depth" ? { specialization?: WebGLSceneDepthVariantDescriptor }
	: K extends "webgl.shadow.depth" | "webgl.shadow.transmittance" ?
		{ specialization?: WebGLDeformationSpecialization }
	: undefined;
export type ShaderSourceResult<K extends ShaderSourceKey> =
	K extends "webgl.scene" | "webgl.scene.depth" | "webgl.shadow.depth" | "webgl.shadow.transmittance" ?
		ShaderProgramSourceArtifact
	: ShaderModuleSourceArtifact;

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
export type ShaderSourceLoader = (descriptor: ShaderSourceFileDescriptor) => Promise<string>;
export type ShaderSourceSyncLoader = (descriptor: ShaderSourceFileDescriptor) => string | undefined;
export interface ShaderSourceConfiguration {
	loader?: ShaderSourceLoader;
	syncLoader?: ShaderSourceSyncLoader;
	preferCustomLoader?: boolean;
}
interface MutableCacheBucketStats { hits: number; misses: number }
const emptyCacheStats = (): MutableCacheBucketStats => ({ hits: 0, misses: 0 });

/** Loads, interprets, and caches built-in shader manifest artifacts. */
export class ShaderSource {
	private static _rawFileCache = new Map<string, Promise<string>>();
	private static _fileCompositeCache = new Map<string, Promise<ReturnType<typeof createInlineCompositeShaderSource>>>();
	private static _resultCache = new Map<string, Promise<ShaderSourceArtifact>>();
	private static _preparedCache = new Map<string, ShaderSourceArtifact>();
	private static _inFlightResultKeys = new Set<string>();
	private static _configuration: ShaderSourceConfiguration = {};
	private static _rawFileStats = emptyCacheStats();
	private static _fileCompositeStats = emptyCacheStats();
	private static _resultStats = emptyCacheStats();
	private static _preparedStats = emptyCacheStats();

	/** Configures global built-in source loaders and clears all source caches. */
	public static configure(configuration: ShaderSourceConfiguration): void {
		this._configuration = { ...configuration };
		this.clearCache();
	}
	/** Restores built-in loading and clears all source caches. */
	public static resetConfiguration(): void {
		this._configuration = {};
		this.clearCache();
	}
	/** Loads and prepares a cloned canonical shader artifact. */
	public static async load<K extends ShaderSourceKey>(key: K, params?: ShaderSourceParams<K>): Promise<ShaderSourceResult<K>> {
		const resolved = this._resolve(key, params);
		const prepared = this._preparedCache.get(resolved.identity);
		if (prepared) {
			this._preparedStats.hits++;
			return cloneShaderSourceArtifact(prepared) as ShaderSourceResult<K>;
		}
		this._preparedStats.misses++;
		const artifact = await this._getArtifactPromise(key, resolved);
		this._preparedCache.set(resolved.identity, cloneShaderSourceArtifact(artifact));
		return cloneShaderSourceArtifact(artifact) as ShaderSourceResult<K>;
	}
	/** Loads an explicitly approved synchronous module artifact. */
	public static getSync<K extends ShaderSourceSyncKey>(key: K): ShaderSourceResult<K> {
		const manifest = this._manifestForKey(key);
		const resolved = resolveShaderManifestRequest(manifest, key, undefined);
		const prepared = this._preparedCache.get(resolved.identity);
		if (prepared) {
			this._preparedStats.hits++;
			return cloneShaderSourceArtifact(prepared) as ShaderSourceResult<K>;
		}
		this._preparedStats.misses++;
		this._resultStats.misses++;
		const definition = manifest.sources[key];
		if (!definition || definition.kind !== "module" || !("asset" in definition.source)) {
			throw new Error(`ShaderSource.getSync("${key}").source.code requires a direct sync asset.`);
		}
		const asset = manifest.assets[definition.source.asset];
		if (!asset?.sync) throw new Error(`ShaderSource.getSync("${key}").source.code is not approved for sync loading.`);
		const descriptor: ShaderSourceFileDescriptor = {
			scope: manifest.backend as "webgpu" | "webgl",
			key: definition.source.asset,
			path: asset.path,
			segmentKind: asset.segmentKind,
		};
		const source = createInlineCompositeShaderSource(this._loadFileRawSync(descriptor), asset.path, asset.segmentKind ?? "source");
		const artifact: ShaderModuleSourceArtifact = {
			kind: "module", key, identity: resolved.identity, language: manifest.language,
			sourceKind: definition.sourceKind, source,
		};
		this._resultCache.set(resolved.identity, Promise.resolve(artifact));
		this._preparedCache.set(resolved.identity, cloneShaderSourceArtifact(artifact));
		return cloneShaderSourceArtifact(artifact) as ShaderSourceResult<K>;
	}
	/** Prepares one canonical artifact for later synchronous access. */
	public static async prepare<K extends ShaderSourceKey>(key: K, params?: ShaderSourceParams<K>): Promise<void> {
		await this.load(key, params);
	}
	/** Prepares canonical artifacts concurrently. */
	public static async prepareMany(requests: readonly ShaderSourceRequest[]): Promise<void> {
		await Promise.all(requests.map((request) => this.prepare(request.key, request.params as never)));
	}
	/** Returns a cloned prepared artifact or throws when it was not prepared. */
	public static get<K extends ShaderSourceKey>(key: K, params?: ShaderSourceParams<K>): ShaderSourceResult<K> {
		const resolved = this._resolve(key, params);
		const prepared = this._preparedCache.get(resolved.identity);
		if (!prepared) {
			this._preparedStats.misses++;
			throw new Error(`Shader source "${resolved.identity}" is not prepared. Call ShaderSource.prepare()/prepareMany() before get().`);
		}
		this._preparedStats.hits++;
		return cloneShaderSourceArtifact(prepared) as ShaderSourceResult<K>;
	}
	/** Reports whether an exact canonical artifact is prepared. */
	public static has<K extends ShaderSourceKey>(key: K, params?: ShaderSourceParams<K>): boolean {
		return this._preparedCache.has(this._resolve(key, params).identity);
	}
	/** Resolves the stable manifest identity for an exact source request. */
	public static getIdentity<K extends ShaderSourceKey>(key: K, params?: ShaderSourceParams<K>): string {
		return this._resolve(key, params).identity;
	}
	/** Clears source caches globally or for one backend scope. */
	public static clearCache(scope: "all" | "webgpu" | "webgl" = "all"): void {
		this._deleteByScope(this._rawFileCache, scope);
		this._deleteByScope(this._fileCompositeCache, scope);
		this._deleteByScope(this._resultCache, scope);
		this._deleteByScope(this._preparedCache, scope);
		this._deleteSetByScope(this._inFlightResultKeys, scope);
		this._rawFileStats = emptyCacheStats(); this._fileCompositeStats = emptyCacheStats();
		this._resultStats = emptyCacheStats(); this._preparedStats = emptyCacheStats();
	}
	/** Returns source-loader and artifact-cache diagnostics. */
	public static getCacheStats(): ShaderSourceCacheStats {
		return {
			rawFiles: { ...this._rawFileStats, size: this._rawFileCache.size },
			fileComposites: { ...this._fileCompositeStats, size: this._fileCompositeCache.size },
			results: { ...this._resultStats, size: this._resultCache.size },
			prepared: { ...this._preparedStats, size: this._preparedCache.size },
			inFlight: this._inFlightResultKeys.size,
		};
	}

	private static _resolve<K extends ShaderSourceKey>(key: K, params: ShaderSourceParams<K> | undefined) {
		return resolveShaderManifestRequest(this._manifestForKey(key), key, params);
	}
	private static _getArtifactPromise<K extends ShaderSourceKey>(key: K, resolved: ReturnType<typeof resolveShaderManifestRequest>): Promise<ShaderSourceArtifact> {
		const cached = this._resultCache.get(resolved.identity);
		if (cached) { this._resultStats.hits++; return cached; }
		this._resultStats.misses++;
		this._inFlightResultKeys.add(resolved.identity);
		const manifest = this._manifestForKey(key);
		let result: Promise<ShaderSourceArtifact>;
		result = buildShaderSourceArtifact(manifest, key, resolved, {
			loadAsset: (owner, assetId) => this._loadManifestAsset(owner, assetId),
			loadSource: (owner, sourceKey) => {
				const nested = resolveShaderManifestRequest(owner, sourceKey, undefined);
				return this._getArtifactPromise(sourceKey as ShaderSourceKey, nested);
			},
		}).catch((error) => {
			if (this._resultCache.get(resolved.identity) === result) this._resultCache.delete(resolved.identity);
			throw error;
		}).finally(() => this._inFlightResultKeys.delete(resolved.identity));
		this._resultCache.set(resolved.identity, result);
		return result;
	}
	private static _loadManifestAsset(manifest: ShaderBackendManifest, assetId: string) {
		const asset = manifest.assets[assetId];
		if (!asset) throw new Error(`Unknown shader manifest asset "${assetId}".`);
		return this._loadFileComposite({
			scope: manifest.backend as "webgpu" | "webgl", key: assetId,
			path: asset.path, segmentKind: asset.segmentKind,
		});
	}
	private static _loadFileRaw(descriptor: ShaderSourceFileDescriptor): Promise<string> {
		const cacheKey = this._buildFileCacheKey(descriptor);
		const cached = this._rawFileCache.get(cacheKey);
		if (cached) { this._rawFileStats.hits++; return cached; }
		this._rawFileStats.misses++;
		const source = this._readShaderSourceFile(descriptor);
		this._rawFileCache.set(cacheKey, source);
		return source;
	}
	private static _loadFileRawSync(descriptor: ShaderSourceFileDescriptor): string {
		const cacheKey = this._buildFileCacheKey(descriptor);
		if (this._rawFileCache.has(cacheKey)) this._rawFileStats.hits++; else this._rawFileStats.misses++;
		const source = this._readShaderSourceFileSync(descriptor);
		this._rawFileCache.set(cacheKey, Promise.resolve(source));
		return source;
	}
	private static _loadFileComposite(descriptor: ShaderSourceFileDescriptor) {
		const cacheKey = this._buildFileCacheKey(descriptor);
		const cached = this._fileCompositeCache.get(cacheKey);
		if (cached) { this._fileCompositeStats.hits++; return cached; }
		this._fileCompositeStats.misses++;
		const composite = this._loadFileRaw(descriptor).then((code) => createInlineCompositeShaderSource(code, descriptor.path, descriptor.segmentKind ?? "source"));
		this._fileCompositeCache.set(cacheKey, composite);
		return composite;
	}
	private static async _readShaderSourceFile(descriptor: ShaderSourceFileDescriptor): Promise<string> {
		const custom = this._configuration.loader;
		if (custom) {
			try { return await custom({ ...descriptor }); }
			catch (error) { if (this._configuration.preferCustomLoader) throw error; }
		}
		if (Platform.isNodeRuntime()) return this._readShaderSourceFileFromNode(descriptor);
		return this._readBrowserBundledShaderSource(descriptor);
	}
	private static async _readShaderSourceFileFromNode(descriptor: ShaderSourceFileDescriptor): Promise<string> {
		const importer = new Function("specifier", "return import(specifier);") as (specifier: string) => Promise<NodeFsModule>;
		const fs = await importer("node:fs/promises");
		return fs.readFile(new URL(descriptor.path, import.meta.url), "utf8");
	}
	private static _readShaderSourceFileSync(descriptor: ShaderSourceFileDescriptor): string {
		const custom = this._configuration.syncLoader;
		if (custom) {
			const result = custom({ ...descriptor });
			if (result !== undefined) return result;
			if (this._configuration.preferCustomLoader) throw new Error(`Custom sync shader loader did not provide "${descriptor.path}".`);
		}
		if (this._configuration.preferCustomLoader && this._configuration.loader) {
			throw new Error(
				`ShaderSource.getSync("${descriptor.key}") requires a syncLoader when preferCustomLoader is true.`,
			);
		}
		if (Platform.isNodeRuntime()) {
			const nodeProcess = (
				globalThis as typeof globalThis & {
					process?: { getBuiltinModule?: (specifier: string) => unknown };
				}
			).process;
			const fs = nodeProcess?.getBuiltinModule?.("fs") as NodeFsSyncModule | undefined;
			if (!fs) throw new Error("Node fs module is unavailable for sync shader loading.");
			return fs.readFileSync(new URL(descriptor.path, import.meta.url), "utf8");
		}
		const bundled = browserSyncShaderSources[descriptor.path] ?? embeddedSyncShaderSources[descriptor.path];
		if (bundled === undefined) throw new Error(`Synchronous shader source "${descriptor.path}" is unavailable.`);
		return bundled;
	}
	private static async _readBrowserBundledShaderSource(descriptor: ShaderSourceFileDescriptor): Promise<string> {
		const loader = browserShaderSources[descriptor.path];
		if (loader) return loader();
		const { embeddedShaderSources } = await import("./generated/embeddedShaderSources");
		const embedded = embeddedShaderSources[descriptor.path];
		if (embedded !== undefined) return embedded;
		throw new Error(`Shader path "${descriptor.path}" is not bundled.`);
	}
	private static _manifestForKey(key: string): ShaderBackendManifest {
		return key.startsWith("webgl.") ? WEBGL_SHADER_MANIFEST : WEBGPU_SHADER_MANIFEST;
	}
	private static _buildFileCacheKey(descriptor: ShaderSourceFileDescriptor): string {
		return `${descriptor.scope}:file:${descriptor.key}:${descriptor.path}`;
	}
	private static _deleteByScope<T>(cache: Map<string, T>, scope: "all" | "webgpu" | "webgl"): void {
		if (scope === "all") { cache.clear(); return; }
		for (const key of cache.keys()) if (key.startsWith(`${scope}:`) || key.startsWith(`${scope}.`)) cache.delete(key);
	}
	private static _deleteSetByScope(cache: Set<string>, scope: "all" | "webgpu" | "webgl"): void {
		if (scope === "all") { cache.clear(); return; }
		for (const key of cache) if (key.startsWith(`${scope}:`) || key.startsWith(`${scope}.`)) cache.delete(key);
	}
}

function createBrowserShaderSources(): ImportMetaGlobLoaderMap {
	if (Platform.isNodeRuntime()) return {};
	try {
		return import.meta.glob<string>(["./webgl/**/*.glsl", "./webgpu/**/*.wgsl", "!./webgpu/utility/mipmapBlit.wgsl"], { query: "?raw", import: "default" });
	} catch { return {}; }
}
function createBrowserSyncShaderSources(): Record<string, string> {
	if (Platform.isNodeRuntime()) return {};
	try {
		return import.meta.glob<string>("./webgpu/utility/mipmapBlit.wgsl", { query: "?raw", import: "default", eager: true });
	} catch { return {}; }
}
