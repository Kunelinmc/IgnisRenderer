import { ShaderRuntime } from "./ShaderRuntime";
import {
	assertShaderDirectiveProfileRegistryComplete,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
} from "./engineDirectives";
import type {
	CompositeShaderSource,
	ShaderBackendCompileResult,
	ShaderBackendId,
	ShaderDiagnostic,
	ShaderDirectiveCompileHook,
	ShaderDirectiveHookContext,
	ShaderDirectiveHookResult,
	ShaderDirectiveProfile,
	ShaderDirectiveProfileRegistry,
	ShaderDirectiveStageRequest,
	ShaderDirectiveStageResult,
	ShaderLanguage,
	ShaderProcessRequest,
	ShaderRuntimeMode,
	ShaderSourceKind,
	ShaderSourceSegmentMap,
	ShaderStage,
} from "./types";

const BASE_HOOK_TOKEN = "base";
const HOOK_TOKEN_PATTERN = /^[A-Za-z0-9._:/-]{1,128}$/;

interface HookPatch {
	includeModules: NonNullable<ShaderDirectiveHookResult["includeModules"]>;
	injectionScripts: NonNullable<ShaderDirectiveHookResult["injectionScripts"]>;
}

interface HookResolution {
	token: string;
	patch: HookPatch | null;
}

interface ShaderDirectiveStageOptions {
	backend: ShaderBackendId;
	profiles?: ShaderDirectiveProfileRegistry;
	hook?: ShaderDirectiveCompileHook | null;
	mode?: ShaderRuntimeMode;
	warn?: ((key: string, message: string) => void) | null;
}

interface ShaderBackendCompileStageOptions {
	backend: ShaderBackendId;
	runtime: ShaderRuntime;
	profiles?: ShaderDirectiveProfileRegistry;
	hook?: ShaderDirectiveCompileHook | null;
	mode?: ShaderRuntimeMode;
	warn?: ((key: string, message: string) => void) | null;
}

interface CachedDirectiveStageResult {
	result: ShaderDirectiveStageResult;
}

function hashStringFNV1a(value: string): string {
	let hash = 0x811c9dc5;
	for (let i = 0; i < value.length; i++) {
		hash ^= value.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}

function hashSourceMap(sourceMap: ShaderSourceSegmentMap | null | undefined): string {
	if (!sourceMap || !Array.isArray(sourceMap.segments)) {
		return "none";
	}
	const payload = [
		`lineCount:${sourceMap.lineCount}`,
		...sourceMap.segments.map((segment) =>
			[
				segment.generatedLineStart,
				segment.generatedLineEnd,
				segment.sourcePath,
				segment.sourceLineStart,
				segment.sourceLineEnd,
				segment.kind,
				segment.label ?? "",
			].join(":")
		),
	].join("|");
	return hashStringFNV1a(payload);
}

function normalizeLanguage(language: ShaderLanguage): ShaderLanguage {
	return language === "glsl" ? "glsl" : "wgsl";
}

function normalizeStage(stage?: ShaderStage): ShaderStage {
	switch (stage) {
		case "vertex":
		case "fragment":
		case "compute":
			return stage;
		default:
			return "unknown";
	}
}

function normalizeSourceKind(sourceKind?: ShaderSourceKind): ShaderSourceKind {
	switch (sourceKind) {
		case "builtin-scene":
		case "builtin-skybox":
		case "builtin-present":
		case "postprocess":
		case "clustered":
		case "shadow":
		case "particle":
		case "custom-material":
			return sourceKind;
		default:
			return "unknown";
	}
}

function cloneSourceMap(sourceMap: ShaderSourceSegmentMap): ShaderSourceSegmentMap {
	return {
		lineCount: sourceMap.lineCount,
		segments: sourceMap.segments.map((segment) => ({ ...segment })),
	};
}

function cloneCompositeSource(
	composite: CompositeShaderSource
): CompositeShaderSource {
	return {
		code: composite.code,
		sourceMap: cloneSourceMap(composite.sourceMap),
	};
}

function cloneDiagnostics(diagnostics: ShaderDiagnostic[]): ShaderDiagnostic[] {
	return diagnostics.map((diagnostic) => ({ ...diagnostic }));
}

function isPromiseLike<T = unknown>(
	value: unknown
): value is PromiseLike<T> {
	return (
		typeof value === "object" &&
		value !== null &&
		"then" in (value as Record<string, unknown>) &&
		typeof (value as { then?: unknown }).then === "function"
	);
}

function hasHookPatch(result: HookResolution): boolean {
	return (
		!!result.patch &&
		(
			result.patch.includeModules.length > 0 ||
			result.patch.injectionScripts.length > 0
		)
	);
}

function buildHookPatchFingerprint(patch: HookPatch): string {
	const includePayload = patch.includeModules
		.map((module) =>
			[
				module.language,
				module.id.trim(),
				module.sourcePath ?? "",
				hashStringFNV1a(module.code),
			].join(":")
		)
		.sort()
		.join("|");
	const scriptPayload = patch.injectionScripts
		.map((script) => script.id.trim())
		.sort()
		.join("|");
	return hashStringFNV1a(`${includePayload}#${scriptPayload}`);
}

function buildDirectiveSourcePath(request: ShaderDirectiveStageRequest): string {
	if (
		typeof request.directiveSourcePath === "string" &&
		request.directiveSourcePath.trim().length > 0
	) {
		return request.directiveSourcePath.trim();
	}
	if (typeof request.label === "string" && request.label.trim().length > 0) {
		return request.label.trim();
	}
	return "<shader>";
}

function createHookContext(
	backend: ShaderBackendId,
	request: ShaderDirectiveStageRequest
): ShaderDirectiveHookContext {
	return {
		backend,
		language: normalizeLanguage(request.language),
		stage: normalizeStage(request.stage),
		sourceKind: normalizeSourceKind(request.sourceKind),
		label:
			typeof request.label === "string" && request.label.trim().length > 0 ?
				request.label.trim()
			:	null,
		directiveSourcePath: buildDirectiveSourcePath(request),
	};
}

function buildDirectiveCacheKey(
	fingerprint: string,
	request: ShaderDirectiveStageRequest
): string {
	const sourceMapHash = hashSourceMap(request.sourceMap ?? null);
	const sourceHash = hashStringFNV1a(request.code);
	return [
		`fingerprint:${fingerprint}`,
		`lang:${normalizeLanguage(request.language)}`,
		`stage:${normalizeStage(request.stage)}`,
		`entry:${request.entryPoint ?? ""}`,
		`kind:${normalizeSourceKind(request.sourceKind)}`,
		`label:${request.label ?? ""}`,
		`sourcePath:${buildDirectiveSourcePath(request)}`,
		`code:${sourceHash}`,
		`sourceMap:${sourceMapHash}`,
	].join("|");
}

function buildFingerprintContextKey(
	request: ShaderDirectiveStageRequest
): string {
	return [
		normalizeLanguage(request.language),
		normalizeStage(request.stage),
		normalizeSourceKind(request.sourceKind),
		request.entryPoint ?? "",
		request.label ?? "",
		buildDirectiveSourcePath(request),
	].join("|");
}

function normalizeHookToken(token: unknown): string {
	if (typeof token !== "string") {
		return "";
	}
	return token.trim();
}

export class ShaderDirectiveStage {
	private _backend: ShaderBackendId;
	private _profile: ShaderDirectiveProfile;
	private _hook: ShaderDirectiveCompileHook | null;
	private _mode: ShaderRuntimeMode;
	private _warn: ((key: string, message: string) => void) | null;
	private _baseRuntime: ShaderRuntime;
	private _runtimeByToken = new Map<string, ShaderRuntime>();
	private _patchFingerprintByToken = new Map<string, string>();
	private _cache = new Map<string, CachedDirectiveStageResult>();
	private _lastFingerprintByContext = new Map<string, string>();
	private _revision = 1;

	public constructor(options: ShaderDirectiveStageOptions) {
		this._backend = options.backend;
		const registry =
			options.profiles ?? DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY;
		assertShaderDirectiveProfileRegistryComplete(registry);
		this._profile = registry[this._backend];
		this._hook = options.hook ?? null;
		this._mode = options.mode ?? "warn";
		this._warn = options.warn ?? null;
		this._baseRuntime = this._createRuntimeForPatch(null);
		this._runtimeByToken.set(BASE_HOOK_TOKEN, this._baseRuntime);
	}

	public get revision(): number {
		return this._revision;
	}

	public get profile(): ShaderDirectiveProfile {
		return this._profile;
	}

	public getCacheFingerprintTag(): string {
		return `${this._profile.id}|rev:${this._profile.revision}`;
	}

	public process(request: ShaderDirectiveStageRequest): ShaderDirectiveStageResult {
		const resolution = this._resolveHookSync(request);
		return this._processWithResolution(request, resolution);
	}

	public async processAsync(
		request: ShaderDirectiveStageRequest
	): Promise<ShaderDirectiveStageResult> {
		const resolution = await this._resolveHookAsync(request);
		return this._processWithResolution(request, resolution);
	}

	private _processWithResolution(
		request: ShaderDirectiveStageRequest,
		resolution: HookResolution
	): ShaderDirectiveStageResult {
		const fingerprint = this._buildDirectiveFingerprint(resolution.token);
		this._trackFingerprint(request, fingerprint);
		const cacheKey = buildDirectiveCacheKey(fingerprint, request);
		const cached = this._cache.get(cacheKey);
		if (cached) {
			return this._cloneStageResult(cached.result);
		}

		const runtime = this._resolveRuntimeForHookResolution(resolution);
		const preprocessed = runtime.preprocessDirectives({
			code: request.code,
			sourceMap: request.sourceMap ?? null,
			language: normalizeLanguage(request.language),
			stage: normalizeStage(request.stage),
			entryPoint: request.entryPoint,
			label: request.label,
			sourceKind: normalizeSourceKind(request.sourceKind),
			directiveSourcePath: buildDirectiveSourcePath(request),
			enableDirectives: true,
		});
		const stageResult: ShaderDirectiveStageResult = {
			code: preprocessed.code,
			sourceMap: preprocessed.sourceMap,
			composite: preprocessed.composite,
			diagnostics: preprocessed.diagnostics,
			hasErrors: preprocessed.hasErrors,
			directiveFingerprint: fingerprint,
		};
		this._cache.set(cacheKey, {
			result: this._cloneStageResult(stageResult),
		});
		return this._cloneStageResult(stageResult);
	}

	private _resolveRuntimeForHookResolution(
		resolution: HookResolution
	): ShaderRuntime {
		if (!hasHookPatch(resolution)) {
			return this._baseRuntime;
		}
		const existing = this._runtimeByToken.get(resolution.token);
		if (existing) {
			return existing;
		}
		const runtime = this._createRuntimeForPatch(resolution.patch);
		this._runtimeByToken.set(resolution.token, runtime);
		return runtime;
	}

	private _createRuntimeForPatch(patch: HookPatch | null): ShaderRuntime {
		const runtime = new ShaderRuntime({ mode: "warn" });
		this._registerProfile(runtime, this._profile);
		if (patch) {
			for (const includeModule of patch.includeModules) {
				runtime.registerIncludeModule(
					includeModule.language,
					includeModule.id,
					includeModule.code,
					includeModule.sourcePath
				);
			}
			for (const script of patch.injectionScripts) {
				runtime.registerInjectionScript(script);
			}
		}
		return runtime;
	}

	private _registerProfile(
		runtime: ShaderRuntime,
		profile: ShaderDirectiveProfile
	): void {
		for (const includeModule of profile.includeModules) {
			runtime.registerIncludeModule(
				includeModule.language,
				includeModule.id,
				includeModule.code,
				includeModule.sourcePath
			);
		}
		for (const script of profile.injectionScripts) {
			runtime.registerInjectionScript(script);
		}
	}

	private _resolveHookSync(request: ShaderDirectiveStageRequest): HookResolution {
		if (!this._hook) {
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
		const context = createHookContext(this._backend, request);
		try {
			const result = this._hook(context);
			if (isPromiseLike(result)) {
				this._handleHookError(
					"hook-async-sync-path",
					`Shader directive hook for "${this._backend}" returned a Promise in sync compile path.`
				);
				return { token: BASE_HOOK_TOKEN, patch: null };
			}
			return this._normalizeHookResult(result, context, false);
		} catch (error) {
			this._handleHookError(
				"hook-sync-failure",
				`Shader directive hook for "${this._backend}" failed: ${String(error)}`
			);
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
	}

	private async _resolveHookAsync(
		request: ShaderDirectiveStageRequest
	): Promise<HookResolution> {
		if (!this._hook) {
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
		const context = createHookContext(this._backend, request);
		try {
			const result = await this._hook(context);
			return this._normalizeHookResult(result, context, true);
		} catch (error) {
			this._handleHookError(
				"hook-async-failure",
				`Shader directive hook for "${this._backend}" failed: ${String(error)}`
			);
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
	}

	private _normalizeHookResult(
		result: ShaderDirectiveHookResult | null | undefined,
		context: ShaderDirectiveHookContext,
		isAsyncPath: boolean
	): HookResolution {
		if (!result) {
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
		const includeModules = Array.isArray(result.includeModules) ?
				result.includeModules.map((module) => ({ ...module }))
			:	[];
		const injectionScripts = Array.isArray(result.injectionScripts) ?
				result.injectionScripts.map((script) => ({ ...script }))
			:	[];
		const patch: HookPatch | null =
			includeModules.length > 0 || injectionScripts.length > 0 ?
				{
					includeModules,
					injectionScripts,
				}
			:	null;
		if (!patch) {
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
		const token = normalizeHookToken(result.token);
		if (!HOOK_TOKEN_PATTERN.test(token)) {
			this._handleHookError(
				"hook-token-invalid",
				`Shader directive hook for "${this._backend}" returned invalid token "${token}" for ${context.directiveSourcePath}.`
			);
			return { token: BASE_HOOK_TOKEN, patch: null };
		}

		const patchFingerprint = buildHookPatchFingerprint(patch);
		const existingPatchFingerprint = this._patchFingerprintByToken.get(token);
		if (
			existingPatchFingerprint &&
			existingPatchFingerprint !== patchFingerprint
		) {
			this._warnWithFallback(
				"hook-token-collision",
				`Shader directive hook token collision for "${token}" on ${context.directiveSourcePath}; disabling hook patch for safety.`,
				isAsyncPath
			);
			return { token: BASE_HOOK_TOKEN, patch: null };
		}
		if (!existingPatchFingerprint) {
			this._patchFingerprintByToken.set(token, patchFingerprint);
		}
		return {
			token,
			patch,
		};
	}

	private _buildDirectiveFingerprint(hookToken: string): string {
		return (
			`${this._profile.id}|rev:${this._profile.revision}` +
			`|hook:${hookToken}`
		);
	}

	private _trackFingerprint(
		request: ShaderDirectiveStageRequest,
		fingerprint: string
	): void {
		const key = buildFingerprintContextKey(request);
		const previous = this._lastFingerprintByContext.get(key);
		if (previous === fingerprint) {
			return;
		}
		this._lastFingerprintByContext.set(key, fingerprint);
		this._revision++;
	}

	private _cloneStageResult(
		result: ShaderDirectiveStageResult
	): ShaderDirectiveStageResult {
		return {
			code: result.code,
			sourceMap: cloneSourceMap(result.sourceMap),
			composite: cloneCompositeSource(result.composite),
			diagnostics: cloneDiagnostics(result.diagnostics),
			hasErrors: result.hasErrors,
			directiveFingerprint: result.directiveFingerprint,
		};
	}

	private _warnWithFallback(
		code: string,
		message: string,
		_isAsyncPath: boolean
	): void {
		if (this._warn) {
			this._warn(`shader-directive-${this._backend}-${code}`, message);
		} else {
			console.warn(message);
		}
	}

	private _handleHookError(code: string, message: string): void {
		if (this._mode === "strict") {
			throw new Error(
				`${message} Migration hint: move directives to backend compile stage hooks with stable tokens.`
			);
		}
		this._warnWithFallback(code, message, false);
	}
}

export class ShaderBackendCompileStage {
	private _runtime: ShaderRuntime;
	private _mode: ShaderRuntimeMode;
	private _directiveStage: ShaderDirectiveStage;

	public constructor(options: ShaderBackendCompileStageOptions) {
		this._runtime = options.runtime;
		this._mode = options.mode ?? "warn";
		this._directiveStage = new ShaderDirectiveStage({
			backend: options.backend,
			profiles: options.profiles,
			hook: options.hook ?? null,
			mode: this._mode,
			warn: options.warn ?? null,
		});
	}

	public get revision(): number {
		return this._directiveStage.revision;
	}

	public getCacheFingerprintTag(): string {
		return this._directiveStage.getCacheFingerprintTag();
	}

	public compile(request: ShaderProcessRequest): ShaderBackendCompileResult {
		const stageA = this._directiveStage.process(this._toDirectiveRequest(request));
		return this._compileStageB(request, stageA);
	}

	public async compileAsync(
		request: ShaderProcessRequest
	): Promise<ShaderBackendCompileResult> {
		const stageA = await this._directiveStage.processAsync(
			this._toDirectiveRequest(request)
		);
		return this._compileStageB(request, stageA);
	}

	private _compileStageB(
		request: ShaderProcessRequest,
		stageA: ShaderDirectiveStageResult
	): ShaderBackendCompileResult {
		this._throwOnDirectiveErrorsIfStrict(stageA, request);
		const stageB = this._runtime.process({
			code: stageA.code,
			sourceMap: stageA.sourceMap,
			language: request.language,
			stage: request.stage,
			entryPoint: request.entryPoint,
			label: request.label,
			sourceKind: request.sourceKind,
			sourceHash: request.sourceHash,
			diagnosticFilter: request.diagnosticFilter,
			enableDirectives: false,
			directiveSourcePath: request.directiveSourcePath,
		});
		const diagnostics = [
			...cloneDiagnostics(stageA.diagnostics),
			...cloneDiagnostics(stageB.diagnostics),
		];
		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
		return {
			code: stageB.code,
			sourceMap: cloneSourceMap(stageB.sourceMap),
			composite: cloneCompositeSource(stageB.composite),
			diagnostics,
			hasErrors,
			fromCache: stageB.fromCache,
			directiveFingerprint: stageA.directiveFingerprint,
			directiveDiagnostics: cloneDiagnostics(stageA.diagnostics),
			backendDiagnostics: cloneDiagnostics(stageB.diagnostics),
		};
	}

	private _throwOnDirectiveErrorsIfStrict(
		stageA: ShaderDirectiveStageResult,
		request: ShaderProcessRequest
	): void {
		if (!stageA.hasErrors || this._mode !== "strict") {
			return;
		}
		const sourceLabel =
			typeof request.label === "string" && request.label.length > 0 ?
				request.label
			:	"unnamed";
		const lines = stageA.diagnostics
			.filter((diagnostic) => diagnostic.severity === "error")
			.slice(0, 16)
			.map((diagnostic) => `${diagnostic.code}: ${diagnostic.message}`);
		const details = lines.length > 0 ? ` ${lines.join(" | ")}` : "";
		throw new Error(
			`Shader directive stage failed in strict mode [${sourceLabel}].${details}` +
				" Migration hint: move directives to backend compile stage hooks with stable tokens."
		);
	}

	private _toDirectiveRequest(
		request: ShaderProcessRequest
	): ShaderDirectiveStageRequest {
		return {
			code: request.code,
			sourceMap: request.sourceMap ?? null,
			language: request.language,
			stage: request.stage,
			entryPoint: request.entryPoint,
			label: request.label,
			sourceKind: request.sourceKind,
			directiveSourcePath: request.directiveSourcePath,
		};
	}
}
