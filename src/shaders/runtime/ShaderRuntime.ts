import {
	createInlineCompositeShaderSource,
	mapShaderGeneratedLocation,
	SOURCE_MAP_SCHEMA_VERSION,
} from "./sourceMap";
import {
	SHADER_RUNTIME_DEFAULT_CACHE_LIMIT,
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	resolveDefaultShaderRuntimeMode,
} from "./constants";
import { createBuiltInShaderRules } from "./builtins";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticFilter,
	ShaderDiagnosticRange,
	ShaderDirectivePreprocessResult,
	ShaderInjectionScript,
	ShaderLanguage,
	ShaderProcessRequest,
	ShaderProcessResult,
	ShaderResolvedInjectionAnchors,
	ShaderRule,
	ShaderRuleContext,
	ShaderRuleInjection,
	ShaderRuleReplaceOutput,
	ShaderRuleReplacePatch,
	ShaderRuleReplaceResolved,
	ShaderRuleReplaceResult,
	ShaderRuleTransformResolved,
	ShaderRuleTransformResult,
	ShaderRuntimeCacheKind,
	ShaderRuntimeCacheStats,
	ShaderRuntimeCacheStatsSnapshot,
	ShaderRuntimeChangeAction,
	ShaderRuntimeChangeEvent,
	ShaderRuntimeMode,
	ShaderSourceSegment,
	ShaderSourceSegmentMap,
} from "./types";
import {
	DirectivePreprocessor,
	type PreprocessResult,
	type RegisteredIncludeModule,
} from "./DirectivePreprocessor";
import {
	injectGLSLSource,
	injectWGSLSource,
	normalizeInjectionAnchorForLanguage,
	type InjectionBlock,
} from "./ShaderSourceInjection";
import {
	canonicalizeModulePath,
	cloneCompositeSource,
	cloneDiagnostics,
	cloneSourceMap,
	createPointRange,
	isPromiseLike,
	normalizeInjectionBlock,
	normalizeInjectionScript,
	normalizeLanguage,
	normalizeSourceKind,
	normalizeStage,
	normalizeSymbols,
} from "./runtimeShared";

interface ShaderRuntimeOptions {
	mode?: ShaderRuntimeMode;
	cacheLimit?: number;
	strictErrorMaxDiagnostics?: number;
}

interface CachedShaderProcessResult {
	result: ShaderProcessResult;
	participatingRuleIds: string[];
}

interface InternalCacheStats {
	hits: number;
	misses: number;
	evictions: number;
	invalidations: number;
}

interface ProcessPreparation {
	context: ShaderRuleContext;
	sourcePath: string;
	baseComposite: CompositeShaderSource;
	preprocessedDiagnostics: ShaderDiagnostic[];
	matchedRules: ShaderRule[];
	matchedRuleIds: string[];
	cacheKey: string;
	sourceMap: ShaderSourceSegmentMap | null | undefined;
}

interface RewritePreparation {
	composite: CompositeShaderSource;
	context: ShaderRuleContext;
	diagnostics: ShaderDiagnostic[];
}

type ShaderRuntimeChangeListener =
	| ((event: ShaderRuntimeChangeEvent) => void)
	| (() => void);

const DEFAULT_STRICT_ERROR_MAX_DIAGNOSTICS = 32;
const IS_DEV_ENVIRONMENT = resolveIsDevelopmentEnvironment();

function resolveIsDevelopmentEnvironment(): boolean {
	const meta = import.meta as ImportMeta & {
		env?: {
			DEV?: boolean;
		};
	};
	if (typeof meta.env?.DEV === "boolean") {
		return meta.env.DEV;
	}

	const nodeEnv =
		(
			globalThis as {
				process?: {
					env?: Record<string, unknown>;
				};
			}
		).process?.env?.NODE_ENV ?? null;
	if (typeof nodeEnv === "string") {
		return nodeEnv.toLowerCase() !== "production";
	}
	return true;
}

function createEmptyCacheStats(): InternalCacheStats {
	return {
		hits: 0,
		misses: 0,
		evictions: 0,
		invalidations: 0,
	};
}

export class ShaderRuntime {
	private _mode: ShaderRuntimeMode;
	private _cacheLimit: number;
	private _strictErrorMaxDiagnostics: number;
	private _revision: number;
	private _builtInRules: Map<string, ShaderRule>;
	private _userRules: Map<string, ShaderRule>;
	private _ruleExecutionOrderCache: ShaderRule[] | null;
	private _builtInSymbols: Set<string>;
	private _syncProcessCache: Map<string, CachedShaderProcessResult>;
	private _asyncProcessCache: Map<string, CachedShaderProcessResult>;
	private _asyncInFlight: Map<string, Promise<ShaderProcessResult>>;
	private _syncCacheStats: InternalCacheStats;
	private _asyncCacheStats: InternalCacheStats;
	private _listeners: Set<ShaderRuntimeChangeListener>;
	private _diagnosticFilters: Set<ShaderDiagnosticFilter>;
	private _includeModulesByLanguage: Map<ShaderLanguage, Map<string, RegisteredIncludeModule>>;
	private _injectionScripts: Map<string, ShaderInjectionScript>;
	private _directiveRegistryRevision: number;
	private _directivePreprocessor: DirectivePreprocessor;

	constructor(options: ShaderRuntimeOptions = {}) {
		this._mode = options.mode ?? resolveDefaultShaderRuntimeMode();
		this._cacheLimit = Math.max(
			1,
			Math.floor(options.cacheLimit ?? SHADER_RUNTIME_DEFAULT_CACHE_LIMIT),
		);
		this._strictErrorMaxDiagnostics = Math.max(
			1,
			Math.floor(options.strictErrorMaxDiagnostics ?? DEFAULT_STRICT_ERROR_MAX_DIAGNOSTICS),
		);
		this._revision = 1;
		this._builtInRules = new Map();
		this._userRules = new Map();
		this._ruleExecutionOrderCache = null;
		this._builtInSymbols = new Set();
		this._syncProcessCache = new Map();
		this._asyncProcessCache = new Map();
		this._asyncInFlight = new Map();
		this._syncCacheStats = createEmptyCacheStats();
		this._asyncCacheStats = createEmptyCacheStats();
		this._listeners = new Set();
		this._diagnosticFilters = new Set();
		this._includeModulesByLanguage = new Map([
			["wgsl", new Map()],
			["glsl", new Map()],
		]);
		this._injectionScripts = new Map();
		this._directiveRegistryRevision = 1;
		this._directivePreprocessor = new DirectivePreprocessor({
			mode: this._mode,
			includeModulesByLanguage: this._includeModulesByLanguage,
			injectionScripts: this._injectionScripts,
		});

		for (const rule of createBuiltInShaderRules()) {
			const normalized = this._normalizeRule(rule);
			this._builtInRules.set(normalized.id, normalized);
			for (const symbol of normalized.symbols ?? []) {
				this._builtInSymbols.add(symbol);
			}
		}
	}

	public get revision(): number {
		return this._revision;
	}

	public getMode(): ShaderRuntimeMode {
		return this._mode;
	}

	public setMode(mode: ShaderRuntimeMode): void {
		if (mode !== "strict" && mode !== "warn" && mode !== "silent") {
			throw new Error(`Unsupported ShaderRuntime mode "${String(mode)}".`);
		}
		if (this._mode === mode) {
			return;
		}
		this._mode = mode;
		this._directivePreprocessor.setMode(mode);
		this._applyMutation("mode", [], { invalidateAll: true });
	}

	public onDidChange(listener: ShaderRuntimeChangeListener): () => void {
		this._listeners.add(listener);
		return () => {
			this._listeners.delete(listener);
		};
	}

	public registerIncludeModule(
		language: ShaderLanguage,
		id: string,
		code: string,
		sourcePath?: string,
	): void {
		const normalizedLanguage = normalizeLanguage(language);
		const normalizedId = typeof id === "string" ? id.trim().replace(/\\/g, "/") : "";
		if (normalizedId.length <= 0) {
			throw new Error("Shader include module id must be a non-empty string.");
		}
		if (typeof code !== "string") {
			throw new Error("Shader include module code must be a string.");
		}
		const canonicalId = canonicalizeModulePath(normalizedId);
		const languageModules = this._includeModulesByLanguage.get(normalizedLanguage) ?? new Map();
		const action: ShaderRuntimeChangeAction = languageModules.has(canonicalId)
			? "update-include-module"
			: "register-include-module";
		languageModules.set(canonicalId, {
			id: normalizedId,
			canonicalId,
			code,
			sourcePath:
				typeof sourcePath === "string" && sourcePath.trim().length > 0
					? sourcePath.trim()
					: canonicalId,
		});
		this._includeModulesByLanguage.set(normalizedLanguage, languageModules);
		this._directiveRegistryRevision++;
		this._applyMutation(action, [], {
			invalidateAll: true,
			includeModuleIds: [formatIncludeModuleEventId(normalizedLanguage, canonicalId)],
		});
	}

	public unregisterIncludeModule(language: ShaderLanguage, id: string): boolean {
		const normalizedLanguage = normalizeLanguage(language);
		const normalizedId = typeof id === "string" ? id.trim().replace(/\\/g, "/") : "";
		if (normalizedId.length <= 0) {
			return false;
		}
		const canonicalId = canonicalizeModulePath(normalizedId);
		const languageModules = this._includeModulesByLanguage.get(normalizedLanguage) ?? null;
		if (!languageModules || !languageModules.has(canonicalId)) {
			return false;
		}
		languageModules.delete(canonicalId);
		this._directiveRegistryRevision++;
		this._applyMutation("unregister-include-module", [], {
			invalidateAll: true,
			includeModuleIds: [formatIncludeModuleEventId(normalizedLanguage, canonicalId)],
		});
		return true;
	}

	public clearIncludeModules(language?: ShaderLanguage): void {
		if (!language) {
			const ids: string[] = [];
			for (const [lang, modules] of this._includeModulesByLanguage) {
				for (const moduleId of modules.keys()) {
					ids.push(formatIncludeModuleEventId(lang, moduleId));
				}
			}
			if (ids.length <= 0) {
				return;
			}
			this._includeModulesByLanguage.set("wgsl", new Map());
			this._includeModulesByLanguage.set("glsl", new Map());
			this._directiveRegistryRevision++;
			this._applyMutation("clear-include-modules", [], {
				invalidateAll: true,
				includeModuleIds: ids,
			});
			return;
		}
		const normalizedLanguage = normalizeLanguage(language);
		const languageModules = this._includeModulesByLanguage.get(normalizedLanguage) ?? null;
		if (!languageModules || languageModules.size <= 0) {
			return;
		}
		const ids = [...languageModules.keys()].map((moduleId) =>
			formatIncludeModuleEventId(normalizedLanguage, moduleId),
		);
		languageModules.clear();
		this._directiveRegistryRevision++;
		this._applyMutation("clear-include-modules", [], {
			invalidateAll: true,
			includeModuleIds: ids,
		});
	}

	public registerInjectionScript(script: ShaderInjectionScript): void {
		const normalized = normalizeInjectionScript(script);
		const action: ShaderRuntimeChangeAction = this._injectionScripts.has(normalized.id)
			? "update-injection-script"
			: "register-injection-script";
		this._injectionScripts.set(normalized.id, normalized);
		this._directiveRegistryRevision++;
		this._applyMutation(action, [], {
			invalidateAll: true,
			injectionScriptIds: [normalized.id],
		});
	}

	public unregisterInjectionScript(id: string): boolean {
		const normalizedId = typeof id === "string" ? id.trim() : "";
		if (!this._injectionScripts.has(normalizedId)) {
			return false;
		}
		this._injectionScripts.delete(normalizedId);
		this._directiveRegistryRevision++;
		this._applyMutation("unregister-injection-script", [], {
			invalidateAll: true,
			injectionScriptIds: [normalizedId],
		});
		return true;
	}

	public clearInjectionScripts(): void {
		if (this._injectionScripts.size <= 0) {
			return;
		}
		const ids = [...this._injectionScripts.keys()];
		this._injectionScripts.clear();
		this._directiveRegistryRevision++;
		this._applyMutation("clear-injection-scripts", [], {
			invalidateAll: true,
			injectionScriptIds: ids,
		});
	}

	public registerRule(rule: ShaderRule): void {
		const normalized = this._normalizeRule(rule);
		if (normalized.id.startsWith(SHADER_RUNTIME_RESERVED_RULE_PREFIX)) {
			throw new Error(
				`ShaderRuntime user rules cannot use reserved prefix "${SHADER_RUNTIME_RESERVED_RULE_PREFIX}".`,
			);
		}
		const action: ShaderRuntimeChangeAction = this._userRules.has(normalized.id)
			? "update-rule"
			: "register-rule";
		const draftUserRules = new Map(this._userRules);
		draftUserRules.set(normalized.id, normalized);
		this._assertUserSymbolConflictForRule(normalized, draftUserRules);
		this._validateDependencyGraph(draftUserRules);
		this._userRules = draftUserRules;
		this._applyMutation(action, [normalized.id], {
			invalidateRuleIds: [normalized.id],
		});
	}

	public unregisterRule(ruleId: string): boolean {
		const normalizedRuleId = typeof ruleId === "string" ? ruleId.trim() : "";
		if (!this._userRules.has(normalizedRuleId)) {
			return false;
		}
		const dependents = this._findDependentUserRules(normalizedRuleId);
		if (dependents.length > 0) {
			throw new Error(
				`ShaderRuntime cannot unregister rule "${normalizedRuleId}" because it is required by ${dependents
					.map((id) => `"${id}"`)
					.join(", ")}.`,
			);
		}
		this._userRules.delete(normalizedRuleId);
		this._applyMutation("unregister-rule", [normalizedRuleId], {
			invalidateRuleIds: [normalizedRuleId],
		});
		return true;
	}

	public clearUserRules(): void {
		if (this._userRules.size <= 0) {
			return;
		}
		const removedRuleIds = [...this._userRules.keys()];
		this._userRules.clear();
		this._applyMutation("clear-rules", removedRuleIds, {
			invalidateRuleIds: removedRuleIds,
		});
	}

	public listRules(): ShaderRule[] {
		return this._collectRulesInExecutionOrder().map((rule) => cloneRule(rule));
	}

	public invalidateProcessCache(): number {
		const removed = this._invalidateAllProcessCaches();
		this._asyncInFlight.clear();
		if (removed > 0) {
			this._applyMutation("invalidate-cache", [], { invalidateAll: false });
		}
		return removed;
	}

	public getCacheStats(kind: ShaderRuntimeCacheKind): ShaderRuntimeCacheStats;
	public getCacheStats(): ShaderRuntimeCacheStatsSnapshot;
	public getCacheStats(
		kind?: ShaderRuntimeCacheKind,
	): ShaderRuntimeCacheStats | ShaderRuntimeCacheStatsSnapshot {
		if (kind) {
			return this._snapshotCacheStats(kind);
		}
		return {
			sync: this._snapshotCacheStats("sync"),
			async: this._snapshotCacheStats("async"),
		};
	}

	public resetCacheStats(kind?: ShaderRuntimeCacheKind): void {
		if (!kind || kind === "sync") {
			this._syncCacheStats = createEmptyCacheStats();
		}
		if (!kind || kind === "async") {
			this._asyncCacheStats = createEmptyCacheStats();
		}
	}

	public filterDiagnostics(predicate: ShaderDiagnosticFilter): () => void {
		if (typeof predicate !== "function") {
			throw new Error("ShaderRuntime diagnostic filter must be a function.");
		}
		this._diagnosticFilters.add(predicate);
		return () => {
			this._diagnosticFilters.delete(predicate);
		};
	}

	public resolveInjectionAnchors(request: ShaderProcessRequest): ShaderResolvedInjectionAnchors {
		return this._directivePreprocessor.resolveInjectionAnchors(request);
	}

	public preprocessDirectives(request: ShaderProcessRequest): ShaderDirectivePreprocessResult {
		const sourcePath = resolveShaderRequestSourcePath(request);
		const initialComposite = request.sourceMap
			? {
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			: createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const preprocessed = this._directivePreprocessor.preprocessSync(request, initialComposite);
		return this._finalizeDirectivePreprocessResult(request, preprocessed);
	}

	public async preprocessDirectivesAsync(
		request: ShaderProcessRequest,
	): Promise<ShaderDirectivePreprocessResult> {
		const sourcePath = resolveShaderRequestSourcePath(request);
		const initialComposite = request.sourceMap
			? {
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			: createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const preprocessed = await this._directivePreprocessor.preprocessAsync(
			request,
			initialComposite,
		);
		return this._finalizeDirectivePreprocessResult(request, preprocessed);
	}

	public process(request: ShaderProcessRequest): ShaderProcessResult {
		const prepared = this._prepareProcessSync(request);
		const cached = this._getCachedResult("sync", prepared.cacheKey);
		if (cached) {
			return this._finalizeProcessResult(
				prepared.context,
				cached.result,
				request.diagnosticFilter,
				true,
			);
		}

		const rawResult = this._executeRulesSync(prepared);
		this._setCachedResult("sync", prepared.cacheKey, rawResult, prepared.matchedRuleIds);
		return this._finalizeProcessResult(
			prepared.context,
			rawResult,
			request.diagnosticFilter,
			false,
		);
	}

	public async processAsync(request: ShaderProcessRequest): Promise<ShaderProcessResult> {
		const prepared = await this._prepareProcessAsync(request);
		const cached = this._getCachedResult("async", prepared.cacheKey);
		if (cached) {
			return this._finalizeProcessResult(
				prepared.context,
				cached.result,
				request.diagnosticFilter,
				true,
			);
		}
		const inFlight = this._asyncInFlight.get(prepared.cacheKey);
		if (inFlight) {
			const shared = await inFlight;
			return this._finalizeProcessResult(
				prepared.context,
				shared,
				request.diagnosticFilter,
				true,
			);
		}

		const startedRevision = this._revision;
		let executionPromise: Promise<ShaderProcessResult>;
		executionPromise = this._executeRulesAsync(prepared)
			.then((rawResult) => {
				if (this._revision === startedRevision) {
					this._setCachedResult(
						"async",
						prepared.cacheKey,
						rawResult,
						prepared.matchedRuleIds,
					);
				}
				return rawResult;
			})
			.finally(() => {
				if (this._asyncInFlight.get(prepared.cacheKey) === executionPromise) {
					this._asyncInFlight.delete(prepared.cacheKey);
				}
			});
		this._asyncInFlight.set(prepared.cacheKey, executionPromise);

		const raw = await executionPromise;
		return this._finalizeProcessResult(prepared.context, raw, request.diagnosticFilter, false);
	}

	private _prepareProcessSync(request: ShaderProcessRequest): ProcessPreparation {
		const initialSourcePath = resolveShaderRequestSourcePath(request);
		const initialComposite = request.sourceMap
			? {
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			: createInlineCompositeShaderSource(request.code, initialSourcePath, "source");
		const preprocessed = this._directivePreprocessor.preprocessSync(request, initialComposite);
		const sourcePath =
			preprocessed.composite.sourceMap.segments[0]?.sourcePath ?? initialSourcePath;
		const context = this._buildRuleContext(request, preprocessed.composite.code);

		const matchedRules: ShaderRule[] = [];
		for (const rule of this._collectRulesInExecutionOrder()) {
			if (!rule.match) {
				matchedRules.push(rule);
				continue;
			}
			const matchResult = rule.match(context);
			if (isPromiseLike(matchResult)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from match() during process(). Use processAsync().`,
				);
			}
			if (matchResult) {
				matchedRules.push(rule);
			}
		}

		const sourceHash = this._resolveSourceHash(
			context.source,
			preprocessed.composite.code === request.code ? request.sourceHash : undefined,
		);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			preprocessed.composite.sourceMap,
			sourceHash,
			matchedRuleIds,
			this._directiveRegistryRevision,
		);

		return {
			context,
			sourcePath,
			baseComposite: preprocessed.composite,
			preprocessedDiagnostics: preprocessed.diagnostics,
			matchedRules,
			matchedRuleIds,
			cacheKey,
			sourceMap: preprocessed.composite.sourceMap,
		};
	}

	private async _prepareProcessAsync(request: ShaderProcessRequest): Promise<ProcessPreparation> {
		const initialSourcePath = resolveShaderRequestSourcePath(request);
		const initialComposite = request.sourceMap
			? {
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			: createInlineCompositeShaderSource(request.code, initialSourcePath, "source");
		const preprocessed = await this._directivePreprocessor.preprocessAsync(
			request,
			initialComposite,
		);
		const sourcePath =
			preprocessed.composite.sourceMap.segments[0]?.sourcePath ?? initialSourcePath;
		const context = this._buildRuleContext(request, preprocessed.composite.code);

		const matchedRules: ShaderRule[] = [];
		for (const rule of this._collectRulesInExecutionOrder()) {
			if (!rule.match) {
				matchedRules.push(rule);
				continue;
			}
			const matchResult = await rule.match(context);
			if (matchResult) {
				matchedRules.push(rule);
			}
		}

		const sourceHash = this._resolveSourceHash(
			context.source,
			preprocessed.composite.code === request.code ? request.sourceHash : undefined,
		);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			preprocessed.composite.sourceMap,
			sourceHash,
			matchedRuleIds,
			this._directiveRegistryRevision,
		);

		return {
			context,
			sourcePath,
			baseComposite: preprocessed.composite,
			preprocessedDiagnostics: preprocessed.diagnostics,
			matchedRules,
			matchedRuleIds,
			cacheKey,
			sourceMap: preprocessed.composite.sourceMap,
		};
	}

	private _buildRuleContext(
		request: ShaderProcessRequest,
		source: string = request.code,
	): ShaderRuleContext {
		return {
			mode: this._mode,
			language: normalizeLanguage(request.language),
			stage: normalizeStage(request.stage),
			entryPoint: request.entryPoint ?? null,
			label: request.label ?? null,
			sourceKind: normalizeSourceKind(request.sourceKind),
			source,
		};
	}

	private _resolveSourceHash(source: string, sourceHash: string | undefined): string {
		const providedHash = typeof sourceHash === "string" ? sourceHash.trim() : "";
		if (providedHash.length <= 0) {
			return hashSourceCode(source);
		}
		if (IS_DEV_ENVIRONMENT) {
			const computedHash = hashSourceCode(source);
			if (providedHash !== computedHash) {
				throw new Error(
					`ShaderRuntime sourceHash mismatch. Provided "${providedHash}" but computed "${computedHash}".`,
				);
			}
		}
		return providedHash;
	}

	private _executeRulesSync(prepared: ProcessPreparation): ShaderProcessResult {
		const rewrite = this._applyRuleRewritesSync(prepared);
		const diagnostics: ShaderDiagnostic[] = [...rewrite.diagnostics];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const dynamicUserSymbols = new Map<string, string>();

		for (const rule of prepared.matchedRules) {
			if (rule.validate) {
				const validateResult = rule.validate(rewrite.context);
				if (isPromiseLike(validateResult)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" returned a Promise from validate() during process(). Use processAsync().`,
					);
				}
				const diagnosticsFromRule = Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
					diagnostics.push(
						this._normalizeDiagnostic(
							{ ...diagnostic, ruleId: rule.id },
							rewrite.composite.sourceMap,
							prepared.sourcePath,
						),
					);
				}
			}

			if (!rule.inject) {
				continue;
			}
			const injection = rule.inject(rewrite.context);
			if (isPromiseLike(injection)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from inject() during process(). Use processAsync().`,
				);
			}
			this._applyInjectionIfAny(
				rewrite.context,
				rewrite.composite.sourceMap,
				prepared.sourcePath,
				rule,
				injection,
				diagnostics,
				headers,
				functions,
				dynamicUserSymbols,
			);
		}

		return this._buildRawProcessResult(
			prepared,
			rewrite.composite,
			rewrite.context,
			diagnostics,
			headers,
			functions,
		);
	}

	private async _executeRulesAsync(prepared: ProcessPreparation): Promise<ShaderProcessResult> {
		const rewrite = await this._applyRuleRewritesAsync(prepared);
		const diagnostics: ShaderDiagnostic[] = [...rewrite.diagnostics];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const dynamicUserSymbols = new Map<string, string>();

		for (const rule of prepared.matchedRules) {
			if (rule.validate) {
				const validateResult = await rule.validate(rewrite.context);
				const diagnosticsFromRule = Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
					diagnostics.push(
						this._normalizeDiagnostic(
							{ ...diagnostic, ruleId: rule.id },
							rewrite.composite.sourceMap,
							prepared.sourcePath,
						),
					);
				}
			}

			if (!rule.inject) {
				continue;
			}
			const injection = await rule.inject(rewrite.context);
			this._applyInjectionIfAny(
				rewrite.context,
				rewrite.composite.sourceMap,
				prepared.sourcePath,
				rule,
				injection,
				diagnostics,
				headers,
				functions,
				dynamicUserSymbols,
			);
		}

		return this._buildRawProcessResult(
			prepared,
			rewrite.composite,
			rewrite.context,
			diagnostics,
			headers,
			functions,
		);
	}

	private _applyRuleRewritesSync(prepared: ProcessPreparation): RewritePreparation {
		let composite = cloneCompositeSource(prepared.baseComposite);
		let context = {
			...prepared.context,
			source: composite.code,
		};
		const diagnostics: ShaderDiagnostic[] = [];
		for (const rule of prepared.matchedRules) {
			if (rule.transform) {
				let transformResult: ShaderRuleTransformResult;
				try {
					transformResult = rule.transform(context);
				} catch (error) {
					throw this._createRuleHookError(rule.id, "transform", error);
				}
				if (isPromiseLike(transformResult)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" returned a Promise from transform() during process(). Use processAsync().`,
					);
				}
				const applied = this._applyRuleTransformResult(
					rule.id,
					transformResult as ShaderRuleTransformResolved,
					composite,
				);
				composite = applied.composite;
				context = {
					...context,
					source: composite.code,
				};
				this._appendRuleDiagnostics(
					diagnostics,
					rule.id,
					applied.diagnostics,
					composite.sourceMap,
					prepared.sourcePath,
				);
			}
			if (!rule.replace) {
				continue;
			}
			let replaceResult: ShaderRuleReplaceResult;
			try {
				replaceResult = rule.replace(context);
			} catch (error) {
				throw this._createRuleHookError(rule.id, "replace", error);
			}
			if (isPromiseLike(replaceResult)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from replace() during process(). Use processAsync().`,
				);
			}
			const applied = this._applyRuleReplaceResult(
				rule.id,
				replaceResult as ShaderRuleReplaceResolved,
				composite,
			);
			composite = applied.composite;
			context = {
				...context,
				source: composite.code,
			};
			this._appendRuleDiagnostics(
				diagnostics,
				rule.id,
				applied.diagnostics,
				composite.sourceMap,
				prepared.sourcePath,
			);
		}
		return {
			composite,
			context,
			diagnostics,
		};
	}

	private async _applyRuleRewritesAsync(
		prepared: ProcessPreparation,
	): Promise<RewritePreparation> {
		let composite = cloneCompositeSource(prepared.baseComposite);
		let context = {
			...prepared.context,
			source: composite.code,
		};
		const diagnostics: ShaderDiagnostic[] = [];
		for (const rule of prepared.matchedRules) {
			if (rule.transform) {
				let transformResult: ShaderRuleTransformResolved;
				try {
					transformResult = await rule.transform(context);
				} catch (error) {
					throw this._createRuleHookError(rule.id, "transform", error);
				}
				const applied = this._applyRuleTransformResult(rule.id, transformResult, composite);
				composite = applied.composite;
				context = {
					...context,
					source: composite.code,
				};
				this._appendRuleDiagnostics(
					diagnostics,
					rule.id,
					applied.diagnostics,
					composite.sourceMap,
					prepared.sourcePath,
				);
			}
			if (!rule.replace) {
				continue;
			}
			let replaceResult: ShaderRuleReplaceResult;
			try {
				replaceResult = await rule.replace(context);
			} catch (error) {
				throw this._createRuleHookError(rule.id, "replace", error);
			}
			const applied = this._applyRuleReplaceResult(
				rule.id,
				replaceResult as ShaderRuleReplaceResolved,
				composite,
			);
			composite = applied.composite;
			context = {
				...context,
				source: composite.code,
			};
			this._appendRuleDiagnostics(
				diagnostics,
				rule.id,
				applied.diagnostics,
				composite.sourceMap,
				prepared.sourcePath,
			);
		}
		return {
			composite,
			context,
			diagnostics,
		};
	}

	private _applyRuleTransformResult(
		ruleId: string,
		result: ShaderRuleTransformResolved,
		previousComposite: CompositeShaderSource,
	): { composite: CompositeShaderSource; diagnostics: ShaderDiagnostic[] } {
		if (!result) {
			return {
				composite: previousComposite,
				diagnostics: [],
			};
		}
		const normalized =
			typeof result === "string"
				? {
						code: result,
						sourceMap: undefined,
						diagnostics: [],
					}
				: {
						code: result.code,
						sourceMap: result.sourceMap,
						diagnostics: Array.isArray(result.diagnostics) ? result.diagnostics : [],
					};
		if (typeof normalized.code !== "string") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" transform() must return a string or { code } object.`,
			);
		}
		const composite = normalized.sourceMap
			? {
					code: normalized.code,
					sourceMap: cloneSourceMap(normalized.sourceMap),
				}
			: createGeneratedCompositeWithColumnSpans(
					normalized.code,
					`<runtime:${ruleId}:transform>`,
					`${ruleId}:transform`,
				);
		return {
			composite,
			diagnostics: normalized.diagnostics,
		};
	}

	private _applyRuleReplaceResult(
		ruleId: string,
		result: ShaderRuleReplaceResolved,
		previousComposite: CompositeShaderSource,
	): { composite: CompositeShaderSource; diagnostics: ShaderDiagnostic[] } {
		if (!result) {
			return {
				composite: previousComposite,
				diagnostics: [],
			};
		}
		const diagnostics = Array.isArray(result)
			? []
			: Array.isArray((result as ShaderRuleReplaceOutput).diagnostics)
				? (result as ShaderRuleReplaceOutput).diagnostics!
				: [];
		const patchesRaw = Array.isArray(result)
			? result
			: (result as ShaderRuleReplaceOutput).patches;
		if (!Array.isArray(patchesRaw)) {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace() must return patch list or { patches } object.`,
			);
		}
		const patches = patchesRaw.map((patch, index) =>
			this._normalizeReplacePatch(ruleId, index, patch),
		);
		if (patches.length <= 0) {
			return {
				composite: previousComposite,
				diagnostics,
			};
		}
		let code = previousComposite.code;
		for (let index = 0; index < patches.length; index++) {
			code = this._applyReplacePatch(code, patches[index]);
		}
		if (code === previousComposite.code) {
			return {
				composite: previousComposite,
				diagnostics,
			};
		}
		return {
			composite: createGeneratedCompositeWithColumnSpans(
				code,
				`<runtime:${ruleId}:replace>`,
				`${ruleId}:replace`,
			),
			diagnostics,
		};
	}

	private _normalizeReplacePatch(
		ruleId: string,
		index: number,
		patch: ShaderRuleReplacePatch,
	): ShaderRuleReplacePatch {
		if (!patch || typeof patch !== "object") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} must be an object.`,
			);
		}
		if (typeof patch.pattern !== "string" && !(patch.pattern instanceof RegExp)) {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} pattern must be string or RegExp.`,
			);
		}
		if (typeof patch.replacement !== "string") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} replacement must be a string.`,
			);
		}
		return {
			pattern: patch.pattern,
			replacement: patch.replacement,
			replaceAll: patch.replaceAll === true,
		};
	}

	private _applyReplacePatch(code: string, patch: ShaderRuleReplacePatch): string {
		if (patch.pattern instanceof RegExp) {
			let expression = patch.pattern;
			if (patch.replaceAll === true && !expression.flags.includes("g")) {
				expression = new RegExp(expression.source, `${expression.flags}g`);
			}
			if (patch.replaceAll !== true && expression.flags.includes("g")) {
				expression = new RegExp(expression.source, expression.flags.replace(/g/g, ""));
			}
			return code.replace(expression, patch.replacement);
		}
		if (patch.pattern.length <= 0) {
			return code;
		}
		if (patch.replaceAll === true) {
			return code.split(patch.pattern).join(patch.replacement);
		}
		const found = code.indexOf(patch.pattern);
		if (found < 0) {
			return code;
		}
		return code.slice(0, found) + patch.replacement + code.slice(found + patch.pattern.length);
	}

	private _appendRuleDiagnostics(
		target: ShaderDiagnostic[],
		ruleId: string,
		diagnostics: ShaderDiagnostic[],
		sourceMap: ShaderSourceSegmentMap | null | undefined,
		fallbackSourcePath: string,
	): void {
		for (const diagnostic of diagnostics) {
			target.push(
				this._normalizeDiagnostic({ ...diagnostic, ruleId }, sourceMap, fallbackSourcePath),
			);
		}
	}

	private _createRuleHookError(
		ruleId: string,
		hookKind: "transform" | "replace",
		error: unknown,
	): Error {
		const message =
			error instanceof Error
				? error.message
				: typeof error === "string"
					? error
					: "Unknown hook failure.";
		return new Error(`ShaderRuntime rule "${ruleId}" ${hookKind} hook failed: ${message}`);
	}

	private _applyInjectionIfAny(
		context: ShaderRuleContext,
		sourceMap: ShaderSourceSegmentMap | null | undefined,
		fallbackSourcePath: string,
		rule: ShaderRule,
		injection: ShaderRuleInjection | null | undefined,
		diagnostics: ShaderDiagnostic[],
		headers: InjectionBlock[],
		functions: InjectionBlock[],
		dynamicUserSymbols: Map<string, string>,
	): void {
		if (!injection) {
			return;
		}
		if (this._isUserRule(rule.id)) {
			const conflict = this._resolveInjectionSymbolConflict(
				rule,
				injection,
				dynamicUserSymbols,
			);
			if (conflict) {
				diagnostics.push(
					this._normalizeDiagnostic(conflict, sourceMap, fallbackSourcePath),
				);
				return;
			}
		}
		this._registerDynamicInjectionSymbols(rule, injection, dynamicUserSymbols);

		const header = normalizeInjectionBlock(injection.header);
		if (header.length > 0) {
			headers.push({
				code: header,
				sourcePath: `<runtime:${rule.id}:header>`,
				label: `${rule.id}:header`,
				anchor: normalizeInjectionAnchorForLanguage(
					context.language,
					injection.headerAnchor,
				),
			});
		}

		const functionBlock = normalizeInjectionBlock(injection.functions);
		if (functionBlock.length > 0) {
			functions.push({
				code: functionBlock,
				sourcePath: `<runtime:${rule.id}:functions>`,
				label: `${rule.id}:functions`,
				anchor: normalizeInjectionAnchorForLanguage(
					context.language,
					injection.functionsAnchor,
				),
			});
		}
	}

	private _buildRawProcessResult(
		prepared: ProcessPreparation,
		rewriteComposite: CompositeShaderSource,
		context: ShaderRuleContext,
		diagnostics: ShaderDiagnostic[],
		headers: InjectionBlock[],
		functions: InjectionBlock[],
	): ShaderProcessResult {
		const composite =
			context.language === "wgsl"
				? injectWGSLSource(rewriteComposite, [...headers, ...functions])
				: injectGLSLSource(rewriteComposite, [...headers, ...functions]);
		const mergedDiagnostics = [...prepared.preprocessedDiagnostics, ...diagnostics];
		const hasErrors = mergedDiagnostics.some((diagnostic) => diagnostic.severity === "error");
		return {
			code: composite.code,
			sourceMap: composite.sourceMap,
			composite,
			diagnostics: mergedDiagnostics,
			hasErrors,
			fromCache: false,
		};
	}

	private _finalizeProcessResult(
		context: ShaderRuleContext,
		rawResult: ShaderProcessResult,
		perCallFilter: ShaderDiagnosticFilter | undefined,
		fromCache: boolean,
	): ShaderProcessResult {
		const diagnostics = this._filterDiagnostics(rawResult.diagnostics, perCallFilter);
		const hasErrors = diagnostics.some((diagnostic) => diagnostic.severity === "error");
		if (hasErrors && this._mode === "strict") {
			throw buildStrictModeError(context, diagnostics, this._strictErrorMaxDiagnostics);
		}
		return {
			code: rawResult.code,
			sourceMap: cloneSourceMap(rawResult.sourceMap),
			composite: cloneCompositeSource(rawResult.composite),
			diagnostics: cloneDiagnostics(diagnostics),
			hasErrors,
			fromCache,
		};
	}

	private _finalizeDirectivePreprocessResult(
		request: ShaderProcessRequest,
		preprocessed: PreprocessResult,
	): ShaderDirectivePreprocessResult {
		const diagnostics = this._filterDiagnostics(
			preprocessed.diagnostics,
			request.diagnosticFilter,
		);
		const hasErrors = diagnostics.some((diagnostic) => diagnostic.severity === "error");
		return {
			code: preprocessed.composite.code,
			sourceMap: cloneSourceMap(preprocessed.composite.sourceMap),
			composite: cloneCompositeSource(preprocessed.composite),
			diagnostics: cloneDiagnostics(diagnostics),
			hasErrors,
		};
	}

	private _filterDiagnostics(
		diagnostics: ShaderDiagnostic[],
		perCallFilter: ShaderDiagnosticFilter | undefined,
	): ShaderDiagnostic[] {
		if (this._mode === "silent") {
			return [];
		}
		const globalFilters = [...this._diagnosticFilters];
		const filtered: ShaderDiagnostic[] = [];
		for (const diagnostic of diagnostics) {
			let keep = true;
			for (const filter of globalFilters) {
				try {
					if (!filter(diagnostic)) {
						keep = false;
						break;
					}
				} catch (error) {
					// Ignore filter failures to avoid breaking shader compilation flow.
				}
			}
			if (!keep) {
				continue;
			}
			if (perCallFilter) {
				try {
					if (!perCallFilter(diagnostic)) {
						continue;
					}
				} catch (error) {
					// Ignore filter failures to avoid breaking shader compilation flow.
				}
			}
			filtered.push(diagnostic);
		}
		return filtered;
	}

	private _normalizeDiagnostic(
		diagnostic: ShaderDiagnostic,
		sourceMap: ShaderSourceSegmentMap | null | undefined,
		fallbackSourcePath: string,
	): ShaderDiagnostic {
		const generatedRange = normalizeDiagnosticRange(diagnostic.range) ?? null;
		const generatedLine =
			normalizePositiveInteger(diagnostic.line) ?? generatedRange?.start.line ?? null;
		const generatedColumn =
			normalizePositiveInteger(diagnostic.column) ?? generatedRange?.start.column ?? 1;
		let resolvedLine = generatedLine ?? undefined;
		let resolvedColumn = generatedLine ? generatedColumn : undefined;
		let resolvedSourcePath =
			typeof diagnostic.sourcePath === "string" && diagnostic.sourcePath.length > 0
				? diagnostic.sourcePath
				: undefined;
		let resolvedRange =
			generatedRange ??
			(generatedLine ? createPointRange(generatedLine, generatedColumn) : undefined);

		if (generatedLine && sourceMap) {
			const mappedStart = mapShaderGeneratedLocation(
				sourceMap,
				generatedLine,
				generatedColumn,
			);
			if (mappedStart) {
				resolvedLine = mappedStart.sourceLine;
				resolvedColumn = mappedStart.sourceColumn;
				resolvedSourcePath = resolvedSourcePath ?? mappedStart.sourcePath;
			}
		}

		if (resolvedRange && sourceMap) {
			const mappedStart = mapShaderGeneratedLocation(
				sourceMap,
				resolvedRange.start.line,
				resolvedRange.start.column,
			);
			const mappedEnd = mapShaderGeneratedLocation(
				sourceMap,
				resolvedRange.end.line,
				resolvedRange.end.column,
			);
			if (mappedStart && mappedEnd) {
				resolvedRange = {
					start: {
						line: mappedStart.sourceLine,
						column: mappedStart.sourceColumn,
					},
					end: {
						line: mappedEnd.sourceLine,
						column: mappedEnd.sourceColumn,
					},
				};
				resolvedSourcePath = resolvedSourcePath ?? mappedStart.sourcePath;
			}
		}

		if (!resolvedSourcePath && sourceMap && sourceMap.segments.length > 0) {
			resolvedSourcePath = sourceMap.segments[0].sourcePath;
		}
		resolvedSourcePath = resolvedSourcePath ?? fallbackSourcePath;

		if (!resolvedRange && typeof resolvedLine === "number") {
			resolvedRange = createPointRange(resolvedLine, resolvedColumn ?? 1);
		}

		return {
			...diagnostic,
			line: resolvedLine,
			column: resolvedColumn,
			sourcePath: resolvedSourcePath,
			range: resolvedRange,
		};
	}

	private _normalizeRule(rule: ShaderRule): ShaderRule {
		if (!rule || typeof rule !== "object") {
			throw new Error("ShaderRuntime rule must be an object.");
		}
		const id = typeof rule.id === "string" ? rule.id.trim() : "";
		if (id.length <= 0) {
			throw new Error("ShaderRuntime rule id must be a non-empty string.");
		}
		const priority =
			typeof rule.priority === "number" && Number.isFinite(rule.priority)
				? Math.floor(rule.priority)
				: 0;
		if (rule.match !== undefined && typeof rule.match !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" match must be a function.`);
		}
		if (rule.transform !== undefined && typeof rule.transform !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" transform must be a function.`);
		}
		if (rule.replace !== undefined && typeof rule.replace !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" replace must be a function.`);
		}
		if (rule.validate !== undefined && typeof rule.validate !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" validate must be a function.`);
		}
		if (rule.inject !== undefined && typeof rule.inject !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" inject must be a function.`);
		}
		if (rule.symbols !== undefined && !Array.isArray(rule.symbols)) {
			throw new Error(`ShaderRuntime rule "${id}" symbols must be a string array.`);
		}
		if (rule.dependsOn !== undefined && !Array.isArray(rule.dependsOn)) {
			throw new Error(`ShaderRuntime rule "${id}" dependsOn must be a string array.`);
		}
		const description =
			typeof rule.description === "string" ? rule.description.trim() : undefined;
		const dependsOn = normalizeDependsOn(rule.dependsOn);
		if (dependsOn.includes(id)) {
			throw new Error(`ShaderRuntime rule "${id}" cannot depend on itself in dependsOn.`);
		}
		return {
			...rule,
			id,
			description: description && description.length > 0 ? description : undefined,
			priority,
			symbols: normalizeSymbols(rule.symbols),
			dependsOn,
		};
	}

	private _collectRulesInExecutionOrder(): ShaderRule[] {
		if (this._ruleExecutionOrderCache) {
			return this._ruleExecutionOrderCache;
		}
		this._ruleExecutionOrderCache = this._computeRuleExecutionOrder(this._userRules);
		return this._ruleExecutionOrderCache;
	}

	private _buildProcessCacheKey(
		context: ShaderRuleContext,
		sourceMap: ShaderSourceSegmentMap | null | undefined,
		sourceHash: string,
		matchedRuleIds: readonly string[],
		directiveRevision: number,
	): string {
		const ruleFingerprint = hashStringFNV1a(matchedRuleIds.join("|"));
		return [
			`mode:${this._mode}`,
			`lang:${context.language}`,
			`stage:${context.stage}`,
			`entry:${context.entryPoint ?? ""}`,
			`label:${hashStringFNV1a(context.label ?? "")}`,
			`kind:${context.sourceKind}`,
			`code:${hashStringFNV1a(sourceHash)}`,
			`sourceMap:${hashSourceMap(sourceMap)}`,
			`rules:${ruleFingerprint}`,
			`directives:${directiveRevision}`,
		].join("|");
	}

	private _getCachedResult(
		kind: ShaderRuntimeCacheKind,
		key: string,
	): CachedShaderProcessResult | null {
		const cache = this._getCacheMap(kind);
		const stats = this._getCacheStats(kind);
		const entry = cache.get(key);
		if (!entry) {
			stats.misses++;
			return null;
		}
		stats.hits++;
		cache.delete(key);
		cache.set(key, entry);
		return entry;
	}

	private _setCachedResult(
		kind: ShaderRuntimeCacheKind,
		key: string,
		result: ShaderProcessResult,
		participatingRuleIds: readonly string[],
	): void {
		const cache = this._getCacheMap(kind);
		const stats = this._getCacheStats(kind);
		cache.set(key, {
			result: cloneProcessResult(result, false),
			participatingRuleIds: [...new Set(participatingRuleIds)],
		});
		while (cache.size > this._cacheLimit) {
			const oldestKey = cache.keys().next().value;
			if (typeof oldestKey !== "string") {
				break;
			}
			cache.delete(oldestKey);
			stats.evictions++;
		}
	}

	private _isUserRule(ruleId: string): boolean {
		return !ruleId.startsWith(SHADER_RUNTIME_RESERVED_RULE_PREFIX);
	}

	private _resolveInjectionSymbolConflict(
		rule: ShaderRule,
		injection: ShaderRuleInjection,
		dynamicUserSymbols: Map<string, string>,
	): ShaderDiagnostic | null {
		const symbols = this._collectRuleSymbols(rule, injection);
		if (symbols.length <= 0) {
			return null;
		}
		for (const symbol of symbols) {
			if (this._builtInSymbols.has(symbol)) {
				return {
					ruleId: rule.id,
					code: "reserved-symbol-conflict",
					severity: this._mode === "strict" ? "error" : "warning",
					message: `Rule "${rule.id}" conflicts with reserved symbol "${symbol}" and was skipped.`,
				};
			}
			const staticOwner = this._findStaticUserSymbolOwner(symbol, rule.id);
			if (staticOwner) {
				return {
					ruleId: rule.id,
					code: "user-symbol-conflict",
					severity: this._mode === "strict" ? "error" : "warning",
					message: `Rule "${rule.id}" conflicts with user rule "${staticOwner}" on symbol "${symbol}" and was skipped.`,
				};
			}
			const dynamicOwner = dynamicUserSymbols.get(symbol);
			if (dynamicOwner && dynamicOwner !== rule.id) {
				return {
					ruleId: rule.id,
					code: "user-symbol-conflict",
					severity: this._mode === "strict" ? "error" : "warning",
					message: `Rule "${rule.id}" conflicts with injected symbol "${symbol}" from "${dynamicOwner}" and was skipped.`,
				};
			}
		}
		return null;
	}

	private _registerDynamicInjectionSymbols(
		rule: ShaderRule,
		injection: ShaderRuleInjection,
		dynamicUserSymbols: Map<string, string>,
	): void {
		if (!this._isUserRule(rule.id)) {
			return;
		}
		for (const symbol of normalizeSymbols(injection.symbols)) {
			if (!dynamicUserSymbols.has(symbol)) {
				dynamicUserSymbols.set(symbol, rule.id);
			}
		}
	}

	private _findStaticUserSymbolOwner(symbol: string, excludeRuleId: string): string | null {
		for (const rule of this._userRules.values()) {
			if (rule.id === excludeRuleId) {
				continue;
			}
			if (normalizeSymbols(rule.symbols).includes(symbol)) {
				return rule.id;
			}
		}
		return null;
	}

	private _collectRuleSymbols(rule: ShaderRule, injection: ShaderRuleInjection): string[] {
		return normalizeSymbols([...(rule.symbols ?? []), ...(injection.symbols ?? [])]);
	}

	private _computeRuleExecutionOrder(userRules: Map<string, ShaderRule>): ShaderRule[] {
		const mergedRules = new Map<string, ShaderRule>();
		for (const rule of this._builtInRules.values()) {
			mergedRules.set(rule.id, rule);
		}
		for (const rule of userRules.values()) {
			mergedRules.set(rule.id, rule);
		}

		const adjacency = new Map<string, string[]>();
		const indegree = new Map<string, number>();
		for (const id of mergedRules.keys()) {
			adjacency.set(id, []);
			indegree.set(id, 0);
		}
		for (const rule of mergedRules.values()) {
			for (const dependencyId of rule.dependsOn ?? []) {
				if (!mergedRules.has(dependencyId)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" depends on missing rule "${dependencyId}".`,
					);
				}
				adjacency.get(dependencyId)?.push(rule.id);
				indegree.set(rule.id, (indegree.get(rule.id) ?? 0) + 1);
			}
		}

		const compareRuleIds = (leftId: string, rightId: string): number => {
			const leftRule = mergedRules.get(leftId);
			const rightRule = mergedRules.get(rightId);
			const leftPriority = leftRule?.priority ?? 0;
			const rightPriority = rightRule?.priority ?? 0;
			if (leftPriority !== rightPriority) {
				return rightPriority - leftPriority;
			}
			return leftId.localeCompare(rightId);
		};

		const ready: string[] = [];
		for (const [id, value] of indegree) {
			if (value <= 0) {
				ready.push(id);
			}
		}
		const ordered: ShaderRule[] = [];
		while (ready.length > 0) {
			ready.sort(compareRuleIds);
			const nextId = ready.shift();
			if (!nextId) {
				break;
			}
			const rule = mergedRules.get(nextId);
			if (rule) {
				ordered.push(rule);
			}
			for (const dependentId of adjacency.get(nextId) ?? []) {
				const nextIndegree = (indegree.get(dependentId) ?? 0) - 1;
				indegree.set(dependentId, nextIndegree);
				if (nextIndegree === 0) {
					ready.push(dependentId);
				}
			}
		}
		if (ordered.length !== mergedRules.size) {
			const unresolved = [...indegree.entries()]
				.filter((entry) => entry[1] > 0)
				.map((entry) => entry[0]);
			throw new Error(
				`ShaderRuntime rule dependency cycle detected: ${unresolved.join(" -> ")}`,
			);
		}
		return ordered;
	}

	private _validateDependencyGraph(userRules: Map<string, ShaderRule>): void {
		this._computeRuleExecutionOrder(userRules);
	}

	private _findDependentUserRules(ruleId: string): string[] {
		const dependents: string[] = [];
		for (const rule of this._userRules.values()) {
			if ((rule.dependsOn ?? []).includes(ruleId)) {
				dependents.push(rule.id);
			}
		}
		dependents.sort((left, right) => left.localeCompare(right));
		return dependents;
	}

	private _assertUserSymbolConflictForRule(
		rule: ShaderRule,
		userRules: Map<string, ShaderRule>,
	): void {
		const ruleSymbols = normalizeSymbols(rule.symbols);
		for (const symbol of ruleSymbols) {
			for (const [otherRuleId, otherRule] of userRules) {
				if (otherRuleId === rule.id) {
					continue;
				}
				if (normalizeSymbols(otherRule.symbols).includes(symbol)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" conflicts with rule "${otherRuleId}" on symbol "${symbol}".`,
					);
				}
			}
		}
	}

	private _snapshotCacheStats(kind: ShaderRuntimeCacheKind): ShaderRuntimeCacheStats {
		const stats = kind === "sync" ? this._syncCacheStats : this._asyncCacheStats;
		const cache = kind === "sync" ? this._syncProcessCache : this._asyncProcessCache;
		return {
			hits: stats.hits,
			misses: stats.misses,
			evictions: stats.evictions,
			invalidations: stats.invalidations,
			size: cache.size,
			limit: this._cacheLimit,
		};
	}

	private _getCacheMap(kind: ShaderRuntimeCacheKind): Map<string, CachedShaderProcessResult> {
		return kind === "sync" ? this._syncProcessCache : this._asyncProcessCache;
	}

	private _getCacheStats(kind: ShaderRuntimeCacheKind): InternalCacheStats {
		return kind === "sync" ? this._syncCacheStats : this._asyncCacheStats;
	}

	private _invalidateAllProcessCaches(): number {
		let removed = 0;
		removed += this._syncProcessCache.size;
		removed += this._asyncProcessCache.size;
		this._syncCacheStats.invalidations += this._syncProcessCache.size;
		this._asyncCacheStats.invalidations += this._asyncProcessCache.size;
		this._syncProcessCache.clear();
		this._asyncProcessCache.clear();
		return removed;
	}

	private _invalidateProcessCachesByRuleIds(ruleIds: readonly string[]): number {
		if (ruleIds.length <= 0) {
			return 0;
		}
		const targets = new Set(ruleIds);
		const removedSync = this._invalidateCacheEntriesByRuleIds(
			this._syncProcessCache,
			this._syncCacheStats,
			targets,
		);
		const removedAsync = this._invalidateCacheEntriesByRuleIds(
			this._asyncProcessCache,
			this._asyncCacheStats,
			targets,
		);
		return removedSync + removedAsync;
	}

	private _invalidateCacheEntriesByRuleIds(
		cache: Map<string, CachedShaderProcessResult>,
		stats: InternalCacheStats,
		targetRuleIds: Set<string>,
	): number {
		let removed = 0;
		for (const [key, entry] of cache) {
			if (entry.participatingRuleIds.some((ruleId) => targetRuleIds.has(ruleId))) {
				cache.delete(key);
				removed++;
			}
		}
		stats.invalidations += removed;
		return removed;
	}

	private _applyMutation(
		action: ShaderRuntimeChangeAction,
		ruleIds: string[],
		options: {
			invalidateAll?: boolean;
			invalidateRuleIds?: string[];
			includeModuleIds?: string[];
			injectionScriptIds?: string[];
		} = {},
	): void {
		this._revision++;
		this._ruleExecutionOrderCache = null;
		if (options.invalidateAll) {
			this._invalidateAllProcessCaches();
		} else if (options.invalidateRuleIds && options.invalidateRuleIds.length > 0) {
			this._invalidateProcessCachesByRuleIds(options.invalidateRuleIds);
		}
		this._asyncInFlight.clear();
		this._emitDidChange({
			revision: this._revision,
			action,
			ruleIds: [...new Set(ruleIds)],
			includeModuleIds: options.includeModuleIds
				? [...new Set(options.includeModuleIds)]
				: undefined,
			injectionScriptIds: options.injectionScriptIds
				? [...new Set(options.injectionScriptIds)]
				: undefined,
		});
	}

	private _emitDidChange(event: ShaderRuntimeChangeEvent): void {
		for (const listener of this._listeners) {
			try {
				if (listener.length <= 0) {
					(listener as () => void)();
					continue;
				}
				(listener as (value: ShaderRuntimeChangeEvent) => void)(event);
			} catch (error) {
				// Ignore listener errors to keep runtime operational.
			}
		}
	}
}

function cloneProcessResult(result: ShaderProcessResult, fromCache: boolean): ShaderProcessResult {
	return {
		code: result.code,
		sourceMap: cloneSourceMap(result.sourceMap),
		composite: cloneCompositeSource(result.composite),
		diagnostics: cloneDiagnostics(result.diagnostics),
		hasErrors: result.hasErrors,
		fromCache,
	};
}

function cloneRule(rule: ShaderRule): ShaderRule {
	return {
		...rule,
		symbols: rule.symbols ? [...rule.symbols] : undefined,
		dependsOn: rule.dependsOn ? [...rule.dependsOn] : undefined,
	};
}

function hashStringFNV1a(value: string): string {
	let hash = 0x811c9dc5;
	for (let index = 0; index < value.length; index++) {
		hash ^= value.charCodeAt(index);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}

function hashStringFNV1aChunked(value: string, chunkSize: number): string {
	let hash = 0x811c9dc5;
	for (let offset = 0; offset < value.length; offset += chunkSize) {
		const end = Math.min(value.length, offset + chunkSize);
		for (let index = offset; index < end; index++) {
			hash ^= value.charCodeAt(index);
			hash = Math.imul(hash, 0x01000193);
		}
	}
	return (hash >>> 0).toString(16);
}

const LARGE_SOURCE_THRESHOLD = 16 * 1024;
const LARGE_SOURCE_CHUNK_SIZE = 4 * 1024;

function hashSourceCode(source: string): string {
	if (source.length > LARGE_SOURCE_THRESHOLD) {
		return hashStringFNV1aChunked(source, LARGE_SOURCE_CHUNK_SIZE);
	}
	return hashStringFNV1a(source);
}

function hashSourceMap(sourceMap: ShaderSourceSegmentMap | null | undefined): string {
	if (!sourceMap || !Array.isArray(sourceMap.segments)) {
		return "none";
	}
	const schemaVersion =
		typeof sourceMap.schemaVersion === "number" ? Math.floor(sourceMap.schemaVersion) : 1;
	const payload = [
		`schema:${SOURCE_MAP_SCHEMA_VERSION}`,
		`sourceSchema:${schemaVersion}`,
		`lineCount:${sourceMap.lineCount}`,
		...sourceMap.segments.map((segment) =>
			[
				segment.generatedLineStart,
				segment.generatedLineEnd,
				segment.generatedColumnStart ?? "",
				segment.generatedColumnEnd ?? "",
				segment.sourcePath,
				segment.sourceLineStart,
				segment.sourceLineEnd,
				segment.sourceColumnStart ?? "",
				segment.sourceColumnEnd ?? "",
				segment.kind,
				segment.label ?? "",
			].join(":"),
		),
	].join("|");
	return hashStringFNV1a(payload);
}

function normalizeDependsOn(dependsOn: string[] | undefined): string[] {
	if (!Array.isArray(dependsOn)) {
		return [];
	}
	const seen = new Set<string>();
	const normalized: string[] = [];
	for (const dependency of dependsOn) {
		if (typeof dependency !== "string") {
			continue;
		}
		const trimmed = dependency.trim();
		if (trimmed.length <= 0 || seen.has(trimmed)) {
			continue;
		}
		seen.add(trimmed);
		normalized.push(trimmed);
	}
	return normalized;
}

function buildStrictModeError(
	context: ShaderRuleContext,
	diagnostics: ShaderDiagnostic[],
	maxDiagnostics: number,
): Error {
	const label = context.label ?? "unnamed-shader";
	const errors = diagnostics.filter((diagnostic) => diagnostic.severity === "error");
	const cap = Number.isFinite(maxDiagnostics)
		? Math.max(1, Math.floor(maxDiagnostics))
		: errors.length;
	const details = errors
		.map((diagnostic) => `- [${diagnostic.code}] ${diagnostic.message}`)
		.slice(0, cap)
		.join("\n");
	return new Error(
		[
			`ShaderRuntime validation failed (${label}, ${context.language}/${context.stage}).`,
			details.length > 0 ? details : "- Unknown validation failure.",
			errors.length > cap ? `- (${errors.length - cap} more diagnostics omitted)` : "",
		].join("\n"),
	);
}

function normalizePositiveInteger(value: number | undefined): number | null {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return null;
	}
	const normalized = Math.floor(value);
	return normalized >= 1 ? normalized : 1;
}

function createGeneratedCompositeWithColumnSpans(
	code: string,
	sourcePath: string,
	label?: string,
): CompositeShaderSource {
	const sourceLines = code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const segments: ShaderSourceSegment[] = [];
	for (let lineIndex = 0; lineIndex < lineCount; lineIndex++) {
		const line = sourceLines[lineIndex] ?? "";
		const lineNumber = lineIndex + 1;
		const columnEnd = Math.max(1, line.length + 1);
		segments.push({
			generatedLineStart: lineNumber,
			generatedLineEnd: lineNumber,
			generatedColumnStart: 1,
			generatedColumnEnd: columnEnd,
			sourcePath,
			sourceLineStart: lineNumber,
			sourceLineEnd: lineNumber,
			sourceColumnStart: 1,
			sourceColumnEnd: columnEnd,
			kind: "generated",
			label,
		});
	}
	return {
		code,
		sourceMap: {
			schemaVersion: SOURCE_MAP_SCHEMA_VERSION,
			lineCount,
			segments,
		},
	};
}

function normalizeDiagnosticRange(
	range: ShaderDiagnosticRange | undefined,
): ShaderDiagnosticRange | null {
	if (!range || typeof range !== "object") {
		return null;
	}
	const startLine = normalizePositiveInteger(range.start?.line);
	const startColumn = normalizePositiveInteger(range.start?.column) ?? 1;
	const endLine = normalizePositiveInteger(range.end?.line);
	const endColumn = normalizePositiveInteger(range.end?.column) ?? 1;
	if (!startLine || !endLine) {
		return null;
	}
	return {
		start: { line: startLine, column: startColumn },
		end: { line: endLine, column: endColumn },
	};
}

function resolveShaderRequestSourcePath(request: ShaderProcessRequest): string {
	const explicitDirectivePath =
		typeof request.directiveSourcePath === "string" ? request.directiveSourcePath.trim() : "";
	if (explicitDirectivePath.length > 0) {
		return explicitDirectivePath;
	}
	const sourceMapPath = request.sourceMap?.segments?.[0]?.sourcePath;
	if (typeof sourceMapPath === "string" && sourceMapPath.length > 0) {
		return sourceMapPath;
	}
	const normalizedLanguage = normalizeLanguage(request.language);
	return (
		request.label ??
		`<runtime:${normalizedLanguage}:${normalizeStage(request.stage)}:${normalizeSourceKind(
			request.sourceKind,
		)}>`
	);
}

function formatIncludeModuleEventId(language: ShaderLanguage, moduleId: string): string {
	return `${language}:${moduleId}`;
}
