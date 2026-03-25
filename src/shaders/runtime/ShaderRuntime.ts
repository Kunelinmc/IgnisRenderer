import {
	SHADER_RUNTIME_DEFAULT_CACHE_LIMIT,
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	resolveDefaultShaderRuntimeMode,
} from "./constants";
import { createBuiltInShaderRules } from "./builtins";
import {
	composeCompositeShaderSources,
	createInlineCompositeShaderSource,
	sliceCompositeShaderSource,
} from "./sourceMap";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderProcessRequest,
	ShaderProcessResult,
	ShaderRule,
	ShaderRuleContext,
	ShaderRuleInjection,
	ShaderSourceSegmentMap,
	ShaderRuntimeMode,
	ShaderSourceKind,
	ShaderStage,
} from "./types";

interface ShaderRuntimeOptions {
	mode?: ShaderRuntimeMode;
	cacheLimit?: number;
}

interface CachedShaderProcessResult {
	result: ShaderProcessResult;
}

type ShaderRuntimeChangeListener = (revision: number) => void;

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
		case "shadow":
		case "particle":
		case "custom-material":
			return sourceKind;
		default:
			return "unknown";
	}
}

function cloneDiagnostics(diagnostics: ShaderDiagnostic[]): ShaderDiagnostic[] {
	return diagnostics.map((diagnostic) => ({ ...diagnostic }));
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

function cloneProcessResult(
	result: ShaderProcessResult,
	fromCache: boolean
): ShaderProcessResult {
	return {
		code: result.code,
		sourceMap: cloneSourceMap(result.sourceMap),
		composite: cloneCompositeSource(result.composite),
		diagnostics: cloneDiagnostics(result.diagnostics),
		hasErrors: result.hasErrors,
		fromCache,
	};
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

function normalizeInjectionBlock(block: string | undefined): string {
	if (typeof block !== "string") {
		return "";
	}
	return block.trim();
}

interface InjectionBlock {
	code: string;
	sourcePath: string;
	label: string;
}

function buildStrictModeError(
	context: ShaderRuleContext,
	diagnostics: ShaderDiagnostic[]
): Error {
	const label = context.label ?? "unnamed-shader";
	const details = diagnostics
		.filter((diagnostic) => diagnostic.severity === "error")
		.map((diagnostic) => `- [${diagnostic.code}] ${diagnostic.message}`)
		.slice(0, 8)
		.join("\n");
	return new Error(
		[
			`ShaderRuntime validation failed (${label}, ${context.language}/${context.stage}).`,
			details.length > 0 ? details : "- Unknown validation failure.",
		].join("\n")
	);
}

function injectWGSLSource(
	source: CompositeShaderSource,
	headers: InjectionBlock[],
	functions: InjectionBlock[]
): CompositeShaderSource {
	const blocks = [...headers, ...functions];
	if (blocks.length <= 0) {
		return source;
	}
	return composeCompositeShaderSources(
		[
			...blocks.map((block) => ({
				code: block.code,
				sourcePath: block.sourcePath,
				kind: "define-block" as const,
				label: block.label,
			})),
			{
				code: source.code,
				sourceMap: source.sourceMap,
				sourcePath: source.sourceMap.segments[0]?.sourcePath ?? "<shader>",
				kind: "source" as const,
			},
		],
		"\n\n"
	);
}

function injectGLSLSource(
	source: CompositeShaderSource,
	headers: InjectionBlock[],
	functions: InjectionBlock[]
): CompositeShaderSource {
	const blocks = [...headers, ...functions];
	if (blocks.length <= 0) {
		return source;
	}

	const injection = composeCompositeShaderSources(
		blocks.map((block) => ({
			code: block.code,
			sourcePath: block.sourcePath,
			kind: "define-block" as const,
			label: block.label,
		})),
		"\n\n"
	);
	const sourceLines = source.code.split(/\r?\n/g);
	const hasVersionLine =
		sourceLines.length > 0 && /^\s*#version[^\n]*$/.test(sourceLines[0]);
	if (!hasVersionLine) {
		return composeCompositeShaderSources(
			[
				{
					code: injection.code,
					sourceMap: injection.sourceMap,
					sourcePath: "<runtime:injection>",
					kind: "define-block",
				},
				{
					code: source.code,
					sourceMap: source.sourceMap,
					sourcePath: source.sourceMap.segments[0]?.sourcePath ?? "<shader>",
					kind: "source",
				},
			],
			"\n\n"
		);
	}

	const beforeVersion = sourceLines[0] ?? "";
	const afterVersion = sliceCompositeShaderSource(source, 2);
	const withInjection = composeCompositeShaderSources(
		[
			createInlineCompositeShaderSource(
				beforeVersion,
				source.sourceMap.segments[0]?.sourcePath ?? "<shader>",
				"source",
				"version"
			),
			{
				code: injection.code,
				sourceMap: injection.sourceMap,
				sourcePath: "<runtime:injection>",
				kind: "define-block",
			},
		],
		"\n"
	);
	if (afterVersion.code.length <= 0) {
		return withInjection;
	}
	return composeCompositeShaderSources(
		[
			{
				code: withInjection.code,
				sourceMap: withInjection.sourceMap,
				sourcePath:
					withInjection.sourceMap.segments[0]?.sourcePath ?? "<shader>",
				kind: "source",
			},
			{
				code: afterVersion.code,
				sourceMap: afterVersion.sourceMap,
				sourcePath:
					afterVersion.sourceMap.segments[0]?.sourcePath ??
					source.sourceMap.segments[0]?.sourcePath ??
					"<shader>",
				kind: "source" as const,
			},
		],
		"\n\n"
	);
}

export class ShaderRuntime {
	private _mode: ShaderRuntimeMode;
	private _cacheLimit: number;
	private _revision: number;
	private _builtInRules: Map<string, ShaderRule>;
	private _userRules: Map<string, ShaderRule>;
	private _builtInSymbols: Set<string>;
	private _processCache: Map<string, CachedShaderProcessResult>;
	private _listeners: Set<ShaderRuntimeChangeListener>;

	public constructor(options: ShaderRuntimeOptions = {}) {
		this._mode = options.mode ?? resolveDefaultShaderRuntimeMode();
		this._cacheLimit = Math.max(
			1,
			Math.floor(options.cacheLimit ?? SHADER_RUNTIME_DEFAULT_CACHE_LIMIT)
		);
		this._revision = 1;
		this._builtInRules = new Map();
		this._userRules = new Map();
		this._builtInSymbols = new Set();
		this._processCache = new Map();
		this._listeners = new Set();

		for (const rule of createBuiltInShaderRules()) {
			this._builtInRules.set(rule.id, rule);
			for (const symbol of rule.symbols ?? []) {
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
		if (mode !== "strict" && mode !== "warn") {
			throw new Error(`Unsupported ShaderRuntime mode "${String(mode)}".`);
		}
		if (this._mode === mode) {
			return;
		}
		this._mode = mode;
		this._bumpRevision();
	}

	public onDidChange(listener: ShaderRuntimeChangeListener): () => void {
		this._listeners.add(listener);
		return () => {
			this._listeners.delete(listener);
		};
	}

	public registerRule(rule: ShaderRule): void {
		const normalized = this._normalizeRule(rule);
		if (normalized.id.startsWith(SHADER_RUNTIME_RESERVED_RULE_PREFIX)) {
			throw new Error(
				`ShaderRuntime user rules cannot use reserved prefix "${SHADER_RUNTIME_RESERVED_RULE_PREFIX}".`
			);
		}
		this._userRules.set(normalized.id, normalized);
		this._bumpRevision();
	}

	public unregisterRule(ruleId: string): boolean {
		const removed = this._userRules.delete(ruleId);
		if (removed) {
			this._bumpRevision();
		}
		return removed;
	}

	public clearUserRules(): void {
		if (this._userRules.size <= 0) {
			return;
		}
		this._userRules.clear();
		this._bumpRevision();
	}

	public listRules(): ShaderRule[] {
		return this._collectRulesInExecutionOrder().map((rule) => ({ ...rule }));
	}

	public process(request: ShaderProcessRequest): ShaderProcessResult {
		const context: ShaderRuleContext = {
			mode: this._mode,
			language: request.language,
			stage: normalizeStage(request.stage),
			entryPoint: request.entryPoint ?? null,
			label: request.label ?? null,
			sourceKind: normalizeSourceKind(request.sourceKind),
			source: request.code,
		};
		const cacheKey = this._buildProcessCacheKey(context, request.sourceMap);
		const cached = this._getCachedResult(cacheKey);
		if (cached) {
			return cloneProcessResult(cached, true);
		}

		const diagnostics: ShaderDiagnostic[] = [];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const rules = this._collectRulesInExecutionOrder();

		for (const rule of rules) {
			if (rule.match && !rule.match(context)) {
				continue;
			}

			if (rule.validate) {
				for (const diagnostic of rule.validate(context)) {
					diagnostics.push({ ...diagnostic, ruleId: rule.id });
				}
			}

			if (!rule.inject) {
				continue;
			}
			const injection = rule.inject(context);
			if (!injection) {
				continue;
			}
			if (this._isUserRule(rule.id)) {
				const conflict = this._resolveBuiltInSymbolConflict(rule, injection);
				if (conflict) {
					diagnostics.push(conflict);
					continue;
				}
			}
			const header = normalizeInjectionBlock(injection.header);
			if (header.length > 0) {
				headers.push({
					code: header,
					sourcePath: `<runtime:${rule.id}:header>`,
					label: `${rule.id}:header`,
				});
			}
			const functionBlock = normalizeInjectionBlock(injection.functions);
			if (functionBlock.length > 0) {
				functions.push({
					code: functionBlock,
					sourcePath: `<runtime:${rule.id}:functions>`,
					label: `${rule.id}:functions`,
				});
			}
		}

		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
		if (hasErrors && this._mode === "strict") {
			throw buildStrictModeError(context, diagnostics);
		}

		const sourcePath =
			context.label ??
			`<runtime:${context.language}:${context.stage}:${context.sourceKind}>`;
		const baseComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const composite =
			context.language === "wgsl" ?
				injectWGSLSource(baseComposite, headers, functions)
			:	injectGLSLSource(baseComposite, headers, functions);
		const result: ShaderProcessResult = {
			code: composite.code,
			sourceMap: cloneSourceMap(composite.sourceMap),
			composite: cloneCompositeSource(composite),
			diagnostics,
			hasErrors,
			fromCache: false,
		};
		this._setCachedResult(cacheKey, result);
		return cloneProcessResult(result, false);
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
			typeof rule.priority === "number" && Number.isFinite(rule.priority) ?
				Math.floor(rule.priority)
			:	0;
		return {
			...rule,
			id,
			priority,
		};
	}

	private _collectRulesInExecutionOrder(): ShaderRule[] {
		return [...this._builtInRules.values(), ...this._userRules.values()].sort(
			(left, right) => {
				const leftPriority = left.priority ?? 0;
				const rightPriority = right.priority ?? 0;
				if (leftPriority !== rightPriority) {
					return rightPriority - leftPriority;
				}
				return left.id.localeCompare(right.id);
			}
		);
	}

	private _buildProcessCacheKey(
		context: ShaderRuleContext,
		sourceMap: ShaderSourceSegmentMap | null | undefined
	): string {
		return [
			`rev:${this._revision}`,
			`mode:${this._mode}`,
			`lang:${context.language}`,
			`stage:${context.stage}`,
			`entry:${context.entryPoint ?? ""}`,
			`kind:${context.sourceKind}`,
			`code:${hashStringFNV1a(context.source)}`,
			`sourceMap:${hashSourceMap(sourceMap)}`,
		].join("|");
	}

	private _getCachedResult(key: string): ShaderProcessResult | null {
		const entry = this._processCache.get(key);
		if (!entry) {
			return null;
		}
		this._processCache.delete(key);
		this._processCache.set(key, entry);
		return cloneProcessResult(entry.result, false);
	}

	private _setCachedResult(key: string, result: ShaderProcessResult): void {
		this._processCache.set(key, {
			result: cloneProcessResult(result, false),
		});
		while (this._processCache.size > this._cacheLimit) {
			const oldestKey = this._processCache.keys().next().value;
			if (typeof oldestKey !== "string") {
				break;
			}
			this._processCache.delete(oldestKey);
		}
	}

	private _isUserRule(ruleId: string): boolean {
		return !ruleId.startsWith(SHADER_RUNTIME_RESERVED_RULE_PREFIX);
	}

	private _resolveBuiltInSymbolConflict(
		rule: ShaderRule,
		injection: ShaderRuleInjection
	): ShaderDiagnostic | null {
		const symbols = [
			...(rule.symbols ?? []),
			...(injection.symbols ?? []),
		].filter((symbol) => typeof symbol === "string" && symbol.length > 0);
		if (symbols.length <= 0) {
			return null;
		}
		for (const symbol of symbols) {
			if (!this._builtInSymbols.has(symbol)) {
				continue;
			}
			return {
				ruleId: rule.id,
				code: "reserved-symbol-conflict",
				severity: this._mode === "strict" ? "error" : "warning",
				message:
					`Rule "${rule.id}" conflicts with reserved symbol "${symbol}" and was skipped.`,
			};
		}
		return null;
	}

	private _bumpRevision(): void {
		this._revision++;
		this._processCache.clear();
		for (const listener of this._listeners) {
			try {
				listener(this._revision);
			} catch (error) {
				// Ignore listener errors to keep runtime operational.
			}
		}
	}
}
