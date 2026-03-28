import {
	SHADER_RUNTIME_DEFAULT_CACHE_LIMIT,
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	resolveDefaultShaderRuntimeMode,
} from "./constants";
import { createBuiltInShaderRules } from "./builtins";
import {
	composeCompositeShaderSources,
	createInlineCompositeShaderSource,
	mapShaderGeneratedLocation,
	sliceCompositeShaderSource,
} from "./sourceMap";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticFilter,
	ShaderDiagnosticRange,
	ShaderGLSLInjectionAnchor,
	ShaderInjectionAnchor,
	ShaderProcessRequest,
	ShaderProcessResult,
	ShaderResolvedInjectionAnchors,
	ShaderRule,
	ShaderRuleContext,
	ShaderRuleInjection,
	ShaderRuntimeCacheKind,
	ShaderRuntimeCacheStats,
	ShaderRuntimeCacheStatsSnapshot,
	ShaderRuntimeChangeAction,
	ShaderRuntimeChangeEvent,
	ShaderRuntimeMode,
	ShaderSourceKind,
	ShaderSourceSegmentMap,
	ShaderStage,
	ShaderWGSLInjectionAnchor,
} from "./types";

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
	matchedRules: ShaderRule[];
	matchedRuleIds: string[];
	cacheKey: string;
	sourceMap: ShaderSourceSegmentMap | null | undefined;
}

type ShaderRuntimeChangeListener =
	| ((event: ShaderRuntimeChangeEvent) => void)
	| (() => void);

const DEFAULT_STRICT_ERROR_MAX_DIAGNOSTICS = 32;
const LARGE_SOURCE_THRESHOLD = 16 * 1024;
const LARGE_SOURCE_CHUNK_SIZE = 4 * 1024;
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

function cloneRule(rule: ShaderRule): ShaderRule {
	return {
		...rule,
		symbols: rule.symbols ? [...rule.symbols] : undefined,
		dependsOn: rule.dependsOn ? [...rule.dependsOn] : undefined,
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

function hashStringFNV1aChunked(value: string, chunkSize: number): string {
	let hash = 0x811c9dc5;
	for (let offset = 0; offset < value.length; offset += chunkSize) {
		const end = Math.min(value.length, offset + chunkSize);
		for (let i = offset; i < end; i++) {
			hash ^= value.charCodeAt(i);
			hash = Math.imul(hash, 0x01000193);
		}
	}
	return (hash >>> 0).toString(16);
}

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

function normalizeSymbols(symbols: string[] | undefined): string[] {
	if (!Array.isArray(symbols)) {
		return [];
	}
	const seen = new Set<string>();
	const normalized: string[] = [];
	for (const symbol of symbols) {
		if (typeof symbol !== "string") {
			continue;
		}
		const trimmed = symbol.trim();
		if (trimmed.length <= 0 || seen.has(trimmed)) {
			continue;
		}
		seen.add(trimmed);
		normalized.push(trimmed);
	}
	return normalized;
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

function isPromiseLike<T = unknown>(value: unknown): value is PromiseLike<T> {
	return (
		typeof value === "object" &&
		value !== null &&
		"then" in value &&
		typeof (value as { then?: unknown }).then === "function"
	);
}

interface InjectionBlock {
	code: string;
	sourcePath: string;
	label: string;
	anchor: ShaderInjectionAnchor;
}

interface GLSLInsertionAnchors {
	afterVersion: number;
	afterPrecision: number;
	afterDefines: number;
	afterStruct: number;
	afterUniforms: number;
	beforeEntryPoint: number;
	endOfFile: number;
}

interface WGSLInsertionAnchors {
	afterEnable: number;
	afterAliases: number;
	afterStruct: number;
	afterBindings: number;
	beforeEntryPoint: number;
	endOfFile: number;
}

function countToken(line: string, token: string): number {
	let count = 0;
	for (let i = 0; i < line.length; i++) {
		if (line[i] === token) {
			count++;
		}
	}
	return count;
}

function findLastStructEndLine(sourceLines: string[], pattern: RegExp): number {
	let lastStructEndLine = 0;
	let inStruct = false;
	let braceDepth = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (!inStruct && pattern.test(line)) {
			inStruct = true;
			braceDepth = countToken(line, "{") - countToken(line, "}");
			if (braceDepth <= 0 && /};?\s*$/.test(line)) {
				lastStructEndLine = lineNumber;
				inStruct = false;
				braceDepth = 0;
			}
			continue;
		}
		if (!inStruct) {
			continue;
		}
		braceDepth += countToken(line, "{");
		braceDepth -= countToken(line, "}");
		if (braceDepth <= 0 && /};?\s*$/.test(line)) {
			lastStructEndLine = lineNumber;
			inStruct = false;
			braceDepth = 0;
		}
	}
	return lastStructEndLine;
}

function normalizeGLSLInjectionAnchor(
	anchor: ShaderInjectionAnchor | undefined
): ShaderGLSLInjectionAnchor {
	switch (anchor) {
		case "afterVersion":
		case "afterPrecision":
		case "afterDefines":
		case "afterStruct":
		case "afterUniforms":
		case "beforeEntryPoint":
		case "endOfFile":
			return anchor;
		default:
			return "afterVersion";
	}
}

function normalizeWGSLInjectionAnchor(
	anchor: ShaderInjectionAnchor | undefined
): ShaderWGSLInjectionAnchor {
	switch (anchor) {
		case "afterEnable":
		case "afterAliases":
		case "afterStruct":
		case "afterBindings":
		case "beforeEntryPoint":
		case "endOfFile":
			return anchor;
		default:
			return "afterEnable";
	}
}

function resolveGLSLInsertionAnchors(
	source: CompositeShaderSource
): GLSLInsertionAnchors {
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	let versionLine = 0;
	let lastPrecisionLine = 0;
	let entryPointLine = 0;
	let lastDefineLine = 0;
	let lastUniformLine = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (versionLine <= 0 && /^\s*#version\b/.test(line)) {
			versionLine = lineNumber;
			continue;
		}
		if (entryPointLine <= 0 && /\bvoid\s+main\s*\(/.test(line)) {
			entryPointLine = lineNumber;
		}
		if (/^\s*precision\b[^;]*;/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastPrecisionLine = lineNumber;
			}
		}
		if (/^\s*#define\b/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastDefineLine = lineNumber;
			}
		}
		if (/^\s*uniform\b/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastUniformLine = lineNumber;
			}
		}
	}

	const lastStructEndLine = findLastStructEndLine(sourceLines, /^\s*struct\b/);
	const afterVersionLine = versionLine > 0 ? versionLine + 1 : 1;
	const afterPrecisionLine =
		lastPrecisionLine > 0 ? lastPrecisionLine + 1 : afterVersionLine;
	const afterDefinesLine =
		lastDefineLine > 0 ? Math.max(lastDefineLine + 1, afterPrecisionLine) : afterPrecisionLine;
	const afterStructLine =
		lastStructEndLine > 0 ? Math.max(lastStructEndLine + 1, afterDefinesLine) : afterDefinesLine;
	const afterUniformsLine =
		lastUniformLine > 0 ? Math.max(lastUniformLine + 1, afterStructLine) : afterStructLine;
	const beforeEntryPointLine = entryPointLine > 0 ? entryPointLine : lineCount + 1;
	return {
		afterVersion: clampInjectionLine(afterVersionLine, lineCount),
		afterPrecision: clampInjectionLine(afterPrecisionLine, lineCount),
		afterDefines: clampInjectionLine(afterDefinesLine, lineCount),
		afterStruct: clampInjectionLine(afterStructLine, lineCount),
		afterUniforms: clampInjectionLine(afterUniformsLine, lineCount),
		beforeEntryPoint: clampInjectionLine(beforeEntryPointLine, lineCount),
		endOfFile: lineCount + 1,
	};
}

function resolveWGSLInsertionAnchors(
	source: CompositeShaderSource
): WGSLInsertionAnchors {
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	let lastEnableLine = 0;
	let lastAliasLine = 0;
	let lastBindingLine = 0;
	let entryPointLine = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (/^\s*enable\b/.test(line)) {
			lastEnableLine = lineNumber;
		}
		if (/^\s*alias\b/.test(line)) {
			lastAliasLine = lineNumber;
		}
		if (/@group\s*\([^)]*\)\s*@binding\s*\([^)]*\)/.test(line)) {
			lastBindingLine = lineNumber;
		}
		if (entryPointLine <= 0 && /@\s*(vertex|fragment|compute)\b/.test(line)) {
			entryPointLine = lineNumber;
		}
	}
	if (entryPointLine <= 0) {
		for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
			if (/\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/.test(sourceLines[lineIndex])) {
				entryPointLine = lineIndex + 1;
				break;
			}
		}
	}
	const lastStructEndLine = findLastStructEndLine(sourceLines, /^\s*struct\b/);
	const afterEnableLine = lastEnableLine > 0 ? lastEnableLine + 1 : 1;
	const afterAliasesLine =
		lastAliasLine > 0 ? Math.max(lastAliasLine + 1, afterEnableLine) : afterEnableLine;
	const afterStructLine =
		lastStructEndLine > 0 ? Math.max(lastStructEndLine + 1, afterAliasesLine) : afterAliasesLine;
	const afterBindingsLine =
		lastBindingLine > 0 ? Math.max(lastBindingLine + 1, afterStructLine) : afterStructLine;
	const beforeEntryPointLine = entryPointLine > 0 ? entryPointLine : lineCount + 1;
	return {
		afterEnable: clampInjectionLine(afterEnableLine, lineCount),
		afterAliases: clampInjectionLine(afterAliasesLine, lineCount),
		afterStruct: clampInjectionLine(afterStructLine, lineCount),
		afterBindings: clampInjectionLine(afterBindingsLine, lineCount),
		beforeEntryPoint: clampInjectionLine(beforeEntryPointLine, lineCount),
		endOfFile: lineCount + 1,
	};
}

function clampInjectionLine(line: number, lineCount: number): number {
	const normalized = Number.isFinite(line) ? Math.floor(line) : 1;
	return Math.min(Math.max(normalized, 1), lineCount + 1);
}

function resolveGLSLInsertionLine(
	anchor: ShaderGLSLInjectionAnchor,
	anchors: GLSLInsertionAnchors
): number {
	switch (anchor) {
		case "afterPrecision":
			return anchors.afterPrecision;
		case "afterDefines":
			return anchors.afterDefines;
		case "afterStruct":
			return anchors.afterStruct;
		case "afterUniforms":
			return anchors.afterUniforms;
		case "beforeEntryPoint":
			return anchors.beforeEntryPoint;
		case "endOfFile":
			return anchors.endOfFile;
		case "afterVersion":
		default:
			return anchors.afterVersion;
	}
}

function resolveWGSLInsertionLine(
	anchor: ShaderWGSLInjectionAnchor,
	anchors: WGSLInsertionAnchors
): number {
	switch (anchor) {
		case "afterAliases":
			return anchors.afterAliases;
		case "afterStruct":
			return anchors.afterStruct;
		case "afterBindings":
			return anchors.afterBindings;
		case "beforeEntryPoint":
			return anchors.beforeEntryPoint;
		case "endOfFile":
			return anchors.endOfFile;
		case "afterEnable":
		default:
			return anchors.afterEnable;
	}
}

function buildStrictModeError(
	context: ShaderRuleContext,
	diagnostics: ShaderDiagnostic[],
	maxDiagnostics: number
): Error {
	const label = context.label ?? "unnamed-shader";
	const errors = diagnostics.filter((diagnostic) => diagnostic.severity === "error");
	const cap = Number.isFinite(maxDiagnostics) ?
		Math.max(1, Math.floor(maxDiagnostics))
	:	errors.length;
	const details = errors
		.map((diagnostic) => `- [${diagnostic.code}] ${diagnostic.message}`)
		.slice(0, cap)
		.join("\n");
	return new Error(
		[
			`ShaderRuntime validation failed (${label}, ${context.language}/${context.stage}).`,
			details.length > 0 ? details : "- Unknown validation failure.",
			errors.length > cap ? `- (${errors.length - cap} more diagnostics omitted)` : "",
		].join("\n")
	);
}

function injectBlocksAtLines(
	source: CompositeShaderSource,
	insertions: Map<number, InjectionBlock[]>
): CompositeShaderSource {
	if (insertions.size <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const sourcePath = source.sourceMap.segments[0]?.sourcePath ?? "<shader>";
	const insertionLines = [...insertions.keys()].sort((left, right) => left - right);
	const parts: {
		code: string;
		sourceMap: ShaderSourceSegmentMap;
		sourcePath: string;
		kind: "source" | "define-block";
	}[] = [];
	let cursorLine = 1;
	for (const insertionLine of insertionLines) {
		if (insertionLine > cursorLine) {
			const sourceSlice = sliceCompositeShaderSource(
				source,
				cursorLine,
				insertionLine - 1
			);
			parts.push({
				code: sourceSlice.code,
				sourceMap: sourceSlice.sourceMap,
				sourcePath: sourceSlice.sourceMap.segments[0]?.sourcePath ?? sourcePath,
				kind: "source",
			});
		}
		const bucket = insertions.get(insertionLine) ?? [];
		const injection = composeCompositeShaderSources(
			bucket.map((block) => ({
				code: block.code,
				sourcePath: block.sourcePath,
				kind: "define-block" as const,
				label: block.label,
			})),
			"\n\n"
		);
		parts.push({
			code: injection.code,
			sourceMap: injection.sourceMap,
			sourcePath: "<runtime:injection>",
			kind: "define-block",
		});
		cursorLine = insertionLine;
	}
	if (cursorLine <= lineCount) {
		const sourceSlice = sliceCompositeShaderSource(source, cursorLine, lineCount);
		parts.push({
			code: sourceSlice.code,
			sourceMap: sourceSlice.sourceMap,
			sourcePath: sourceSlice.sourceMap.segments[0]?.sourcePath ?? sourcePath,
			kind: "source",
		});
	}
	return composeCompositeShaderSources(parts, "\n");
}

function injectGLSLSource(
	source: CompositeShaderSource,
	blocks: InjectionBlock[]
): CompositeShaderSource {
	if (blocks.length <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const anchors = resolveGLSLInsertionAnchors(source);
	const insertions = new Map<number, InjectionBlock[]>();
	for (const block of blocks) {
		const anchor = normalizeGLSLInjectionAnchor(block.anchor);
		const insertionLine = clampInjectionLine(
			resolveGLSLInsertionLine(anchor, anchors),
			lineCount
		);
		const bucket = insertions.get(insertionLine);
		if (bucket) {
			bucket.push(block);
			continue;
		}
		insertions.set(insertionLine, [block]);
	}
	return injectBlocksAtLines(source, insertions);
}

function injectWGSLSource(
	source: CompositeShaderSource,
	blocks: InjectionBlock[]
): CompositeShaderSource {
	if (blocks.length <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const anchors = resolveWGSLInsertionAnchors(source);
	const insertions = new Map<number, InjectionBlock[]>();
	for (const block of blocks) {
		const anchor = normalizeWGSLInjectionAnchor(block.anchor);
		const insertionLine = clampInjectionLine(
			resolveWGSLInsertionLine(anchor, anchors),
			lineCount
		);
		const bucket = insertions.get(insertionLine);
		if (bucket) {
			bucket.push(block);
			continue;
		}
		insertions.set(insertionLine, [block]);
	}
	return injectBlocksAtLines(source, insertions);
}

function normalizePositiveInteger(value: number | undefined): number | null {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return null;
	}
	const normalized = Math.floor(value);
	return normalized >= 1 ? normalized : 1;
}

function createPointRange(line: number, column: number): ShaderDiagnosticRange {
	return {
		start: { line, column },
		end: { line, column },
	};
}

function normalizeDiagnosticRange(
	range: ShaderDiagnosticRange | undefined
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
		start: {
			line: startLine,
			column: startColumn,
		},
		end: {
			line: endLine,
			column: endColumn,
		},
	};
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

	public constructor(options: ShaderRuntimeOptions = {}) {
		this._mode = options.mode ?? resolveDefaultShaderRuntimeMode();
		this._cacheLimit = Math.max(
			1,
			Math.floor(options.cacheLimit ?? SHADER_RUNTIME_DEFAULT_CACHE_LIMIT)
		);
		this._strictErrorMaxDiagnostics = Math.max(
			1,
			Math.floor(
				options.strictErrorMaxDiagnostics ?? DEFAULT_STRICT_ERROR_MAX_DIAGNOSTICS
			)
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
		this._applyMutation("mode", [], { invalidateAll: true });
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
		const action: ShaderRuntimeChangeAction =
			this._userRules.has(normalized.id) ? "update-rule" : "register-rule";
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
					.join(", ")}.`
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
		kind?: ShaderRuntimeCacheKind
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

	public resolveInjectionAnchors(
		request: ShaderProcessRequest
	): ShaderResolvedInjectionAnchors {
		const stage = normalizeStage(request.stage);
		const sourceKind = normalizeSourceKind(request.sourceKind);
		const sourcePath =
			request.label ??
			`<runtime:${request.language}:${stage}:${sourceKind}>`;
		const baseComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		if (request.language === "glsl") {
			const anchors = resolveGLSLInsertionAnchors(baseComposite);
			return {
				language: "glsl",
				lineCount: Math.max(1, baseComposite.code.split(/\r?\n/g).length),
				anchors,
			};
		}
		const anchors = resolveWGSLInsertionAnchors(baseComposite);
		return {
			language: "wgsl",
			lineCount: Math.max(1, baseComposite.code.split(/\r?\n/g).length),
			anchors,
		};
	}

	public process(request: ShaderProcessRequest): ShaderProcessResult {
		const prepared = this._prepareProcessSync(request);
		const cached = this._getCachedResult("sync", prepared.cacheKey);
		if (cached) {
			return this._finalizeProcessResult(
				prepared.context,
				cached.result,
				request.diagnosticFilter,
				true
			);
		}

		const rawResult = this._executeRulesSync(prepared);
		this._setCachedResult(
			"sync",
			prepared.cacheKey,
			rawResult,
			prepared.matchedRuleIds
		);
		return this._finalizeProcessResult(
			prepared.context,
			rawResult,
			request.diagnosticFilter,
			false
		);
	}

	public async processAsync(
		request: ShaderProcessRequest
	): Promise<ShaderProcessResult> {
		const prepared = await this._prepareProcessAsync(request);
		const cached = this._getCachedResult("async", prepared.cacheKey);
		if (cached) {
			return this._finalizeProcessResult(
				prepared.context,
				cached.result,
				request.diagnosticFilter,
				true
			);
		}
		const inFlight = this._asyncInFlight.get(prepared.cacheKey);
		if (inFlight) {
			const shared = await inFlight;
			return this._finalizeProcessResult(
				prepared.context,
				shared,
				request.diagnosticFilter,
				true
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
						prepared.matchedRuleIds
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
		return this._finalizeProcessResult(
			prepared.context,
			raw,
			request.diagnosticFilter,
			false
		);
	}

	private _prepareProcessSync(request: ShaderProcessRequest): ProcessPreparation {
		const context = this._buildRuleContext(request);
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

		const matchedRules: ShaderRule[] = [];
		for (const rule of this._collectRulesInExecutionOrder()) {
			if (!rule.match) {
				matchedRules.push(rule);
				continue;
			}
			const matchResult = rule.match(context);
			if (isPromiseLike(matchResult)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from match() during process(). Use processAsync().`
				);
			}
			if (matchResult) {
				matchedRules.push(rule);
			}
		}

		const sourceHash = this._resolveSourceHash(context.source, request.sourceHash);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			request.sourceMap,
			sourceHash,
			matchedRuleIds
		);

		return {
			context,
			sourcePath,
			baseComposite,
			matchedRules,
			matchedRuleIds,
			cacheKey,
			sourceMap: request.sourceMap,
		};
	}

	private async _prepareProcessAsync(
		request: ShaderProcessRequest
	): Promise<ProcessPreparation> {
		const context = this._buildRuleContext(request);
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

		const sourceHash = this._resolveSourceHash(context.source, request.sourceHash);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			request.sourceMap,
			sourceHash,
			matchedRuleIds
		);

		return {
			context,
			sourcePath,
			baseComposite,
			matchedRules,
			matchedRuleIds,
			cacheKey,
			sourceMap: request.sourceMap,
		};
	}

	private _buildRuleContext(request: ShaderProcessRequest): ShaderRuleContext {
		return {
			mode: this._mode,
			language: request.language,
			stage: normalizeStage(request.stage),
			entryPoint: request.entryPoint ?? null,
			label: request.label ?? null,
			sourceKind: normalizeSourceKind(request.sourceKind),
			source: request.code,
		};
	}

	private _resolveSourceHash(source: string, sourceHash: string | undefined): string {
		const providedHash =
			typeof sourceHash === "string" ? sourceHash.trim() : "";
		if (providedHash.length <= 0) {
			return hashSourceCode(source);
		}
		if (IS_DEV_ENVIRONMENT) {
			const computedHash = hashSourceCode(source);
			if (providedHash !== computedHash) {
				throw new Error(
					`ShaderRuntime sourceHash mismatch. Provided "${providedHash}" but computed "${computedHash}".`
				);
			}
		}
		return providedHash;
	}

	private _executeRulesSync(prepared: ProcessPreparation): ShaderProcessResult {
		const diagnostics: ShaderDiagnostic[] = [];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const dynamicUserSymbols = new Map<string, string>();

		for (const rule of prepared.matchedRules) {
			if (rule.validate) {
				const validateResult = rule.validate(prepared.context);
				if (isPromiseLike(validateResult)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" returned a Promise from validate() during process(). Use processAsync().`
					);
				}
				const diagnosticsFromRule =
					Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
					diagnostics.push(
						this._normalizeDiagnostic(
							{ ...diagnostic, ruleId: rule.id },
							prepared.sourceMap,
							prepared.sourcePath
						)
					);
				}
			}

			if (!rule.inject) {
				continue;
			}
			const injection = rule.inject(prepared.context);
			if (isPromiseLike(injection)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from inject() during process(). Use processAsync().`
				);
			}
			this._applyInjectionIfAny(
				prepared,
				rule,
				injection,
				diagnostics,
				headers,
				functions,
				dynamicUserSymbols
			);
		}

		return this._buildRawProcessResult(prepared, diagnostics, headers, functions);
	}

	private async _executeRulesAsync(
		prepared: ProcessPreparation
	): Promise<ShaderProcessResult> {
		const diagnostics: ShaderDiagnostic[] = [];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const dynamicUserSymbols = new Map<string, string>();

		for (const rule of prepared.matchedRules) {
			if (rule.validate) {
				const validateResult = await rule.validate(prepared.context);
				const diagnosticsFromRule =
					Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
					diagnostics.push(
						this._normalizeDiagnostic(
							{ ...diagnostic, ruleId: rule.id },
							prepared.sourceMap,
							prepared.sourcePath
						)
					);
				}
			}

			if (!rule.inject) {
				continue;
			}
			const injection = await rule.inject(prepared.context);
			this._applyInjectionIfAny(
				prepared,
				rule,
				injection,
				diagnostics,
				headers,
				functions,
				dynamicUserSymbols
			);
		}

		return this._buildRawProcessResult(prepared, diagnostics, headers, functions);
	}

	private _applyInjectionIfAny(
		prepared: ProcessPreparation,
		rule: ShaderRule,
		injection: ShaderRuleInjection | null | undefined,
		diagnostics: ShaderDiagnostic[],
		headers: InjectionBlock[],
		functions: InjectionBlock[],
		dynamicUserSymbols: Map<string, string>
	): void {
		if (!injection) {
			return;
		}
		if (this._isUserRule(rule.id)) {
			const conflict = this._resolveInjectionSymbolConflict(
				rule,
				injection,
				dynamicUserSymbols
			);
			if (conflict) {
				diagnostics.push(
					this._normalizeDiagnostic(
						conflict,
						prepared.sourceMap,
						prepared.sourcePath
					)
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
				anchor: this._normalizeInjectionAnchorForLanguage(
					prepared.context.language,
					injection.headerAnchor
				),
			});
		}

		const functionBlock = normalizeInjectionBlock(injection.functions);
		if (functionBlock.length > 0) {
			functions.push({
				code: functionBlock,
				sourcePath: `<runtime:${rule.id}:functions>`,
				label: `${rule.id}:functions`,
				anchor: this._normalizeInjectionAnchorForLanguage(
					prepared.context.language,
					injection.functionsAnchor
				),
			});
		}
	}

	private _buildRawProcessResult(
		prepared: ProcessPreparation,
		diagnostics: ShaderDiagnostic[],
		headers: InjectionBlock[],
		functions: InjectionBlock[]
	): ShaderProcessResult {
		const composite =
			prepared.context.language === "wgsl" ?
				injectWGSLSource(prepared.baseComposite, [...headers, ...functions])
			:	injectGLSLSource(prepared.baseComposite, [...headers, ...functions]);
		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
		return {
			code: composite.code,
			sourceMap: composite.sourceMap,
			composite,
			diagnostics,
			hasErrors,
			fromCache: false,
		};
	}

	private _normalizeInjectionAnchorForLanguage(
		language: ShaderRuleContext["language"],
		anchor: ShaderInjectionAnchor | undefined
	): ShaderInjectionAnchor {
		return language === "wgsl" ?
				normalizeWGSLInjectionAnchor(anchor)
			:	normalizeGLSLInjectionAnchor(anchor);
	}

	private _finalizeProcessResult(
		context: ShaderRuleContext,
		rawResult: ShaderProcessResult,
		perCallFilter: ShaderDiagnosticFilter | undefined,
		fromCache: boolean
	): ShaderProcessResult {
		const diagnostics = this._filterDiagnostics(rawResult.diagnostics, perCallFilter);
		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
		if (hasErrors && this._mode === "strict") {
			throw buildStrictModeError(
				context,
				diagnostics,
				this._strictErrorMaxDiagnostics
			);
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

	private _filterDiagnostics(
		diagnostics: ShaderDiagnostic[],
		perCallFilter: ShaderDiagnosticFilter | undefined
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
		fallbackSourcePath: string
	): ShaderDiagnostic {
		const generatedRange = normalizeDiagnosticRange(diagnostic.range) ?? null;
		const generatedLine =
			normalizePositiveInteger(diagnostic.line) ?? generatedRange?.start.line ?? null;
		const generatedColumn =
			normalizePositiveInteger(diagnostic.column) ??
			generatedRange?.start.column ??
			1;
		let resolvedLine = generatedLine ?? undefined;
		let resolvedColumn = generatedLine ? generatedColumn : undefined;
		let resolvedSourcePath =
			typeof diagnostic.sourcePath === "string" &&
			diagnostic.sourcePath.length > 0 ?
				diagnostic.sourcePath
			:	undefined;
		let resolvedRange =
			generatedRange ??
			(generatedLine ? createPointRange(generatedLine, generatedColumn) : undefined);

		if (generatedLine && sourceMap) {
			const mappedStart = mapShaderGeneratedLocation(
				sourceMap,
				generatedLine,
				generatedColumn
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
				resolvedRange.start.column
			);
			const mappedEnd = mapShaderGeneratedLocation(
				sourceMap,
				resolvedRange.end.line,
				resolvedRange.end.column
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
			typeof rule.priority === "number" && Number.isFinite(rule.priority) ?
				Math.floor(rule.priority)
			:	0;
		if (rule.match !== undefined && typeof rule.match !== "function") {
			throw new Error(`ShaderRuntime rule "${id}" match must be a function.`);
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
			throw new Error(
				`ShaderRuntime rule "${id}" cannot depend on itself in dependsOn.`
			);
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
		matchedRuleIds: readonly string[]
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
		].join("|");
	}

	private _getCachedResult(
		kind: ShaderRuntimeCacheKind,
		key: string
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
		participatingRuleIds: readonly string[]
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
		dynamicUserSymbols: Map<string, string>
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
					message:
						`Rule "${rule.id}" conflicts with reserved symbol "${symbol}" and was skipped.`,
				};
			}
			const staticOwner = this._findStaticUserSymbolOwner(symbol, rule.id);
			if (staticOwner) {
				return {
					ruleId: rule.id,
					code: "user-symbol-conflict",
					severity: this._mode === "strict" ? "error" : "warning",
					message:
						`Rule "${rule.id}" conflicts with user rule "${staticOwner}" on symbol "${symbol}" and was skipped.`,
				};
			}
			const dynamicOwner = dynamicUserSymbols.get(symbol);
			if (dynamicOwner && dynamicOwner !== rule.id) {
				return {
					ruleId: rule.id,
					code: "user-symbol-conflict",
					severity: this._mode === "strict" ? "error" : "warning",
					message:
						`Rule "${rule.id}" conflicts with injected symbol "${symbol}" from "${dynamicOwner}" and was skipped.`,
				};
			}
		}
		return null;
	}

	private _registerDynamicInjectionSymbols(
		rule: ShaderRule,
		injection: ShaderRuleInjection,
		dynamicUserSymbols: Map<string, string>
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

	private _findStaticUserSymbolOwner(
		symbol: string,
		excludeRuleId: string
	): string | null {
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

	private _collectRuleSymbols(
		rule: ShaderRule,
		injection: ShaderRuleInjection
	): string[] {
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
						`ShaderRuntime rule "${rule.id}" depends on missing rule "${dependencyId}".`
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
				`ShaderRuntime rule dependency cycle detected: ${unresolved.join(" -> ")}`
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
		userRules: Map<string, ShaderRule>
	): void {
		const ruleSymbols = normalizeSymbols(rule.symbols);
		for (const symbol of ruleSymbols) {
			for (const [otherRuleId, otherRule] of userRules) {
				if (otherRuleId === rule.id) {
					continue;
				}
				if (normalizeSymbols(otherRule.symbols).includes(symbol)) {
					throw new Error(
						`ShaderRuntime rule "${rule.id}" conflicts with rule "${otherRuleId}" on symbol "${symbol}".`
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

	private _getCacheMap(
		kind: ShaderRuntimeCacheKind
	): Map<string, CachedShaderProcessResult> {
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
			targets
		);
		const removedAsync = this._invalidateCacheEntriesByRuleIds(
			this._asyncProcessCache,
			this._asyncCacheStats,
			targets
		);
		return removedSync + removedAsync;
	}

	private _invalidateCacheEntriesByRuleIds(
		cache: Map<string, CachedShaderProcessResult>,
		stats: InternalCacheStats,
		targetRuleIds: Set<string>
	): number {
		let removed = 0;
		for (const [key, entry] of cache) {
			if (
				entry.participatingRuleIds.some((ruleId) => targetRuleIds.has(ruleId))
			) {
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
		} = {}
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
