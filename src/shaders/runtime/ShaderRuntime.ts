import {
	SHADER_RUNTIME_DEFAULT_CACHE_LIMIT,
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	resolveDefaultShaderRuntimeMode,
} from "./constants";
import { createBuiltInShaderRules } from "./builtins";
import {
	compressLineOriginsToSourceMap,
	composeCompositeShaderSources,
	expandSourceMapToLineOrigins,
	createInlineCompositeShaderSource,
	mapShaderGeneratedLocation,
	SOURCE_MAP_SCHEMA_VERSION,
	sliceCompositeShaderSource,
} from "./sourceMap";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticFilter,
	ShaderDiagnosticSeverity,
	ShaderDiagnosticRange,
	ShaderDirectivePreprocessResult,
	ShaderGLSLInjectionAnchor,
	ShaderInjectionArgValue,
	ShaderInjectionAnchor,
	ShaderInjectionScript,
	ShaderInjectionScriptContext,
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
	ShaderRuleTransformOutput,
	ShaderRuleTransformResolved,
	ShaderRuleTransformResult,
	ShaderRuntimeCacheKind,
	ShaderRuntimeCacheStats,
	ShaderRuntimeCacheStatsSnapshot,
	ShaderRuntimeChangeAction,
	ShaderRuntimeChangeEvent,
	ShaderRuntimeMode,
	ShaderSourceKind,
	ShaderSourceSegment,
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

interface LineOrigin {
	sourcePath: string;
	sourceLine: number;
	kind: "source" | "template" | "include" | "define-block" | "generated";
	label?: string;
}

interface PreprocessContext {
	request: ShaderProcessRequest;
	contextTemplate: Omit<ShaderRuleContext, "source">;
	mode: ShaderRuntimeMode;
	language: ShaderLanguage;
	sourcePath: string;
	diagnostics: ShaderDiagnostic[];
	macros: Map<string, MacroDefinition>;
	expandedModules: Set<string>;
	processingStack: string[];
}

interface PreprocessResult {
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
}

interface DirectiveLineScanState {
	inBlockComment: boolean;
	stringQuote: '"' | "'" | null;
	escape: boolean;
}

interface DirectiveLine {
	name: string;
	body: string;
	column: number;
	raw: string;
}

type IncludeSpecifierKind = "angle" | "quote";

interface IncludeSpecifier {
	kind: IncludeSpecifierKind;
	path: string;
}

interface InjectInvocation {
	id: string;
	args: Record<string, ShaderInjectionArgValue>;
}

type MacroDefinitionKind = "object" | "function";

interface BaseMacroDefinition {
	kind: MacroDefinitionKind;
	name: string;
	replacement: string;
	sourcePath: string;
	sourceLine: number;
}

interface ObjectMacroDefinition extends BaseMacroDefinition {
	kind: "object";
}

interface FunctionMacroDefinition extends BaseMacroDefinition {
	kind: "function";
	params: string[];
}

type MacroDefinition = ObjectMacroDefinition | FunctionMacroDefinition;

interface RegisteredIncludeModule {
	id: string;
	canonicalId: string;
	code: string;
	sourcePath: string;
}

interface ConditionalBranchState {
	parentActive: boolean;
	branchTaken: boolean;
	currentActive: boolean;
	elseSeen: boolean;
	sourcePath: string;
	sourceLine: number;
	column: number;
}

type ConditionTokenKind =
	| "eof"
	| "identifier"
	| "number"
	| "operator"
	| "leftParen"
	| "rightParen";

interface ConditionToken {
	kind: ConditionTokenKind;
	text: string;
	column: number;
}

interface DirectiveConditionParserOptions {
	expression: string;
	baseColumn: number;
	isDefined: (identifier: string) => boolean;
	resolveIdentifier: (identifier: string) => bigint;
}

type ShaderRuntimeChangeListener =
	| ((event: ShaderRuntimeChangeEvent) => void)
	| (() => void);

const DEFAULT_STRICT_ERROR_MAX_DIAGNOSTICS = 32;
const LARGE_SOURCE_THRESHOLD = 16 * 1024;
const LARGE_SOURCE_CHUNK_SIZE = 4 * 1024;
const DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH = 32;
const DIRECTIVE_CONDITIONAL_NAMES = new Set([
	"if",
	"ifdef",
	"ifndef",
	"elif",
	"else",
	"endif",
]);
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
		case "builtin-environment":
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
		schemaVersion: sourceMap.schemaVersion,
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
	const schemaVersion =
		typeof sourceMap.schemaVersion === "number" ?
			Math.floor(sourceMap.schemaVersion)
		:	1;
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

function isWhitespaceCharacter(char: string): boolean {
	return char === " " || char === "\t" || char === "\r" || char === "\n";
}

function isIdentifierStartCharacter(char: string): boolean {
	if (!char || char.length <= 0) {
		return false;
	}
	const code = char.charCodeAt(0);
	return (
		(code >= 65 && code <= 90) ||
		(code >= 97 && code <= 122) ||
		char === "_"
	);
}

function isIdentifierPartCharacter(char: string): boolean {
	if (!char || char.length <= 0) {
		return false;
	}
	const code = char.charCodeAt(0);
	return (
		(code >= 65 && code <= 90) ||
		(code >= 97 && code <= 122) ||
		(code >= 48 && code <= 57) ||
		char === "_"
	);
}

function isPromiseLike<T = unknown>(value: unknown): value is PromiseLike<T> {
	return (
		typeof value === "object" &&
		value !== null &&
		"then" in value &&
		typeof (value as { then?: unknown }).then === "function"
	);
}

class ConditionParseError extends Error {
	public column: number;

	public constructor(message: string, column: number) {
		super(message);
		this.name = "ConditionParseError";
		this.column = Math.max(1, Math.floor(column));
	}
}

class DirectiveConditionParser {
	private _expression: string;
	private _baseColumn: number;
	private _isDefined: (identifier: string) => boolean;
	private _resolveIdentifier: (identifier: string) => bigint;
	private _tokens: ConditionToken[];
	private _index = 0;

	public constructor(options: DirectiveConditionParserOptions) {
		this._expression = options.expression;
		this._baseColumn = options.baseColumn;
		this._isDefined = options.isDefined;
		this._resolveIdentifier = options.resolveIdentifier;
		this._tokens = this._tokenize();
	}

	public parse(): bigint {
		const value = this._parseLogicalOr();
		const token = this._peek();
		if (token.kind !== "eof") {
			throw new ConditionParseError(
				`Unexpected token "${token.text}" in directive expression.`,
				token.column
			);
		}
		return value;
	}

	private _parseLogicalOr(): bigint {
		let value = this._parseLogicalAnd();
		while (this._matchOperator("||")) {
			const right = this._parseLogicalAnd();
			value = value !== 0n || right !== 0n ? 1n : 0n;
		}
		return value;
	}

	private _parseLogicalAnd(): bigint {
		let value = this._parseEquality();
		while (this._matchOperator("&&")) {
			const right = this._parseEquality();
			value = value !== 0n && right !== 0n ? 1n : 0n;
		}
		return value;
	}

	private _parseEquality(): bigint {
		let value = this._parseRelational();
		while (true) {
			if (this._matchOperator("==")) {
				const right = this._parseRelational();
				value = value === right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator("!=")) {
				const right = this._parseRelational();
				value = value !== right ? 1n : 0n;
				continue;
			}
			return value;
		}
	}

	private _parseRelational(): bigint {
		let value = this._parseAdditive();
		while (true) {
			if (this._matchOperator("<")) {
				const right = this._parseAdditive();
				value = value < right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator(">")) {
				const right = this._parseAdditive();
				value = value > right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator("<=")) {
				const right = this._parseAdditive();
				value = value <= right ? 1n : 0n;
				continue;
			}
			if (this._matchOperator(">=")) {
				const right = this._parseAdditive();
				value = value >= right ? 1n : 0n;
				continue;
			}
			return value;
		}
	}

	private _parseAdditive(): bigint {
		let value = this._parseMultiplicative();
		while (true) {
			if (this._matchOperator("+")) {
				value += this._parseMultiplicative();
				continue;
			}
			if (this._matchOperator("-")) {
				value -= this._parseMultiplicative();
				continue;
			}
			return value;
		}
	}

	private _parseMultiplicative(): bigint {
		let value = this._parseUnary();
		while (true) {
			if (this._matchOperator("*")) {
				value *= this._parseUnary();
				continue;
			}
			if (this._matchOperator("/")) {
				const right = this._parseUnary();
				if (right === 0n) {
					throw new ConditionParseError(
						"Division by zero in directive condition expression.",
						this._previous().column
					);
				}
				value /= right;
				continue;
			}
			if (this._matchOperator("%")) {
				const right = this._parseUnary();
				if (right === 0n) {
					throw new ConditionParseError(
						"Modulo by zero in directive condition expression.",
						this._previous().column
					);
				}
				value %= right;
				continue;
			}
			return value;
		}
	}

	private _parseUnary(): bigint {
		if (this._matchOperator("!")) {
			const value = this._parseUnary();
			return value === 0n ? 1n : 0n;
		}
		if (this._matchOperator("-")) {
			return -this._parseUnary();
		}
		if (this._matchOperator("+")) {
			return this._parseUnary();
		}
		return this._parsePrimary();
	}

	private _parsePrimary(): bigint {
		const token = this._peek();
		if (token.kind === "number") {
			this._consume();
			return this._parseIntegerLiteral(token);
		}
		if (token.kind === "identifier") {
			this._consume();
			if (token.text === "defined") {
				return this._parseDefinedOperator(token.column);
			}
			return this._resolveIdentifier(token.text);
		}
		if (token.kind === "leftParen") {
			this._consume();
			const value = this._parseLogicalOr();
			const closing = this._peek();
			if (closing.kind !== "rightParen") {
				throw new ConditionParseError(
					`Expected ")" but got "${closing.text}".`,
					closing.column
				);
			}
			this._consume();
			return value;
		}
		throw new ConditionParseError(
			`Unexpected token "${token.text}" in directive expression.`,
			token.column
		);
	}

	private _parseDefinedOperator(operatorColumn: number): bigint {
		if (this._peek().kind === "leftParen") {
			this._consume();
			const identifier = this._peek();
			if (identifier.kind !== "identifier") {
				throw new ConditionParseError(
					`Expected identifier after "defined(" but got "${identifier.text}".`,
					identifier.column
				);
			}
			this._consume();
			const closing = this._peek();
			if (closing.kind !== "rightParen") {
				throw new ConditionParseError(
					`Expected ")" after "defined(${identifier.text}" but got "${closing.text}".`,
					closing.column
				);
			}
			this._consume();
			return this._isDefined(identifier.text) ? 1n : 0n;
		}
		const identifier = this._peek();
		if (identifier.kind !== "identifier") {
			throw new ConditionParseError(
				`Expected identifier after "defined" but got "${identifier.text}".`,
				identifier.column
			);
		}
		this._consume();
		if (identifier.column < operatorColumn) {
			throw new ConditionParseError(
				"Invalid defined() expression in directive condition.",
				operatorColumn
			);
		}
		return this._isDefined(identifier.text) ? 1n : 0n;
	}

	private _parseIntegerLiteral(token: ConditionToken): bigint {
		if (
			!/^(?:0[xX][0-9A-Fa-f]+|0[bB][01]+|0[oO][0-7]+|[0-9]+)$/.test(token.text)
		) {
			throw new ConditionParseError(
				`Invalid integer literal "${token.text}" in directive expression.`,
				token.column
			);
		}
		try {
			return BigInt(token.text);
		} catch {
			throw new ConditionParseError(
				`Invalid integer literal "${token.text}" in directive expression.`,
				token.column
			);
		}
	}

	private _peek(): ConditionToken {
		return this._tokens[this._index];
	}

	private _previous(): ConditionToken {
		return this._tokens[Math.max(0, this._index - 1)];
	}

	private _consume(): ConditionToken {
		const current = this._tokens[this._index];
		if (this._index < this._tokens.length - 1) {
			this._index++;
		}
		return current;
	}

	private _matchOperator(operator: string): boolean {
		const token = this._peek();
		if (token.kind !== "operator" || token.text !== operator) {
			return false;
		}
		this._consume();
		return true;
	}

	private _tokenize(): ConditionToken[] {
		const tokens: ConditionToken[] = [];
		const expression = this._expression;
		let index = 0;
		while (index < expression.length) {
			const char = expression[index];
			if (isWhitespaceCharacter(char)) {
				index++;
				continue;
			}
			const column = this._baseColumn + index;
			const twoChars = expression.slice(index, index + 2);
			if (
				twoChars === "&&" ||
				twoChars === "||" ||
				twoChars === "==" ||
				twoChars === "!=" ||
				twoChars === "<=" ||
				twoChars === ">="
			) {
				tokens.push({
					kind: "operator",
					text: twoChars,
					column,
				});
				index += 2;
				continue;
			}
			if (char === "(") {
				tokens.push({
					kind: "leftParen",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if (char === ")") {
				tokens.push({
					kind: "rightParen",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if ("!<>+-*/%".includes(char)) {
				tokens.push({
					kind: "operator",
					text: char,
					column,
				});
				index++;
				continue;
			}
			if (isIdentifierStartCharacter(char)) {
				let end = index + 1;
				while (end < expression.length && isIdentifierPartCharacter(expression[end])) {
					end++;
				}
				tokens.push({
					kind: "identifier",
					text: expression.slice(index, end),
					column,
				});
				index = end;
				continue;
			}
			if (char >= "0" && char <= "9") {
				let end = index + 1;
				while (
					end < expression.length &&
					/[A-Za-z0-9]/.test(expression[end] ?? "")
				) {
					end++;
				}
				tokens.push({
					kind: "number",
					text: expression.slice(index, end),
					column,
				});
				index = end;
				continue;
			}
			throw new ConditionParseError(
				`Unexpected character "${char}" in directive expression.`,
				column
			);
		}
		tokens.push({
			kind: "eof",
			text: "<eof>",
			column: this._baseColumn + expression.length,
		});
		return tokens;
	}
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

function countNewlinesUntilOffset(source: string, offset: number): number {
	if (source.length <= 0) {
		return 0;
	}
	const limit = Math.max(0, Math.min(source.length, Math.floor(offset)));
	let count = 0;
	for (let i = 0; i < limit; i++) {
		if (source.charCodeAt(i) === 10) {
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

function normalizeLanguage(language?: ShaderLanguage): ShaderLanguage {
	return language === "glsl" ? "glsl" : "wgsl";
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
	let lastBindingEndLine = 0;
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
	const preEntryLines =
		entryPointLine > 0 ? sourceLines.slice(0, entryPointLine - 1) : sourceLines;
	const preEntrySource = preEntryLines.join("\n");
	const bindingDeclarationPattern =
		/(?:@\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?\s*)*var(?:<[^>]+>)?\s+[A-Za-z_][A-Za-z0-9_]*\s*:[^;]*;/gms;
	let bindingMatch: RegExpExecArray | null = bindingDeclarationPattern.exec(
		preEntrySource
	);
	while (bindingMatch) {
		const declaration = bindingMatch[0];
		if (
			/@group\s*\([^)]*\)/.test(declaration) &&
			/@binding\s*\([^)]*\)/.test(declaration)
		) {
			const declarationEnd = (bindingMatch.index ?? 0) + declaration.length;
			lastBindingEndLine = Math.max(
				lastBindingEndLine,
				1 + countNewlinesUntilOffset(preEntrySource, declarationEnd)
			);
		}
		bindingMatch = bindingDeclarationPattern.exec(preEntrySource);
	}
	const lastStructEndLine = findLastStructEndLine(
		preEntryLines,
		/^\s*struct\b/
	);
	const afterEnableLine = lastEnableLine > 0 ? lastEnableLine + 1 : 1;
	const afterAliasesLine =
		lastAliasLine > 0 ? Math.max(lastAliasLine + 1, afterEnableLine) : afterEnableLine;
	const afterStructLine =
		lastStructEndLine > 0 ? Math.max(lastStructEndLine + 1, afterAliasesLine) : afterAliasesLine;
	const afterBindingsLine =
		lastBindingEndLine > 0 ?
			Math.max(lastBindingEndLine + 1, afterStructLine)
		:	afterStructLine;
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

function createGeneratedCompositeWithColumnSpans(
	code: string,
	sourcePath: string,
	label?: string
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
	private _includeModulesByLanguage: Map<
		ShaderLanguage,
		Map<string, RegisteredIncludeModule>
	>;
	private _injectionScripts: Map<string, ShaderInjectionScript>;
	private _directiveRegistryRevision: number;

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
		this._includeModulesByLanguage = new Map([
			["wgsl", new Map()],
			["glsl", new Map()],
		]);
		this._injectionScripts = new Map();
		this._directiveRegistryRevision = 1;

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

	public registerIncludeModule(
		language: ShaderLanguage,
		id: string,
		code: string,
		sourcePath?: string
	): void {
		const normalizedLanguage = normalizeLanguage(language);
		const normalizedId =
			typeof id === "string" ? id.trim().replace(/\\/g, "/") : "";
		if (normalizedId.length <= 0) {
			throw new Error("Shader include module id must be a non-empty string.");
		}
		if (typeof code !== "string") {
			throw new Error("Shader include module code must be a string.");
		}
		const canonicalId = this._canonicalizeModulePath(normalizedId);
		const languageModules =
			this._includeModulesByLanguage.get(normalizedLanguage) ?? new Map();
		const action: ShaderRuntimeChangeAction =
			languageModules.has(canonicalId) ?
				"update-include-module"
			:	"register-include-module";
		languageModules.set(canonicalId, {
			id: normalizedId,
			canonicalId,
			code,
			sourcePath:
				typeof sourcePath === "string" && sourcePath.trim().length > 0 ?
					sourcePath.trim()
				:	canonicalId,
		});
		this._includeModulesByLanguage.set(normalizedLanguage, languageModules);
		this._directiveRegistryRevision++;
		this._applyMutation(action, [], {
			invalidateAll: true,
			includeModuleIds: [this._formatIncludeModuleEventId(normalizedLanguage, canonicalId)],
		});
	}

	public unregisterIncludeModule(language: ShaderLanguage, id: string): boolean {
		const normalizedLanguage = normalizeLanguage(language);
		const normalizedId =
			typeof id === "string" ? id.trim().replace(/\\/g, "/") : "";
		if (normalizedId.length <= 0) {
			return false;
		}
		const canonicalId = this._canonicalizeModulePath(normalizedId);
		const languageModules =
			this._includeModulesByLanguage.get(normalizedLanguage) ?? null;
		if (!languageModules || !languageModules.has(canonicalId)) {
			return false;
		}
		languageModules.delete(canonicalId);
		this._directiveRegistryRevision++;
		this._applyMutation("unregister-include-module", [], {
			invalidateAll: true,
			includeModuleIds: [this._formatIncludeModuleEventId(normalizedLanguage, canonicalId)],
		});
		return true;
	}

	public clearIncludeModules(language?: ShaderLanguage): void {
		if (!language) {
			const ids: string[] = [];
			for (const [lang, modules] of this._includeModulesByLanguage) {
				for (const moduleId of modules.keys()) {
					ids.push(this._formatIncludeModuleEventId(lang, moduleId));
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
		const languageModules =
			this._includeModulesByLanguage.get(normalizedLanguage) ?? null;
		if (!languageModules || languageModules.size <= 0) {
			return;
		}
		const ids = [...languageModules.keys()].map((moduleId) =>
			this._formatIncludeModuleEventId(normalizedLanguage, moduleId)
		);
		languageModules.clear();
		this._directiveRegistryRevision++;
		this._applyMutation("clear-include-modules", [], {
			invalidateAll: true,
			includeModuleIds: ids,
		});
	}

	public registerInjectionScript(script: ShaderInjectionScript): void {
		const normalized = this._normalizeInjectionScript(script);
		const action: ShaderRuntimeChangeAction =
			this._injectionScripts.has(normalized.id) ?
				"update-injection-script"
			:	"register-injection-script";
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
		const sourcePath = this._resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const baseComposite =
			request.enableDirectives === false ?
				initialComposite
			:	this._preprocessDirectivesSync(request, initialComposite).composite;
		if (normalizeLanguage(request.language) === "glsl") {
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

	public preprocessDirectives(
		request: ShaderProcessRequest
	): ShaderDirectivePreprocessResult {
		const sourcePath = this._resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const preprocessed = this._preprocessDirectivesSync(request, initialComposite);
		return this._finalizeDirectivePreprocessResult(request, preprocessed);
	}

	public async preprocessDirectivesAsync(
		request: ShaderProcessRequest
	): Promise<ShaderDirectivePreprocessResult> {
		const sourcePath = this._resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const preprocessed = await this._preprocessDirectivesAsync(
			request,
			initialComposite
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

	private _resolveRequestSourcePath(request: ShaderProcessRequest): string {
		const explicitDirectivePath =
			typeof request.directiveSourcePath === "string" ?
				request.directiveSourcePath.trim()
			:	"";
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
				request.sourceKind
			)}>`
		);
	}

	private _createPreprocessContext(
		request: ShaderProcessRequest,
		sourcePath: string
	): PreprocessContext {
		const language = normalizeLanguage(request.language);
		return {
			request,
			mode: this._mode,
			language,
			sourcePath,
			contextTemplate: {
				mode: this._mode,
				language,
				stage: normalizeStage(request.stage),
				entryPoint: request.entryPoint ?? null,
				label: request.label ?? null,
				sourceKind: normalizeSourceKind(request.sourceKind),
			},
			diagnostics: [],
			macros: new Map(),
			expandedModules: new Set(),
			processingStack: [],
		};
	}

	private _preprocessDirectivesSync(
		request: ShaderProcessRequest,
		initialComposite: CompositeShaderSource
	): PreprocessResult {
		if (request.enableDirectives === false) {
			return {
				composite: initialComposite,
				diagnostics: [],
			};
		}
		const sourcePath =
			initialComposite.sourceMap.segments[0]?.sourcePath ??
			this._resolveRequestSourcePath(request);
		const preprocessContext = this._createPreprocessContext(request, sourcePath);
		const expanded = this._expandDirectiveComposite(
			initialComposite,
			this._canonicalizeModulePathSafe(sourcePath),
			preprocessContext
		);
		const macroExpanded = this._expandMacrosInComposite(expanded, preprocessContext);
		const injected = this._resolveDirectiveInjectsSync(
			macroExpanded,
			preprocessContext
		);
		return {
			composite: injected,
			diagnostics: [...preprocessContext.diagnostics],
		};
	}

	private async _preprocessDirectivesAsync(
		request: ShaderProcessRequest,
		initialComposite: CompositeShaderSource
	): Promise<PreprocessResult> {
		if (request.enableDirectives === false) {
			return {
				composite: initialComposite,
				diagnostics: [],
			};
		}
		const sourcePath =
			initialComposite.sourceMap.segments[0]?.sourcePath ??
			this._resolveRequestSourcePath(request);
		const preprocessContext = this._createPreprocessContext(request, sourcePath);
		const expanded = this._expandDirectiveComposite(
			initialComposite,
			this._canonicalizeModulePathSafe(sourcePath),
			preprocessContext
		);
		const macroExpanded = this._expandMacrosInComposite(expanded, preprocessContext);
		const injected = await this._resolveDirectiveInjectsAsync(
			macroExpanded,
			preprocessContext
		);
		return {
			composite: injected,
			diagnostics: [...preprocessContext.diagnostics],
		};
	}

	private _splitCompositeLines(composite: CompositeShaderSource): {
		lines: string[];
		origins: LineOrigin[];
	} {
		const lines = composite.code.split(/\r?\n/g);
		const fallbackPath = composite.sourceMap.segments[0]?.sourcePath ?? "<generated>";
		const origins = expandSourceMapToLineOrigins(
			composite.sourceMap,
			lines.length,
			fallbackPath,
			"source"
		) as LineOrigin[];
		return { lines, origins };
	}

	private _composeLinesToComposite(
		lines: string[],
		origins: LineOrigin[]
	): CompositeShaderSource {
		const effectiveLines = lines.length > 0 ? lines : [""];
		const effectiveOrigins =
			origins.length > 0 ?
				origins
			:	[
					{
						sourcePath: "<generated>",
						sourceLine: 1,
						kind: "generated" as const,
					},
				];
		return {
			code: effectiveLines.join("\n"),
			sourceMap: compressLineOriginsToSourceMap(
				effectiveOrigins as unknown as Array<{
					sourcePath: string;
					sourceLine: number;
					kind: "source" | "template" | "include" | "define-block" | "generated";
					label?: string;
				}>
			),
		};
	}

	private _scanDirectiveFromLine(
		line: string,
		state: DirectiveLineScanState
	): DirectiveLine | null {
		let index = 0;
		while (index < line.length && (line[index] === " " || line[index] === "\t")) {
			index++;
		}
		if (index >= line.length || line[index] !== "#") {
			this._updateDirectiveStateFromLine(line, state);
			return null;
		}
		if (state.inBlockComment || state.stringQuote) {
			this._updateDirectiveStateFromLine(line, state);
			return null;
		}
		const raw = line.slice(index + 1).trim();
		const match = /^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(.*))?$/.exec(raw);
		this._updateDirectiveStateFromLine(line, state);
		if (!match) {
			return null;
		}
		return {
			name: match[1].toLowerCase(),
			body: (match[2] ?? "").trim(),
			column: index + 1,
			raw: line.trim(),
		};
	}

	private _updateDirectiveStateFromLine(
		line: string,
		state: DirectiveLineScanState
	): void {
		for (let i = 0; i < line.length; i++) {
			const char = line[i];
			const next = i + 1 < line.length ? line[i + 1] : "";
			if (state.inBlockComment) {
				if (char === "*" && next === "/") {
					state.inBlockComment = false;
					i++;
				}
				continue;
			}
			if (state.stringQuote) {
				if (state.escape) {
					state.escape = false;
					continue;
				}
				if (char === "\\") {
					state.escape = true;
					continue;
				}
				if (char === state.stringQuote) {
					state.stringQuote = null;
					continue;
				}
				continue;
			}
			if (char === "/" && next === "/") {
				break;
			}
			if (char === "/" && next === "*") {
				state.inBlockComment = true;
				i++;
				continue;
			}
			if (char === "\"" || char === "'") {
				state.stringQuote = char as '"' | "'";
				state.escape = false;
			}
		}
		if (state.stringQuote && !state.inBlockComment) {
			state.escape = false;
		}
	}

	private _expandDirectiveComposite(
		composite: CompositeShaderSource,
		modulePath: string,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		const conditionalStack: ConditionalBranchState[] = [];
		const firstVersionLine = this._findFirstGLSLVersionLine(lines);
		const isBranchActive = (): boolean =>
			conditionalStack.length <= 0 ?
				true
			:	conditionalStack[conditionalStack.length - 1].currentActive;

		for (let index = 0; index < lines.length; index++) {
			const lineNumber = index + 1;
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: modulePath,
				sourceLine: lineNumber,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive) {
				if (isBranchActive()) {
					outputLines.push(line);
					outputOrigins.push(origin);
				}
				continue;
			}
			if (DIRECTIVE_CONDITIONAL_NAMES.has(directive.name)) {
				this._applyConditionalDirective(
					directive,
					origin,
					preprocessContext,
					conditionalStack
				);
				continue;
			}
			if (!isBranchActive()) {
				continue;
			}
			if (
				preprocessContext.language === "glsl" &&
				firstVersionLine > 0 &&
				lineNumber < firstVersionLine &&
				(directive.name === "include" || directive.name === "import")
			) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-include-before-version",
					`Directive "#${directive.name}" appears before "#version" and was skipped.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			switch (directive.name) {
				case "include":
				case "import": {
					const specifier = this._parseIncludeSpecifier(
						directive,
						origin,
						preprocessContext
					);
					if (!specifier) {
						continue;
					}
					if (directive.name === "import" && specifier.kind !== "angle") {
						this._pushDirectiveDiagnostic(
							preprocessContext,
							"directive-import-invalid-path",
							`Directive "#import" only supports angle-bracket paths.`,
							origin.sourcePath,
							origin.sourceLine,
							directive.column
						);
						continue;
					}
					const includeComposite = this._resolveIncludeComposite(
						specifier,
						modulePath,
						preprocessContext,
						origin
					);
					if (!includeComposite) {
						continue;
					}
					const includeLines = includeComposite.code.split(/\r?\n/g);
					const includeOrigins = expandSourceMapToLineOrigins(
						includeComposite.sourceMap,
						includeLines.length,
						includeComposite.sourceMap.segments[0]?.sourcePath ?? modulePath,
						"include"
					) as LineOrigin[];
					for (let includeIndex = 0; includeIndex < includeLines.length; includeIndex++) {
						outputLines.push(includeLines[includeIndex]);
						outputOrigins.push(
							includeOrigins[includeIndex] ?? {
								sourcePath: modulePath,
								sourceLine: lineNumber,
								kind: "include",
							}
						);
					}
					continue;
				}
				case "define": {
					const macro = this._parseMacroDefinition(
						directive,
						origin,
						preprocessContext
					);
					if (!macro) {
						continue;
					}
					if (preprocessContext.macros.has(macro.name)) {
						this._pushDirectiveDiagnostic(
							preprocessContext,
							"directive-define-redefined",
							`Macro "${macro.name}" was redefined; latest definition wins.`,
							origin.sourcePath,
							origin.sourceLine,
							directive.column,
							"warning"
						);
					}
						preprocessContext.macros.set(macro.name, macro);
						continue;
					}
					case "undef": {
						const macroName = this._parseSingleDirectiveIdentifier(
							directive,
							origin,
							preprocessContext,
							"directive-undef-invalid"
						);
						if (!macroName) {
							continue;
						}
						preprocessContext.macros.delete(macroName);
						continue;
					}
					case "inject":
						outputLines.push(line);
						outputOrigins.push(origin);
						continue;
					default:
						outputLines.push(line);
						outputOrigins.push(origin);
				}
			}
		if (conditionalStack.length > 0) {
			for (const state of conditionalStack) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-conditional-unterminated",
					`Directive conditional block starting at "${state.sourcePath}:${state.sourceLine}" was not terminated with "#endif".`,
					state.sourcePath,
					state.sourceLine,
					state.column
				);
			}
		}

			return this._composeLinesToComposite(outputLines, outputOrigins);
		}

	private _applyConditionalDirective(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext,
		stack: ConditionalBranchState[]
	): void {
		switch (directive.name) {
			case "if": {
				const parentActive =
					stack.length <= 0 ? true : stack[stack.length - 1].currentActive;
				const branchValue =
					parentActive ?
						this._evaluateDirectiveConditionExpression(
							directive.body,
							directive,
							origin,
							preprocessContext
						)
					:	false;
				stack.push({
					parentActive,
					branchTaken: parentActive && branchValue,
					currentActive: parentActive && branchValue,
					elseSeen: false,
					sourcePath: origin.sourcePath,
					sourceLine: origin.sourceLine,
					column: directive.column,
				});
				return;
			}
			case "ifdef":
			case "ifndef": {
				const parentActive =
					stack.length <= 0 ? true : stack[stack.length - 1].currentActive;
				if (!parentActive) {
					stack.push({
						parentActive,
						branchTaken: false,
						currentActive: false,
						elseSeen: false,
						sourcePath: origin.sourcePath,
						sourceLine: origin.sourceLine,
						column: directive.column,
					});
					return;
				}
				const identifier = this._parseSingleDirectiveIdentifier(
					directive,
					origin,
					preprocessContext,
					"directive-conditional-invalid-identifier"
				);
				const branchValue =
					identifier ?
						directive.name === "ifdef" ?
							preprocessContext.macros.has(identifier)
						:	!preprocessContext.macros.has(identifier)
					:	false;
				stack.push({
					parentActive,
					branchTaken: parentActive && branchValue,
					currentActive: parentActive && branchValue,
					elseSeen: false,
					sourcePath: origin.sourcePath,
					sourceLine: origin.sourceLine,
					column: directive.column,
				});
				return;
			}
			case "elif": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-elif-without-if",
						`Directive "#elif" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				const branch = stack[stack.length - 1];
				if (branch.elseSeen) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-elif-after-else",
						`Directive "#elif" cannot appear after "#else" in the same conditional block.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					branch.currentActive = false;
					return;
				}
				if (!branch.parentActive || branch.branchTaken) {
					branch.currentActive = false;
					return;
				}
				const value = this._evaluateDirectiveConditionExpression(
					directive.body,
					directive,
					origin,
					preprocessContext
				);
				branch.currentActive = branch.parentActive && value;
				if (branch.currentActive) {
					branch.branchTaken = true;
				}
				return;
			}
			case "else": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-else-without-if",
						`Directive "#else" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				const branch = stack[stack.length - 1];
				if (branch.elseSeen) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-else-duplicate",
						`Directive "#else" can appear only once per conditional block.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					branch.currentActive = false;
					return;
				}
				branch.elseSeen = true;
				branch.currentActive = branch.parentActive && !branch.branchTaken;
				if (branch.currentActive) {
					branch.branchTaken = true;
				}
				return;
			}
			case "endif": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-endif-without-if",
						`Directive "#endif" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				stack.pop();
				return;
			}
		}
	}

	private _parseSingleDirectiveIdentifier(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext,
		errorCode: string
	): string | null {
		const body = directive.body.trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(body)) {
			return body;
		}
		this._pushDirectiveDiagnostic(
			preprocessContext,
			errorCode,
			`Directive "#${directive.name}" expects a single macro identifier.`,
			origin.sourcePath,
			origin.sourceLine,
			directive.column
		);
		return null;
	}

	private _evaluateDirectiveConditionExpression(
		expression: string,
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): boolean {
		const trimmedExpression = expression.trim();
		if (trimmedExpression.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-conditional-expression-invalid",
				`Directive "#${directive.name}" requires a condition expression.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return false;
		}
		const resolverStack = new Set<string>();
		const baseColumn = directive.column + directive.name.length + 2;
		try {
			const parser = new DirectiveConditionParser({
				expression: trimmedExpression,
				baseColumn,
				isDefined: (identifier) => preprocessContext.macros.has(identifier),
				resolveIdentifier: (identifier) =>
					this._resolveDirectiveConditionIdentifier(
						identifier,
						preprocessContext,
						origin,
						baseColumn,
						resolverStack
					),
			});
			return parser.parse() !== 0n;
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "Invalid directive condition expression.";
			const column =
				error instanceof ConditionParseError ? error.column : directive.column;
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-conditional-expression-invalid",
				message,
				origin.sourcePath,
				origin.sourceLine,
				column
			);
			return false;
		}
	}

	private _resolveDirectiveConditionIdentifier(
		identifier: string,
		preprocessContext: PreprocessContext,
		origin: LineOrigin,
		column: number,
		resolverStack: Set<string>
	): bigint {
		if (resolverStack.has(identifier)) {
			return 0n;
		}
		const macro = preprocessContext.macros.get(identifier);
		if (!macro) {
			return 0n;
		}
		if (macro.kind === "function") {
			return 0n;
		}
		const expanded = this._expandMacroText(
			macro.replacement,
			preprocessContext,
			origin.sourcePath,
			origin.sourceLine,
			1
		).trim();
		if (expanded.length <= 0) {
			return 0n;
		}
		resolverStack.add(identifier);
		try {
			const parser = new DirectiveConditionParser({
				expression: expanded,
				baseColumn: column,
				isDefined: (name) => preprocessContext.macros.has(name),
				resolveIdentifier: (name) =>
					this._resolveDirectiveConditionIdentifier(
						name,
						preprocessContext,
						origin,
						column,
						resolverStack
					),
			});
			return parser.parse();
		} finally {
			resolverStack.delete(identifier);
		}
	}

	private _findFirstGLSLVersionLine(lines: string[]): number {
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i].trim();
			if (line.length <= 0) {
				continue;
			}
			if (line.startsWith("#version")) {
				return i + 1;
			}
			return 0;
		}
		return 0;
	}

	private _parseIncludeSpecifier(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): IncludeSpecifier | null {
		const body = directive.body.trim();
		if (body.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-empty",
				`Directive "#${directive.name}" requires a module path.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		if (body.startsWith("<") && body.endsWith(">")) {
			return {
				kind: "angle",
				path: body.slice(1, -1).trim(),
			};
		}
		if (body.startsWith("\"") && body.endsWith("\"")) {
			return {
				kind: "quote",
				path: body.slice(1, -1),
			};
		}
		this._pushDirectiveDiagnostic(
			preprocessContext,
			"directive-include-invalid-path",
			`Directive "#${directive.name}" expects <path> or "path".`,
			origin.sourcePath,
			origin.sourceLine,
			directive.column
		);
		return null;
	}

	private _resolveIncludeComposite(
		specifier: IncludeSpecifier,
		currentModulePath: string,
		preprocessContext: PreprocessContext,
		origin: LineOrigin
	): CompositeShaderSource | null {
		const canonicalModuleId = this._resolveIncludeModuleId(
			specifier,
			currentModulePath,
			preprocessContext.language
		);
		if (!canonicalModuleId) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-invalid-target",
				`Include target "${specifier.path}" is invalid.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		if (preprocessContext.processingStack.includes(canonicalModuleId)) {
			const chain = [...preprocessContext.processingStack, canonicalModuleId].join(
				" -> "
			);
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-cycle",
				`Detected cyclic include/import chain: ${chain}.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		if (preprocessContext.expandedModules.has(canonicalModuleId)) {
			return null;
		}
		const module = this._resolveRegisteredIncludeModule(
			preprocessContext.language,
			canonicalModuleId
		);
		if (!module) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-not-found",
				`Include module "${canonicalModuleId}" was not registered for ${preprocessContext.language}.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		preprocessContext.processingStack.push(canonicalModuleId);
		preprocessContext.expandedModules.add(canonicalModuleId);
		const expanded = this._expandDirectiveComposite(
			createInlineCompositeShaderSource(
				module.code,
				module.sourcePath,
				"include"
			),
			canonicalModuleId,
			preprocessContext
		);
		preprocessContext.processingStack.pop();
		return expanded;
	}

	private _resolveIncludeModuleId(
		specifier: IncludeSpecifier,
		currentModulePath: string,
		language: ShaderLanguage
	): string | null {
		const normalizedPath = specifier.path.replace(/\\/g, "/").trim();
		if (normalizedPath.length <= 0) {
			return null;
		}
		if (specifier.kind === "angle") {
			const canonical = this._canonicalizeModulePathSafe(normalizedPath);
			if (!canonical) {
				return null;
			}
			if (this._resolveRegisteredIncludeModule(language, canonical)) {
				return canonical;
			}
			const withExtension = this._withLanguageDefaultExtension(
				canonical,
				language
			);
			if (this._resolveRegisteredIncludeModule(language, withExtension)) {
				return withExtension;
			}
			return withExtension;
		}
		const joined = this._joinModulePath(currentModulePath, normalizedPath);
		const canonical = this._canonicalizeModulePathSafe(joined);
		if (!canonical) {
			return null;
		}
		if (this._resolveRegisteredIncludeModule(language, canonical)) {
			return canonical;
		}
		const withExtension = this._withLanguageDefaultExtension(canonical, language);
		if (this._resolveRegisteredIncludeModule(language, withExtension)) {
			return withExtension;
		}
		return withExtension;
	}

	private _resolveRegisteredIncludeModule(
		language: ShaderLanguage,
		moduleId: string
	): RegisteredIncludeModule | null {
		const modules = this._includeModulesByLanguage.get(language);
		if (!modules) {
			return null;
		}
		return modules.get(moduleId) ?? null;
	}

	private _withLanguageDefaultExtension(
		moduleId: string,
		language: ShaderLanguage
	): string {
		const slashIndex = moduleId.lastIndexOf("/");
		const fileName = slashIndex >= 0 ? moduleId.slice(slashIndex + 1) : moduleId;
		if (fileName.includes(".")) {
			return moduleId;
		}
		return `${moduleId}.${language === "wgsl" ? "wgsl" : "glsl"}`;
	}

	private _joinModulePath(baseModulePath: string, relativePath: string): string {
		const base = baseModulePath.replace(/\\/g, "/");
		const slashIndex = base.lastIndexOf("/");
		const directory = slashIndex >= 0 ? base.slice(0, slashIndex) : "";
		return directory.length > 0 ? `${directory}/${relativePath}` : relativePath;
	}

	private _canonicalizeModulePathSafe(value: string): string {
		try {
			return this._canonicalizeModulePath(value);
		} catch (error) {
			return value
				.replace(/\\/g, "/")
				.replace(/^\/+/, "")
				.replace(/\/{2,}/g, "/")
				.trim();
		}
	}

	private _canonicalizeModulePath(value: string): string {
		const normalized = value.replace(/\\/g, "/").trim();
		if (normalized.length <= 0) {
			throw new Error("Shader module path cannot be empty.");
		}
		const segments: string[] = [];
		for (const rawSegment of normalized.split("/")) {
			const segment = rawSegment.trim();
			if (segment.length <= 0 || segment === ".") {
				continue;
			}
			if (segment === "..") {
				if (segments.length <= 0) {
					throw new Error(
						`Shader module path "${value}" escapes outside include root.`
					);
				}
				segments.pop();
				continue;
			}
			segments.push(segment);
		}
		if (segments.length <= 0) {
			throw new Error(`Shader module path "${value}" resolved to an empty path.`);
		}
		return segments.join("/");
	}

	private _formatIncludeModuleEventId(
		language: ShaderLanguage,
		moduleId: string
	): string {
		return `${language}:${moduleId}`;
	}

	private _pushDirectiveDiagnostic(
		preprocessContext: PreprocessContext,
		code: string,
		message: string,
		sourcePath: string,
		line: number,
		column: number,
		overrideSeverity?: ShaderDiagnosticSeverity
	): void {
		preprocessContext.diagnostics.push({
			ruleId: "ignis/directive-runtime",
			code,
			severity:
				overrideSeverity ?? this._resolveDirectiveSeverityByMode(preprocessContext.mode),
			message,
			sourcePath,
			line: Math.max(1, Math.floor(line)),
			column: Math.max(1, Math.floor(column)),
			range: createPointRange(
				Math.max(1, Math.floor(line)),
				Math.max(1, Math.floor(column))
			),
		});
	}

	private _resolveDirectiveSeverityByMode(
		mode: ShaderRuntimeMode
	): ShaderDiagnosticSeverity {
		return mode === "strict" ? "error" : "warning";
	}

	private _parseMacroDefinition(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): MacroDefinition | null {
		const body = directive.body;
		if (body.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-define-invalid",
				`Directive "#define" requires a macro name.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const match = /^([A-Za-z_][A-Za-z0-9_]*)(.*)$/.exec(body);
		if (!match) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-define-invalid",
				`Directive "#define" has invalid syntax.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const macroName = match[1];
		const remainder = match[2] ?? "";
		if (remainder.startsWith("(")) {
			const closeIndex = remainder.indexOf(")");
			if (closeIndex < 0) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-define-function-invalid",
					`Function macro "${macroName}" is missing closing ")".`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				return null;
			}
			const parameterList = remainder.slice(1, closeIndex).trim();
			const params =
				parameterList.length <= 0 ?
					[]
				:	parameterList
						.split(",")
						.map((parameter) => parameter.trim())
						.filter((parameter) => parameter.length > 0);
			for (const parameter of params) {
				if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(parameter)) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-define-function-param-invalid",
						`Function macro "${macroName}" has invalid parameter "${parameter}".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return null;
				}
			}
			const replacement = remainder.slice(closeIndex + 1).trimStart();
			return {
				kind: "function",
				name: macroName,
				params,
				replacement,
				sourcePath: origin.sourcePath,
				sourceLine: origin.sourceLine,
			};
		}
		return {
			kind: "object",
			name: macroName,
			replacement: remainder.trimStart(),
			sourcePath: origin.sourcePath,
			sourceLine: origin.sourceLine,
		};
	}

	private _expandMacrosInComposite(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		if (preprocessContext.macros.size <= 0) {
			return composite;
		}
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const macroState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			outputLines.push(
				this._expandMacrosInLine(
					lines[index],
					macroState,
					preprocessContext,
					origin.sourcePath,
					origin.sourceLine
				)
			);
		}
		return this._composeLinesToComposite(outputLines, origins);
	}

	private _expandMacrosInLine(
		line: string,
		state: DirectiveLineScanState,
		preprocessContext: PreprocessContext,
		sourcePath: string,
		sourceLine: number
	): string {
		let output = "";
		let index = 0;
		while (index < line.length) {
			const char = line[index];
			const next = index + 1 < line.length ? line[index + 1] : "";
			if (state.inBlockComment) {
				output += char;
				if (char === "*" && next === "/") {
					output += "/";
					state.inBlockComment = false;
					index += 2;
					continue;
				}
				index++;
				continue;
			}
			if (state.stringQuote) {
				output += char;
				if (state.escape) {
					state.escape = false;
					index++;
					continue;
				}
				if (char === "\\") {
					state.escape = true;
					index++;
					continue;
				}
				if (char === state.stringQuote) {
					state.stringQuote = null;
					index++;
					continue;
				}
				index++;
				continue;
			}
			if (char === "/" && next === "/") {
				output += line.slice(index);
				break;
			}
			if (char === "/" && next === "*") {
				output += "/*";
				state.inBlockComment = true;
				index += 2;
				continue;
			}
			if (char === "\"" || char === "'") {
				output += char;
				state.stringQuote = char as '"' | "'";
				state.escape = false;
				index++;
				continue;
			}
			if (isIdentifierStartCharacter(char)) {
				let end = index + 1;
				while (
					end < line.length &&
					isIdentifierPartCharacter(line[end])
				) {
					end++;
				}
				const token = line.slice(index, end);
				const macro = preprocessContext.macros.get(token);
				if (!macro) {
					output += token;
					index = end;
					continue;
				}
				if (macro.kind === "object") {
					output += this._expandMacroText(
						macro.replacement,
						preprocessContext,
						sourcePath,
						sourceLine,
						1
					);
					index = end;
					continue;
				}
				const invocation = this._parseFunctionMacroInvocation(line, end);
				if (!invocation) {
					output += token;
					index = end;
					continue;
				}
				if (invocation.args.length !== macro.params.length) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-macro-arg-count",
						`Macro "${macro.name}" expected ${macro.params.length} argument(s) but got ${invocation.args.length}.`,
						sourcePath,
						sourceLine,
						index + 1,
						"warning"
					);
				}
				const substituted = this._substituteFunctionMacro(
					macro,
					invocation.args.map((argument) =>
						this._expandMacroText(
							argument.trim(),
							preprocessContext,
							sourcePath,
							sourceLine,
							1
						)
					)
				);
				output += this._expandMacroText(
					substituted,
					preprocessContext,
					sourcePath,
					sourceLine,
					1
				);
				index = invocation.endIndex + 1;
				continue;
			}
			output += char;
			index++;
		}
		return output;
	}

	private _expandMacroText(
		text: string,
		preprocessContext: PreprocessContext,
		sourcePath: string,
		sourceLine: number,
		depth: number
	): string {
		if (text.length <= 0 || preprocessContext.macros.size <= 0) {
			return text;
		}
		if (depth > DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-macro-depth-limit",
				`Macro expansion exceeded maximum depth (${DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH}).`,
				sourcePath,
				sourceLine,
				1,
				"warning"
			);
			return text;
		}
		let output = "";
		let index = 0;
		while (index < text.length) {
			const char = text[index];
			if (!isIdentifierStartCharacter(char)) {
				output += char;
				index++;
				continue;
			}
			let end = index + 1;
			while (end < text.length && isIdentifierPartCharacter(text[end])) {
				end++;
			}
			const token = text.slice(index, end);
			const macro = preprocessContext.macros.get(token);
			if (!macro) {
				output += token;
				index = end;
				continue;
			}
			if (macro.kind === "object") {
				output += this._expandMacroText(
					macro.replacement,
					preprocessContext,
					sourcePath,
					sourceLine,
					depth + 1
				);
				index = end;
				continue;
			}
			output += token;
			index = end;
		}
		return output;
	}

	private _parseFunctionMacroInvocation(
		line: string,
		identifierEndIndex: number
	): { args: string[]; endIndex: number } | null {
		let index = identifierEndIndex;
		while (index < line.length && isWhitespaceCharacter(line[index])) {
			index++;
		}
		if (index >= line.length || line[index] !== "(") {
			return null;
		}
		let depth = 1;
		let cursor = index + 1;
		let current = "";
		const args: string[] = [];
		let stringQuote: '"' | "'" | null = null;
		let escape = false;
		while (cursor < line.length) {
			const char = line[cursor];
			if (stringQuote) {
				current += char;
				if (escape) {
					escape = false;
					cursor++;
					continue;
				}
				if (char === "\\") {
					escape = true;
					cursor++;
					continue;
				}
				if (char === stringQuote) {
					stringQuote = null;
				}
				cursor++;
				continue;
			}
			if (char === "\"" || char === "'") {
				current += char;
				stringQuote = char as '"' | "'";
				escape = false;
				cursor++;
				continue;
			}
			if (char === "(") {
				depth++;
				current += char;
				cursor++;
				continue;
			}
			if (char === ")") {
				depth--;
				if (depth === 0) {
					if (current.trim().length > 0 || args.length > 0) {
						args.push(current.trim());
					}
					return {
						args,
						endIndex: cursor,
					};
				}
				current += char;
				cursor++;
				continue;
			}
			if (char === "," && depth === 1) {
				args.push(current.trim());
				current = "";
				cursor++;
				continue;
			}
			current += char;
			cursor++;
		}
		return null;
	}

	private _substituteFunctionMacro(
		macro: FunctionMacroDefinition,
		args: string[]
	): string {
		const parameterMap = new Map<string, string>();
		for (let index = 0; index < macro.params.length; index++) {
			parameterMap.set(macro.params[index], args[index] ?? "");
		}
		let output = "";
		let cursor = 0;
		while (cursor < macro.replacement.length) {
			const char = macro.replacement[cursor];
			if (!isIdentifierStartCharacter(char)) {
				output += char;
				cursor++;
				continue;
			}
			let end = cursor + 1;
			while (
				end < macro.replacement.length &&
				isIdentifierPartCharacter(macro.replacement[end])
			) {
				end++;
			}
			const token = macro.replacement.slice(cursor, end);
			output += parameterMap.get(token) ?? token;
			cursor = end;
		}
		return output;
	}

	private _resolveDirectiveInjectsSync(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const headerBlocks: InjectionBlock[] = [];
		const functionBlocks: InjectionBlock[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive || directive.name !== "inject") {
				outputLines.push(line);
				outputOrigins.push(origin);
				continue;
			}
			const invocation = this._parseInjectInvocation(
				directive,
				preprocessContext,
				origin
			);
			if (!invocation) {
				continue;
			}
			const script = this._injectionScripts.get(invocation.id);
			if (!script) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-not-found",
					`Injection script "${invocation.id}" was not registered.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			if (script.language && script.language !== preprocessContext.language) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-language-mismatch",
					`Injection script "${invocation.id}" does not support ${preprocessContext.language}.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			const scriptContext = this._createInjectionScriptContext(
				preprocessContext,
				composite.code
			);
			const injection = script.run(invocation.args, scriptContext);
			if (isPromiseLike(injection)) {
				throw new Error(
					`Injection script "${script.id}" returned a Promise during process(). Use processAsync().`
				);
			}
			this._appendDirectiveInjectionBlocks(
				preprocessContext,
				script,
				injection,
				headerBlocks,
				functionBlocks
			);
		}
		const baseComposite = this._composeLinesToComposite(outputLines, outputOrigins);
		const mergedBlocks = [...headerBlocks, ...functionBlocks];
		if (mergedBlocks.length <= 0) {
			return baseComposite;
		}
		return preprocessContext.language === "wgsl" ?
				injectWGSLSource(baseComposite, mergedBlocks)
			:	injectGLSLSource(baseComposite, mergedBlocks);
	}

	private async _resolveDirectiveInjectsAsync(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): Promise<CompositeShaderSource> {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const headerBlocks: InjectionBlock[] = [];
		const functionBlocks: InjectionBlock[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive || directive.name !== "inject") {
				outputLines.push(line);
				outputOrigins.push(origin);
				continue;
			}
			const invocation = this._parseInjectInvocation(
				directive,
				preprocessContext,
				origin
			);
			if (!invocation) {
				continue;
			}
			const script = this._injectionScripts.get(invocation.id);
			if (!script) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-not-found",
					`Injection script "${invocation.id}" was not registered.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			if (script.language && script.language !== preprocessContext.language) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-language-mismatch",
					`Injection script "${invocation.id}" does not support ${preprocessContext.language}.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			const scriptContext = this._createInjectionScriptContext(
				preprocessContext,
				composite.code
			);
			const injection = await script.run(invocation.args, scriptContext);
			this._appendDirectiveInjectionBlocks(
				preprocessContext,
				script,
				injection,
				headerBlocks,
				functionBlocks
			);
		}
		const baseComposite = this._composeLinesToComposite(outputLines, outputOrigins);
		const mergedBlocks = [...headerBlocks, ...functionBlocks];
		if (mergedBlocks.length <= 0) {
			return baseComposite;
		}
		return preprocessContext.language === "wgsl" ?
				injectWGSLSource(baseComposite, mergedBlocks)
			:	injectGLSLSource(baseComposite, mergedBlocks);
	}

	private _appendDirectiveInjectionBlocks(
		preprocessContext: PreprocessContext,
		script: ShaderInjectionScript,
		injection: ShaderRuleInjection | null | undefined,
		headers: InjectionBlock[],
		functions: InjectionBlock[]
	): void {
		if (!injection) {
			return;
		}
		const header = normalizeInjectionBlock(injection.header);
		if (header.length > 0) {
			headers.push({
				code: header,
				sourcePath: `<directive:inject:${script.id}:header>`,
				label: `directive-inject:${script.id}:header`,
				anchor: this._normalizeInjectionAnchorForLanguage(
					preprocessContext.language,
					injection.headerAnchor
				),
			});
		}
		const functionsBlock = normalizeInjectionBlock(injection.functions);
		if (functionsBlock.length > 0) {
			functions.push({
				code: functionsBlock,
				sourcePath: `<directive:inject:${script.id}:functions>`,
				label: `directive-inject:${script.id}:functions`,
				anchor: this._normalizeInjectionAnchorForLanguage(
					preprocessContext.language,
					injection.functionsAnchor
				),
			});
		}
	}

	private _parseInjectInvocation(
		directive: DirectiveLine,
		preprocessContext: PreprocessContext,
		origin: LineOrigin
	): InjectInvocation | null {
		const body = directive.body.trim();
		const match =
			/^<([^>]+)>\s*(?:\((.*)\))?$/.exec(body) ??
			/^([A-Za-z_][A-Za-z0-9_\/\.-]*)\s*(?:\((.*)\))?$/.exec(body);
		if (!match) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-inject-invalid",
				`Directive "#inject" expects <script-id>(key=value, ...).`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const id = match[1].trim();
		const argsPayload = (match[2] ?? "").trim();
		const args: Record<string, ShaderInjectionArgValue> = {};
		if (argsPayload.length > 0) {
			for (const part of this._splitInjectArguments(argsPayload)) {
				const equalIndex = part.indexOf("=");
				if (equalIndex <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-inject-arg-invalid",
						`Invalid inject argument "${part}". Expected key=value.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column,
						"warning"
					);
					continue;
				}
				const key = part.slice(0, equalIndex).trim();
				if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-inject-arg-key-invalid",
						`Invalid inject argument key "${key}".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column,
						"warning"
					);
					continue;
				}
				const valueRaw = part.slice(equalIndex + 1).trim();
				args[key] = this._parseInjectArgumentValue(valueRaw);
			}
		}
		return {
			id,
			args,
		};
	}

	private _splitInjectArguments(payload: string): string[] {
		const parts: string[] = [];
		let current = "";
		let stringQuote: '"' | "'" | null = null;
		let escape = false;
		let depth = 0;
		for (let index = 0; index < payload.length; index++) {
			const char = payload[index];
			if (stringQuote) {
				current += char;
				if (escape) {
					escape = false;
					continue;
				}
				if (char === "\\") {
					escape = true;
					continue;
				}
				if (char === stringQuote) {
					stringQuote = null;
				}
				continue;
			}
			if (char === "\"" || char === "'") {
				current += char;
				stringQuote = char as '"' | "'";
				escape = false;
				continue;
			}
			if (char === "(") {
				depth++;
				current += char;
				continue;
			}
			if (char === ")") {
				depth = Math.max(0, depth - 1);
				current += char;
				continue;
			}
			if (char === "," && depth === 0) {
				const value = current.trim();
				if (value.length > 0) {
					parts.push(value);
				}
				current = "";
				continue;
			}
			current += char;
		}
		const tail = current.trim();
		if (tail.length > 0) {
			parts.push(tail);
		}
		return parts;
	}

	private _parseInjectArgumentValue(
		rawValue: string
	): ShaderInjectionArgValue {
		const trimmed = rawValue.trim();
		if (
			(trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
			(trimmed.startsWith("'") && trimmed.endsWith("'"))
		) {
			return trimmed.slice(1, -1);
		}
		if (trimmed === "true") {
			return true;
		}
		if (trimmed === "false") {
			return false;
		}
		const numeric = Number(trimmed);
		if (Number.isFinite(numeric) && trimmed.length > 0) {
			return numeric;
		}
		return trimmed;
	}

	private _createInjectionScriptContext(
		preprocessContext: PreprocessContext,
		source: string
	): ShaderInjectionScriptContext {
		return {
			...preprocessContext.contextTemplate,
			source,
		};
	}

	private _normalizeInjectionScript(
		script: ShaderInjectionScript
	): ShaderInjectionScript {
		if (!script || typeof script !== "object") {
			throw new Error("Shader injection script must be an object.");
		}
		const id = typeof script.id === "string" ? script.id.trim() : "";
		if (id.length <= 0) {
			throw new Error("Shader injection script id must be a non-empty string.");
		}
		if (typeof script.run !== "function") {
			throw new Error(`Shader injection script "${id}" run must be a function.`);
		}
		const language =
			script.language === "wgsl" || script.language === "glsl" ?
				script.language
			:	undefined;
		return {
			...script,
			id,
			language,
			description:
				typeof script.description === "string" ?
					script.description.trim() || undefined
				:	undefined,
			symbols: normalizeSymbols(script.symbols),
		};
	}

	private _prepareProcessSync(request: ShaderProcessRequest): ProcessPreparation {
		const initialSourcePath = this._resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, initialSourcePath, "source");
		const preprocessed = this._preprocessDirectivesSync(request, initialComposite);
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
					`ShaderRuntime rule "${rule.id}" returned a Promise from match() during process(). Use processAsync().`
				);
			}
			if (matchResult) {
				matchedRules.push(rule);
			}
		}

		const sourceHash = this._resolveSourceHash(
			context.source,
			preprocessed.composite.code === request.code ? request.sourceHash : undefined
		);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			preprocessed.composite.sourceMap,
			sourceHash,
			matchedRuleIds,
			this._directiveRegistryRevision
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

	private async _prepareProcessAsync(
		request: ShaderProcessRequest
	): Promise<ProcessPreparation> {
		const initialSourcePath = this._resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: cloneSourceMap(request.sourceMap),
				}
			:	createInlineCompositeShaderSource(request.code, initialSourcePath, "source");
		const preprocessed = await this._preprocessDirectivesAsync(
			request,
			initialComposite
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
			preprocessed.composite.code === request.code ? request.sourceHash : undefined
		);
		const matchedRuleIds = matchedRules.map((rule) => rule.id);
		const cacheKey = this._buildProcessCacheKey(
			context,
			preprocessed.composite.sourceMap,
			sourceHash,
			matchedRuleIds,
			this._directiveRegistryRevision
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
		source: string = request.code
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
						`ShaderRuntime rule "${rule.id}" returned a Promise from validate() during process(). Use processAsync().`
					);
				}
				const diagnosticsFromRule =
					Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
						diagnostics.push(
							this._normalizeDiagnostic(
								{ ...diagnostic, ruleId: rule.id },
								rewrite.composite.sourceMap,
								prepared.sourcePath
							)
						);
					}
				}

			if (!rule.inject) {
				continue;
			}
			const injection = rule.inject(rewrite.context);
			if (isPromiseLike(injection)) {
				throw new Error(
					`ShaderRuntime rule "${rule.id}" returned a Promise from inject() during process(). Use processAsync().`
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
				dynamicUserSymbols
			);
		}

		return this._buildRawProcessResult(
			prepared,
			rewrite.composite,
			rewrite.context,
			diagnostics,
			headers,
			functions
		);
	}

	private async _executeRulesAsync(
		prepared: ProcessPreparation
	): Promise<ShaderProcessResult> {
		const rewrite = await this._applyRuleRewritesAsync(prepared);
		const diagnostics: ShaderDiagnostic[] = [...rewrite.diagnostics];
		const headers: InjectionBlock[] = [];
		const functions: InjectionBlock[] = [];
		const dynamicUserSymbols = new Map<string, string>();

		for (const rule of prepared.matchedRules) {
			if (rule.validate) {
				const validateResult = await rule.validate(rewrite.context);
				const diagnosticsFromRule =
					Array.isArray(validateResult) ? validateResult : [];
				for (const diagnostic of diagnosticsFromRule) {
					diagnostics.push(
						this._normalizeDiagnostic(
							{ ...diagnostic, ruleId: rule.id },
							rewrite.composite.sourceMap,
							prepared.sourcePath
						)
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
				dynamicUserSymbols
			);
		}

		return this._buildRawProcessResult(
			prepared,
			rewrite.composite,
			rewrite.context,
			diagnostics,
			headers,
			functions
		);
	}

	private _applyRuleRewritesSync(
		prepared: ProcessPreparation
	): RewritePreparation {
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
						`ShaderRuntime rule "${rule.id}" returned a Promise from transform() during process(). Use processAsync().`
					);
				}
				const applied = this._applyRuleTransformResult(
					rule.id,
					transformResult as ShaderRuleTransformResolved,
					composite
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
					prepared.sourcePath
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
					`ShaderRuntime rule "${rule.id}" returned a Promise from replace() during process(). Use processAsync().`
				);
			}
			const applied = this._applyRuleReplaceResult(
				rule.id,
				replaceResult as ShaderRuleReplaceResolved,
				composite
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
				prepared.sourcePath
			);
		}
		return {
			composite,
			context,
			diagnostics,
		};
	}

	private async _applyRuleRewritesAsync(
		prepared: ProcessPreparation
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
					const applied = this._applyRuleTransformResult(
						rule.id,
						transformResult,
						composite
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
					prepared.sourcePath
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
				composite
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
				prepared.sourcePath
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
		previousComposite: CompositeShaderSource
	): { composite: CompositeShaderSource; diagnostics: ShaderDiagnostic[] } {
		if (!result) {
			return {
				composite: previousComposite,
				diagnostics: [],
			};
		}
		const normalized =
			typeof result === "string" ?
				{
					code: result,
					sourceMap: undefined,
					diagnostics: [],
				}
			:	{
					code: result.code,
					sourceMap: result.sourceMap,
					diagnostics: Array.isArray(result.diagnostics) ? result.diagnostics : [],
				};
		if (typeof normalized.code !== "string") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" transform() must return a string or { code } object.`
			);
		}
		const composite =
			normalized.sourceMap ?
				{
					code: normalized.code,
					sourceMap: cloneSourceMap(normalized.sourceMap),
				}
			:	createGeneratedCompositeWithColumnSpans(
					normalized.code,
					`<runtime:${ruleId}:transform>`,
					`${ruleId}:transform`
				);
		return {
			composite,
			diagnostics: normalized.diagnostics,
		};
	}

	private _applyRuleReplaceResult(
		ruleId: string,
		result: ShaderRuleReplaceResolved,
		previousComposite: CompositeShaderSource
	): { composite: CompositeShaderSource; diagnostics: ShaderDiagnostic[] } {
		if (!result) {
			return {
				composite: previousComposite,
				diagnostics: [],
			};
		}
		const diagnostics =
			Array.isArray(result) ? []
			: Array.isArray((result as ShaderRuleReplaceOutput).diagnostics) ?
				(result as ShaderRuleReplaceOutput).diagnostics!
			:	[];
		const patchesRaw =
			Array.isArray(result) ? result : (result as ShaderRuleReplaceOutput).patches;
		if (!Array.isArray(patchesRaw)) {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace() must return patch list or { patches } object.`
			);
		}
		const patches = patchesRaw.map((patch, index) =>
			this._normalizeReplacePatch(ruleId, index, patch)
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
				`${ruleId}:replace`
			),
			diagnostics,
		};
	}

	private _normalizeReplacePatch(
		ruleId: string,
		index: number,
		patch: ShaderRuleReplacePatch
	): ShaderRuleReplacePatch {
		if (!patch || typeof patch !== "object") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} must be an object.`
			);
		}
		if (
			typeof patch.pattern !== "string" &&
			!(patch.pattern instanceof RegExp)
		) {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} pattern must be string or RegExp.`
			);
		}
		if (typeof patch.replacement !== "string") {
			throw new Error(
				`ShaderRuntime rule "${ruleId}" replace patch #${index + 1} replacement must be a string.`
			);
		}
		return {
			pattern: patch.pattern,
			replacement: patch.replacement,
			replaceAll: patch.replaceAll === true,
		};
	}

	private _applyReplacePatch(
		code: string,
		patch: ShaderRuleReplacePatch
	): string {
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
		return (
			code.slice(0, found) +
			patch.replacement +
			code.slice(found + patch.pattern.length)
		);
	}

	private _appendRuleDiagnostics(
		target: ShaderDiagnostic[],
		ruleId: string,
		diagnostics: ShaderDiagnostic[],
		sourceMap: ShaderSourceSegmentMap | null | undefined,
		fallbackSourcePath: string
	): void {
		for (const diagnostic of diagnostics) {
			target.push(
				this._normalizeDiagnostic(
					{ ...diagnostic, ruleId },
					sourceMap,
					fallbackSourcePath
				)
			);
		}
	}

	private _createRuleHookError(
		ruleId: string,
		hookKind: "transform" | "replace",
		error: unknown
	): Error {
		const message =
			error instanceof Error ?
				error.message
			:	typeof error === "string" ?
				error
			:	"Unknown hook failure.";
		return new Error(
			`ShaderRuntime rule "${ruleId}" ${hookKind} hook failed: ${message}`
		);
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
						sourceMap,
						fallbackSourcePath
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
						context.language,
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
						context.language,
						injection.functionsAnchor
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
		functions: InjectionBlock[]
	): ShaderProcessResult {
		const composite =
			context.language === "wgsl" ?
				injectWGSLSource(rewriteComposite, [...headers, ...functions])
			:	injectGLSLSource(rewriteComposite, [...headers, ...functions]);
		const mergedDiagnostics = [
			...prepared.preprocessedDiagnostics,
			...diagnostics,
		];
		const hasErrors = mergedDiagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
		return {
			code: composite.code,
			sourceMap: composite.sourceMap,
			composite,
			diagnostics: mergedDiagnostics,
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

	private _finalizeDirectivePreprocessResult(
		request: ShaderProcessRequest,
		preprocessed: PreprocessResult
	): ShaderDirectivePreprocessResult {
		const diagnostics = this._filterDiagnostics(
			preprocessed.diagnostics,
			request.diagnosticFilter
		);
		const hasErrors = diagnostics.some(
			(diagnostic) => diagnostic.severity === "error"
		);
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
		matchedRuleIds: readonly string[],
		directiveRevision: number
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
			includeModuleIds?: string[];
			injectionScriptIds?: string[];
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
			includeModuleIds:
				options.includeModuleIds ? [...new Set(options.includeModuleIds)] : undefined,
			injectionScriptIds:
				options.injectionScriptIds ?
					[...new Set(options.injectionScriptIds)]
				:	undefined,
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
