import { SOURCE_MAP_SCHEMA_VERSION } from "./sourceMap";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticRange,
	ShaderInjectionScript,
	ShaderLanguage,
	ShaderProcessRequest,
	ShaderProcessResult,
	ShaderRule,
	ShaderRuleContext,
	ShaderRuntimeMode,
	ShaderSourceKind,
	ShaderSourceSegment,
	ShaderSourceSegmentMap,
	ShaderStage,
} from "./types";

export function normalizeStage(stage?: ShaderStage): ShaderStage {
	switch (stage) {
		case "vertex":
		case "fragment":
		case "compute":
			return stage;
		default:
			return "unknown";
	}
}

export function normalizeSourceKind(sourceKind?: ShaderSourceKind): ShaderSourceKind {
	switch (sourceKind) {
		case "builtin-scene":
		case "builtin-environment":
		case "builtin-present":
		case "postprocess":
		case "clustered":
		case "shadow":
		case "decal":
		case "particle":
		case "custom-material":
			return sourceKind;
		default:
			return "unknown";
	}
}

export function cloneDiagnostics(diagnostics: ShaderDiagnostic[]): ShaderDiagnostic[] {
	return diagnostics.map((diagnostic) => ({ ...diagnostic }));
}

export function cloneSourceMap(sourceMap: ShaderSourceSegmentMap): ShaderSourceSegmentMap {
	return {
		schemaVersion: sourceMap.schemaVersion,
		lineCount: sourceMap.lineCount,
		segments: sourceMap.segments.map((segment) => ({ ...segment })),
	};
}

export function cloneCompositeSource(
	composite: CompositeShaderSource
): CompositeShaderSource {
	return {
		code: composite.code,
		sourceMap: cloneSourceMap(composite.sourceMap),
	};
}

export function cloneProcessResult(
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

export function cloneRule(rule: ShaderRule): ShaderRule {
	return {
		...rule,
		symbols: rule.symbols ? [...rule.symbols] : undefined,
		dependsOn: rule.dependsOn ? [...rule.dependsOn] : undefined,
	};
}

export function hashStringFNV1a(value: string): string {
	let hash = 0x811c9dc5;
	for (let i = 0; i < value.length; i++) {
		hash ^= value.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}

export function hashStringFNV1aChunked(value: string, chunkSize: number): string {
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

const LARGE_SOURCE_THRESHOLD = 16 * 1024;
const LARGE_SOURCE_CHUNK_SIZE = 4 * 1024;

export function hashSourceCode(source: string): string {
	if (source.length > LARGE_SOURCE_THRESHOLD) {
		return hashStringFNV1aChunked(source, LARGE_SOURCE_CHUNK_SIZE);
	}
	return hashStringFNV1a(source);
}

export function hashSourceMap(sourceMap: ShaderSourceSegmentMap | null | undefined): string {
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

export function normalizeInjectionBlock(block: string | undefined): string {
	if (typeof block !== "string") {
		return "";
	}
	return block.trim();
}

export function normalizeSymbols(symbols: string[] | undefined): string[] {
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

export function normalizeDependsOn(dependsOn: string[] | undefined): string[] {
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

export function isWhitespaceCharacter(char: string): boolean {
	return char === " " || char === "\t" || char === "\r" || char === "\n";
}

export function isIdentifierStartCharacter(char: string): boolean {
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

export function isIdentifierPartCharacter(char: string): boolean {
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

export function isPromiseLike<T = unknown>(value: unknown): value is PromiseLike<T> {
	return (
		typeof value === "object" &&
		value !== null &&
		"then" in value &&
		typeof (value as { then?: unknown }).then === "function"
	);
}



export function normalizeLanguage(language?: ShaderLanguage): ShaderLanguage {
	return language === "glsl" ? "glsl" : "wgsl";
}

export function buildStrictModeError(
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

export function normalizePositiveInteger(value: number | undefined): number | null {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return null;
	}
	const normalized = Math.floor(value);
	return normalized >= 1 ? normalized : 1;
}

export function createPointRange(line: number, column: number): ShaderDiagnosticRange {
	return {
		start: { line, column },
		end: { line, column },
	};
}

export function createGeneratedCompositeWithColumnSpans(
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

export function normalizeDiagnosticRange(
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

export function resolveShaderRequestSourcePath(request: ShaderProcessRequest): string {
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

export function withLanguageDefaultExtension(
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

export function joinModulePath(baseModulePath: string, relativePath: string): string {
	const base = baseModulePath.replace(/\\/g, "/");
	const slashIndex = base.lastIndexOf("/");
	const directory = slashIndex >= 0 ? base.slice(0, slashIndex) : "";
	return directory.length > 0 ? `${directory}/${relativePath}` : relativePath;
}

export function canonicalizeModulePathSafe(value: string): string {
	try {
		return canonicalizeModulePath(value);
	} catch (error) {
		return value
			.replace(/\\/g, "/")
			.replace(/^\/+/, "")
			.replace(/\/{2,}/g, "/")
			.trim();
	}
}

export function canonicalizeModulePath(value: string): string {
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

export function formatIncludeModuleEventId(
	language: ShaderLanguage,
	moduleId: string
): string {
	return `${language}:${moduleId}`;
}

export function normalizeInjectionScript(
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
