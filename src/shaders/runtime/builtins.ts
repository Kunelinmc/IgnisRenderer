import { SHADER_RUNTIME_RULE_IDS } from "./constants";
import type {
	ShaderDiagnostic,
	ShaderDiagnosticRange,
	ShaderRule,
	ShaderRuleContext,
} from "./types";

const OPEN_TO_CLOSE = new Map<string, string>([
	["(", ")"],
	["[", "]"],
	["{", "}"],
]);

const RUNTIME_METADATA_INJECTION_SOURCE_KINDS = new Set([
	"builtin-scene",
	"builtin-environment",
	"builtin-present",
	"postprocess",
	"clustered",
	"shadow",
	"decal",
	"particle",
]);
const RUNTIME_METADATA_MARKER = "IGNIS_RUNTIME_INJECTION_ENABLED";

interface ErrorLocation {
	line?: number;
	column?: number;
	sourcePath?: string;
	range?: ShaderDiagnosticRange;
}

interface UnmatchedBracketToken {
	token: string;
	offset: number;
	length: number;
}

function createError(
	ruleId: string,
	code: string,
	message: string,
	location?: ErrorLocation
): ShaderDiagnostic {
	return {
		ruleId,
		code,
		severity: "error",
		message,
		line: location?.line,
		column: location?.column,
		sourcePath: location?.sourcePath,
		range: location?.range,
	};
}

function findFirstUnmatchedBracket(
	source: string
): UnmatchedBracketToken | null {
	const stack: { token: string; offset: number }[] = [];
	for (let i = 0; i < source.length; i++) {
		const char = source[i];
		if (OPEN_TO_CLOSE.has(char)) {
			stack.push({ token: char, offset: i });
			continue;
		}

		if (char !== ")" && char !== "]" && char !== "}") {
			continue;
		}

		const open = stack.pop();
		if (!open) {
			return {
				token: char,
				offset: i,
				length: 1,
			};
		}
		const expected = OPEN_TO_CLOSE.get(open.token);
		if (expected !== char) {
			return {
				token: `${open.token} ... ${char}`,
				offset: i,
				length: 1,
			};
		}
	}

	if (stack.length > 0) {
		const unmatched = stack[stack.length - 1];
		return {
			token: unmatched.token,
			offset: unmatched.offset,
			length: 1,
		};
	}
	return null;
}

function escapeRegExp(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function hasWGSLStageEntryPoint(context: ShaderRuleContext): boolean {
	const entryPoint = context.entryPoint;
	if (!entryPoint) {
		return true;
	}

	const stagePattern =
		context.stage === "unknown" ? "(vertex|fragment|compute)" : context.stage;
	const pattern = new RegExp(
		`@${stagePattern}\\s+fn\\s+${escapeRegExp(entryPoint)}\\s*\\(`,
		"m"
	);
	return pattern.test(context.source);
}

function getLineColumnAtOffset(
	source: string,
	offset: number
): { line: number; column: number } {
	const safeOffset = Math.max(0, Math.min(source.length, Math.floor(offset)));
	let line = 1;
	let column = 1;
	for (let i = 0; i < safeOffset; i++) {
		if (source.charCodeAt(i) === 10) {
			line++;
			column = 1;
			continue;
		}
		column++;
	}
	return { line, column };
}

function createRangeFromOffset(
	source: string,
	offset: number,
	length: number = 1
): ShaderDiagnosticRange {
	const safeLength = Math.max(1, Math.floor(length));
	const start = getLineColumnAtOffset(source, offset);
	const endOffset = Math.max(offset, offset + safeLength - 1);
	const end = getLineColumnAtOffset(source, endOffset);
	return {
		start,
		end,
	};
}

function createSourceExtentRange(source: string): ShaderDiagnosticRange {
	if (source.length <= 0) {
		return {
			start: { line: 1, column: 1 },
			end: { line: 1, column: 1 },
		};
	}
	const lines = source.split(/\r?\n/g);
	const lastLine = Math.max(1, lines.length);
	const lastLineLength = lines[lastLine - 1]?.length ?? 0;
	return {
		start: { line: 1, column: 1 },
		end: { line: lastLine, column: Math.max(1, lastLineLength + 1) },
	};
}

function normalizeMacroToken(value: string): string {
	const normalized = value
		.trim()
		.replace(/[^A-Za-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.toUpperCase();
	return normalized.length > 0 ? normalized : "UNKNOWN";
}

function shouldInjectRuntimeMetadata(context: ShaderRuleContext): boolean {
	if (!RUNTIME_METADATA_INJECTION_SOURCE_KINDS.has(context.sourceKind)) {
		return false;
	}
	return !context.source.includes(RUNTIME_METADATA_MARKER);
}

function createWGSLRuntimeMetadataHeader(context: ShaderRuleContext): string {
	const stageToken = normalizeMacroToken(context.stage);
	const sourceKindToken = normalizeMacroToken(context.sourceKind);
	return [
		"const IGNIS_RUNTIME_INJECTION_ENABLED: bool = true;",
		"const IGNIS_RUNTIME_LANGUAGE_WGSL: bool = true;",
		`const IGNIS_RUNTIME_STAGE_${stageToken}: bool = true;`,
		`const IGNIS_RUNTIME_SOURCE_KIND_${sourceKindToken}: bool = true;`,
	].join("\n");
}

function createGLSLRuntimeMetadataHeader(context: ShaderRuleContext): string {
	const stageToken = normalizeMacroToken(context.stage);
	const sourceKindToken = normalizeMacroToken(context.sourceKind);
	return [
		"#define IGNIS_RUNTIME_INJECTION_ENABLED 1",
		"#define IGNIS_RUNTIME_LANGUAGE_GLSL 1",
		`#define IGNIS_RUNTIME_STAGE_${stageToken} 1`,
		`#define IGNIS_RUNTIME_SOURCE_KIND_${sourceKindToken} 1`,
	].join("\n");
}

export function createBuiltInShaderRules(): ShaderRule[] {
	return [
		{
			id: "ignis/reserved-symbols",
			priority: 1100,
			symbols: ["ignis_runtime_reserved_symbol"],
		},
		{
			id: "ignis/inject-runtime-metadata",
			priority: 600,
			match(context) {
				return shouldInjectRuntimeMetadata(context);
			},
			inject(context) {
				if (context.language === "wgsl") {
					return {
						header: createWGSLRuntimeMetadataHeader(context),
						headerAnchor: "afterEnable",
					};
				}
				return {
					header: createGLSLRuntimeMetadataHeader(context),
					headerAnchor: "afterPrecision",
				};
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.EMPTY_SOURCE,
			priority: 1000,
			validate(context) {
				if (context.source.trim().length > 0) {
					return [];
				}
				const range = createSourceExtentRange(context.source);
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.EMPTY_SOURCE,
						"empty-source",
						"Shader source is empty.",
						{
							line: range.start.line,
							column: range.start.column,
							range,
						}
					),
				];
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.PLACEHOLDER,
			priority: 900,
			validate(context) {
				const match = /__[A-Za-z0-9_]+__/.exec(context.source);
				if (!match) {
					return [];
				}
				const offset = typeof match.index === "number" ? match.index : 0;
				const range = createRangeFromOffset(
					context.source,
					offset,
					match[0].length
				);
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.PLACEHOLDER,
						"placeholder-not-resolved",
						`Shader contains unresolved placeholder ${match[0]}.`,
						{
							line: range.start.line,
							column: range.start.column,
							range,
						}
					),
				];
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.BRACKET_BALANCE,
			priority: 800,
			validate(context) {
				const unmatched = findFirstUnmatchedBracket(context.source);
				if (!unmatched) {
					return [];
				}
				const range = createRangeFromOffset(
					context.source,
					unmatched.offset,
					unmatched.length
				);
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.BRACKET_BALANCE,
						"unbalanced-brackets",
						`Shader contains unmatched bracket token "${unmatched.token}".`,
						{
							line: range.start.line,
							column: range.start.column,
							range,
						}
					),
				];
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.GLSL_ENTRY_POINT,
			priority: 700,
			match(context) {
				return context.language === "glsl";
			},
			validate(context) {
				if (/\bvoid\s+main\s*\(/m.test(context.source)) {
					return [];
				}
				const range = createSourceExtentRange(context.source);
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.GLSL_ENTRY_POINT,
						"missing-main",
						"GLSL shader is missing entry point function `main`.",
						{
							line: range.start.line,
							column: range.start.column,
							range,
						}
					),
				];
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.WGSL_ENTRY_POINT,
			priority: 700,
			match(context) {
				return context.language === "wgsl";
			},
			validate(context) {
				if (!context.entryPoint) {
					return [];
				}
				if (hasWGSLStageEntryPoint(context)) {
					return [];
				}
				const range = createSourceExtentRange(context.source);
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.WGSL_ENTRY_POINT,
						"missing-entry-point",
						`WGSL shader is missing entry point "${context.entryPoint}".`,
						{
							line: range.start.line,
							column: range.start.column,
							range,
						}
					),
				];
			},
		},
	];
}
