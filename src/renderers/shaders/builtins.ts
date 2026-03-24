import { SHADER_RUNTIME_RULE_IDS } from "./constants";
import type { ShaderDiagnostic, ShaderRule, ShaderRuleContext } from "./types";

const OPEN_TO_CLOSE = new Map<string, string>([
	["(", ")"],
	["[", "]"],
	["{", "}"],
]);

function createError(
	ruleId: string,
	code: string,
	message: string
): ShaderDiagnostic {
	return {
		ruleId,
		code,
		severity: "error",
		message,
	};
}

function findFirstUnmatchedBracket(source: string): string | null {
	const stack: string[] = [];
	for (let i = 0; i < source.length; i++) {
		const char = source[i];
		if (OPEN_TO_CLOSE.has(char)) {
			stack.push(char);
			continue;
		}

		if (char !== ")" && char !== "]" && char !== "}") {
			continue;
		}

		const open = stack.pop();
		if (!open) {
			return char;
		}
		const expected = OPEN_TO_CLOSE.get(open);
		if (expected !== char) {
			return `${open} ... ${char}`;
		}
	}

	if (stack.length > 0) {
		return stack[stack.length - 1];
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

export function createBuiltInShaderRules(): ShaderRule[] {
	return [
		{
			id: "ignis/reserved-symbols",
			priority: 1100,
			symbols: ["ignis_runtime_reserved_symbol"],
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.EMPTY_SOURCE,
			priority: 1000,
			validate(context) {
				if (context.source.trim().length > 0) {
					return [];
				}
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.EMPTY_SOURCE,
						"empty-source",
						"Shader source is empty."
					),
				];
			},
		},
		{
			id: SHADER_RUNTIME_RULE_IDS.PLACEHOLDER,
			priority: 900,
			validate(context) {
				const match = context.source.match(/__[A-Za-z0-9_]+__/);
				if (!match) {
					return [];
				}
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.PLACEHOLDER,
						"placeholder-not-resolved",
						`Shader contains unresolved placeholder ${match[0]}.`
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
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.BRACKET_BALANCE,
						"unbalanced-brackets",
						`Shader contains unmatched bracket token "${unmatched}".`
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
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.GLSL_ENTRY_POINT,
						"missing-main",
						"GLSL shader is missing entry point function `main`."
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
				return [
					createError(
						SHADER_RUNTIME_RULE_IDS.WGSL_ENTRY_POINT,
						"missing-entry-point",
						`WGSL shader is missing entry point "${context.entryPoint}".`
					),
				];
			},
		},
	];
}
