import type { ShaderRuntimeMode } from "./types";

export const SHADER_RUNTIME_RESERVED_RULE_PREFIX = "ignis/";
export const SHADER_RUNTIME_DEFAULT_CACHE_LIMIT = 256;

export const SHADER_RUNTIME_RULE_IDS = {
	EMPTY_SOURCE: "ignis/validate-empty-source",
	PLACEHOLDER: "ignis/validate-placeholder",
	BRACKET_BALANCE: "ignis/validate-bracket-balance",
	GLSL_ENTRY_POINT: "ignis/validate-glsl-entry-point",
	WGSL_ENTRY_POINT: "ignis/validate-wgsl-entry-point",
} as const;

export function resolveDefaultShaderRuntimeMode(): ShaderRuntimeMode {
	const meta = import.meta as ImportMeta & {
		env?: {
			DEV?: boolean;
			PROD?: boolean;
			NODE_ENV?: string;
		};
	};
	const devFlag = meta.env?.DEV;
	if (typeof devFlag === "boolean") {
		return devFlag ? "strict" : "warn";
	}

	const nodeEnv =
		(
			globalThis as {
				process?: {
					env?: Record<string, unknown>;
				};
			}
		).process?.env?.NODE_ENV ?? null;
	if (typeof nodeEnv === "string" && nodeEnv.toLowerCase() === "production") {
		return "warn";
	}
	return "strict";
}
