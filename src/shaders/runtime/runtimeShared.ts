import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticRange,
	ShaderInjectionArgumentDefinition,
	ShaderInjectionArgumentSchema,
	ShaderInjectionScript,
	ShaderLanguage,
	ShaderSourceKind,
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

export function normalizeInjectionBlock(block: string | undefined): string {
	if (typeof block !== "string") {
		return "";
	}
	return block.trim();
}

export function normalizeSymbols(symbols: readonly string[] | undefined): string[] {
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

export function createPointRange(line: number, column: number): ShaderDiagnosticRange {
	return {
		start: { line, column },
		end: { line, column },
	};
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

export function normalizeInjectionScript<
	Schema extends ShaderInjectionArgumentSchema,
>(script: ShaderInjectionScript<Schema>): ShaderInjectionScript<Schema> {
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
	if (!script.arguments || typeof script.arguments !== "object") {
		throw new Error(
			`Shader injection script "${id}" arguments must be an object schema.`,
		);
	}
	const language =
		script.language === "wgsl" || script.language === "glsl" ?
			script.language
		:	undefined;
	return {
		...script,
		id,
		language,
		arguments: normalizeInjectionArgumentSchema(id, script.arguments) as Schema,
		description:
			typeof script.description === "string" ?
				script.description.trim() || undefined
			:	undefined,
		symbols: normalizeSymbols(script.symbols),
	};
}

function normalizeInjectionArgumentSchema(
	scriptId: string,
	schema: ShaderInjectionArgumentSchema,
): ShaderInjectionArgumentSchema {
	const normalized: Record<string, ShaderInjectionArgumentDefinition> = {};
	for (const [rawName, rawDefinition] of Object.entries(schema)) {
		const name = rawName.trim();
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
			throw new Error(
				`Shader injection script "${scriptId}" argument name "${rawName}" is invalid.`,
			);
		}
		if (!rawDefinition || typeof rawDefinition !== "object") {
			throw new Error(
				`Shader injection script "${scriptId}" argument "${name}" definition must be an object.`,
			);
		}
		const definition = normalizeInjectionArgumentDefinition(
			scriptId,
			name,
			rawDefinition,
		);
		normalized[name] = Object.freeze(definition);
	}
	return Object.freeze(normalized);
}

function normalizeInjectionArgumentDefinition(
	scriptId: string,
	name: string,
	definition: ShaderInjectionArgumentDefinition,
): ShaderInjectionArgumentDefinition {
	const label = `Shader injection script "${scriptId}" argument "${name}"`;
	switch (definition.type) {
		case "string":
			if (definition.default !== undefined && typeof definition.default !== "string") {
				throw new Error(`${label} default must be a string.`);
			}
			return { ...definition, required: definition.required === true };
		case "boolean":
			if (
				definition.default !== undefined &&
				typeof definition.default !== "boolean"
			) {
				throw new Error(`${label} default must be a boolean.`);
			}
			return { ...definition, required: definition.required === true };
		case "number":
		case "integer": {
			for (const [key, value] of [
				["default", definition.default],
				["min", definition.min],
				["max", definition.max],
			] as const) {
				if (value !== undefined && !Number.isFinite(value)) {
					throw new Error(`${label} ${key} must be finite.`);
				}
			}
			if (
				definition.type === "integer" &&
				definition.default !== undefined &&
				!Number.isInteger(definition.default)
			) {
				throw new Error(`${label} default must be an integer.`);
			}
			if (
				definition.min !== undefined &&
				definition.max !== undefined &&
				definition.min > definition.max
			) {
				throw new Error(`${label} min must not exceed max.`);
			}
			return { ...definition, required: definition.required === true };
		}
		case "enum": {
			const values = [...definition.values];
			if (
				values.length <= 0 ||
				values.some((value) => typeof value !== "string" || value.length <= 0)
			) {
				throw new Error(`${label} enum values must contain non-empty strings.`);
			}
			if (
				definition.default !== undefined &&
				!values.includes(definition.default)
			) {
				throw new Error(`${label} default must be one of its enum values.`);
			}
			return {
				...definition,
				required: definition.required === true,
				values: Object.freeze(values),
			};
		}
		default:
			throw new Error(`${label} has an unsupported type.`);
	}
}
