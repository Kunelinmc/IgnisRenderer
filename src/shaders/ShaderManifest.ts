import { createBuiltinInjectionFeaturePacks } from "./features/builtinInjectionScripts";
import { composeShaderDirectiveProfile } from "./runtime/DirectiveProfile";
import {
	composeCompositeShaderSources,
	createInlineCompositeShaderSource,
} from "./runtime/sourceMap";
import type {
	CompositeShaderSource,
	ShaderBackendId,
	ShaderDirectiveFeaturePack,
	ShaderDirectiveProfile,
	ShaderDirectiveProfileBase,
	ShaderLanguage,
	ShaderSourceKind,
	ShaderSourceSegmentKind,
	ShaderStage,
} from "./runtime/types";

export type ShaderManifestPrimitive = string | number | boolean;

export type ShaderManifestExpression =
	| { readonly literal: ShaderManifestPrimitive }
	| { readonly parameter: string }
	| {
			readonly equals: readonly [
				ShaderManifestExpression,
				ShaderManifestExpression,
			];
	  }
	| { readonly all: readonly ShaderManifestExpression[] }
	| { readonly any: readonly ShaderManifestExpression[] }
	| { readonly not: ShaderManifestExpression }
	| {
			readonly multiply: readonly [
				ShaderManifestExpression,
				ShaderManifestExpression,
			];
	  }
	| {
			readonly select: {
				readonly cases: readonly {
					readonly when: ShaderManifestExpression;
					readonly value: ShaderManifestExpression;
				}[];
				readonly fallback: ShaderManifestExpression;
			};
	  };

export type ShaderParameterSchema =
	| { readonly type: "boolean"; readonly default?: boolean }
	| {
			readonly type: "enum";
			readonly values: readonly string[];
			readonly default: string;
	  }
	| {
			readonly type: "integer";
			readonly default?: number;
			readonly required?: boolean;
			readonly min?: number;
			readonly max?: number;
			readonly bitMask?: number;
	  }
	| {
			readonly type: "record";
			readonly fields: Readonly<Record<string, ShaderParameterSchema>>;
			readonly default?: Readonly<Record<string, unknown>>;
	  };

export interface ShaderManifestAsset {
	readonly path: string;
	readonly sync?: boolean;
	readonly segmentKind?: ShaderSourceSegmentKind;
}

export interface ShaderManifestConditionalBlock {
	readonly start: string;
	readonly end: string;
	readonly when: ShaderManifestExpression;
}

export interface ShaderManifestReplacement {
	readonly marker: string;
	readonly value: ShaderSourceNode;
}

export type ShaderSourceNode =
	| { readonly asset: string }
	| { readonly source: string }
	| {
			readonly concat: readonly ShaderSourceNode[];
			readonly separator?: string;
			readonly fallbackSourcePath?: string;
	  }
	| {
			readonly when: ShaderManifestExpression;
			readonly then: ShaderSourceNode;
			readonly else?: ShaderSourceNode;
	  }
	| {
			readonly defines: Readonly<
				Record<string, ShaderManifestExpression>
			>;
			readonly sourcePath: string;
	  }
	| {
			readonly template: ShaderSourceNode;
			readonly blocks?: readonly ShaderManifestConditionalBlock[];
			readonly replacements?: readonly ShaderManifestReplacement[];
	  };

interface ShaderManifestSourceBase {
	readonly sourceKind: ShaderSourceKind;
	readonly parameters?: ShaderParameterSchema;
}

export interface ShaderManifestModuleSource
	extends ShaderManifestSourceBase {
	readonly kind: "module";
	readonly source: ShaderSourceNode;
}

export interface ShaderManifestProgramSource
	extends ShaderManifestSourceBase {
	readonly kind: "program";
	readonly stages: Readonly<
		Partial<Record<Exclude<ShaderStage, "unknown">, ShaderSourceNode>>
	>;
}

export type ShaderManifestSource =
	| ShaderManifestModuleSource
	| ShaderManifestProgramSource;

export interface ShaderManifestProfile {
	readonly baseId: string;
	readonly assetPackId: string;
	readonly assetPackRevision: number;
	readonly includes: readonly {
		readonly id: string;
		readonly source: string;
	}[];
	readonly featurePacks: readonly "builtin-injections"[];
	readonly overlay: {
		readonly id: string;
		readonly includeId: string;
		readonly sourcePath: string;
		readonly baseInclude?: string;
		readonly parameters: ShaderParameterSchema;
		readonly defines: Readonly<
			Record<string, ShaderManifestExpression>
		>;
	};
}

export interface ShaderBackendManifest {
	readonly backend: ShaderBackendId;
	readonly language: ShaderLanguage;
	readonly assets: Readonly<Record<string, ShaderManifestAsset>>;
	readonly sources: Readonly<Record<string, ShaderManifestSource>>;
	readonly preloadGroups?: Readonly<Record<string, readonly string[]>>;
	readonly profile: ShaderManifestProfile;
}

interface ShaderSourceArtifactBase {
	readonly key: string;
	readonly identity: string;
	readonly language: ShaderLanguage;
	readonly sourceKind: ShaderSourceKind;
}

export interface ShaderModuleSourceArtifact extends ShaderSourceArtifactBase {
	readonly kind: "module";
	readonly source: CompositeShaderSource;
}

export interface ShaderProgramSourceArtifact extends ShaderSourceArtifactBase {
	readonly kind: "program";
	readonly stages: Readonly<
		Partial<
			Record<Exclude<ShaderStage, "unknown">, CompositeShaderSource>
		>
	>;
}

export type ShaderSourceArtifact =
	| ShaderModuleSourceArtifact
	| ShaderProgramSourceArtifact;

export interface ResolvedShaderManifestRequest {
	readonly parameters: unknown;
	readonly identity: string;
}

export interface ShaderManifestBuildContext {
	loadAsset(
		manifest: ShaderBackendManifest,
		assetId: string,
	): Promise<CompositeShaderSource>;
	loadSource(
		manifest: ShaderBackendManifest,
		key: string,
	): Promise<ShaderSourceArtifact>;
}

const validatedManifests = new WeakSet<object>();

export function validateShaderBackendManifest(
	manifest: ShaderBackendManifest,
): void {
	if (validatedManifests.has(manifest)) return;
	const sourceKeys = new Set(Object.keys(manifest.sources));
	for (const [assetId, asset] of Object.entries(manifest.assets)) {
		if (!assetId || !asset.path) {
			throw new Error("Shader manifest asset ids and paths must be non-empty.");
		}
		if (asset.sync !== undefined && typeof asset.sync !== "boolean") {
			throw new Error(`Shader manifest asset "${assetId}" has invalid sync metadata.`);
		}
	}
	const visiting = new Set<string>();
	const visited = new Set<string>();
	const visitNode = (node: ShaderSourceNode, owner: string): void => {
		if ("asset" in node) {
			if (!manifest.assets[node.asset]) {
				throw new Error(
					`Shader manifest source "${owner}" references unknown asset "${node.asset}".`,
				);
			}
			return;
		}
		if ("source" in node) {
			if (!sourceKeys.has(node.source)) {
				throw new Error(
					`Shader manifest source "${owner}" references unknown source "${node.source}".`,
				);
			}
			visitSource(node.source);
			return;
		}
		if ("concat" in node) {
			for (const child of node.concat) visitNode(child, owner);
			return;
		}
		if ("when" in node) {
			validateExpression(node.when, owner);
			visitNode(node.then, owner);
			if (node.else) visitNode(node.else, owner);
			return;
		}
		if ("template" in node) {
			visitNode(node.template, owner);
			for (const block of node.blocks ?? []) {
				validateExpression(block.when, owner);
			}
			for (const replacement of node.replacements ?? []) {
				visitNode(replacement.value, owner);
			}
			return;
		}
		if ("defines" in node) {
			for (const expression of Object.values(node.defines)) {
				validateExpression(expression, owner);
			}
			return;
		}
		throw new Error(`Unsupported shader source node in "${owner}".`);
	};
	const visitSource = (key: string): void => {
		if (visited.has(key)) return;
		if (visiting.has(key)) {
			throw new Error(`Shader manifest composition cycle includes "${key}".`);
		}
		visiting.add(key);
		const source = manifest.sources[key];
		if (source.parameters) validateParameterSchema(source.parameters, key);
		if (source.kind === "module") {
			visitNode(source.source, key);
		} else {
			if (Object.keys(source.stages).length === 0) {
				throw new Error(`Shader program source "${key}" has no stages.`);
			}
			for (const node of Object.values(source.stages)) {
				if (node) visitNode(node, key);
			}
		}
		visiting.delete(key);
		visited.add(key);
	};
	for (const key of sourceKeys) visitSource(key);
	const includeIds = new Set<string>();
	for (const include of manifest.profile.includes) {
		if (includeIds.has(include.id)) {
			throw new Error(`Duplicate shader profile include "${include.id}".`);
		}
		includeIds.add(include.id);
		const source = manifest.sources[include.source];
		if (!source || source.kind !== "module") {
			throw new Error(
				`Shader profile include "${include.id}" requires module source "${include.source}".`,
			);
		}
	}
	for (const group of Object.values(manifest.preloadGroups ?? {})) {
		for (const key of group) {
			if (!sourceKeys.has(key)) {
				throw new Error(`Shader preload group references unknown source "${key}".`);
			}
		}
	}
	validateParameterSchema(
		manifest.profile.overlay.parameters,
		`${manifest.backend} profile`,
	);
	for (const expression of Object.values(manifest.profile.overlay.defines)) {
		validateExpression(expression, `${manifest.backend} profile`);
	}
	validatedManifests.add(manifest);
}

function validateExpression(expression: unknown, owner: string): void {
	if (!expression || typeof expression !== "object") {
		throw new Error(`Shader manifest expression in "${owner}" must be an object.`);
	}
	const value = expression as Record<string, unknown>;
	if ("literal" in value || "parameter" in value) return;
	if ("equals" in value || "multiply" in value) {
		const pair = (value.equals ?? value.multiply) as unknown;
		if (!Array.isArray(pair) || pair.length !== 2) {
			throw new Error(`Shader manifest expression pair in "${owner}" is invalid.`);
		}
		validateExpression(pair[0], owner);
		validateExpression(pair[1], owner);
		return;
	}
	if ("all" in value || "any" in value) {
		const entries = (value.all ?? value.any) as unknown;
		if (!Array.isArray(entries)) {
			throw new Error(`Shader manifest expression list in "${owner}" is invalid.`);
		}
		for (const entry of entries) validateExpression(entry, owner);
		return;
	}
	if ("not" in value) {
		validateExpression(value.not, owner);
		return;
	}
	if ("select" in value) {
		const select = value.select as {
			cases?: unknown;
			fallback?: unknown;
		};
		if (!Array.isArray(select.cases) || select.fallback === undefined) {
			throw new Error(`Shader manifest select expression in "${owner}" is invalid.`);
		}
		for (const entry of select.cases as Array<Record<string, unknown>>) {
			validateExpression(entry.when, owner);
			validateExpression(entry.value, owner);
		}
		validateExpression(select.fallback, owner);
		return;
	}
	throw new Error(`Unsupported shader manifest expression in "${owner}".`);
}

function validateParameterSchema(schema: ShaderParameterSchema, path: string): void {
	if (schema.type === "boolean") {
		if (schema.default !== undefined && typeof schema.default !== "boolean") {
			throw new Error(`Shader parameter default "${path}" must be boolean.`);
		}
		return;
	}
	if (schema.type === "enum") {
		if (!schema.values.includes(schema.default)) {
			throw new Error(`Shader parameter enum default "${path}" is invalid.`);
		}
		return;
	}
	if (schema.type === "integer") {
		if (schema.default !== undefined && !Number.isInteger(schema.default)) {
			throw new Error(`Shader parameter default "${path}" must be an integer.`);
		}
		return;
	}
	for (const [key, field] of Object.entries(schema.fields)) {
		validateParameterSchema(field, `${path}.${key}`);
	}
}

export function resolveShaderManifestRequest(
	manifest: ShaderBackendManifest,
	key: string,
	params: unknown,
): ResolvedShaderManifestRequest {
	validateShaderBackendManifest(manifest);
	const definition = manifest.sources[key];
	if (!definition) throw new Error(`Unsupported ShaderSource key "${key}".`);
	const parameters = definition.parameters ?
		normalizeParameter(definition.parameters, params, key)
	:	undefined;
	return {
		parameters,
		identity:
			parameters === undefined ? key : `${key}|${stableSerialize(parameters)}`,
	};
}

export async function buildShaderSourceArtifact(
	manifest: ShaderBackendManifest,
	key: string,
	resolved: ResolvedShaderManifestRequest,
	context: ShaderManifestBuildContext,
): Promise<ShaderSourceArtifact> {
	const definition = manifest.sources[key];
	if (!definition) throw new Error(`Unsupported ShaderSource key "${key}".`);
	const base = {
		key,
		identity: resolved.identity,
		language: manifest.language,
		sourceKind: definition.sourceKind,
	};
	if (definition.kind === "module") {
		return {
			...base,
			kind: "module",
			source: await buildNode(
				manifest,
				definition.source,
				resolved.parameters,
				context,
			),
		};
	}
	const stages: Partial<
		Record<Exclude<ShaderStage, "unknown">, CompositeShaderSource>
	> = {};
	for (const [stage, node] of Object.entries(definition.stages)) {
		if (!node) continue;
		stages[stage as Exclude<ShaderStage, "unknown">] = await buildNode(
			manifest,
			node,
			resolved.parameters,
			context,
		);
	}
	return { ...base, kind: "program", stages };
}

export function cloneShaderSourceArtifact<T extends ShaderSourceArtifact>(
	artifact: T,
): T {
	if (artifact.kind === "module") {
		return { ...artifact, source: cloneComposite(artifact.source) } as T;
	}
	const stages: ShaderProgramSourceArtifact["stages"] = Object.fromEntries(
		Object.entries(artifact.stages).map(([stage, source]) => [
			stage,
			cloneComposite(source),
		]),
	);
	return { ...artifact, stages } as T;
}

export async function prepareShaderDirectiveProfileBase(
	manifest: ShaderBackendManifest,
	load: (key: string) => Promise<ShaderSourceArtifact>,
): Promise<ShaderDirectiveProfileBase> {
	validateShaderBackendManifest(manifest);
	const modules = await Promise.all(
		manifest.profile.includes.map(async (include) => {
			const artifact = await load(include.source);
			if (artifact.kind !== "module") {
				throw new Error(
					`Shader profile include "${include.id}" did not resolve to a module.`,
				);
			}
			return {
				language: manifest.language,
				id: include.id,
				code: artifact.source.code,
				sourcePath:
					artifact.source.sourceMap.segments[0]?.sourcePath ??
					`runtime://ignis/includes/${manifest.language}/${include.id}`,
			};
		}),
	);
	const assetPack: ShaderDirectiveFeaturePack = {
		id: manifest.profile.assetPackId,
		backend: manifest.backend,
		revision: manifest.profile.assetPackRevision,
		includeModules: modules,
		injectionScripts: [],
	};
	const packs = manifest.profile.featurePacks.flatMap((pack) =>
		pack === "builtin-injections" ?
			createBuiltinInjectionFeaturePacks(manifest.backend)
		:	[],
	);
	return {
		id: manifest.profile.baseId,
		backend: manifest.backend,
		packs: [assetPack, ...packs],
	};
}

export function createShaderDirectiveProfileFromManifest(
	manifest: ShaderBackendManifest,
	base: ShaderDirectiveProfileBase,
	params: unknown,
): ShaderDirectiveProfile {
	validateShaderBackendManifest(manifest);
	const overlay = manifest.profile.overlay;
	const parameters = normalizeParameter(
		overlay.parameters,
		params,
		`${manifest.backend} shader profile`,
	);
	const lines: string[] = [];
	if (overlay.baseInclude) lines.push(`#include <${overlay.baseInclude}>`);
	for (const [name, expression] of Object.entries(overlay.defines)) {
		lines.push(`#define ${name} ${formatDefineValue(evaluate(expression, parameters))}`);
	}
	return composeShaderDirectiveProfile(base, {
		id: overlay.id,
		backend: manifest.backend,
		includeModules: [
			{
				language: manifest.language,
				id: overlay.includeId,
				code: lines.join("\n"),
				sourcePath: overlay.sourcePath,
			},
		],
	});
}

async function buildNode(
	manifest: ShaderBackendManifest,
	node: ShaderSourceNode,
	parameters: unknown,
	context: ShaderManifestBuildContext,
): Promise<CompositeShaderSource> {
	if ("asset" in node) return context.loadAsset(manifest, node.asset);
	if ("source" in node) {
		const artifact = await context.loadSource(manifest, node.source);
		if (artifact.kind !== "module") {
			throw new Error(`Shader source node "${node.source}" is not a module.`);
		}
		return artifact.source;
	}
	if ("concat" in node) {
		const parts = await Promise.all(
			node.concat.map((child) => buildNode(manifest, child, parameters, context)),
		);
		return composeCompositeShaderSources(
			parts.map((part) => ({
				code: part.code,
				sourceMap: part.sourceMap,
				sourcePath:
					part.sourceMap.segments[0]?.sourcePath ??
					node.fallbackSourcePath ??
					"<shader-manifest-part>",
				kind: "template" as const,
			})),
			node.separator ?? "\n\n",
			"template",
		);
	}
	if ("when" in node) {
		const selected = evaluate(node.when, parameters) ? node.then : node.else;
		return selected ?
			buildNode(manifest, selected, parameters, context)
		:	createInlineCompositeShaderSource("", "<shader-manifest-empty>", "generated");
	}
	if ("defines" in node) {
		const code = Object.entries(node.defines)
			.map(
				([name, expression]) =>
					`#define ${name} ${formatDefineValue(evaluate(expression, parameters))}`,
			)
			.join("\n");
		return createInlineCompositeShaderSource(code, node.sourcePath, "generated");
	}
	const template = await buildNode(manifest, node.template, parameters, context);
	let code = template.code;
	for (const block of node.blocks ?? []) {
		code = replaceConditionalBlock(
			code,
			block.start,
			block.end,
			Boolean(evaluate(block.when, parameters)),
		);
	}
	for (const replacement of node.replacements ?? []) {
		const value = await buildNode(
			manifest,
			replacement.value,
			parameters,
			context,
		);
		if (!code.includes(replacement.marker)) {
			throw new Error(`Shader template marker "${replacement.marker}" was not found.`);
		}
		code = code.replaceAll(replacement.marker, value.code);
	}
	return createInlineCompositeShaderSource(
		code,
		template.sourceMap.segments[0]?.sourcePath ?? "<shader-manifest-template>",
		"template",
	);
}

function replaceConditionalBlock(
	code: string,
	start: string,
	end: string,
	enabled: boolean,
): string {
	const startIndex = code.indexOf(start);
	const endIndex = code.indexOf(end, startIndex + start.length);
	if (startIndex < 0 || endIndex < 0) {
		throw new Error(`Shader conditional block "${start}" / "${end}" was not found.`);
	}
	const contentStart = startIndex + start.length;
	return enabled ?
		code.slice(0, startIndex) + code.slice(contentStart)
	:	code.slice(0, startIndex) + code.slice(endIndex);
}

function normalizeParameter(
	schema: ShaderParameterSchema,
	value: unknown,
	path: string,
): unknown {
	if (schema.type === "boolean") {
		return value === undefined ? schema.default === true : value === true;
	}
	if (schema.type === "enum") {
		return typeof value === "string" && schema.values.includes(value) ?
			value
		:	schema.default;
	}
	if (schema.type === "integer") {
		if (value === undefined && schema.required && schema.default === undefined) {
			throw new Error(`${path} is required.`);
		}
		let result = Number.isFinite(value) ? Math.floor(value as number) :
			schema.default ?? 0;
		if (schema.min !== undefined) result = Math.max(schema.min, result);
		if (schema.max !== undefined) result = Math.min(schema.max, result);
		if (schema.bitMask !== undefined) result &= schema.bitMask;
		return result;
	}
	const input =
		value && typeof value === "object" ?
			(value as Record<string, unknown>)
		: schema.default ?? {};
	const result: Record<string, unknown> = {};
	for (const [key, field] of Object.entries(schema.fields)) {
		result[key] = normalizeParameter(field, input[key], `${path}.${key}`);
	}
	return result;
}

function evaluate(
	expression: ShaderManifestExpression,
	parameters: unknown,
): ShaderManifestPrimitive {
	if ("literal" in expression) return expression.literal;
	if ("parameter" in expression) return readParameter(parameters, expression.parameter);
	if ("equals" in expression) {
		return evaluate(expression.equals[0], parameters) ===
			evaluate(expression.equals[1], parameters);
	}
	if ("all" in expression) {
		return expression.all.every((value) => Boolean(evaluate(value, parameters)));
	}
	if ("any" in expression) {
		return expression.any.some((value) => Boolean(evaluate(value, parameters)));
	}
	if ("not" in expression) return !Boolean(evaluate(expression.not, parameters));
	if ("multiply" in expression) {
		return Number(evaluate(expression.multiply[0], parameters)) *
			Number(evaluate(expression.multiply[1], parameters));
	}
	for (const entry of expression.select.cases) {
		if (evaluate(entry.when, parameters)) return evaluate(entry.value, parameters);
	}
	return evaluate(expression.select.fallback, parameters);
}

function readParameter(parameters: unknown, path: string): ShaderManifestPrimitive {
	let value: unknown = parameters;
	for (const part of path.split(".")) {
		if (!value || typeof value !== "object") {
			throw new Error(`Shader manifest parameter "${path}" is unavailable.`);
		}
		value = (value as Record<string, unknown>)[part];
	}
	if (
		typeof value !== "string" &&
		typeof value !== "number" &&
		typeof value !== "boolean"
	) {
		throw new Error(`Shader manifest parameter "${path}" is not primitive.`);
	}
	return value;
}

function formatDefineValue(value: ShaderManifestPrimitive): string {
	return typeof value === "boolean" ? (value ? "1" : "0") : String(value);
}

function stableSerialize(value: unknown): string {
	if (!value || typeof value !== "object") return JSON.stringify(value);
	if (Array.isArray(value)) return `[${value.map(stableSerialize).join(",")}]`;
	return `{${Object.keys(value as Record<string, unknown>)
		.sort()
		.map(
			(key) =>
				`${JSON.stringify(key)}:${stableSerialize((value as Record<string, unknown>)[key])}`,
		)
		.join(",")}}`;
}

function cloneComposite(source: CompositeShaderSource): CompositeShaderSource {
	return {
		code: source.code,
		sourceMap: {
			schemaVersion: source.sourceMap.schemaVersion,
			lineCount: source.sourceMap.lineCount,
			segments: source.sourceMap.segments.map((segment) => ({ ...segment })),
		},
	};
}
