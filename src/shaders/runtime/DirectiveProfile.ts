import {
	canonicalizeModulePath,
	normalizeInjectionScript,
} from "./runtimeShared";
import type {
	ShaderDirectiveFeaturePack,
	ShaderDirectiveProfile,
	ShaderDirectiveProfileBase,
	ShaderDirectiveProfileOverlay,
	ShaderIncludeModule,
	ShaderInjectionArgumentSchema,
	ShaderInjectionScript,
} from "./types";

function hashStringFNV1a(value: string): string {
	let hash = 0x811c9dc5;
	for (let index = 0; index < value.length; index++) {
		hash ^= value.charCodeAt(index);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16).padStart(8, "0");
}

function normalizeId(value: string, label: string): string {
	const normalized = typeof value === "string" ? value.trim() : "";
	if (normalized.length <= 0) {
		throw new Error(`${label} must be a non-empty string.`);
	}
	return normalized;
}

function normalizeModule(module: ShaderIncludeModule): ShaderIncludeModule {
	const id = normalizeId(module.id, "Shader directive include module id");
	if (module.language !== "wgsl" && module.language !== "glsl") {
		throw new Error(
			`Shader directive include module "${id}" language must be wgsl or glsl.`,
		);
	}
	if (typeof module.code !== "string") {
		throw new Error(
			`Shader directive include module "${id}" code must be a string.`,
		);
	}
	return Object.freeze({
		language: module.language,
		id,
		code: module.code,
		sourcePath:
			typeof module.sourcePath === "string" && module.sourcePath.trim().length > 0 ?
				module.sourcePath.trim()
			:	undefined,
	});
}

function stableSchemaValue(schema: ShaderInjectionArgumentSchema): unknown {
	return Object.keys(schema)
		.sort()
		.map((key) => {
			const definition = schema[key];
			return [
				key,
				definition.type,
				definition.required === true,
				"default" in definition ? definition.default ?? null : null,
				"min" in definition ? definition.min ?? null : null,
				"max" in definition ? definition.max ?? null : null,
				"values" in definition ? [...definition.values] : null,
			];
		});
}

function scriptFingerprintValue(script: ShaderInjectionScript): unknown {
	return [
		script.id,
		script.language ?? null,
		script.description ?? null,
		[...(script.symbols ?? [])],
		stableSchemaValue(script.arguments),
	];
}

function moduleFingerprintValue(module: ShaderIncludeModule): unknown {
	return [
		module.language,
		canonicalizeModulePath(module.id),
		module.sourcePath ?? null,
		module.code,
	];
}

function registerModule(
	module: ShaderIncludeModule,
	modules: ShaderIncludeModule[],
	moduleKeys: Set<string>,
): void {
	const normalized = normalizeModule(module);
	const key = `${normalized.language}:${canonicalizeModulePath(normalized.id)}`;
	if (moduleKeys.has(key)) {
		throw new Error(`Duplicate shader directive include module "${key}".`);
	}
	moduleKeys.add(key);
	modules.push(normalized);
}

function registerScript(
	script: ShaderInjectionScript,
	scripts: ShaderInjectionScript[],
	scriptIds: Set<string>,
): void {
	const prepared = normalizeInjectionScript(script);
	const normalized = Object.freeze({
		...prepared,
		symbols: prepared.symbols ? Object.freeze([...prepared.symbols]) : undefined,
	});
	if (scriptIds.has(normalized.id)) {
		throw new Error(`Duplicate shader injection script "${normalized.id}".`);
	}
	scriptIds.add(normalized.id);
	scripts.push(normalized);
}

/** Defines an injection script while preserving argument-schema inference. */
export function defineShaderInjectionScript<
	const Schema extends ShaderInjectionArgumentSchema,
>(script: ShaderInjectionScript<Schema>): ShaderInjectionScript<Schema> {
	return script;
}

/** @internal Composes the immutable profile consumed by a backend compile stage. */
export function composeShaderDirectiveProfile(
	base: ShaderDirectiveProfileBase,
	overlay: ShaderDirectiveProfileOverlay,
): ShaderDirectiveProfile {
	if (base.backend !== overlay.backend) {
		throw new Error(
			`Shader directive profile overlay backend "${overlay.backend}" does not match base backend "${base.backend}".`,
		);
	}
	const baseId = normalizeId(base.id, "Shader directive profile base id");
	const overlayId = normalizeId(overlay.id, "Shader directive profile overlay id");
	const packIds = new Set<string>();
	const moduleKeys = new Set<string>();
	const scriptIds = new Set<string>();
	const modules: ShaderIncludeModule[] = [];
	const scripts: ShaderInjectionScript[] = [];
	const packs: Array<[string, number]> = [];

	for (const pack of base.packs) {
		registerFeaturePack(
			pack,
			base.backend,
			packIds,
			packs,
			modules,
			moduleKeys,
			scripts,
			scriptIds,
		);
	}
	for (const module of overlay.includeModules) {
		registerModule(module, modules, moduleKeys);
	}

	const fingerprint = hashStringFNV1a(
		JSON.stringify({
			backend: base.backend,
			baseId,
			overlayId,
			packs,
			modules: modules.map(moduleFingerprintValue),
			scripts: scripts.map(scriptFingerprintValue),
		}),
	);
	return Object.freeze({
		id: `${baseId}+${overlayId}`,
		backend: base.backend,
		fingerprint,
		includeModules: Object.freeze([...modules]),
		injectionScripts: Object.freeze([...scripts]),
	});
}

function registerFeaturePack(
	pack: ShaderDirectiveFeaturePack,
	backend: ShaderDirectiveProfileBase["backend"],
	packIds: Set<string>,
	packs: Array<[string, number]>,
	modules: ShaderIncludeModule[],
	moduleKeys: Set<string>,
	scripts: ShaderInjectionScript[],
	scriptIds: Set<string>,
): void {
	const id = normalizeId(pack.id, "Shader directive feature pack id");
	if (pack.backend !== backend) {
		throw new Error(
			`Shader directive feature pack "${id}" targets "${pack.backend}" instead of "${backend}".`,
		);
	}
	if (packIds.has(id)) {
		throw new Error(`Duplicate shader directive feature pack "${id}".`);
	}
	if (!Number.isInteger(pack.revision) || pack.revision < 0) {
		throw new Error(
			`Shader directive feature pack "${id}" revision must be a non-negative integer.`,
		);
	}
	packIds.add(id);
	packs.push([id, pack.revision]);
	for (const module of pack.includeModules) {
		registerModule(module, modules, moduleKeys);
	}
	for (const script of pack.injectionScripts) {
		registerScript(script, scripts, scriptIds);
	}
}
