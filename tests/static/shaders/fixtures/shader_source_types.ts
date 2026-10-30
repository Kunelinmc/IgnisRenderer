import {
	ShaderSource,
	type ShaderSourceKey,
} from "../../../../src/shaders/ShaderSource.ts";
import { WEBGL_SHADER_MANIFEST } from "../../../../src/shaders/webgl/sources.ts";
import { WEBGPU_SHADER_MANIFEST } from "../../../../src/shaders/webgpu/sources.ts";
import {
	WEBGL_ALPHA_MAP_DEPTH_VARIANT,
	WEBGL_FULL_SCENE_VARIANT,
} from "../../../../src/shaders/webgl/sceneVariants.ts";
import {
	prepareShaderDirectiveProfileBase,
	type ShaderBackendManifest,
} from "../../../../src/shaders/ShaderManifest.ts";

await prepareShaderDirectiveProfileBase(WEBGL_SHADER_MANIFEST, (key) => {
	const include: typeof WEBGL_SHADER_MANIFEST.profile.includes[number]["source"] = key;
	return ShaderSource.load(include);
});
await prepareShaderDirectiveProfileBase(WEBGPU_SHADER_MANIFEST,
	(key) => ShaderSource.load(key));
await ShaderSource.prepareMany(
	WEBGL_SHADER_MANIFEST.preloadGroups.backendInit.map((key) => ({ key })),
);
await ShaderSource.prepareMany(
	WEBGPU_SHADER_MANIFEST.preloadGroups.backendInit.map((key) => ({ key })),
);

const invalidProfileSource: ShaderBackendManifest<typeof WEBGL_SHADER_MANIFEST.sources> = {
	...WEBGL_SHADER_MANIFEST,
	profile: {
		...WEBGL_SHADER_MANIFEST.profile,
		includes: [{
			id: "unknown-source",
			// @ts-expect-error Profile references must exist in the owning manifest.
			source: "webgl.directive.typo",
		}],
	},
};
const invalidProfileProgram: ShaderBackendManifest<typeof WEBGL_SHADER_MANIFEST.sources> = {
	...WEBGL_SHADER_MANIFEST,
	profile: {
		...WEBGL_SHADER_MANIFEST.profile,
		includes: [{
			id: "program-source",
			// @ts-expect-error Profiles can include modules, not multi-stage programs.
			source: "webgl.scene",
		}],
	},
};
const invalidPreload: ShaderBackendManifest<typeof WEBGL_SHADER_MANIFEST.sources> = {
	...WEBGL_SHADER_MANIFEST,
	preloadGroups: {
		backendInit: [
			// @ts-expect-error Preload references must exist in the owning manifest.
			"webgl.part.typo",
			// @ts-expect-error Another backend's sources do not belong to this manifest.
			"webgpu.scene",
		],
	},
};
await prepareShaderDirectiveProfileBase(WEBGL_SHADER_MANIFEST,
	// @ts-expect-error The include loader must return module artifacts.
	() => ShaderSource.load("webgl.scene"));

const unknownIncludeManifest = {
	...WEBGL_SHADER_MANIFEST,
	profile: {
		...WEBGL_SHADER_MANIFEST.profile,
		includes: [{ id: "unknown-source", source: "webgl.directive.typo" }],
	},
} as const;
await prepareShaderDirectiveProfileBase(
	// @ts-expect-error Profile preparation must not widen the source map to accept a typo.
	unknownIncludeManifest,
	() => ShaderSource.load("webgl.directive.animation"),
);

// Every manifest source must be usable without a separately maintained key type.
export function loadManifestSource(
	key: keyof typeof WEBGL_SHADER_MANIFEST.sources
		| keyof typeof WEBGPU_SHADER_MANIFEST.sources,
): void {
	void ShaderSource.load(key);
}

await ShaderSource.load("webgpu.directive.animation");
export async function prepareDynamicKey(key: ShaderSourceKey): Promise<void> {
	await ShaderSource.prepareMany([{ key }]);
}

await ShaderSource.prepareMany([
	{ key: "webgl.scene", params: { specialization: WEBGL_FULL_SCENE_VARIANT } },
	{ key: "webgpu.scene" },
]);
const moduleSource = await ShaderSource.load("webgpu.scene");
moduleSource.source.code;
// @ts-expect-error Module artifacts do not contain program stages.
moduleSource.stages;

const programSource = await ShaderSource.load("webgl.scene", {
	specialization: WEBGL_FULL_SCENE_VARIANT,
});
programSource.stages.vertex?.code;
// @ts-expect-error Program artifacts do not contain a single module source.
programSource.source;

await ShaderSource.load("webgl.scene.depth", {
	specialization: WEBGL_ALPHA_MAP_DEPTH_VARIANT,
});
await ShaderSource.load("webgl.shadow.depth", {
	specialization: { skinProfile: "skin4", morphPosition: true },
});
ShaderSource.getSync("webgpu.utility.mipmapBlit").source.code;
ShaderSource.getSync("webgl.directive.animation").source.code;

// @ts-expect-error Shader names must exist in a backend manifest.
await ShaderSource.load("webgpu.postprocess.typo");
// @ts-expect-error There is no WebGL fog directive in the manifest.
await ShaderSource.load("webgl.directive.fog");
// @ts-expect-error Fixed WebGPU modules do not accept specialization parameters.
await ShaderSource.load("webgpu.scene", { specialization: WEBGL_FULL_SCENE_VARIANT });
// @ts-expect-error Scene parameters must use the scene schema, not the depth schema.
await ShaderSource.load("webgl.scene", { specialization: WEBGL_ALPHA_MAP_DEPTH_VARIANT });
// @ts-expect-error Enum values must stay restricted to the schema's literals.
await ShaderSource.load("webgl.shadow.depth", { specialization: { skinProfile: "skin16", morphPosition: false } });
// @ts-expect-error Nested parameters must reject unknown properties.
await ShaderSource.load("webgl.shadow.depth", { specialization: { skinProfile: "static", morphPosition: false, typo: true } });
// @ts-expect-error Sources without a direct synchronous asset require preparation.
ShaderSource.getSync("webgpu.scene");
// @ts-expect-error An asynchronous direct asset must not become a sync source.
ShaderSource.getSync("webgpu.utility.present");
// @ts-expect-error Batch requests must preserve the relationship between key and params.
await ShaderSource.prepareMany([{ key: "webgpu.scene", params: { specialization: WEBGL_FULL_SCENE_VARIANT } }]);
