import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
	ShaderSource,
} from "../../../src/shaders/ShaderSource.ts";
import { WEBGL_SHADER_MANIFEST } from "../../../src/shaders/webgl/sources.ts";
import { embeddedShaderSources } from "../../../src/shaders/generated/embeddedShaderSources.ts";
import {
	resolveShaderManifestRequest,
	validateShaderBackendManifest,
} from "../../../src/shaders/shaderManifest.ts";
import {
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../../src/shaders/runtime/index.ts";
import {
	WEBGL_TEST_PROFILE,
	WEBGPU_TEST_PROFILE,
} from "./shaderDirectiveTestProfiles.mjs";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import { WEBGL_FULL_SCENE_VARIANT } from "../../../src/shaders/webgl/sceneVariants.ts";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_QUALITY,
	VOLUMETRIC_SIGMA_T_SCALE,
} from "../../../src/postprocess/constants.ts";

const REPO_ROOT = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"..",
	"..",
	".."
);
const SHADER_ROOT = path.join(REPO_ROOT, "src", "shaders");

const WEBGL_SCENE_LIMITS = {
	maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: MAX_POINT_LIGHTS,
	maxSpotLights: MAX_SPOT_LIGHTS,
};

async function testLoadsRawAndCompositeParts() {
	ShaderSource.clearCache();

	const wgslRaw = (await ShaderSource.load("webgpu.scene.part.lightData")).source;
	const wgslComposite = (await ShaderSource.load(
		"webgpu.scene.part.lightData"
	)).source;
	const shadowRaw = (await ShaderSource.load("webgpu.shadow.depth")).source;
	const shadowComposite = (await ShaderSource.load(
		"webgpu.shadow.depth"
	)).source;
	const planarReflectionComposite = (await ShaderSource.load(
		"webgpu.utility.planarReflectionComposite"
	)).source;
	const glslRaw = (await ShaderSource.load("webgl.part.sceneVertex")).source;
	const glslComposite = (await ShaderSource.load(
		"webgl.part.sceneVertex"
	)).source;

	assert.ok(wgslRaw.code.includes("struct DirectionalLightData"));
	assert.ok(wgslComposite.code.includes("struct DirectionalLightData"));
	assert.ok(shadowRaw.code.includes("fn vsMain"));
	assert.ok(shadowRaw.code.includes("fn fsDepthClip"));
	assert.ok(shadowRaw.code.includes("discardOutsideAtlasPage(input);"));
	assert.ok(shadowRaw.code.includes("pixel.x >= input.atlasClipRect.z"));
	assert.ok(shadowComposite.code.includes("fn fsTransmittance"));
	assert.ok(
		planarReflectionComposite.code.includes(
			"morphPositionDeltas: array<f32>"
		)
	);
	assert.ok(planarReflectionComposite.code.includes("vertexCount: f32"));
	assert.ok(planarReflectionComposite.code.includes("morphSemanticMask: f32"));
	assert.ok(glslRaw.code.includes("layout(location = 0) in vec3 aPosition;"));
	assert.ok(glslComposite.code.includes("layout(location = 0) in vec3 aPosition;"));
	assert.equal(
		wgslComposite.sourceMap.segments[0].sourcePath,
		"./webgpu/common/lightData.wgsl"
	);
	assert.equal(
		shadowComposite.sourceMap.segments[0].sourcePath,
		"./webgpu/shadow/depth.wgsl"
	);
	assert.equal(
		glslComposite.sourceMap.segments[0].sourcePath,
		"./webgl/scene/sceneVertex.glsl"
	);
}

async function testConcurrentLoadsShareResultCache() {
	ShaderSource.clearCache();

	const [first, second] = await Promise.all([
		ShaderSource.load("webgpu.utility.present"),
		ShaderSource.load("webgpu.utility.present"),
	]);
	const stats = ShaderSource.getCacheStats();

	assert.equal(first.source.code, second.source.code);
	assert.ok(stats.results.misses >= 1);
	assert.ok(stats.results.hits >= 1);
}

async function testGetRequiresPrepare() {
	ShaderSource.clearCache();

	assert.throws(
		() => ShaderSource.get("webgpu.utility.present"),
		/Shader source "webgpu\.utility\.present" is not prepared/
	);
	await ShaderSource.prepare("webgpu.utility.present");
	const prepared = ShaderSource.get("webgpu.utility.present");

	assert.ok(prepared.source.code.includes("fn"));
}

async function testWebGLSceneVariants() {
	ShaderSource.clearCache();
	assert.ok(!WEBGL_SHADER_MANIFEST.sources["webgl.part.sceneFragment"]);
	const withoutShadowTransmittance = {
		...WEBGL_FULL_SCENE_VARIANT,
		scene: {
			...WEBGL_FULL_SCENE_VARIANT.scene,
			shadowTransmittance: false,
		},
	};
	await ShaderSource.prepareMany([
		...WEBGL_SHADER_MANIFEST.preloadGroups.backendInit.map((key) => ({ key })),
		{
			key: "webgl.scene",
			params: {
				specialization: withoutShadowTransmittance,
			},
		},
		{
			key: "webgl.scene",
			params: {
				specialization: WEBGL_FULL_SCENE_VARIANT,
			},
		},
		{
			key: "webgl.scene",
			params: {
				specialization: withoutShadowTransmittance,
			},
		},
	]);

	const raw = ShaderSource.get("webgl.scene", {
		specialization: withoutShadowTransmittance,
	});
	const withShadowTransmittance = ShaderSource.get("webgl.scene", {
		specialization: WEBGL_FULL_SCENE_VARIANT,
	});
	const composite = ShaderSource.get("webgl.scene", {
		specialization: withoutShadowTransmittance,
	});

	assert.ok(raw.stages.vertex.code.includes("layout(location = 0) in vec3 aPosition;"));
	assert.ok(raw.stages.vertex.code.includes("#import <ignis/webgl/animation>"));
	assert.ok(raw.stages.vertex.code.includes("#define IGNIS_WEBGL_SKIN_INFLUENCES 8"));
	assert.ok(raw.stages.fragment.code.includes("#import <ignis/webgl/constants>"));
	assert.ok(
		raw.stages.fragment.code.includes(
			"const int MAX_DIRECTIONAL_LIGHTS = __WEBGL_MAX_DIRECTIONAL_LIGHTS__;"
		)
	);
	assert.ok(
		raw.stages.fragment.code.includes(
			"const int MAX_POINT_LIGHTS = __WEBGL_MAX_POINT_LIGHTS__;"
		)
	);
	assert.ok(
		raw.stages.fragment.code.includes(
			"const int MAX_SPOT_LIGHTS = __WEBGL_MAX_SPOT_LIGHTS__;"
		)
	);
	assert.ok(
		raw.stages.fragment.code.includes(
			"const int MAX_LOCAL_LIGHT_PROBES = __WEBGL_MAX_LOCAL_LIGHT_PROBES__;"
		)
	);
	assert.ok(
		raw.stages.fragment.code.includes(
			"const int MAX_REFLECTION_PROBES = __WEBGL_MAX_REFLECTION_PROBES__;"
		)
	);
	assert.ok(!raw.stages.fragment.code.includes("__MAX_DIRECTIONAL_LIGHTS__"));
	assert.ok(
		raw.stages.fragment.code.includes("vec3 calculateIrradianceFromSH(vec3 normal)")
	);
	assert.ok(
		raw.stages.fragment.code.includes("vec4 sampleBlendedLocalLightProbeIrradiance(")
	);
	assert.ok(!raw.stages.fragment.code.includes("WEBGL_SHADOW_TRANSMITTANCE 1"));
	assert.ok(
		withShadowTransmittance.stages.fragment.code.includes(
			"#define WEBGL_SHADOW_TRANSMITTANCE 1"
		)
	);
	assert.ok(
		withShadowTransmittance.stages.fragment.code.includes(
			"uniform sampler2D uShadowTransmittanceAtlas;"
		)
	);
	assert.equal(
		composite.stages.fragment.sourceMap.segments[0].sourcePath,
		"./webgl/scene/fragmentPrelude.glsl"
	);
	assert.ok(
		composite.stages.fragment.sourceMap.segments.some(
			(segment) =>
				segment.sourcePath === "./webgl/scene/fragmentMainOutput.glsl"
		)
	);

	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGL_TEST_PROFILE,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.stages.fragment.code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragment",
		sourceKind: "builtin-scene",
		sourceMap: composite.stages.fragment.sourceMap,
	});
	assert.equal(compiled.hasErrors, false);
	assert.ok(
		compiled.code.includes(
			`const int MAX_DIRECTIONAL_LIGHTS = ${MAX_DIRECTIONAL_LIGHTS};`
		)
	);
	assert.ok(
		compiled.code.includes(
			`const int MAX_POINT_LIGHTS = ${MAX_POINT_LIGHTS};`
		)
	);
	assert.ok(
		compiled.code.includes(
			`const int MAX_SPOT_LIGHTS = ${MAX_SPOT_LIGHTS};`
		)
	);
	assert.ok(
		compiled.code.includes(
			`const int MAX_LOCAL_LIGHT_PROBES = ${MAX_LOCAL_LIGHT_PROBES};`
		)
	);
	assert.ok(
		compiled.code.includes(
			`const int MAX_REFLECTION_PROBES = ${MAX_REFLECTION_PROBES};`
		)
	);
	assert.ok(!compiled.code.includes("__WEBGL_MAX_DIRECTIONAL_LIGHTS__"));
}

async function testWebGLScenePrunedVariant() {
	ShaderSource.clearCache();
	const variant = {
		output: "single",
		oit: false,
		scene: {
			shadows: false,
			shadowTransmittance: false,
			clusteredLighting: false,
			sh: false,
			localLightProbes: false,
			irradianceProbeGrid: false,
			reflectionProbes: false,
			environmentSpecular: false,
		},
		material: {
			model: "unlit",
			baseMap: false,
			metallicRoughnessMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			iridescence: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropy: false,
			anisotropyMap: false,
			transmission: false,
			alphaMask: false,
		},
	};
	await ShaderSource.prepareMany([
		{ key: "webgl.scene", params: { specialization: variant } },
		{
			key: "webgl.scene",
			params: { specialization: variant },
		},
	]);
	const raw = ShaderSource.get("webgl.scene", {
		specialization: variant,
	});
	const composite = ShaderSource.get("webgl.scene", {
		specialization: variant,
	});

	assert.ok(!raw.stages.fragment.code.includes("uniform sampler2D uShadowAtlas;"));
	assert.ok(!raw.stages.fragment.code.includes("uniform int uEnableClusteredLighting;"));
	assert.ok(!raw.stages.fragment.code.includes("uniform int uEnableSH;"));
	assert.ok(!raw.stages.fragment.code.includes("uniform sampler2D uNormalMap;"));
	assert.ok(!raw.stages.fragment.code.includes("vec3 shadePBR("));
	assert.ok(!raw.stages.fragment.code.includes("vec3 shadePhong("));
	assert.ok(
		composite.stages.fragment.sourceMap.segments.some(
			(segment) =>
				segment.sourcePath === "./webgl/scene/fragmentUniforms.glsl"
		)
	);

	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGL_TEST_PROFILE,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.stages.fragment.code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragmentPruned",
		sourceKind: "builtin-scene",
		sourceMap: composite.stages.fragment.sourceMap,
	});
	assert.equal(compiled.hasErrors, false);
	assert.ok(!compiled.code.includes("WEBGL_MATERIAL_MODEL_FULL"));
	assert.ok(!compiled.code.includes("uShadowAtlas"));
	assert.ok(!compiled.code.includes("uEnableClusteredLighting"));
}

async function testWebGLPhongSceneVariantIncludesMaterialBlock() {
	ShaderSource.clearCache();
	const variant = {
		output: "mrt",
		materialGBuffer: false,
		oit: false,
		scene: {
			shadows: true,
			shadowTransmittance: false,
			clusteredLighting: false,
			sh: false,
			localLightProbes: false,
			irradianceProbeGrid: false,
			reflectionProbes: false,
			environmentSpecular: false,
		},
		material: {
			model: "phong",
			baseMap: false,
			metallicRoughnessMap: false,
			specularMap: false,
			specularColorMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			clearcoat: false,
			clearcoatMap: false,
			clearcoatRoughnessMap: false,
			clearcoatNormalMap: false,
			sheen: false,
			sheenColorMap: false,
			sheenRoughnessMap: false,
			iridescence: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropy: false,
			anisotropyMap: false,
			transmission: false,
			transmissionMap: false,
			thicknessMap: false,
			alphaMask: false,
		},
	};
	const raw = await ShaderSource.load("webgl.scene", {
		specialization: variant,
	});
	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGL_TEST_PROFILE,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.stages.fragment.code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragmentPhong",
		sourceKind: "builtin-scene",
	});

	assert.ok(raw.stages.fragment.code.includes("uniform IgnisPhongMaterial"));
	assert.ok(raw.stages.fragment.code.includes("vec3 shadePhong("));
	assert.ok(raw.stages.fragment.code.includes("uSpecular.rgb"));
	assert.ok(!raw.stages.fragment.code.includes("uniform vec4 uPBR;"));
	assert.equal(compiled.hasErrors, false);
	assert.ok(compiled.code.includes("uniform IgnisPhongMaterial"));
	assert.ok(compiled.code.includes("vec4 ignisSpecular;"));
	assert.ok(compiled.code.includes("vec3 shadowNormal = normal;"));
	assert.ok(compiled.code.includes(
		"color = shadePhong(albedo, normal, shadowNormal, viewDir);"
	));
}

async function testWebGLPBRSceneVariantCompilesExactUniformBlocks() {
	ShaderSource.clearCache();
	const material = Object.fromEntries(
		Object.keys(WEBGL_FULL_SCENE_VARIANT.material).map((name) => [name, false]),
	);
	material.model = "pbr";
	material.normalMap = true;
	const variant = {
		...WEBGL_FULL_SCENE_VARIANT,
		output: "single",
		materialGBuffer: false,
		oit: false,
		scene: Object.fromEntries(
			Object.keys(WEBGL_FULL_SCENE_VARIANT.scene).map((name) => [name, false]),
		),
		material,
		skinProfile: "static",
		morphSemanticMask: 0,
	};
	const raw = await ShaderSource.load("webgl.scene", {
		specialization: variant,
	});
	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGL_TEST_PROFILE,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.stages.fragment.code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragmentPBRUniformBlocks",
		sourceKind: "builtin-scene",
	});
	assert.equal(compiled.hasErrors, false);
	assert.ok(compiled.code.includes("uniform IgnisMaterialCommon"));
	assert.ok(compiled.code.includes("uniform IgnisPBRMaterial"));
	assert.ok(compiled.code.includes("ignisNormalMapTransformA"));
	assert.ok(!compiled.code.includes("ignisSheenColorMapTransformA"));
	assert.ok(!compiled.code.includes("uniform vec4 uPBR;"));
}

async function testWebGPUCompositeIncludesSharedParts() {
	ShaderSource.clearCache();

	const ssr = (await ShaderSource.load("webgpu.postprocess.ssr")).source;
	const deferred = (await ShaderSource.load("webgpu.deferredLighting")).source;

	assert.ok(ssr.code.includes("struct DirectionalLightData"));
	assert.ok(ssr.code.includes("struct TraceParams"));
	assert.ok(deferred.code.includes("struct DirectionalLightData"));
	assert.ok(deferred.code.includes("fn activeClusteredLightCount() -> u32"));
}

async function testSrgbDirectiveSupportsBuiltInConsumers() {
	ShaderSource.clearCache();
	const [webgpuIbl, webgpuDecal, webglIbl] = await Promise.all([
		ShaderSource.load("webgpu.iblPrefilter"),
		ShaderSource.load("webgpu.utility.decal"),
		ShaderSource.load("webgl.part.iblPrefilterFragment"),
	]);
	const webgpuCompileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGPU_TEST_PROFILE,
		mode: "strict",
	});
	const webglCompileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGL_TEST_PROFILE,
		mode: "strict",
	});
	const compiledWebgpuIbl = webgpuCompileStage.compile({
		code: webgpuIbl.source.code,
		language: "wgsl",
		stage: "compute",
		entryPoint: "csMain",
		label: "WebGPUIblPrefilter",
		sourceKind: "builtin-environment",
	});
	const compiledWebgpuDecal = webgpuCompileStage.compile({
		code: webgpuDecal.source.code,
		language: "wgsl",
		stage: "unknown",
		entryPoint: "fsMain",
		label: "WebGPUDecal",
		sourceKind: "builtin-scene",
		sourceMap: webgpuDecal.source.sourceMap,
	});
	const compiledWebglIbl = webglCompileStage.compile({
		code: webglIbl.source.code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLIblPrefilter",
		sourceKind: "builtin-environment",
	});

	assert.equal(compiledWebgpuIbl.hasErrors, false);
	assert.equal(compiledWebgpuDecal.hasErrors, false);
	assert.equal(compiledWebglIbl.hasErrors, false);
	assert.equal(
		compiledWebgpuIbl.code.match(/fn srgbToLinear\b/g)?.length,
		1
	);
	assert.equal(
		compiledWebgpuDecal.code.match(/fn srgbToLinear\b/g)?.length,
		1
	);
	assert.equal(
		compiledWebglIbl.code.match(/vec3 srgbToLinear\b/g)?.length,
		1
	);
	assert.ok(!compiledWebgpuIbl.code.includes("sRGBToLinear"));
	assert.ok(!compiledWebglIbl.code.includes("sRGBToLinear"));
}

async function testWebGPUSharedNumericalConstants() {
	ShaderSource.clearCache();
	const [constants, particleSimulation, ssao] = await Promise.all([
		ShaderSource.load("webgpu.directive.constants"),
		ShaderSource.load("webgpu.particleSimulation"),
		ShaderSource.load("webgpu.postprocess.ssao"),
	]);
	const compileStage = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }),
		profile: WEBGPU_TEST_PROFILE,
		mode: "strict",
	});
	const compiledParticleSimulation = compileStage.compile({
		code: particleSimulation.source.code,
		language: "wgsl",
		stage: "compute",
		entryPoint: "simulateMain",
		label: "WebGPUParticleSimulation",
		sourceKind: "unknown",
	});
	const compiledSsao = compileStage.compile({
		code: ssao.source.code,
		language: "wgsl",
		stage: "compute",
		entryPoint: "csRaw",
		label: "WebGPUSsao",
		sourceKind: "postprocess",
	});

	assert.ok(constants.source.code.includes("const PI_SQUARED: f32 = 9.86960440109;"));
	assert.ok(constants.source.code.includes("const GOLDEN_RATIO_CONJUGATE: f32 = 0.61803398875;"));
	assert.ok(
		constants.source.code.includes("const STEFAN_BOLTZMANN_CONSTANT: f32 = 5.670374419e-8;")
	);
	assert.ok(particleSimulation.source.code.includes("#import <ignis/webgpu/constants>"));
	assert.ok(particleSimulation.source.code.includes("nextRandom(seed) * TWO_PI"));
	assert.ok(!particleSimulation.source.code.includes("6.28318530718"));
	assert.ok(!ssao.source.code.includes("const GOLDEN_RATIO_CONJUGATE"));
	assert.equal(compiledParticleSimulation.hasErrors, false);
	assert.equal(compiledSsao.hasErrors, false);
}

function testCompositeResultsAreCloned() {
	ShaderSource.clearCache();
	return ShaderSource.load("webgpu.utility.present").then((first) => {
		first.source.sourceMap.segments[0].sourcePath = "mutated";
		return ShaderSource.load("webgpu.utility.present").then((second) => {
			assert.notEqual(second.source.sourceMap.segments[0].sourcePath, "mutated");
		});
	});
}

function testSyncLoadPopulatesPreparedCache() {
	ShaderSource.clearCache();

	assert.equal(ShaderSource.has("webgpu.utility.mipmapBlit"), false);
	const source = ShaderSource.getSync("webgpu.utility.mipmapBlit");

	assert.ok(source.source.code.includes("fn vsMain"));
	assert.equal(ShaderSource.has("webgpu.utility.mipmapBlit"), true);
	assert.equal(ShaderSource.get("webgpu.utility.mipmapBlit").source.code, source.source.code);
}

async function testCustomAsyncLoaderOverridesBuiltInSource() {
	ShaderSource.configure({
		loader: async (descriptor) => {
			assert.equal(descriptor.path, "./webgpu/common/lightData.wgsl");
			return "struct DirectionalLightData { value: f32, };";
		},
		preferCustomLoader: true,
	});

	try {
		const source = (await ShaderSource.load("webgpu.scene.part.lightData")).source;
		assert.equal(source.code, "struct DirectionalLightData { value: f32, };");
	} finally {
		ShaderSource.resetConfiguration();
	}
}

async function testCustomLoaderFailureFallsBackToBuiltIns() {
	ShaderSource.configure({
		loader: async () => {
			throw new Error("custom loader unavailable");
		},
	});

	try {
		const source = (await ShaderSource.load("webgpu.utility.present")).source;
		assert.ok(source.code.includes("fn"));
	} finally {
		ShaderSource.resetConfiguration();
	}
}

function testCustomSyncLoaderOverridesBuiltInSource() {
	ShaderSource.configure({
		syncLoader: (descriptor) =>
			descriptor.path === "./webgpu/utility/mipmapBlit.wgsl" ?
				"@compute @workgroup_size(1) fn csMain() {}"
			:	undefined,
		preferCustomLoader: true,
	});

	try {
		const source = ShaderSource.getSync("webgpu.utility.mipmapBlit");
		assert.equal(source.source.code, "@compute @workgroup_size(1) fn csMain() {}");
	} finally {
		ShaderSource.resetConfiguration();
	}
}

function testMissingCustomSyncLoaderReportsClearError() {
	ShaderSource.configure({
		loader: async () => "unused",
		preferCustomLoader: true,
	});

	try {
		assert.throws(
			() => ShaderSource.getSync("webgpu.utility.mipmapBlit"),
			/requires a syncLoader/
		);
	} finally {
		ShaderSource.resetConfiguration();
	}
}

function testClearCacheDoesNotTouchShaderRuntime() {
	ShaderSource.clearCache();
	const runtime = new ShaderRuntime({ mode: "warn" });
	const request = {
		code: "fn main() -> void {}",
		language: "wgsl",
		stage: "compute",
		entryPoint: "main",
		label: "shader-source-cache-isolation",
	};

	runtime.process(request);
	runtime.process(request);
	const before = runtime.getCacheStats("sync");
	ShaderSource.clearCache();
	const after = runtime.getCacheStats("sync");

	assert.equal(before.size, after.size);
	assert.equal(before.hits, after.hits);
}

async function testWebGLDepthAndShadowArtifacts() {
	const depthSpecialization = {
		alphaMask: true,
		baseMap: true,
		skinProfile: "skin8",
		morphPosition: true,
	};
	const shadowSpecialization = {
		skinProfile: "skin4",
		morphPosition: true,
	};
	const [depth, shadow] = await Promise.all([
		ShaderSource.load("webgl.scene.depth", {
			specialization: depthSpecialization,
		}),
		ShaderSource.load("webgl.shadow.depth", {
			specialization: shadowSpecialization,
		}),
	]);
	assert.equal(depth.kind, "program");
	assert.match(depth.stages.vertex.code, /IGNIS_WEBGL_SKIN_INFLUENCES 8/);
	assert.match(depth.stages.fragment.code, /WEBGL_DEPTH_ALPHA_MASK 1/);
	assert.match(shadow.stages.vertex.code, /IGNIS_WEBGL_SKIN_INFLUENCES 4/);
	assert.doesNotMatch(shadow.stages.vertex.code, /__IGNIS_WEBGL_ANIMATION_DEFINES__/);
	assert.notEqual(depth.identity, shadow.identity);
}

function testShaderManifestValidation() {
	const profile = {
		baseId: "test-base",
		revision: 1,
		includes: [],
		overlay: {
			id: "test-overlay",
			includeId: "test/constants",
			sourcePath: "runtime://test/constants",
			parameters: { type: "record", fields: {} },
			defines: {},
		},
	};
	const base = {
		backend: "webgpu",
		language: "wgsl",
		assets: { main: { path: "./webgpu/test.wgsl" } },
		profile,
	};
	assert.throws(
		() => validateShaderBackendManifest({
			...base,
			sources: {
				broken: { kind: "module", sourceKind: "unknown", source: { asset: "missing" } },
			},
		}),
		/unknown asset "missing"/,
	);
	assert.throws(
		() => validateShaderBackendManifest({
			...base,
			sources: {
				a: { kind: "module", sourceKind: "unknown", source: { source: "b" } },
				b: { kind: "module", sourceKind: "unknown", source: { source: "a" } },
			},
		}),
		/composition cycle/,
	);
	assert.throws(
		() => validateShaderBackendManifest({
			...base,
			sources: {
				broken: {
					kind: "module",
					sourceKind: "unknown",
					source: {
						when: { unsupported: true },
						then: { asset: "main" },
					},
				},
			},
		}),
		/Unsupported shader manifest expression/,
	);
}

function testShaderManifestIdentityUsesNormalizedCanonicalParameters() {
	const manifest = {
		backend: "webgpu",
		language: "wgsl",
		assets: { main: { path: "./webgpu/test.wgsl" } },
		sources: {
			test: {
				kind: "module",
				sourceKind: "unknown",
				parameters: {
					type: "record",
					fields: {
						zeta: { type: "integer", default: 3, min: 0, max: 7 },
						nested: {
							type: "record",
							fields: {
								toggle: { type: "boolean", default: false },
								mode: {
									type: "enum",
									values: ["a", "b"],
									default: "a",
								},
							},
						},
					},
				},
				source: { asset: "main" },
			},
		},
		profile: {
			baseId: "test-base",
			revision: 1,
			includes: [],
			overlay: {
				id: "test-overlay",
				includeId: "test/constants",
				sourcePath: "runtime://test/constants",
				parameters: { type: "record", fields: {} },
				defines: {},
			},
		},
	};
	const first = resolveShaderManifestRequest(manifest, "test", {
		zeta: 99,
		nested: { toggle: true, mode: "b", ignored: true },
	});
	const second = resolveShaderManifestRequest(manifest, "test", {
		nested: { mode: "b", toggle: true },
		zeta: 7,
	});

	assert.deepEqual(first.parameters, {
		nested: { mode: "b", toggle: true },
		zeta: 7,
	});
	assert.equal(first.identity, second.identity);
	assert.equal(
		first.identity,
		'test|{"nested":{"mode":"b","toggle":true},"zeta":7}',
	);
}

async function collectShaderFiles(relativeRoot, extension) {
	const root = path.join(SHADER_ROOT, relativeRoot);
	const files = [];

	async function walk(directory) {
		const entries = await readdir(directory, { withFileTypes: true });
		for (const entry of entries) {
			const fullPath = path.join(directory, entry.name);
			if (entry.isDirectory()) {
				await walk(fullPath);
				continue;
			}
			if (!entry.isFile() || !entry.name.endsWith(extension)) {
				continue;
			}
			const relativePath = path
				.relative(SHADER_ROOT, fullPath)
				.split(path.sep)
				.join("/");
			files.push(`./${relativePath}`);
		}
	}

	await walk(root);
	return files;
}

async function testEmbeddedManifestMatchesShaderFiles() {
	const shaderPaths = [
		...(await collectShaderFiles("webgpu", ".wgsl")),
		...(await collectShaderFiles("webgl", ".glsl")),
	].sort();
	const manifestPaths = Object.keys(embeddedShaderSources).sort();

	assert.deepEqual(manifestPaths, shaderPaths);
	for (const shaderPath of shaderPaths) {
		const content = await readFile(
			path.join(SHADER_ROOT, shaderPath.slice(2)),
			"utf8"
		);
		const normalized = content.replace(/\r\n/g, "\n");
		assert.equal(embeddedShaderSources[shaderPath], normalized);
	}
}

function testPostProcessShaderConstantsMatchCPUContract() {
	const compactWGSLFXAA = embeddedShaderSources[
		"./webgpu/postprocess/fxaa.wgsl"
	].replace(/\s+/g, "");
	const quality = FXAA_QUALITY
		.map((value) => value.toFixed(1))
		.join(",");
	assert.ok(
		compactWGSLFXAA.includes(
			`array<f32,${FXAA_QUALITY.length}>(${quality})`,
		),
	);
	const compactGLSLFXAA = embeddedShaderSources[
		"./webgl/postprocess/fxaaFragment.glsl"
	].replace(/\s+/g, "");
	assert.ok(
		compactGLSLFXAA.includes(
			`constfloatIGNIS_FXAA_EDGE_THRESHOLD_MIN=${FXAA_EDGE_THRESHOLD_MIN};`,
		),
	);
	const compactVolumetric = embeddedShaderSources[
		"./webgpu/postprocess/volumetric.wgsl"
	].replace(/\s+/g, "");
	assert.ok(
		compactVolumetric.includes(
			`constIGNIS_VOLUMETRIC_SIGMA_T_SCALE:f32=${VOLUMETRIC_SIGMA_T_SCALE};`,
		),
	);
}

function testShadowSamplingRotationIsSpatiallyStable() {
	const shadowSources = [
		embeddedShaderSources["./webgpu/common/utils.wgsl"],
		embeddedShaderSources["./webgpu/particles/render.wgsl"],
		embeddedShaderSources["./webgl/scene/fragmentShadows.glsl"],
	];
	for (const source of shadowSources) {
		assert.equal(source.includes("floor(texelPosition)"), false);
		assert.match(source, /ShadowFilterDiskSample/i);
		assert.match(source, /ShadowSearchDiskSample/i);
	}
}

async function run() {
	await testLoadsRawAndCompositeParts();
	await testConcurrentLoadsShareResultCache();
	await testGetRequiresPrepare();
	await testWebGLSceneVariants();
	await testWebGLScenePrunedVariant();
	await testWebGLPhongSceneVariantIncludesMaterialBlock();
	await testWebGLPBRSceneVariantCompilesExactUniformBlocks();
	await testWebGPUCompositeIncludesSharedParts();
	await testSrgbDirectiveSupportsBuiltInConsumers();
	await testWebGPUSharedNumericalConstants();
	await testCompositeResultsAreCloned();
	testSyncLoadPopulatesPreparedCache();
	await testCustomAsyncLoaderOverridesBuiltInSource();
	await testCustomLoaderFailureFallsBackToBuiltIns();
	testCustomSyncLoaderOverridesBuiltInSource();
	testMissingCustomSyncLoaderReportsClearError();
	testClearCacheDoesNotTouchShaderRuntime();
	await testWebGLDepthAndShadowArtifacts();
	testShaderManifestValidation();
	testShaderManifestIdentityUsesNormalizedCanonicalParameters();
	testPostProcessShaderConstantsMatchCPUContract();
	testShadowSamplingRotationIsSpatiallyStable();
	await testEmbeddedManifestMatchesShaderFiles();
	console.log("ShaderSource tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
