import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
	ShaderSource,
	WEBGL_SHADER_PARTS,
} from "../../../src/shaders/ShaderSource.ts";
import { embeddedShaderSources } from "../../../src/shaders/generated/embeddedShaderSources.ts";
import {
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../../src/shaders/runtime/index.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";

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

	const wgslRaw = await ShaderSource.load("webgpu.scene.part.lightData.raw");
	const wgslComposite = await ShaderSource.load(
		"webgpu.scene.part.lightData.composite"
	);
	const shadowRaw = await ShaderSource.load("webgpu.shadow.depth.raw");
	const shadowComposite = await ShaderSource.load(
		"webgpu.shadow.depth.composite"
	);
	const pagedShadowCompute = await ShaderSource.load(
		"webgpu.shadow.pagedShadowRequestMark.composite"
	);
	const pagedShadowDrawBuild = await ShaderSource.load(
		"webgpu.shadow.pagedShadowDrawBuild.composite"
	);
	const pagedShadowFeedback = await ShaderSource.load(
		"webgpu.shadow.pagedShadowFeedback.composite"
	);
	const pagedShadowPageTableCopy = await ShaderSource.load(
		"webgpu.shadow.pagedShadowPageTableCopy.composite"
	);
	const glslRaw = await ShaderSource.load("webgl.part.sceneVertex.raw");
	const glslComposite = await ShaderSource.load(
		"webgl.part.sceneVertex.composite"
	);

	assert.ok(wgslRaw.includes("struct DirectionalLightData"));
	assert.ok(wgslComposite.code.includes("struct DirectionalLightData"));
	assert.ok(shadowRaw.includes("fn vsMain"));
	assert.ok(shadowRaw.includes("fn fsDepthClip"));
	assert.ok(shadowRaw.includes("discardOutsideAtlasPage(input);"));
	assert.ok(shadowRaw.includes("pixel.x >= input.atlasClipRect.z"));
	assert.ok(shadowComposite.code.includes("fn fsTransmittance"));
	assert.ok(pagedShadowCompute.code.includes("fn csMain"));
	assert.ok(pagedShadowDrawBuild.code.includes("PagedShadowDrawParams"));
	assert.ok(pagedShadowFeedback.code.includes("texture_depth_2d"));
	assert.ok(pagedShadowPageTableCopy.code.includes("texture_storage_2d<r32uint"));
	assert.ok(glslRaw.includes("layout(location = 0) in vec3 aPosition;"));
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
		ShaderSource.load("webgpu.utility.present.composite"),
		ShaderSource.load("webgpu.utility.present.composite"),
	]);
	const stats = ShaderSource.getCacheStats();

	assert.equal(first.code, second.code);
	assert.ok(stats.results.misses >= 1);
	assert.ok(stats.results.hits >= 1);
}

async function testGetRequiresPrepare() {
	ShaderSource.clearCache();

	assert.throws(
		() => ShaderSource.get("webgpu.utility.present.raw"),
		/ShaderSource "webgpu\.utility\.present\.raw" is not prepared/
	);
	await ShaderSource.prepare("webgpu.utility.present.raw");
	const prepared = ShaderSource.get("webgpu.utility.present.raw");

	assert.ok(prepared.includes("fn"));
}

async function testWebGLSceneVariants() {
	ShaderSource.clearCache();
	assert.ok(!WEBGL_SHADER_PARTS.includes("sceneFragment"));
	await ShaderSource.prepareMany([
		...WEBGL_SHADER_PARTS.flatMap((part) => [
			{ key: `webgl.part.${part}.raw` },
			{ key: `webgl.part.${part}.composite` },
		]),
		{ key: "webgl.scene.raw", params: { limits: WEBGL_SCENE_LIMITS } },
		{
			key: "webgl.scene.raw",
			params: {
				limits: {
					...WEBGL_SCENE_LIMITS,
					enableShadowTransmittance: true,
				},
			},
		},
		{ key: "webgl.scene.composite", params: { limits: WEBGL_SCENE_LIMITS } },
	]);

	const raw = ShaderSource.get("webgl.scene.raw", {
		limits: WEBGL_SCENE_LIMITS,
	});
	const withShadowTransmittance = ShaderSource.get("webgl.scene.raw", {
		limits: {
			...WEBGL_SCENE_LIMITS,
			enableShadowTransmittance: true,
		},
	});
	const composite = ShaderSource.get("webgl.scene.composite", {
		limits: WEBGL_SCENE_LIMITS,
	});

	assert.ok(raw.vertex.includes("layout(location = 0) in vec3 aPosition;"));
	assert.ok(raw.fragment.includes("#import <ignis/webgl/constants>"));
	assert.ok(
		raw.fragment.includes(
			"const int MAX_DIRECTIONAL_LIGHTS = __WEBGL_MAX_DIRECTIONAL_LIGHTS__;"
		)
	);
	assert.ok(
		raw.fragment.includes(
			"const int MAX_POINT_LIGHTS = __WEBGL_MAX_POINT_LIGHTS__;"
		)
	);
	assert.ok(
		raw.fragment.includes(
			"const int MAX_SPOT_LIGHTS = __WEBGL_MAX_SPOT_LIGHTS__;"
		)
	);
	assert.ok(
		raw.fragment.includes(
			"const int MAX_LOCAL_LIGHT_PROBES = __WEBGL_MAX_LOCAL_LIGHT_PROBES__;"
		)
	);
	assert.ok(
		raw.fragment.includes(
			"const int MAX_REFLECTION_PROBES = __WEBGL_MAX_REFLECTION_PROBES__;"
		)
	);
	assert.ok(!raw.fragment.includes("__MAX_DIRECTIONAL_LIGHTS__"));
	assert.ok(
		raw.fragment.includes("vec3 calculateIrradianceFromSH(vec3 normal)")
	);
	assert.ok(
		raw.fragment.includes("vec4 sampleBlendedLocalLightProbeIrradiance(")
	);
	assert.ok(!raw.fragment.includes("WEBGL_SHADOW_TRANSMITTANCE 1"));
	assert.ok(
		withShadowTransmittance.fragment.includes(
			"#define WEBGL_SHADOW_TRANSMITTANCE 1"
		)
	);
	assert.ok(
		withShadowTransmittance.fragment.includes(
			"uniform sampler2D uShadowTransmittanceAtlas;"
		)
	);
	assert.equal(
		composite.fragment.sourceMap.segments[0].sourcePath,
		"./webgl/scene/fragmentPrelude.glsl"
	);
	assert.ok(
		composite.fragment.sourceMap.segments.some(
			(segment) =>
				segment.sourcePath === "./webgl/scene/fragmentMainOutput.glsl"
		)
	);

	const compileStage = new ShaderBackendCompileStage({
		backend: "webgl",
		runtime: new ShaderRuntime({ mode: "strict" }),
		profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.fragment,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragment",
		sourceKind: "builtin-scene",
		sourceMap: composite.fragment.sourceMap,
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
		{ key: "webgl.scene.raw", params: { limits: WEBGL_SCENE_LIMITS, variant } },
		{
			key: "webgl.scene.composite",
			params: { limits: WEBGL_SCENE_LIMITS, variant },
		},
	]);
	const raw = ShaderSource.get("webgl.scene.raw", {
		limits: WEBGL_SCENE_LIMITS,
		variant,
	});
	const composite = ShaderSource.get("webgl.scene.composite", {
		limits: WEBGL_SCENE_LIMITS,
		variant,
	});

	assert.ok(!raw.fragment.includes("uniform sampler2D uShadowAtlas;"));
	assert.ok(!raw.fragment.includes("uniform int uEnableClusteredLighting;"));
	assert.ok(!raw.fragment.includes("uniform int uEnableSH;"));
	assert.ok(!raw.fragment.includes("uniform sampler2D uNormalMap;"));
	assert.ok(!raw.fragment.includes("vec3 shadePBR("));
	assert.ok(!raw.fragment.includes("vec3 shadePhong("));
	assert.ok(
		composite.fragment.sourceMap.segments.some(
			(segment) =>
				segment.sourcePath === "./webgl/scene/fragmentUniforms.glsl"
		)
	);

	const compileStage = new ShaderBackendCompileStage({
		backend: "webgl",
		runtime: new ShaderRuntime({ mode: "strict" }),
		profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
		mode: "strict",
	});
	const compiled = compileStage.compile({
		code: raw.fragment,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLSceneFragmentPruned",
		sourceKind: "builtin-scene",
		sourceMap: composite.fragment.sourceMap,
	});
	assert.equal(compiled.hasErrors, false);
	assert.ok(!compiled.code.includes("WEBGL_MATERIAL_MODEL_FULL"));
	assert.ok(!compiled.code.includes("uShadowAtlas"));
	assert.ok(!compiled.code.includes("uEnableClusteredLighting"));
}

async function testWebGPUCompositeIncludesSharedParts() {
	ShaderSource.clearCache();

	const ssr = await ShaderSource.load("webgpu.postprocess.ssr.composite");
	const deferred = await ShaderSource.load("webgpu.deferredLighting.composite");

	assert.ok(ssr.code.includes("struct DirectionalLightData"));
	assert.ok(ssr.code.includes("struct TraceParams"));
	assert.ok(deferred.code.includes("struct DirectionalLightData"));
	assert.ok(deferred.code.includes("fn activeClusteredLightCount() -> u32"));
}

async function testPagedShadowDrawBuildUsesConservativePlaneCulling() {
	ShaderSource.clearCache();

	const source = await ShaderSource.load(
		"webgpu.shadow.pagedShadowDrawBuild.raw"
	);

	assert.ok(source.includes("var outsideNear = true;"));
	assert.ok(source.includes("var outsideFar = true;"));
	assert.ok(source.includes("atomicAdd(&counters[3], intersectingPageCount)"));
	assert.ok(source.includes("atomicAdd(&counters[4]"));
	assert.ok(!source.includes("candidateIndex * params.physicalPageCount"));
	assert.ok(!source.includes("let ndc = clip.xyz / vec3<f32>(clip.w);"));
	assert.ok(source.includes("let matrixIndex = dirtyPhysicalPages[dirtyBase + 2u];"));
	assert.ok(source.includes("arrayLength(&drawInstanceMeta)"));
	assert.ok(source.includes("array<CascadeUVRange, 4>"));
	assert.ok(source.includes("@group(0) @binding(11) var<storage, read> dirtyGridOffsets"));
	assert.ok(source.includes("@group(0) @binding(12) var<storage, read> dirtyGridIndices"));
	assert.ok(source.includes("@group(0) @binding(13) var<storage, read> dirtyPageUvRanges"));
	assert.ok(source.includes("var<workgroup> g_cachedDirtyPages"));
	assert.ok(source.includes("workgroupUniformLoad(&g_cachedCellCount)"));
	assert.ok(source.includes("rangeIntersectsCoarseCell"));
	assert.ok(source.includes("CachedDirtyPage("));
	assert.ok(source.includes("dirtyPageUvRanges[dirtyIndex]"));
	assert.ok(!source.includes("fn dirtyPageUvRange"));
	assert.ok(!source.includes("PAGE_CLIP_XY_MARGIN / atlasSize"));
	assert.ok(!source.includes("(px - 1.5) / gridSize"));
}

async function testPagedShadowDirtyGridBuildUsesGlobalGridBuffers() {
	ShaderSource.clearCache();

	const source = await ShaderSource.load(
		"webgpu.shadow.pagedShadowDirtyGridBuild.raw"
	);

	assert.ok(source.includes("var<storage, read_write> dirtyGridCounts"));
	assert.ok(source.includes("var<storage, read_write> dirtyGridOffsets"));
	assert.ok(source.includes("var<storage, read_write> dirtyGridIndices"));
	assert.ok(source.includes("var<storage, read_write> dirtyPageUvRanges"));
	assert.ok(source.includes("var<storage, read_write> clearDrawIndirectArgs"));
	assert.ok(source.includes("const PAGE_CLIP_XY_MARGIN: f32 = 4.0;"));
	assert.ok(source.includes("fn dirtyPageUvRange"));
	assert.ok(source.includes("PAGE_CLIP_XY_MARGIN / atlasSize"));
	assert.ok(source.includes("clearDrawIndirectArgs[0] = 6u"));
	assert.ok(source.includes("clearDrawIndirectArgs[1] = dirtyCount"));
	assert.ok(source.includes("dirtyPageUvRanges[i] = dirtyPageUvRange(dirtyBase)"));
	assert.ok(source.includes("dirtyGridOffsets[DIRTY_GRID_CELL_COUNT] = sum"));
	assert.ok(source.includes("dirtyGridIndices[insertIndex] = i"));
}

async function testPagedShadowClearLayoutMatchesShaderBindings() {
	ShaderSource.clearCache();

	const source = await ShaderSource.load("webgpu.shadow.pagedShadowClear.raw");
	const shadowPass = await readFile(
		path.join(
			REPO_ROOT,
			"src",
			"backends",
			"webgpu",
			"WebGPUShadowCasterRenderer.ts",
		),
		"utf8"
	);
	const layoutStart = shadowPass.indexOf(
		"label: \"WebGPUPagedShadowClearBindGroupLayout\""
	);
	const layoutEnd = shadowPass.indexOf(
		"label: \"WebGPUPagedShadowClearPipelineLayout\"",
		layoutStart
	);
	const layoutSource = shadowPass.slice(layoutStart, layoutEnd);

	assert.ok(source.includes("@group(0) @binding(0) var<uniform> params"));
	assert.ok(
		source.includes(
			"@group(0) @binding(1) var<storage, read> dirtyPhysicalPages"
		)
	);
	assert.ok(!source.includes("@group(0) @binding(2)"));
	assert.ok(layoutSource.includes("binding: 0"));
	assert.ok(layoutSource.includes("binding: 1"));
	assert.ok(!layoutSource.includes("binding: 2"));
}

async function testPagedShadowRequestCompactUsesLayoutAddresses() {
	ShaderSource.clearCache();

	const source = await ShaderSource.load(
		"webgpu.shadow.pagedShadowRequestCompact.raw"
	);

	assert.ok(source.includes("struct PagedShadowPageAddress"));
	assert.ok(source.includes("@group(0) @binding(4) var<storage, read> pageAddresses"));
	assert.ok(source.includes("let address = pageAddresses[tableIndex]"));
	assert.ok(source.includes("compactedRequests[base + 1u] = address.matrixIndex"));
	assert.ok(!source.includes("fn resolvePageTableAddress"));
	assert.ok(!source.includes("tableIndex / (gridSize * gridSize)"));
}

async function testPagedShadowSamplingUsesGlobalPageTableStride() {
	ShaderSource.clearCache();
	const source = await ShaderSource.load("webgpu.scene.part.utils.raw");

	assert.ok(source.includes("textureDimensions(pagedShadowPageTable).x"));
	assert.ok(source.includes("let pageTableIndex ="));
	assert.ok(source.includes("pageTableIndex % pageTableWidth"));
	assert.ok(source.includes("pageTableIndex / pageTableWidth"));
	assert.ok(!source.includes("let pageTableBaseY ="));
}

function testCompositeResultsAreCloned() {
	ShaderSource.clearCache();
	return ShaderSource.load("webgpu.utility.present.composite").then((first) => {
		first.sourceMap.segments[0].sourcePath = "mutated";
		return ShaderSource.load("webgpu.utility.present.composite").then((second) => {
			assert.notEqual(second.sourceMap.segments[0].sourcePath, "mutated");
		});
	});
}

function testSyncLoadPopulatesPreparedCache() {
	ShaderSource.clearCache();

	assert.equal(ShaderSource.has("webgpu.utility.mipmapBlit.raw"), false);
	const source = ShaderSource.getSync("webgpu.utility.mipmapBlit.raw");

	assert.ok(source.includes("fn vsMain"));
	assert.equal(ShaderSource.has("webgpu.utility.mipmapBlit.raw"), true);
	assert.equal(ShaderSource.get("webgpu.utility.mipmapBlit.raw"), source);
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
		const source = await ShaderSource.load("webgpu.scene.part.lightData.raw");
		assert.equal(source, "struct DirectionalLightData { value: f32, };");
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
		const source = await ShaderSource.load("webgpu.utility.present.raw");
		assert.ok(source.includes("fn"));
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
		const source = ShaderSource.getSync("webgpu.utility.mipmapBlit.raw");
		assert.equal(source, "@compute @workgroup_size(1) fn csMain() {}");
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
			() => ShaderSource.getSync("webgpu.utility.mipmapBlit.raw"),
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

async function run() {
	await testLoadsRawAndCompositeParts();
	await testConcurrentLoadsShareResultCache();
	await testGetRequiresPrepare();
	await testWebGLSceneVariants();
	await testWebGLScenePrunedVariant();
	await testWebGPUCompositeIncludesSharedParts();
	await testPagedShadowDrawBuildUsesConservativePlaneCulling();
	await testPagedShadowDirtyGridBuildUsesGlobalGridBuffers();
	await testPagedShadowClearLayoutMatchesShaderBindings();
	await testPagedShadowRequestCompactUsesLayoutAddresses();
	await testPagedShadowSamplingUsesGlobalPageTableStride();
	await testCompositeResultsAreCloned();
	testSyncLoadPopulatesPreparedCache();
	await testCustomAsyncLoaderOverridesBuiltInSource();
	await testCustomLoaderFailureFallsBackToBuiltIns();
	testCustomSyncLoaderOverridesBuiltInSource();
	testMissingCustomSyncLoaderReportsClearError();
	testClearCacheDoesNotTouchShaderRuntime();
	await testEmbeddedManifestMatchesShaderFiles();
	console.log("ShaderSource tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exitCode = 1;
});
