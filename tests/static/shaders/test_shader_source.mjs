import assert from "node:assert/strict";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
	ShaderSource,
	WEBGL_SHADER_PARTS,
} from "../../../src/shaders/ShaderSource.ts";
import { embeddedShaderSources } from "../../../src/shaders/generated/embeddedShaderSources.ts";
import { ShaderRuntime } from "../../../src/shaders/runtime/index.ts";

const REPO_ROOT = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"..",
	"..",
	".."
);
const SHADER_ROOT = path.join(REPO_ROOT, "src", "shaders");

const WEBGL_SCENE_LIMITS = {
	maxDirectionalLights: 2,
	maxPointLights: 3,
	maxSpotLights: 4,
};

async function testLoadsRawAndCompositeParts() {
	ShaderSource.clearCache();

	const wgslRaw = await ShaderSource.load("webgpu.scene.part.lightData.raw");
	const wgslComposite = await ShaderSource.load(
		"webgpu.scene.part.lightData.composite"
	);
	const glslRaw = await ShaderSource.load("webgl.part.sceneVertex.raw");
	const glslComposite = await ShaderSource.load(
		"webgl.part.sceneVertex.composite"
	);

	assert.ok(wgslRaw.includes("struct DirectionalLightData"));
	assert.ok(wgslComposite.code.includes("struct DirectionalLightData"));
	assert.ok(glslRaw.includes("layout(location = 0) in vec3 aPosition;"));
	assert.ok(glslComposite.code.includes("layout(location = 0) in vec3 aPosition;"));
	assert.equal(
		wgslComposite.sourceMap.segments[0].sourcePath,
		"./webgpu/common/lightData.wgsl"
	);
	assert.equal(
		glslComposite.sourceMap.segments[0].sourcePath,
		"./webgl/parts/sceneVertex.glsl"
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
	assert.ok(raw.fragment.includes("const int MAX_DIRECTIONAL_LIGHTS = 2;"));
	assert.ok(raw.fragment.includes("const int MAX_POINT_LIGHTS = 3;"));
	assert.ok(raw.fragment.includes("const int MAX_SPOT_LIGHTS = 4;"));
	assert.ok(!raw.fragment.includes("__MAX_DIRECTIONAL_LIGHTS__"));
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
		"./webgl/parts/sceneFragment.glsl"
	);
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
		...(await collectShaderFiles(path.join("webgl", "parts"), ".glsl")),
	].sort();
	const manifestPaths = Object.keys(embeddedShaderSources).sort();

	assert.deepEqual(manifestPaths, shaderPaths);
	for (const shaderPath of shaderPaths) {
		const expected = await readFile(
			path.join(SHADER_ROOT, shaderPath.slice(2)),
			"utf8"
		);
		assert.equal(embeddedShaderSources[shaderPath], expected);
	}
}

async function run() {
	await testLoadsRawAndCompositeParts();
	await testConcurrentLoadsShareResultCache();
	await testGetRequiresPrepare();
	await testWebGLSceneVariants();
	await testWebGPUCompositeIncludesSharedParts();
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
