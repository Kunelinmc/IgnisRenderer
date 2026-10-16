import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { createWebGPUPipelineLayouts } from "../../../src/backends/webgpu/WebGPUPipelineLayouts.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import { WebGPUScenePipelineResources } from "../../../src/backends/webgpu/WebGPUScenePipelineResources.ts";
import {
	WebGPUMaterialPipelineResolver,
	readWebGPUShaderRuntimeView,
} from "../../../src/backends/webgpu/WebGPUMaterialPipelineResolver.ts";
import { createWebGPUMaterialUniformData } from "../../../src/backends/webgpu/material.ts";
import { resolveWebGPUScenePassDescriptor } from "../../../src/backends/webgpu/WebGPUScenePassDescriptors.ts";
import { ShaderRuntime, ShaderBackendCompileStage } from "../../../src/shaders/runtime/index.ts";
import { createShaderMaterialSourceBlocks } from "../../../src/shaders/ShaderMaterialSource.ts";
import { WEBGPU_TEST_PROFILE } from "./shaderDirectiveTestProfiles.mjs";

import { FakeWebGPUBackend as FakeBackend } from "../../helpers/fakes.mjs";

function createWebGPUShaderMaterialSourceBlocks(material, stage, source) {
	return createShaderMaterialSourceBlocks({
		material,
		language: "wgsl",
		stage,
		source,
		wgslUniformBinding: 39,
	});
}

function createWebGLShaderMaterialSourceBlocks(material, stage, source) {
	return createShaderMaterialSourceBlocks({
		material,
		language: "glsl",
		stage,
		source,
	});
}

function createLayouts() {
	const families = ["pbr", "phong", "flat", "unlit"];
	const familyLayouts = (prefix) => Object.fromEntries(
		families.map((family) => [family, { id: `${prefix}:${family}` }])
	);
	return {
		scenePipelineLayouts: familyLayouts("scene-layout"),
		sceneGBufferPipelineLayouts: familyLayouts("scene-gbuffer-layout"),
		sceneDepthPrepassPipelineLayout: { id: "scene-depth-layout" },
		planarReflectionPipelineLayouts: familyLayouts("planar-layout"),
		environmentPipelineLayout: { id: "environment-layout" },
	};
}

const scenePipelineResolvers = new WeakMap();

async function getScenePipeline(
	resources,
	backend,
	material,
	mode = "single",
	wireframe = false,
	topology = "triangle-list",
	transparentMode = "default",
	drawMode = "default",
	sampleCount,
	deferredGBufferLayout = "extended",
	geometry,
	materialData,
) {
	material.refreshRevision();
	const data = materialData ?? createWebGPUMaterialUniformData(material, wireframe);
	let resolver = scenePipelineResolvers.get(resources);
	if (!resolver) {
		resolver = new WebGPUMaterialPipelineResolver();
		scenePipelineResolvers.set(resources, resolver);
	}
	const shaderMode = mode === "gbuffer" ? "deferred" : mode === "color" ? "single" : mode;
	const state = resolver.resolve(
		material,
		data,
		wireframe,
		shaderMode,
		"scene",
		readWebGPUShaderRuntimeView(backend),
	);
	return resources.resolvePipeline({
		materialState: state,
		pass: resolveWebGPUScenePassDescriptor(
			mode,
			transparentMode,
			drawMode,
			deferredGBufferLayout,
		),
		topology,
		geometryLayout: geometry ?? {
			layoutKey: "test-fallback",
			sceneVertexLayouts: [],
		},
		sampleCount,
	});
}

function testWebGPUMaterialPipelineResolverProducesImmutableRuntimeVariants() {
	const material = new ShaderMaterial({
		chunks: [
			{ backend: "webgpu", language: "wgsl", stage: "vertex", code: WGSL_VERTEX },
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
		],
	});
	material.refreshRevision();
	const data = createWebGPUMaterialUniformData(material, false);
	const resolver = new WebGPUMaterialPipelineResolver();
	const runtime = {
		revision: 4,
		mode: "warn",
		directiveCacheTag: "profile:a",
		supportsRuntimeInjects: true,
	};
	const first = resolver.resolve(material, data, false, "single", "scene", runtime);
	const cached = resolver.resolve(material, data, false, "single", "scene", runtime);
	assert.equal(first, cached);
	assert.equal(first.program.kind, "custom");
	assert.equal(first.program.regularProgram.fragmentTargetMode, "single");
	assert.equal(first.pipelineKey, data.pipelineKey);
	assert.equal(first.shaderRuntime.directiveCacheTag, "profile:a");
	assert.ok(!("material" in first));

	const changedRuntime = resolver.resolve(material, data, false, "single", "scene", {
		...runtime,
		revision: runtime.revision + 1,
	});
	assert.notEqual(changedRuntime, first);
	assert.notEqual(changedRuntime.shaderCacheKey, first.shaderCacheKey);

	resolver.clear();
	assert.notEqual(
		resolver.resolve(material, data, false, "single", "scene", runtime),
		first,
	);
}

const WGSL_VERTEX = /* wgsl */ `
@vertex
fn customVs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(position, 1.0);
}
`;

const WGSL_FRAGMENT_SINGLE = /* wgsl */ `
@fragment
fn customFsSingle() -> @location(0) vec4<f32> {
	return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
`;

const WGSL_FRAGMENT_MRT = /* wgsl */ `
struct MRTOut {
	@location(0) color: vec4<f32>,
	@location(1) g0: vec4<f32>,
	@location(2) g1: vec4<f32>,
	@location(3) g2: vec4<f32>,
	@location(4) g3: vec4<f32>,
}

@fragment
fn customFsMRT() -> MRTOut {
	return MRTOut(
		vec4<f32>(1.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0),
		vec4<f32>(0.0, 0.0, 0.0, 1.0)
	);
}
`;

const WGSL_FRAGMENT_DEFERRED = /* wgsl */ `
struct DeferredOut {
	@location(0) g0: vec4<f32>,
	@location(1) g1: vec4<f32>,
	@location(2) g2: vec4<f32>,
	@location(3) g3: vec4<f32>,
	@location(4) g4: vec4<f32>,
	@location(5) g5: vec4<f32>,
	@location(6) g6: vec4<f32>,
}

@fragment
fn customFsDeferred() -> DeferredOut {
	return DeferredOut(
		vec4<f32>(1.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0),
		vec4<f32>(0.0)
	);
}
`;

const WGSL_FRAGMENT_DEPTH = /* wgsl */ `
@fragment
fn customDepth() {
}
`;

const WEBGL_VERTEX = /* glsl */ `
#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;

void main() {
	gl_Position = vec4(aPosition, 1.0);
}
`;

const WEBGL_FRAGMENT = /* glsl */ `
#version 300 es
precision highp float;

out vec4 outColor;

void main() {
	outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

const WEBGL_FRAGMENT_MRT = /* glsl */ `
#version 300 es
precision highp float;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotionDepth;
layout(location = 2) out vec4 outNormal;

void main() {
	outColor = vec4(0.0, 1.0, 0.0, 1.0);
	outMotionDepth = vec4(0.0, 0.0, 0.0, 1.0);
	outNormal = vec4(0.5, 0.5, 1.0, 1.0);
}
`;

const WEBGL_FRAGMENT_DEPTH = /* glsl */ `
#version 300 es
precision highp float;

void main() {
}
`;

function getChunkCode(material, selector) {
	const chunk = material.chunks.find((entry) => {
		const backend = entry.backend ?? "webgpu";
		const mode = entry.mode ?? "single";
		return (
			backend === selector.backend &&
			entry.language === selector.language &&
			entry.stage === selector.stage &&
			mode === selector.mode
		);
	});
	return chunk?.code;
}

async function captureWarnMessagesAsync(run) {
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: {
			warn: (...args) => {
				warnings.push(args.map((arg) => String(arg)).join(" "));
			},
		},
		resetOnceKeys: true,
	});
	try {
		await run();
	} finally {
		Logger.reset();
	}
	return warnings;
}

async function testWGSLProgramSelection() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "WGSLMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "mrt",
				code: WGSL_FRAGMENT_MRT,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
		fragmentMRTEntryPoint: "customFsMRT",
		uniformBindings: [{ name: "time", type: "f32" }],
	});

	const singlePipeline = await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1,
	);
	assert.equal(singlePipeline.desc.layout.id, "scene-layout:flat");
	assert.equal(singlePipeline.desc.vertex.entryPoint, "customVs");
	assert.equal(singlePipeline.desc.fragment.entryPoint, "customFsSingle");
	assert.equal(singlePipeline.desc.fragment.targets.length, 1);

	const mrtPipeline = await getScenePipeline(library, backend,
		material, "mrt", false, undefined, undefined, undefined, 1,
	);
	assert.equal(mrtPipeline.desc.layout.id, "scene-layout:flat");
	assert.equal(mrtPipeline.desc.vertex.entryPoint, "customVs");
	assert.equal(mrtPipeline.desc.fragment.entryPoint, "customFsMRT");
	assert.equal(mrtPipeline.desc.fragment.targets.length, 5);
	assert.equal(mrtPipeline.desc.fragment.targets[1].writeMask, undefined);

	const moduleCodes = backend.shaderModules.map((module) => module.desc.code);
	assert.ok(moduleCodes.includes(WGSL_VERTEX));
	assert.ok(moduleCodes.includes(WGSL_FRAGMENT_SINGLE));
	assert.ok(moduleCodes.includes(WGSL_FRAGMENT_MRT));
	assert.ok(
		backend.shaderModules.some((module) =>
			module.desc.generatedSourceBlocks?.some((block) =>
				block.code.includes("@group(1) @binding(39)")
			)
		),
	);

	const pipelineCountBefore = backend.pipelines.length;
	await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1,
	);
	await getScenePipeline(library, backend,
		material, "mrt", false, undefined, undefined, undefined, 1,
	);
	assert.equal(backend.pipelines.length, pipelineCountBefore);
}

async function testWebGPUPipelineCacheIncludesGeometryLayout() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
	});
	const empty = { arrayStride: 0, attributes: [] };
	const geometryA = {
		layoutKey: "position-12",
		sceneVertexLayouts: [
			{
				arrayStride: 12,
				attributes: [{ shaderLocation: 0, format: "float32x3", offset: 0 }],
			},
			empty,
			empty,
			empty,
		],
	};
	const geometryB = {
		...geometryA,
		layoutKey: "position-16",
		sceneVertexLayouts: [
			{
				arrayStride: 16,
				attributes: geometryA.sceneVertexLayouts[0].attributes,
			},
			empty,
			empty,
			empty,
		],
	};
	const first = await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1, "extended", geometryA,
	);
	const second = await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1, "extended", geometryB,
	);
	const firstAgain = await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1, "extended", geometryA,
	);

	assert.notEqual(first, second);
	assert.equal(firstAgain, first);
}

async function testWebGPUMRTFallsBackToSingleFragmentEntryPoint() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "WGSLMRTFallbackMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
		fragmentMRTEntryPoint: "customFsMRT",
	});

	const program = material.resolveWebGPUProgram("mrt");
	assert.equal(program.fragmentCode, WGSL_FRAGMENT_SINGLE);
	assert.equal(program.fragmentEntryPoint, "customFsSingle");
	assert.equal(program.fragmentTargetMode, "single");

	const pipeline = await getScenePipeline(library, backend,
		material, "mrt", false, undefined, undefined, undefined, 1,
	);
	assert.equal(pipeline.desc.fragment.entryPoint, "customFsSingle");
	assert.equal(pipeline.desc.fragment.targets.length, 5);
	for (let index = 1; index < pipeline.desc.fragment.targets.length; index++) {
		assert.equal(pipeline.desc.fragment.targets[index].writeMask, 0);
	}
}

async function testWebGPUDeferredProgramSelection() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "DeferredCustomMaterial",
		deferredLighting: true,
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "deferred",
				code: WGSL_FRAGMENT_DEFERRED,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentDeferredEntryPoint: "customFsDeferred",
	});

	assert.equal(material.hasWebGPUDeferredProgram(), true);
	const program = material.resolveWebGPUProgram("deferred");
	assert.equal(program.fragmentEntryPoint, "customFsDeferred");
	assert.equal(program.fragmentTargetMode, "deferred");
	assert.equal(program.fragmentCode.includes("DeferredOut"), true);

	const pipeline = await getScenePipeline(library, backend,
		material, "gbuffer", false, undefined, undefined, undefined, 1,
	);
	assert.equal(pipeline.desc.layout.id, "scene-gbuffer-layout:flat");
	assert.equal(pipeline.desc.fragment.entryPoint, "customFsDeferred");
	assert.equal(pipeline.desc.fragment.targets.length, 7);

	const missingOptIn = new ShaderMaterial({
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "deferred",
				code: WGSL_FRAGMENT_DEFERRED,
			},
		],
	});
	assert.equal(missingOptIn.hasWebGPUDeferredProgram(), false);
}

async function testBuiltinBaseGBufferPipelineSelection() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const pipeline = await getScenePipeline(library, backend,
		new PBRMaterial(),
		"gbuffer",
		false,
		undefined,
		undefined,
		undefined,
		1,
		"base",
	);
	assert.equal(pipeline.desc.fragment.entryPoint, "fsMainGBufferBasePBR");
	assert.equal(pipeline.desc.fragment.targets.length, 4);
}

function testResolveWebGPUDepthPrepassProgramContract() {
	const material = new ShaderMaterial({
		name: "DepthPrepassMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
		],
		vertexEntryPoint: "customVs",
		depthFragmentEntryPoint: "customDepth",
		depthFragmentCode: WGSL_FRAGMENT_DEPTH,
	});

	const depthProgram = material.resolveWebGPUDepthPrepassProgram("single");
	assert.ok(depthProgram);
	assert.equal(depthProgram.vertexEntryPoint, "customVs");
	assert.equal(depthProgram.fragmentEntryPoint, "customDepth");
	assert.equal(depthProgram.fragmentCode, WGSL_FRAGMENT_DEPTH);

	const missingDepthContract = new ShaderMaterial({
		name: "DepthPrepassMissing",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
		],
		vertexEntryPoint: "customVs",
	});
	assert.equal(
		missingDepthContract.resolveWebGPUDepthPrepassProgram("single"),
		null
	);
}

function testDepthWriteParamIsInheritedFromMaterial() {
	const defaultMaterial = new ShaderMaterial();
	assert.equal(defaultMaterial.depthWrite, true);

	const depthReadMaterial = new ShaderMaterial({
		name: "DepthReadShader",
		depthWrite: false,
	});
	assert.equal(depthReadMaterial.depthWrite, false);
}

function testTransparentOutputCompatibilityOptIn() {
	assert.equal(new ShaderMaterial().transparentOutputCompatible, false);
	assert.equal(new ShaderMaterial({
		transparentOutputCompatible: true,
	}).transparentOutputCompatible, true);
}

async function testGLSLProgramUsesTranspiler() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const transpilerCalls = [];
	const material = new ShaderMaterial({
		name: "GLSLMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "glsl",
				stage: "vertex",
				code: "void main() { gl_Position = vec4(0.0); }",
			},
			{
				backend: "webgpu",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: "void main() { }",
			},
			{
				backend: "webgpu",
				language: "glsl",
				stage: "fragment",
				mode: "mrt",
				code: "void main() { }",
			},
		],
		glslToWgsl(source, stage) {
			transpilerCalls.push({ source, stage });
			switch (stage) {
				case "vertex":
					return WGSL_VERTEX;
				case "fragment-single":
					return WGSL_FRAGMENT_SINGLE;
				case "fragment-mrt":
				default:
					return WGSL_FRAGMENT_MRT;
			}
		},
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
		fragmentMRTEntryPoint: "customFsMRT",
	});

	await getScenePipeline(library, backend,
		material, "mrt", false, undefined, undefined, undefined, 1,
	);
	await getScenePipeline(library, backend,
		material, "single", false, undefined, undefined, undefined, 1,
	);

	assert.ok(transpilerCalls.some((call) => call.stage === "vertex"));
	assert.ok(transpilerCalls.some((call) => call.stage === "fragment-mrt"));
	assert.ok(transpilerCalls.some((call) => call.stage === "fragment-single"));
}

async function testGLSLWithoutTranspilerThrows() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenGLSLMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "glsl",
				stage: "vertex",
				code: "void main() { gl_Position = vec4(0.0); }",
			},
			{
				backend: "webgpu",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: "void main() { }",
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
	});

	await assert.rejects(
		() => getScenePipeline(library, backend,
			material, "single", false, undefined, undefined, undefined, 1,
		),
		/glslToWgsl transpiler/
	);
}

async function testWarnModeFallbackToBuiltinShader() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	backend.shaderRuntime.setMode("warn");
	backend.failCustomShaderModules = true;
	const library = new WebGPUScenePipelineResources(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenCustomShader",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
	});

	let pipeline = null;
	const warnings = await captureWarnMessagesAsync(async () => {
		pipeline = await getScenePipeline(library, backend,
			material, "single", false, undefined, undefined, undefined, 1,
		);
	});
	assert.equal(pipeline.desc.vertex.entryPoint, "vsMain");
	assert.equal(pipeline.desc.fragment.entryPoint, "fsMainSingleFlat");
	assert.ok(
		backend.shaderModules.some((module) => module.label === "WebGPUSceneShader")
	);
	assert.ok(
		warnings.some((warning) =>
			warning.includes("[webgpu-shader-material-compile-failed-")
		)
	);
}

function testResolveWebGLProgramPrefersWebGLSource() {
	const material = new ShaderMaterial({
		name: "WebGLCustomMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "glsl",
				stage: "vertex",
				code: "legacy-vertex",
			},
			{
				backend: "webgpu",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: "legacy-fragment",
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: WEBGL_FRAGMENT,
			},
		],
	});

	const program = material.resolveWebGLProgram();
	assert.equal(program.vertexCode, WEBGL_VERTEX);
	assert.equal(program.fragmentCode, WEBGL_FRAGMENT);
}

function testResolveWebGLProgramSupportsModeAwareWebGLFragments() {
	const material = new ShaderMaterial({
		name: "WebGLModeAwareMaterial",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: WEBGL_FRAGMENT,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "mrt",
				code: WEBGL_FRAGMENT_MRT,
			},
		],
	});

	const singleProgram = material.resolveWebGLProgram("single");
	const mrtProgram = material.resolveWebGLProgram("mrt");
	assert.equal(singleProgram.fragmentCode, WEBGL_FRAGMENT);
	assert.equal(mrtProgram.fragmentCode, WEBGL_FRAGMENT_MRT);
}

function testResolveWebGLProgramMRTFallsBackToSingleFragment() {
	const material = new ShaderMaterial({
		name: "WebGLMRTFallbackMaterial",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: WEBGL_FRAGMENT,
			},
		],
	});

	const program = material.resolveWebGLProgram("mrt");
	assert.equal(program.fragmentCode, WEBGL_FRAGMENT);
}

function testResolveWebGLProgramFallsBackToWebGPUGLSL() {
	const material = new ShaderMaterial({
		name: "WebGLFallbackMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "glsl",
				stage: "vertex",
				code: "legacy-vertex",
			},
			{
				backend: "webgpu",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: "legacy-fragment",
			},
		],
	});

	const program = material.resolveWebGLProgram();
	assert.equal(program.vertexCode, "legacy-vertex");
	assert.equal(program.fragmentCode, "legacy-fragment");
}

function testResolveWebGLProgramMissingSourceThrows() {
	const material = new ShaderMaterial({
		name: "MissingWebGLSource",
	});

	assert.throws(
		() => material.resolveWebGLProgram(),
		/missing WebGL GLSL source/
	);
}

function testResolveWebGLDepthPrepassProgramContract() {
	const material = new ShaderMaterial({
		name: "WebGLDepthPrepassMaterial",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment-depth",
				code: WEBGL_FRAGMENT_DEPTH,
			},
		],
	});

	const depthProgram = material.resolveWebGLDepthPrepassProgram("single");
	assert.ok(depthProgram);
	assert.equal(depthProgram.vertexCode, WEBGL_VERTEX);
	assert.equal(depthProgram.fragmentCode, WEBGL_FRAGMENT_DEPTH);
	assert.equal(
		getChunkCode(material, {
			backend: "webgl",
			language: "glsl",
			stage: "fragment-depth",
			mode: "single",
		}),
		WEBGL_FRAGMENT_DEPTH
	);

	const missingDepth = new ShaderMaterial({
		name: "WebGLDepthMissing",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
		],
	});
	assert.equal(missingDepth.resolveWebGLDepthPrepassProgram("single"), null);
}

function testChunkBackendMustBeExplicit() {
	assert.throws(
		() => new ShaderMaterial({
			chunks: [
				{
					language: "wgsl",
					stage: "vertex",
					code: WGSL_VERTEX,
				},
			],
		}),
		/backend must be explicitly set to "webgpu" or "webgl"/
	);
}

function testChunkApiSupportsUnifiedShaderUpdates() {
	const material = new ShaderMaterial({
		name: "ChunkMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "mrt",
				code: WGSL_FRAGMENT_MRT,
			},
		],
	});
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "wgsl",
			stage: "vertex",
			mode: "single",
		}),
		WGSL_VERTEX
	);
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "wgsl",
			stage: "fragment",
			mode: "single",
		}),
		WGSL_FRAGMENT_SINGLE
	);
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "wgsl",
			stage: "fragment",
			mode: "mrt",
		}),
		WGSL_FRAGMENT_MRT
	);

	material.upsertChunk({
		backend: "webgpu",
		language: "glsl",
		stage: "vertex",
		code: "legacy-vertex",
	});
	material.upsertChunk({
		backend: "webgpu",
		language: "glsl",
		stage: "fragment",
		mode: "single",
		code: "legacy-fragment",
	});
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "glsl",
			stage: "vertex",
			mode: "single",
		}),
		"legacy-vertex"
	);
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "glsl",
			stage: "fragment",
			mode: "single",
		}),
		"legacy-fragment"
	);
	assert.equal(
		material.removeChunk({
			backend: "webgpu",
			language: "glsl",
			stage: "fragment",
			mode: "single",
		}),
		true
	);
	assert.equal(
		getChunkCode(material, {
			backend: "webgpu",
			language: "glsl",
			stage: "fragment",
			mode: "single",
		}),
		undefined
	);
}

function testTextureBindingAutoSlotAndUniformDefaults() {
	const material = new ShaderMaterial({ name: "TextureBindingDefaults" });
	material.setTextureBindings([
		{
			name: "noise-map",
			texture: { colorSpace: "Linear" },
		},
		{
			name: "detail",
			texture: null,
		},
	]);
	const bindings = material.getTextureBindings();
	assert.equal(bindings.length, 2);
	assert.equal(bindings[0].slot, 0);
	assert.equal(bindings[1].slot, 1);
	assert.equal(bindings[0].webglUniform, "uShaderTex_noise_map");
	assert.equal(bindings[0].linear, true);
	assert.equal(bindings[1].linear, false);

	assert.throws(
		() =>
			material.setTextureBindings([
				{ name: "slotA", texture: null, slot: 3 },
				{ name: "slotB", texture: null, slot: 3 },
			]),
		/duplicate texture slot/
	);
}

function testTextureBindingGeneratedSourceBlocks() {
	const material = new ShaderMaterial({
		name: "InjectBindingMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: WEBGL_FRAGMENT,
			},
		],
	});
	material.setTextureBinding({
		name: "noise-map",
		texture: null,
		slot: 13,
		uvSet: 1,
		linear: true,
		webglUniform: "uNoiseTex",
	});

	const webgpu = material.resolveWebGPUProgram("single");
	assert.equal(webgpu.fragmentCode, WGSL_FRAGMENT_SINGLE);
	const webgpuBlocks = createWebGPUShaderMaterialSourceBlocks(
		material,
		"fragment",
		webgpu.fragmentCode,
	);
	const webgpuGenerated = webgpuBlocks.map((block) => block.code).join("\n");
	assert.ok(webgpuGenerated.includes("@group(1) @binding(27)"));
	assert.ok(webgpuGenerated.includes("@group(1) @binding(28)"));
	assert.ok(webgpuGenerated.includes("ignisSampleTexture_noise_map"));
	assert.ok(webgpuGenerated.includes("IGNIS_TEXTURE_UVSET_NOISE_MAP: u32 = 1u"));

	const webgl = material.resolveWebGLProgram();
	assert.equal(webgl.fragmentCode, WEBGL_FRAGMENT);
	const webglBlocks = createWebGLShaderMaterialSourceBlocks(
		material,
		"fragment",
		webgl.fragmentCode,
	);
	const webglGenerated = webglBlocks.map((block) => block.code).join("\n");
	assert.ok(webglGenerated.includes("uniform sampler2D uNoiseTex;"));
	assert.ok(webglGenerated.includes("ignisSampleTexture_noise_map"));
}

function testTextureBindingUvSetGreaterThanOneIsPreserved() {
	const material = new ShaderMaterial({ name: "TextureBindingUvClamp" });
	material.setTextureBinding({
		name: "detail",
		texture: null,
		uvSet: 2,
	});
	const bindings = material.getTextureBindings();
	assert.equal(bindings.length, 1);
	assert.equal(bindings[0].uvSet, 2);
}

function testUniformBindingSchemaAndValueRevision() {
	const material = new ShaderMaterial({ name: "UniformBindingMaterial" });
	const initialShaderRevision = material.shaderRevision;
	material.setUniformBindings([
		{ name: "time", type: "f32", value: 1 },
		{ name: "mode", type: "i32", value: 2, stage: "fragment" },
		{ name: "flags", type: "u32", value: 3 },
		{ name: "uvScale", type: "vec2f", value: [1, 2] },
		{ name: "offset", type: "vec3i", value: [1, 2, 3] },
		{ name: "mask", type: "vec4u", value: [1, 2, 3, 4] },
		{ name: "transform", type: "mat4x4f", value: Matrix4.identity() },
	]);
	assert.ok(material.shaderRevision > initialShaderRevision);
	const schemaRevision = material.shaderRevision;
	const valueRevision = material.uniformValueRevision;
	const bindings = material.getUniformBindings();
	assert.equal(bindings.length, 7);
	assert.equal(bindings[0].webglUniform, "uShaderUniform_time");
	assert.equal(bindings[0].wgslField, "time");
	assert.equal(bindings[1].stage, "fragment");
	assert.deepEqual(bindings[3].value, [1, 2]);

	const external = [4, 5];
	material.setUniform("uvScale", external);
	external[0] = 99;
	assert.deepEqual(
		material.getUniformBindings().find((binding) => binding.name === "uvScale")
			.value,
		[4, 5]
	);
	assert.equal(material.shaderRevision, schemaRevision);
	assert.ok(material.uniformValueRevision > valueRevision);

	assert.throws(
		() => material.setUniform("missing", 1),
		/not declared/
	);
	assert.throws(
		() => material.setUniform("mode", 1.5),
		/requires integer/
	);
	assert.throws(
		() => material.setUniform("transform", Array(16).fill(1)),
		/Matrix4 or number\[4\]\[4\]/
	);
}

function testUniformBindingDuplicateDiagnostics() {
	const material = new ShaderMaterial({ name: "UniformDuplicateDiagnostics" });
	assert.throws(
		() =>
			material.setUniformBindings([
				{ name: "a", type: "f32" },
				{ name: "a", type: "i32" },
			]),
		/duplicate uniform binding/
	);
	assert.throws(
		() =>
			material.setUniformBindings([
				{ name: "a", type: "f32", wgslField: "same" },
				{ name: "b", type: "f32", wgslField: "same" },
			]),
		/duplicate WGSL field/
	);
	assert.throws(
		() =>
			material.setUniformBindings([
				{ name: "a", type: "f32", webglUniform: "uSame" },
				{ name: "b", type: "f32", webglUniform: "uSame" },
			]),
		/duplicate WebGL uniform/
	);
}

function testUniformBindingGeneratedSourceBlocks() {
	const material = new ShaderMaterial({
		name: "UniformInjectMaterial",
		chunks: [
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: WEBGL_FRAGMENT,
			},
		],
		uniformBindings: [
			{
				name: "vertexTime",
				type: "f32",
				stage: "vertex",
				webglUniform: "uVertexTime",
			},
			{
				name: "fragmentTint",
				type: "vec4f",
				stage: "fragment",
				wgslField: "fragmentTint",
				webglUniform: "uFragmentTint",
			},
		],
	});

	const webgpu = material.resolveWebGPUProgram("single");
	const webgpuVertex = createWebGPUShaderMaterialSourceBlocks(
		material,
		"vertex",
		webgpu.vertexCode,
	).map((block) => block.code).join("\n");
	const webgpuFragment = createWebGPUShaderMaterialSourceBlocks(
		material,
		"fragment",
		webgpu.fragmentCode,
	).map((block) => block.code).join("\n");
	assert.ok(webgpuVertex.includes("vertexTime: f32"));
	assert.ok(webgpuVertex.includes("__ignisPad_vertex_1: vec4<f32>"));
	assert.ok(webgpuFragment.includes("fragmentTint: vec4<f32>"));
	assert.ok(webgpuFragment.includes("__ignisPad_fragment_0: f32"));
	assert.ok(webgpuFragment.includes("@group(1) @binding(39)"));

	const webgl = material.resolveWebGLProgram();
	const webglVertex = createWebGLShaderMaterialSourceBlocks(
		material,
		"vertex",
		webgl.vertexCode,
	).map((block) => block.code).join("\n");
	const webglFragment = createWebGLShaderMaterialSourceBlocks(
		material,
		"fragment",
		webgl.fragmentCode,
	).map((block) => block.code).join("\n");
	assert.ok(webglVertex.includes("uniform float uVertexTime;"));
	assert.equal(webglVertex.includes("uFragmentTint"), false);
	assert.ok(webglFragment.includes("uniform vec4 uFragmentTint;"));
	assert.equal(webglFragment.includes("uVertexTime"), false);
	assert.equal(webglGeneratedIncludesPadding(webglVertex, webglFragment), false);
}

function webglGeneratedIncludesPadding(...sources) {
	return sources.some((source) => source.includes("__ignisPad"));
}

function testGeneratedMaterialABIOwnershipRejectsRedeclarations() {
	const material = new ShaderMaterial({
		uniformBindings: [{ name: "time", type: "f32" }],
		textureBindings: [{ name: "noise", texture: null, webglUniform: "uNoise" }],
	});
	assert.throws(
		() => createWebGPUShaderMaterialSourceBlocks(
			material,
			"fragment",
			`@group(1) @binding(39) var<uniform> customUniforms: vec4<f32>;
			@fragment fn fsMain() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }`,
		),
		/redeclares reserved @group\(1\) @binding\(39\)/,
	);
	assert.throws(
		() => createWebGLShaderMaterialSourceBlocks(
			material,
			"fragment",
			`#version 300 es
			uniform float uShaderUniform_time;
			void main() {}`,
		),
		/redeclares generated uniform/,
	);
}

async function testBuiltinDepthIsolation() {
	globalThis.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
	const layouts = createWebGPUPipelineLayouts({
		createBindGroupLayout: (desc) => ({ desc }),
		createPipelineLayout: (desc) => ({ desc }),
	});
	assert.ok(layouts.depthFrameBindGroupLayout, "depth frame must use an isolated binding layout");
	assert.deepEqual(layouts.depthFrameBindGroupLayout.desc.entries.map((e) => e.binding), [0]);
	assert.deepEqual(layouts.depthModelBindGroupLayout.desc.entries.map((e) => e.binding),
		[0, 34, 35, 36, 37, 40]);
	assert.ok(layouts.depthModelBindGroupLayout.desc.entries.every(
		(e) => e.visibility === GPUShaderStage.VERTEX,
	));
	assert.deepEqual(layouts.depthMaterialBindGroupLayout.desc.entries.map((e) => e.binding),
		[0, 1, 2]);
	assert.equal(layouts.depthMaterialBindGroupLayout.desc.entries[0].buffer.minBindingSize, 48);
	assert.deepEqual(layouts.sceneDepthPrepassPipelineLayout.desc.bindGroupLayouts,
		[layouts.depthFrameBindGroupLayout, layouts.depthModelBindGroupLayout,
			layouts.depthMaterialBindGroupLayout]);

	const backend = new FakeBackend();
	const resources = new WebGPUScenePipelineResources(backend, layouts);
	const resolver = new WebGPUMaterialPipelineResolver();
	const runtime = readWebGPUShaderRuntimeView(backend);
	const pbr = new PBRMaterial();
	const flat = new Material();
	const policy = resolver.resolveBuiltinDepth(pbr, runtime);
	assert.equal(policy, resolver.resolveBuiltinDepth(pbr, runtime));
	assert.equal(policy.program.kind, "builtin");
	assert.equal(resolver.resolveBuiltinDepth(new Material({ wireframe: true }), runtime).wireframe,
		false, "authored wireframe retains the solid early-z pass");
	const directPolicyMaterial = new Material();
	const directPolicy = resolver.resolveBuiltinDepth(directPolicyMaterial, runtime);
	directPolicyMaterial.depthWrite = false;
	assert.equal(resolver.resolveBuiltinDepth(directPolicyMaterial, runtime).depthWrite, false,
		"depth policy observes direct mutation without a refreshed general material revision");
	directPolicyMaterial.depthWrite = true;
	directPolicyMaterial.cullMode = "front";
	assert.equal(resolver.resolveBuiltinDepth(directPolicyMaterial, runtime).cullMode, "front");
	directPolicyMaterial.alphaMode = AlphaMode.Blend;
	assert.equal(resolver.resolveBuiltinDepth(directPolicyMaterial, runtime).transparent, true);
	directPolicyMaterial.alphaMode = AlphaMode.Mask;
	assert.equal(resolver.resolveBuiltinDepth(directPolicyMaterial, runtime).alphaMode, AlphaMode.Mask);
	directPolicyMaterial.alphaMode = AlphaMode.Opaque;
	directPolicyMaterial.cullMode = directPolicy.cullMode;
	assert.equal(resolver.resolveBuiltinDepth(directPolicyMaterial, runtime), directPolicy);
	const directTransmission = new PBRMaterial();
	resolver.resolveBuiltinDepth(directTransmission, runtime);
	directTransmission.transmissionFactor = 0.5;
	assert.equal(resolver.resolveBuiltinDepth(directTransmission, runtime).usesTransmission, true);
	const request = {
		materialState: policy,
		pass: resolveWebGPUScenePassDescriptor("single", "default", "early-z-prepass"),
		topology: "triangle-list",
		geometryLayout: { layoutKey: "depth-geometry", sceneVertexLayouts: [] },
		sampleCount: 1,
	};
	const [first, shared] = await Promise.all([
		resources.resolvePipeline(request),
		resources.resolvePipeline({ ...request,
			materialState: resolver.resolveBuiltinDepth(flat, runtime) }),
	]);
	assert.equal(first, shared, "compatible builtin material families share depth creation");
	assert.equal(backend.pipelines.length, 1);
	assert.equal(first.desc.layout, layouts.sceneDepthPrepassPipelineLayout);
	assert.equal(first.desc.fragment, undefined);
	assert.equal(backend.shaderModules[0].label, "WebGPUSceneDepthShader");
	assert.equal(await resources.resolvePipeline({ ...request,
		materialState: { ...policy, shadingFamily: "pbr", pipelineKey: "pbr:features",
			shaderCacheKey: "scene:unrelated-features" } }), first,
		"scene shading family and material features do not specialize builtin depth");

	for (const variant of [
		{ geometryLayout: { layoutKey: "other", sceneVertexLayouts: [] } },
		{ topology: "line-list" },
		{ pass: { ...request.pass, frontFace: "cw" } },
		{ materialState: { ...policy, cullMode: "none" } },
		{ materialState: { ...policy, shaderRuntime: { ...runtime, revision: 2 } } },
		{ materialState: { ...policy, shaderRuntime: { ...runtime, directiveCacheTag: "other" } } },
	]) {
		assert.notEqual(await resources.resolvePipeline({ ...request, ...variant }), first);
	}
	const mask = new PBRMaterial({ alphaMode: AlphaMode.Mask });
	const maskPipeline = await resources.resolvePipeline({ ...request,
		materialState: resolver.resolveBuiltinDepth(mask, runtime) });
	assert.notEqual(maskPipeline, first);
	assert.equal(maskPipeline.desc.fragment.entryPoint, "fsMainDepthMask");
	const transparent = new PBRMaterial({ alphaMode: AlphaMode.Blend });
	assert.equal(await resources.resolvePipeline({ ...request,
		materialState: resolver.resolveBuiltinDepth(transparent, runtime) }), null);
	assert.equal(await resources.resolvePipeline({ ...request,
		materialState: resolver.resolveBuiltinDepth(new Material({ depthWrite: false }), runtime) }),
		null);
	const transmission = new PBRMaterial({ transmissionFactor: 0.5 });
	assert.equal(await resources.resolvePipeline({ ...request,
		materialState: resolver.resolveBuiltinDepth(transmission, runtime) }), null);
	const custom = new ShaderMaterial({
		vertexEntryPoint: "customVs", fragmentEntryPoint: "customFsSingle", chunks: [
		{ backend: "webgpu", language: "wgsl", stage: "vertex", code: WGSL_VERTEX },
		{ backend: "webgpu", language: "wgsl", stage: "fragment", mode: "single",
			code: WGSL_FRAGMENT_SINGLE },
	] });
	const customPolicy = resolver.resolve(custom, createWebGPUMaterialUniformData(custom, false),
		false, "single", "early-z", runtime);
	const customPipeline = await resources.resolvePipeline({ ...request, materialState: customPolicy });
	assert.equal(customPipeline.desc.layout, layouts.scenePipelineLayouts[customPolicy.shadingFamily],
		"custom depth preserves original scene ABI");
	assert.equal(customPipeline.desc.vertex.entryPoint, "customVs");

	const shader = await ShaderSource.load("webgpu.scene.depth");
	const compileUniformContract = (source) => new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }), profile: WEBGPU_TEST_PROFILE, mode: "strict",
	}).compile({
		code: source.code, language: "wgsl", stage: "unknown", sourceKind: "builtin-scene",
	}).code;
	const definitions = compileUniformContract((await ShaderSource.load("webgpu.scene")).source);
	const depthDefinitions = compileUniformContract(shader.source);
	for (const name of ["FrameCameraUniforms", "ObjectUniforms", "AnimationParams", "StaticInstance"]) {
		const struct = new RegExp(`struct ${name} \\{[^}]*\\}`, "s");
		assert.equal(depthDefinitions.match(struct)?.[0].replace(/\s+/g, " "),
			definitions.match(struct)?.[0].replace(/\s+/g, " "),
			`${name} preserves scene buffer ABI`);
	}
	const bindings = [...shader.source.code.matchAll(/@group\((\d+)\) @binding\((\d+)\)/g)]
		.map((m) => `${m[1]}:${m[2]}`);
	assert.deepEqual(bindings, ["0:0", "1:0", "1:34", "1:35", "1:36", "1:37", "1:40", "2:0", "2:1", "2:2"]);
	assert.ok(shader.source.code.includes("#define IGNIS_WEBGPU_ANIMATION_POSITION_ONLY 1"));
	assert.ok(shader.source.code.includes("staticInstances[input.instanceIndex]"));
	assert.ok(shader.source.code.includes("clipPosition.z * 0.5 + clipPosition.w * 0.5"));
	assert.ok(shader.source.code.includes("jitterNdc * clipPosition.w"));
	assert.ok(shader.source.code.includes("clamp(depthMaterial.params.x * baseAlpha, 0.0, 1.0)"));
	const compiled = new ShaderBackendCompileStage({
		runtime: new ShaderRuntime({ mode: "strict" }), profile: WEBGPU_TEST_PROFILE, mode: "strict",
	}).compile({
		code: shader.source.code, language: "wgsl", stage: "unknown", sourceKind: "builtin-scene",
		sourceMap: shader.source.sourceMap,
	});
	assert.ok(!compiled.code.includes("arrayLength(&morphNormalDeltas)"));
	assert.ok(!compiled.code.includes("safeNormalize("));
	assert.ok(compiled.code.includes("fn applySkinning"));
	assert.ok(!compiled.code.includes("#import"));
	resources.destroy();
	assert.ok(backend.shaderModules.every((module) => module.destroyed));
	assert.ok(backend.pipelines.every((pipeline) => pipeline.destroyed));
}

async function testBuiltinDepthInvalidationDuringCreation() {
	const backend = new FakeBackend();
	const resources = new WebGPUScenePipelineResources(backend, createLayouts());
	const resolver = new WebGPUMaterialPipelineResolver();
	const request = {
		materialState: resolver.resolveBuiltinDepth(new PBRMaterial(), readWebGPUShaderRuntimeView(backend)),
		pass: resolveWebGPUScenePassDescriptor("single", "default", "early-z-prepass"),
		topology: "triangle-list",
		geometryLayout: { layoutKey: "depth-race", sceneVertexLayouts: [] },
		sampleCount: 1,
	};
	let releaseCreation;
	let enteredCreation;
	const creationGate = new Promise((resolve) => { releaseCreation = resolve; });
	const creationEntered = new Promise((resolve) => { enteredCreation = resolve; });
	const createModule = backend.createShaderModule.bind(backend);
	let calls = 0;
	backend.createShaderModule = async (desc) => {
		if (calls++ === 0) {
			enteredCreation();
			await creationGate;
		}
		return createModule(desc);
	};
	const first = resources.resolvePipeline(request);
	const second = resources.resolvePipeline(request);
	await creationEntered;
	resources.invalidateShaderRuntimeCaches();
	releaseCreation();
	const [pipeline, shared] = await Promise.all([first, second]);
	assert.equal(pipeline, shared);
	assert.equal(backend.shaderModules.length, 2);
	assert.equal(backend.shaderModules[0].destroyed, true,
		"module created under invalidated generation is destroyed");
	assert.equal(pipeline.desc.vertex.module, backend.shaderModules[1]);
	assert.equal(backend.pipelines.filter((entry) => !entry.destroyed).length, 1,
		"only the winning pipeline stays live after retry");
	resources.destroy();
	assert.ok(backend.shaderModules.every((module) => module.destroyed));
	assert.ok(backend.pipelines.every((entry) => entry.destroyed));
}

async function testBuiltinDepthDestructionDuringCreation(stage = "createShaderModule") {
	const backend = new FakeBackend();
	const resources = new WebGPUScenePipelineResources(backend, createLayouts());
	const resolver = new WebGPUMaterialPipelineResolver();
	const request = {
		materialState: resolver.resolveBuiltinDepth(new Material(), readWebGPUShaderRuntimeView(backend)),
		pass: resolveWebGPUScenePassDescriptor("single", "default", "early-z-prepass"),
		topology: "triangle-list",
		geometryLayout: { layoutKey: "depth-teardown", sceneVertexLayouts: [] },
		sampleCount: 1,
	};
	let releaseCreation;
	let enteredCreation;
	const creationGate = new Promise((resolve) => { releaseCreation = resolve; });
	const creationEntered = new Promise((resolve) => { enteredCreation = resolve; });
	const createResource = backend[stage].bind(backend);
	let calls = 0;
	backend[stage] = async (desc) => {
		if (calls++ === 0) {
			enteredCreation();
			await creationGate;
		}
		return createResource(desc);
	};
	const pending = resources.resolvePipeline(request);
	await creationEntered;
	resources.destroy();
	releaseCreation();
	assert.equal(await pending, null);
	assert.equal(backend.shaderModules.length, 1,
		"pending creation must not retry after permanent teardown");
	assert.equal(backend.shaderModules[0].destroyed, true);
	assert.equal(backend.pipelines.length, stage === "createPipeline" ? 1 : 0);
	assert.ok(backend.pipelines.every((pipeline) => pipeline.destroyed),
		"pending pipeline must be destroyed without publishing after teardown");
	assert.equal(await resources.resolvePipeline(request), null);
	assert.equal(backend.shaderModules.length, 1);
}

async function run() {
	await testBuiltinDepthIsolation();
	await testBuiltinDepthInvalidationDuringCreation();
	await testBuiltinDepthDestructionDuringCreation();
	await testBuiltinDepthDestructionDuringCreation("createPipeline");
	testWebGPUMaterialPipelineResolverProducesImmutableRuntimeVariants();
	await testWGSLProgramSelection();
	await testWebGPUPipelineCacheIncludesGeometryLayout();
	await testWebGPUMRTFallsBackToSingleFragmentEntryPoint();
	await testWebGPUDeferredProgramSelection();
	await testBuiltinBaseGBufferPipelineSelection();
	testResolveWebGPUDepthPrepassProgramContract();
	testDepthWriteParamIsInheritedFromMaterial();
	testTransparentOutputCompatibilityOptIn();
	await testGLSLProgramUsesTranspiler();
	await testGLSLWithoutTranspilerThrows();
	await testWarnModeFallbackToBuiltinShader();
	testResolveWebGLProgramPrefersWebGLSource();
	testResolveWebGLProgramSupportsModeAwareWebGLFragments();
	testResolveWebGLProgramMRTFallsBackToSingleFragment();
	testResolveWebGLProgramFallsBackToWebGPUGLSL();
	testResolveWebGLProgramMissingSourceThrows();
	testResolveWebGLDepthPrepassProgramContract();
	testChunkBackendMustBeExplicit();
	testChunkApiSupportsUnifiedShaderUpdates();
	testTextureBindingAutoSlotAndUniformDefaults();
	testTextureBindingGeneratedSourceBlocks();
	testTextureBindingUvSetGreaterThanOneIsPreserved();
	testUniformBindingSchemaAndValueRevision();
	testUniformBindingDuplicateDiagnostics();
	testUniformBindingGeneratedSourceBlocks();
	testGeneratedMaterialABIOwnershipRejectsRedeclarations();
	console.log("ShaderMaterial tests passed");
}

await run();
