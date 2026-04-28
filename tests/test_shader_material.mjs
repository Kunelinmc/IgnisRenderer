import assert from "node:assert/strict";
import { Logger } from "../src/foundation/Logger.ts";
import { ShaderMaterial } from "../src/materials/ShaderMaterial.ts";
import { WebGPUPipelineLibrary } from "../src/renderers/webgpu/WebGPUPipelineLibrary.ts";
import { ShaderRuntime } from "../src/shaders/runtime/index.ts";

import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";

function createLayouts() {
	return {
		scenePipelineLayout: { id: "scene-layout" },
		skyboxPipelineLayout: { id: "skybox-layout" },
	};
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
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "WGSLMaterial",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "mrt",
				code: WGSL_FRAGMENT_MRT,
			},
		],
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
		fragmentMRTEntryPoint: "customFsMRT",
	});

	const singlePipeline = await library.getPipeline(material, "single");
	assert.equal(singlePipeline.desc.vertex.entryPoint, "customVs");
	assert.equal(singlePipeline.desc.fragment.entryPoint, "customFsSingle");
	assert.equal(singlePipeline.desc.fragment.targets.length, 1);

	const mrtPipeline = await library.getPipeline(material, "mrt");
	assert.equal(mrtPipeline.desc.vertex.entryPoint, "customVs");
	assert.equal(mrtPipeline.desc.fragment.entryPoint, "customFsMRT");
	assert.equal(mrtPipeline.desc.fragment.targets.length, 5);

	const moduleCodes = backend.shaderModules.map((module) => module.desc.code);
	assert.ok(moduleCodes.includes(WGSL_VERTEX));
	assert.ok(moduleCodes.includes(WGSL_FRAGMENT_SINGLE));
	assert.ok(moduleCodes.includes(WGSL_FRAGMENT_MRT));

	const pipelineCountBefore = backend.pipelines.length;
	await library.getPipeline(material, "single");
	await library.getPipeline(material, "mrt");
	assert.equal(backend.pipelines.length, pipelineCountBefore);
}

async function testGLSLProgramUsesTranspiler() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const transpilerCalls = [];
	const material = new ShaderMaterial({
		name: "GLSLMaterial",
		chunks: [
			{
				language: "glsl",
				stage: "vertex",
				code: "void main() { gl_Position = vec4(0.0); }",
			},
			{
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: "void main() { }",
			},
			{
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

	await library.getPipeline(material, "mrt");
	await library.getPipeline(material, "single");

	assert.ok(transpilerCalls.some((call) => call.stage === "vertex"));
	assert.ok(transpilerCalls.some((call) => call.stage === "fragment-mrt"));
	assert.ok(transpilerCalls.some((call) => call.stage === "fragment-single"));
}

async function testGLSLWithoutTranspilerThrows() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenGLSLMaterial",
		chunks: [
			{
				language: "glsl",
				stage: "vertex",
				code: "void main() { gl_Position = vec4(0.0); }",
			},
			{
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
		() => library.getPipeline(material, "single"),
		/glslToWgsl transpiler/
	);
}

async function testWarnModeFallbackToBuiltinShader() {
	const backend = new FakeBackend();
	backend.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	backend.shaderRuntime.setMode("warn");
	backend.failCustomShaderModules = true;
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenCustomShader",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
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
		pipeline = await library.getPipeline(material, "single");
	});
	assert.equal(pipeline.desc.vertex.entryPoint, "vsMain");
	assert.equal(pipeline.desc.fragment.entryPoint, "fsMainSingle");
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
				language: "glsl",
				stage: "vertex",
				code: "legacy-vertex",
			},
			{
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

function testChunkApiSupportsUnifiedShaderUpdates() {
	const material = new ShaderMaterial({
		name: "ChunkMaterial",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: WGSL_FRAGMENT_SINGLE,
			},
			{
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

function testTextureBindingInjectDirectivesDecoratePrograms() {
	const material = new ShaderMaterial({
		name: "InjectBindingMaterial",
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: WGSL_VERTEX,
			},
			{
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

	const webgpuWithoutInject = material.resolveWebGPUProgram("single");
	assert.equal(
		webgpuWithoutInject.fragmentCode.includes(
			"ignis/material/texture-binding"
		),
		false
	);
	const webgpuWithInject = material.resolveWebGPUProgram("single", {
		enableRuntimeInjects: true,
	});
	assert.ok(
		webgpuWithInject.fragmentCode.includes(
			`#inject <ignis/material/texture-binding>(name="noise-map", slot=13`
		)
	);

	const webglWithoutInject = material.resolveWebGLProgram();
	assert.equal(
		webglWithoutInject.fragmentCode.includes(
			"ignis/material/texture-binding"
		),
		false
	);
	const webglWithInject = material.resolveWebGLProgram({
		enableRuntimeInjects: true,
	});
	assert.ok(webglWithInject.fragmentCode.trimStart().startsWith("#version 300 es"));
	assert.ok(
		webglWithInject.fragmentCode.includes('uniform="uNoiseTex"')
	);
}

function testTextureBindingUvSetGreaterThanOneUsesSecondSet() {
	const material = new ShaderMaterial({ name: "TextureBindingUvClamp" });
	material.setTextureBinding({
		name: "detail",
		texture: null,
		uvSet: 2,
	});
	const bindings = material.getTextureBindings();
	assert.equal(bindings.length, 1);
	assert.equal(bindings[0].uvSet, 1);
}

async function run() {
	await testWGSLProgramSelection();
	await testGLSLProgramUsesTranspiler();
	await testGLSLWithoutTranspilerThrows();
	await testWarnModeFallbackToBuiltinShader();
	testResolveWebGLProgramPrefersWebGLSource();
	testResolveWebGLProgramSupportsModeAwareWebGLFragments();
	testResolveWebGLProgramMRTFallsBackToSingleFragment();
	testResolveWebGLProgramFallsBackToWebGPUGLSL();
	testResolveWebGLProgramMissingSourceThrows();
	testChunkApiSupportsUnifiedShaderUpdates();
	testTextureBindingAutoSlotAndUniformDefaults();
	testTextureBindingInjectDirectivesDecoratePrograms();
	testTextureBindingUvSetGreaterThanOneUsesSecondSet();
	console.log("ShaderMaterial tests passed");
}

await run();
