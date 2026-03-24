import assert from "node:assert/strict";
import { ShaderMaterial } from "../src/materials/ShaderMaterial.ts";
import { WebGPUPipelineLibrary } from "../src/renderers/webgpu/WebGPUPipelineLibrary.ts";
import { ShaderRuntime } from "../src/shaders/runtime/index.ts";

class FakeBackend {
	constructor() {
		this.canvasFormat = "rgba8unorm";
		this.shaderModules = [];
		this.pipelines = [];
		this.warnings = [];
		this.failCustomShaderModules = false;
		this.shaderRuntime = new ShaderRuntime({ mode: "strict" });
	}

	async createShaderModule(desc) {
		if (
			this.failCustomShaderModules &&
			typeof desc.label === "string" &&
			desc.label.startsWith("WebGPUShaderMaterial")
		) {
			throw new Error("simulated custom shader module compile failure");
		}
		const module = { label: desc.label, desc };
		this.shaderModules.push(module);
		return module;
	}

	createPipeline(desc) {
		const pipeline = { label: desc.label, desc };
		this.pipelines.push(pipeline);
		return pipeline;
	}

	warnOnce(key, message) {
		this.warnings.push({ key, message });
	}
}

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

async function testWGSLProgramSelection() {
	const backend = new FakeBackend();
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "WGSLMaterial",
		webgpuWGSL: {
			vertex: WGSL_VERTEX,
			fragmentSingle: WGSL_FRAGMENT_SINGLE,
			fragmentMRT: WGSL_FRAGMENT_MRT,
		},
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
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const transpilerCalls = [];
	const material = new ShaderMaterial({
		name: "GLSLMaterial",
		webgpuGLSL: {
			vertex: "void main() { gl_Position = vec4(0.0); }",
			fragmentSingle: "void main() { }",
			fragmentMRT: "void main() { }",
		},
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
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenGLSLMaterial",
		webgpuGLSL: {
			vertex: "void main() { gl_Position = vec4(0.0); }",
			fragmentSingle: "void main() { }",
		},
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
	backend.shaderRuntime.setMode("warn");
	backend.failCustomShaderModules = true;
	const library = new WebGPUPipelineLibrary(backend, createLayouts());
	const material = new ShaderMaterial({
		name: "BrokenCustomShader",
		webgpuWGSL: {
			vertex: WGSL_VERTEX,
			fragmentSingle: WGSL_FRAGMENT_SINGLE,
		},
		vertexEntryPoint: "customVs",
		fragmentSingleEntryPoint: "customFsSingle",
	});

	const pipeline = await library.getPipeline(material, "single");
	assert.equal(pipeline.desc.vertex.entryPoint, "vsMain");
	assert.equal(pipeline.desc.fragment.entryPoint, "fsMainSingle");
	assert.ok(
		backend.shaderModules.some((module) => module.label === "WebGPUSceneShader")
	);
	assert.ok(
		backend.warnings.some((warning) =>
			warning.key.startsWith("webgpu-shader-material-compile-failed-")
		)
	);
}

function testResolveWebGLProgramPrefersWebGLSource() {
	const material = new ShaderMaterial({
		name: "WebGLCustomMaterial",
		webgpuGLSL: {
			vertex: "legacy-vertex",
			fragmentSingle: "legacy-fragment",
		},
		webglGLSL: {
			vertex: WEBGL_VERTEX,
			fragment: WEBGL_FRAGMENT,
		},
	});

	const program = material.resolveWebGLProgram();
	assert.equal(program.vertexCode, WEBGL_VERTEX);
	assert.equal(program.fragmentCode, WEBGL_FRAGMENT);
}

function testResolveWebGLProgramFallsBackToWebGPUGLSL() {
	const material = new ShaderMaterial({
		name: "WebGLFallbackMaterial",
		webgpuGLSL: {
			vertex: "legacy-vertex",
			fragmentSingle: "legacy-fragment",
		},
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

async function run() {
	await testWGSLProgramSelection();
	await testGLSLProgramUsesTranspiler();
	await testGLSLWithoutTranspilerThrows();
	await testWarnModeFallbackToBuiltinShader();
	testResolveWebGLProgramPrefersWebGLSource();
	testResolveWebGLProgramFallsBackToWebGPUGLSL();
	testResolveWebGLProgramMissingSourceThrows();
	console.log("ShaderMaterial tests passed");
}

await run();
