import assert from "node:assert/strict";

import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { WebGLAuxiliaryRasterRuntime } from "../../../src/backends/webgl/WebGLAuxiliaryRasterRuntime.ts";
import {
	WEBGL_AUXILIARY_RASTER_EXTENSION,
} from "../../../src/backends/webgl/WebGLAuxiliaryRaster.ts";
import {
	WebGLScopedRasterEncoder,
} from "../../../src/backends/webgl/WebGLCustomRenderTargetRuntime.ts";
import { WebGLResourceStore } from "../../../src/backends/webgl/WebGLResourceStore.ts";

function testTypedUniformArraysAndClosedScope() {
	const calls = [];
	const gl = {
		getUniformLocation: (_program, name) => ({ name }),
		uniform1fv: (_location, value) => calls.push(["f32", Array.from(value)]),
		uniform2iv: (_location, value) => calls.push(["vec2i", Array.from(value)]),
		uniformMatrix4fv: (_location, transpose, value) =>
			calls.push(["mat4", transpose, value.length]),
		viewport: (...values) => calls.push(["viewport", ...values]),
	};
	const scope = { active: true };
	const resources = new WebGLResourceStore();
	const encoder = new WebGLScopedRasterEncoder(gl, scope, resources);
	encoder._pipeline = resources.markScope(
		resources.createPipeline(undefined, { program: {}, desc: {} }, () => {}), scope,
	);
	encoder.setUniforms([
		{ name: "uWeights", type: "f32", value: new Float32Array([1, 2]) },
		{ name: "uPairs", type: "vec2i", value: new Int32Array([1, 2, 3, 4]) },
		{ name: "uTransforms", type: "mat4x4f", value: new Float32Array(32) },
	]);
	assert.deepEqual(calls[0], ["f32", [1, 2]]);
	assert.deepEqual(calls[1], ["vec2i", [1, 2, 3, 4]]);
	assert.deepEqual(calls[2], ["mat4", false, 32]);
	assert.throws(
		() => encoder.setUniforms([{
			name: "uBad",
			type: "vec3f",
			value: [1, 2],
		}]),
		/positive multiple of 3/,
	);
	scope.active = false;
	assert.throws(
		() => encoder.setViewport(0, 0, 1, 1),
		/scope is no longer active/,
	);
}

function testAuxiliaryUploadPreservesSrgbStorageAndAlpha() {
	const uploads = [];
	const gl = {
		RGBA8: 0x8058,
		SRGB8_ALPHA8: 0x8c43,
		RGBA: 0x1908,
		UNSIGNED_BYTE: 0x1401,
		TEXTURE_2D: 0x0de1,
		UNPACK_ALIGNMENT: 0x0cf5,
		TEXTURE_MAX_LEVEL: 0x813d,
		bindTexture() {},
		pixelStorei() {},
		texParameteri() {},
		texImage2D(...args) { uploads.push(args); },
	};
	const runtime = new WebGLAuxiliaryRasterRuntime(gl, {}, {});
	const source = new Uint8Array([128, 64, 32, 127]);
	const texture = runtime._resources.createTexture({
		width: 1, height: 1, format: TextureFormat.RGBA8UnormSrgb,
	}, { texture: {}, maxMipLevel: 0 }, () => {});
	runtime._writeTexture(
		texture,
		source, {}, { width: 1, height: 1 }
	);
	assert.equal(uploads[0][2], gl.SRGB8_ALPHA8);
	assert.deepEqual(Array.from(uploads[0][8]), [128, 64, 32, 127]);
	delete gl.SRGB8_ALPHA8;
	assert.throws(() => runtime._writeTexture(
		texture,
		source, {}, { width: 1, height: 1 }
	), /sRGB/);
}

async function testPrivateScopeOwnershipAndLifetime() {
	const deleted = [];
	const gl = {
		ARRAY_BUFFER: 1, ELEMENT_ARRAY_BUFFER: 2, STATIC_DRAW: 3,
		createBuffer: () => ({}), bindBuffer() {}, bufferData() {},
		deleteBuffer: (buffer) => deleted.push(buffer),
	};
	const runtime = new WebGLAuxiliaryRasterRuntime(gl, {}, {});
	const foreignRuntime = new WebGLAuxiliaryRasterRuntime(gl, {}, {});
	const foreign = foreignRuntime._resources.createBuffer(4, {}, () => {});
	let previous;
	await runtime.execute(1, new AbortController().signal, ({ resources, encoder }) => {
		previous = resources.createBuffer({ size: 4, usage: 1 });
		assert.equal(previous.kind, "buffer");
		assert.equal(Object.values(previous).some((value) => value === foreign), false);
		assert.throws(() => encoder.setVertexBuffer(0, foreign), /another owner/);
		assert.throws(() => encoder.setVertexBuffer(0, { ...previous }), /another owner/);
		assert.throws(() => resources.createBindingGroup({
			entries: [{ binding: 0, resource: foreign }],
		}), /another owner/);
		encoder.setVertexBuffer(0, previous);
		resources.createBindingGroup({ entries: [{ binding: 0,
			resource: { buffer: previous, offset: 0, size: 4 },
		}] });
		previous.destroy();
		previous.destroy();
		assert.throws(() => encoder.setVertexBuffer(0, previous), /destroyed|unavailable/);
	});
	assert.equal(deleted.length, 1, "scope cleanup must not delete an explicitly released buffer twice");
	await runtime.execute(1, new AbortController().signal, ({ encoder }) => {
		assert.throws(() => encoder.setVertexBuffer(0, previous), /destroyed|unavailable/);
	});
	runtime.destroy();
	foreignRuntime.destroy();
}

async function testCachedProgramsUsePrivateShaderSourcesAndScopedAliases() {
	let createdPrograms = 0;
	let deletedPrograms = 0;
	const gl = {
		VERTEX_SHADER: 1, FRAGMENT_SHADER: 2, COMPILE_STATUS: 3, LINK_STATUS: 4,
		createShader: () => ({}), shaderSource() {}, compileShader() {},
		getShaderParameter: () => true, deleteShader() {},
		createProgram: () => { createdPrograms++; return {}; },
		attachShader() {}, linkProgram() {}, getProgramParameter: () => true,
		deleteProgram: () => { deletedPrograms++; },
	};
	const compiler = { revision: 0, compile: (request) => request };
	const runtime = new WebGLAuxiliaryRasterRuntime(gl, { revision: 0 }, compiler);
	const handles = [];
	const execute = (code) => runtime.execute(1, new AbortController().signal, ({ resources }) => {
		const module = resources.createShaderModule({ language: "glsl", stage: "vertex", code });
		const pipeline = resources.createRenderPipeline({ vertex: { module, entryPoint: "main" } });
		handles.push(pipeline);
		assert.deepEqual(Object.keys(pipeline).sort(), ["destroy", "kind", "label"]);
	});
	await execute("void main() { gl_Position = vec4(0.0); }");
	await execute("void main() { gl_Position = vec4(0.0); }");
	assert.equal(createdPrograms, 1, "equivalent private shader sources reuse the native program");
	assert.notEqual(handles[0], handles[1], "each scope receives its own opaque pipeline handle");
	assert.equal(deletedPrograms, 0, "scope alias cleanup must retain the cached native program");
	await execute("void main() { gl_Position = vec4(1.0); }");
	assert.equal(createdPrograms, 2, "changed private shader source must produce a new program");
	compiler.revision++;
	await execute("void main() { gl_Position = vec4(1.0); }");
	assert.equal(createdPrograms, 3);
	assert.equal(deletedPrograms, 2, "compiler invalidation releases each cached program once");
	runtime.destroy();
	runtime.destroy();
	assert.equal(deletedPrograms, 3);
}

function createServices(id, calls, extensions) {
	return {
		frame: {},
		auxiliaryRaster: {
			hasExtension: (name) => extensions.includes(name),
			async execute(generation, signal, task) {
				calls.push([id, generation]);
				return task({ generation, signal, encoder: {}, resources: {} });
			},
			destroy() {},
		},
		restoreContextWorkBaseline() {},
		destroy() {},
	};
}

async function testRequestAwareAvailabilityAndRetainedExecution() {
	const calls = [];
	const backend = new WebGLBackend();
	const facade = backend.extensions.requireBackendExtension(
		WEBGL_AUXILIARY_RASTER_EXTENSION,
	);
	assert.deepEqual(facade.getAvailability(), {
		state: "temporarily-unavailable",
		acceptsRequests: false,
		reason: "WebGL backend must be initialized before requesting auxiliary raster work.",
	});

	backend._contextServices = createServices(
		"first",
		calls,
		["EXT_color_buffer_float", "OES_texture_half_float_linear"],
	);
	backend._contextWorkQueue.bindContext();
	assert.equal(facade.getAvailability({
		requiredExtensions: ["EXT_color_buffer_float"],
		alternativeExtensionGroups: [[
			"OES_texture_float_linear",
			"OES_texture_half_float_linear",
		]],
	}).state, "ready");
	assert.match(
		facade.getAvailability({ requiredExtensions: ["MISSING_extension"] })
			.reason,
		/requires MISSING_extension/,
	);

	assert.equal(await facade.execute({
		label: "ready-task",
		task: ({ generation }) => generation,
	}), 1);
	backend._contextWorkQueue.suspend();
	assert.equal(facade.getAvailability().acceptsRequests, false);
	assert.equal(facade.getAvailability({
		contextLossPolicy: "retain-pending",
	}).acceptsRequests, true);

	const retained = facade.execute({
		label: "retained-task",
		contextLossPolicy: "retain-pending",
		task: () => "restored",
	});
	backend._contextServices = createServices("restored", calls, []);
	backend._contextWorkQueue.bindContext();
	assert.equal(await retained, "restored");
	assert.deepEqual(calls, [["first", 1], ["restored", 2]]);
	backend.destroy();
}

testAuxiliaryUploadPreservesSrgbStorageAndAlpha();
testTypedUniformArraysAndClosedScope();
await testPrivateScopeOwnershipAndLifetime();
await testCachedProgramsUsePrivateShaderSourcesAndScopedAliases();
await testRequestAwareAvailabilityAndRetainedExecution();

console.log("WebGL auxiliary raster tests passed");
