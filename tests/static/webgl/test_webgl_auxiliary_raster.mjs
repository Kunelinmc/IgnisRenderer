import assert from "node:assert/strict";

import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import {
	WEBGL_AUXILIARY_RASTER_EXTENSION,
} from "../../../src/backends/webgl/WebGLAuxiliaryRaster.ts";
import {
	WebGLScopedRasterEncoder,
} from "../../../src/backends/webgl/WebGLCustomRenderTargetRuntime.ts";

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
	const encoder = new WebGLScopedRasterEncoder(gl, scope);
	encoder._pipeline = { _webglProgram: {}, _webglDesc: {} };
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

testTypedUniformArraysAndClosedScope();
await testRequestAwareAvailabilityAndRetainedExecution();

console.log("WebGL auxiliary raster tests passed");
