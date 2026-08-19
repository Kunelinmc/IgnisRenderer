import assert from "node:assert/strict";

import {
	WebGPUShadowCasterRenderer,
} from "../../../src/backends/webgpu/WebGPUShadowCasterRenderer.ts";

function createResource(id) {
	return { id, _gpuResource: { id: `gpu:${id}`, destroy() {} } };
}

const fallback = createResource("fallback");
const staticGroup = { id: "group:static" };
const bindGroups = [];
const device = {
	createBindGroup(desc) {
		const group = { id: `group:${bindGroups.length}`, desc };
		bindGroups.push(group);
		return group;
	},
};
const backend = { device };
let payload = {
	generation: 0,
	paramsBuffer: fallback,
	jointMatricesBuffer: fallback,
	morphWeightsBuffer: fallback,
	jointCount: 0,
	morphCount: 0,
};
const pool = {
	getShadowPayload() {
		return payload;
	},
	getFallbackStorageBuffer() {
		return fallback;
	},
};
const renderer = new WebGPUShadowCasterRenderer(backend, {}, pool);
Object.assign(renderer, {
	_animationBindGroupLayout: { id: "layout:animation" },
	_staticAnimationBindGroup: staticGroup,
});
const packet = { id: "packet:shadow" };
const geometry = { morphPositionBuffer: null };
const context = { transient: { get() { return null; } } };

assert.strictEqual(
	renderer._resolveAnimationBinding(packet, geometry, context),
	staticGroup
);
assert.equal(bindGroups.length, 0);

payload = {
	generation: 1,
	paramsBuffer: createResource("params:1"),
	jointMatricesBuffer: createResource("joint:shared"),
	morphWeightsBuffer: fallback,
	jointCount: 1,
	morphCount: 0,
};
const first = renderer._resolveAnimationBinding(packet, geometry, context);
assert.equal(bindGroups.length, 1);
assert.strictEqual(
	first.desc.entries[1].resource.buffer,
	payload.jointMatricesBuffer._gpuResource
);
assert.strictEqual(
	renderer._resolveAnimationBinding(packet, geometry, context),
	first
);
assert.equal(bindGroups.length, 1);

payload = {
	...payload,
	generation: 2,
	jointMatricesBuffer: createResource("joint:grown"),
};
const rebuilt = renderer._resolveAnimationBinding(packet, geometry, context);
assert.notStrictEqual(rebuilt, first);
assert.equal(bindGroups.length, 2);

console.log("WebGPU shadow animation payload tests passed");
