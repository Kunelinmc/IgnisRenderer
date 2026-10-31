import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
const resourceManager = createWebGPUResourceManager();
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

import {
	WebGPUShadowCasterRenderer,
} from "../../../src/backends/webgpu/WebGPUShadowCasterRenderer.ts";

function createResource(id) {
	return Object.assign(resourceManager.registerBuffer(
		{ id: `gpu:${id}`, destroy() {} }, { size: 64 },
	), { id });
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
const backend = { device, resourceManager };
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
const packet = createTestDrawPacket({ id: "packet:shadow" });
const submission = packet.submission;
const geometry = { morphPositionBuffer: null };
const context = { transient: { get() { return null; } } };

function testGeometryResourcesBelongToShadowOwner() {
	const own = createResource("geometry:own");
	const foreignStore = createWebGPUResourceManager();
	const foreign = foreignStore.registerBuffer({ destroy() {} }, { size: 64 });
	const geometry = {
		shadowVertexBindings: [{ slot: 0, buffer: foreign }],
		indexBuffer: own,
		indexFormat: "uint32",
	};
	const caster = new WebGPUShadowCasterRenderer(backend, {
		getGeometry() { return geometry; },
	}, pool);
	assert.throws(() => caster._collectShadowDrawCandidate(submission), /not owned/i,
		"foreign vertex buffers must not enter a shadow candidate");
	geometry.shadowVertexBindings[0].buffer = own;
	geometry.indexBuffer = foreign;
	assert.throws(() => caster._collectShadowDrawCandidate(submission), /not owned/i,
		"foreign index buffers must not enter a shadow candidate");
	geometry.indexBuffer = own;
	const candidate = caster._collectShadowDrawCandidate(submission);
	assert.equal(candidate.vertexBindings[0].buffer, resourceManager.getBuffer(own));
	assert.equal(candidate.indexBuffer, resourceManager.getBuffer(own));
	own.destroy();
	assert.throws(() => caster._collectShadowDrawCandidate(submission), /not live|destroyed/i);
	foreign.destroy();
}

testGeometryResourcesBelongToShadowOwner();

assert.strictEqual(
	renderer._resolveAnimationBinding(submission, geometry, context),
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
const first = renderer._resolveAnimationBinding(submission, geometry, context);
assert.equal(bindGroups.length, 1);
assert.strictEqual(
	first.desc.entries[1].resource.buffer,
	resourceManager.getBuffer(payload.jointMatricesBuffer)
);
assert.strictEqual(
	renderer._resolveAnimationBinding(submission, geometry, context),
	first
);
assert.equal(bindGroups.length, 1);

payload = {
	...payload,
	generation: 2,
	jointMatricesBuffer: createResource("joint:grown"),
};
const rebuilt = renderer._resolveAnimationBinding(submission, geometry, context);
assert.notStrictEqual(rebuilt, first);
assert.equal(bindGroups.length, 2);

console.log("WebGPU shadow animation payload tests passed");
