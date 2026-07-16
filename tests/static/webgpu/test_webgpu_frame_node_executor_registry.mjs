import assert from "node:assert/strict";

import { WebGPUFrameNodeExecutorRegistry } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameNodeExecutorRegistry.ts";

function createNode(kind = "opaque-scene") {
	return {
		id: `main-opaque:${kind}`,
		stage: "main-opaque",
		kind,
		label: "RegistryTest",
	};
}

async function testRegistryDispatchesByNodeKind() {
	const calls = [];
	const executor = async (node, session) => {
		calls.push([node.id, session.context.id]);
	};
	const registry = new WebGPUFrameNodeExecutorRegistry(
		new Proxy({}, { get: () => executor })
	);
	const session = { context: { id: "frame-context" } };

	await registry.execute(createNode(), session);

	assert.deepEqual(calls, [["main-opaque:opaque-scene", "frame-context"]]);
}

async function testRegistryRejectsMissingExecutor() {
	const registry = new WebGPUFrameNodeExecutorRegistry({});

	await assert.rejects(
		registry.execute(createNode("hiz-build"), { context: {} }),
		/node kind "hiz-build" has no executor/
	);
}

await testRegistryDispatchesByNodeKind();
await testRegistryRejectsMissingExecutor();
console.log("WebGPU frame node executor registry tests passed");
