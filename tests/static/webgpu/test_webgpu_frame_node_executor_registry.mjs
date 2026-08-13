import assert from "node:assert/strict";

import { WebGPUFrameNodeExecutorRegistry } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameNodeExecutorRegistry.ts";
import { WEBGPU_FRAME_GRAPH_NODE_KINDS } from "../../../src/backends/webgpu/rendergraph/types.ts";

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
	const complete = Object.fromEntries(
		WEBGPU_FRAME_GRAPH_NODE_KINDS.map((kind) => [kind, executor]),
	);
	const registry = WebGPUFrameNodeExecutorRegistry.fromModules([
		{ id: "owner", executors: complete, destroy() {} },
	]);
	const session = { context: { id: "frame-context" } };

	await registry.execute({ ...createNode(), ownerId: "owner" }, session);

	assert.deepEqual(calls, [["main-opaque:opaque-scene", "frame-context"]]);
}

async function testRegistryRejectsMissingExecutor() {
	const registry = WebGPUFrameNodeExecutorRegistry.fromModules([
		{
			id: "owner",
			executors: Object.fromEntries(
				WEBGPU_FRAME_GRAPH_NODE_KINDS.map((kind) => [kind, async () => {}]),
			),
			destroy() {},
		},
	]);

	await assert.rejects(
		registry.execute({ ...createNode("hiz-build"), ownerId: "missing" }, { context: {} }),
		/no owner-aware executor/,
	);
}

function testRuntimeCompositionRejectsDuplicateAndMissingOwners() {
	assert.throws(
		() => WebGPUFrameNodeExecutorRegistry.fromModules([]),
		/missing executors/,
	);
	const complete = Object.fromEntries(
		WEBGPU_FRAME_GRAPH_NODE_KINDS.map((kind) => [kind, async () => {}]),
	);
	assert.doesNotThrow(() => WebGPUFrameNodeExecutorRegistry.fromModules([
		{ id: "all", executors: complete, destroy() {} },
		{ id: "another-shadow-owner", executors: { shadow: async () => {} }, destroy() {} },
	]));
}

await testRegistryDispatchesByNodeKind();
await testRegistryRejectsMissingExecutor();
testRuntimeCompositionRejectsDuplicateAndMissingOwners();
console.log("WebGPU frame node executor registry tests passed");
