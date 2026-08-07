import assert from "node:assert/strict";

import { WebGPUFrameGraphModuleRegistry } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphModuleRegistry.ts";
import {
	WebGPUFrameModuleStateStore,
	defineWebGPUFrameModuleStateKey,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphModule.ts";
import { WebGPUFrameGraphCompiler } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphCompiler.ts";
import { WEBGPU_FRAME_GRAPH_NODE_KINDS } from "../../../src/backends/webgpu/rendergraph/types.ts";

const COMPLETE_EXECUTORS = Object.fromEntries(
	WEBGPU_FRAME_GRAPH_NODE_KINDS.map((kind) => [kind, async () => {}]),
);
const PASS = {
	stage: "main-opaque",
	executor: "backend",
	enabled: true,
	dependsOn: [],
};

function createNode(id, kind = "opaque-scene") {
	return { id, stage: PASS.stage, kind, label: id, retention: "always" };
}

function createRegistry(modules) {
	const registry = new WebGPUFrameGraphModuleRegistry();
	registry.register({ id: "executors", executors: COMPLETE_EXECUTORS, destroy() {} });
	for (const module of modules) registry.register(module);
	registry.seal();
	return registry;
}

function plan(registry) {
	return registry.planStage({
		pass: PASS,
		context: {},
		state: {},
		moduleState: new WebGPUFrameModuleStateStore(),
	});
}

function compileNodeIds(registry) {
	const compiler = new WebGPUFrameGraphCompiler();
	compiler.beginFrame([]);
	return compiler.compileStage(plan(registry)).nodes.map((node) => node.id);
}

function testRegistrationAndSealingValidation() {
	const duplicate = new WebGPUFrameGraphModuleRegistry();
	duplicate.register({ id: "same", executors: COMPLETE_EXECUTORS, destroy() {} });
	assert.throws(
		() => duplicate.register({ id: "same", executors: {}, destroy() {} }),
		/already registered/,
	);

	const sealed = createRegistry([]);
	assert.throws(
		() => sealed.register({ id: "late", executors: {}, destroy() {} }),
		/registry is sealed/,
	);
	assert.equal(Object.isFrozen(sealed.modules), true);
	assert.throws(
		() => sealed.modules.push({ id: "late", executors: {}, destroy() {} }),
		TypeError,
	);

	const missing = new WebGPUFrameGraphModuleRegistry();
	assert.throws(() => missing.seal(), /missing executors/);

	const duplicateExecutor = new WebGPUFrameGraphModuleRegistry();
	duplicateExecutor.register({
		id: "complete",
		executors: COMPLETE_EXECUTORS,
		destroy() {},
	});
	duplicateExecutor.register({
		id: "duplicate-executor",
		executors: { shadow: async () => {} },
		destroy() {},
	});
	assert.throws(() => duplicateExecutor.seal(), /duplicate module owners/);
}

function testAnalysisStateOwnershipAndSealing() {
	const ownerKey = defineWebGPUFrameModuleStateKey("owner", "owner:value");
	const foreignKey = defineWebGPUFrameModuleStateKey("foreign", "foreign:value");
	const forgedOwnerKey = defineWebGPUFrameModuleStateKey("intruder", "owner:value");
	const registry = createRegistry([
		{
			id: "owner",
			executors: {},
			analyze(_input, state) {
				state.set(ownerKey, 42);
				assert.throws(() => state.set(foreignKey, 1), /cannot write state/);
				assert.throws(() => state.get(foreignKey), /cannot read state/);
			},
			destroy() {},
		},
		{
			id: "intruder",
			executors: {},
			analyze(_input, state) {
				assert.throws(
					() => state.get(forgedOwnerKey),
					/conflicting identities/,
				);
			},
			destroy() {},
		},
	]);
	const state = registry.analyze({
		context: {},
		framePackets: {},
		postProcessPasses: [],
	});
	assert.equal(state.require(ownerKey), 42);
	assert.throws(() => state.set(ownerKey, 43), /state is sealed/);
}

function testContributionOrderDoesNotDependOnRegistrationOrder() {
	const first = {
		id: "first",
		executors: {},
		planStage: () => [{ order: 100, nodes: [createNode("main-opaque:first")] }],
		destroy() {},
	};
	const second = {
		id: "second",
		executors: {},
		planStage: () => [{ order: 200, nodes: [createNode("main-opaque:second")] }],
		destroy() {},
	};
	const forward = compileNodeIds(createRegistry([first, second]));
	const reversed = compileNodeIds(createRegistry([second, first]));
	assert.deepEqual(forward, ["main-opaque:first", "main-opaque:second"]);
	assert.deepEqual(reversed, forward);
}

function testContributionConflictsAreRejected() {
	const module = (id, contribution) => ({
		id,
		executors: {},
		planStage: () => [contribution],
		destroy() {},
	});
	assert.throws(
		() => plan(createRegistry([
			module("a", { order: 100, nodes: [createNode("a")] }),
			module("b", { order: 100, nodes: [createNode("b")] }),
		])),
		/order 100 is owned by both/,
	);
	assert.throws(
		() => plan(createRegistry([
			module("a", { order: 100, nodes: [createNode("same-node")] }),
			module("b", { order: 200, nodes: [createNode("same-node")] }),
		])),
		/is contributed by both/,
	);
	const composition = { namespace: "test", definition: {}, inputs: {} };
	assert.throws(
		() => plan(createRegistry([
			module("a", { order: 100, composition }),
			module("b", { order: 200, composition }),
		])),
		/has composed subgraphs from both/,
	);
}

testRegistrationAndSealingValidation();
testAnalysisStateOwnershipAndSealing();
testContributionOrderDoesNotDependOnRegistrationOrder();
testContributionConflictsAreRejected();
console.log("WebGPU frame graph module registry tests passed");
