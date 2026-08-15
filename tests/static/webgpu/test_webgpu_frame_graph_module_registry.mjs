import assert from "node:assert/strict";

import { WebGPUFrameGraphModuleRegistry } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphModuleRegistry.ts";
import { WebGPUFrameMessageSnapshot } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessage.ts";
import {
	WebGPUDeferredFrameModule,
	WebGPUDeferredOpaqueStatePort,
} from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredFrameModule.ts";
import { WebGPUReflectionFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUReflectionFrameModule.ts";
import { WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessages.ts";
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

function createNode(kind = "opaque-scene", localId = kind) {
	return {
		id: `${PASS.stage}:${localId}`,
		localId,
		stage: PASS.stage,
		kind,
		label: localId,
		retention: "always",
	};
}

function createRegistry(modules) {
	const registry = new WebGPUFrameGraphModuleRegistry();
	registry.register({ id: "executors", executors: COMPLETE_EXECUTORS, destroy() {} });
	for (const module of modules) registry.register(module);
	registry.seal();
	return registry;
}

function plan(registry) {
	return registry.planStage({ pass: PASS, context: {}, state: {} },
		new WebGPUFrameMessageSnapshot());
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

	const missing = new WebGPUFrameGraphModuleRegistry();
	assert.throws(() => missing.seal(), /missing executors/);
}

async function testLaneOrderDoesNotDependOnRegistrationOrder() {
	const geometry = {
		id: "geometry",
		executors: {},
		planStage: () => [{ lane: "geometry", nodes: [createNode()] }],
		destroy() {},
	};
	const lighting = {
		id: "lighting",
		executors: {},
		planStage: () => [{ lane: "lighting", nodes: [createNode("deferred-lighting")] }],
		destroy() {},
	};
	const forward = await plan(createRegistry([geometry, lighting]));
	const reversed = await plan(createRegistry([lighting, geometry]));
	assert.deepEqual(
		forward.nodes.map((node) => node.id),
		["main-opaque:geometry:opaque-scene", "main-opaque:lighting:deferred-lighting"],
	);
	assert.deepEqual(reversed.nodes.map((node) => node.id), forward.nodes.map((node) => node.id));
}

async function testStaticEdgesAndOwnerLocalIdentity() {
	const module = (id, contribution) => ({
		id,
		executors: {},
		planStage: () => [contribution],
		destroy() {},
	});
	await assert.rejects(
		plan(createRegistry([
			module("a", { lane: "geometry", nodes: [createNode()] }),
			module("b", { lane: "geometry", nodes: [createNode()] }),
		])),
		/require a static before\/after edge/,
	);
	const ordered = await plan(createRegistry([
		module("b", { lane: "geometry", after: ["a"], nodes: [createNode()] }),
		module("a", {
			lane: "geometry",
			before: ["b"],
			nodes: [createNode("opaque-scene", "first"), createNode("opaque-scene", "second")],
		}),
	]));
	assert.deepEqual(ordered.nodes.map((node) => node.id), [
		"main-opaque:a:first",
		"main-opaque:a:second",
		"main-opaque:b:opaque-scene",
	]);
}

async function testExclusiveAndCompositionConflicts() {
	const module = (id, contribution) => ({
		id,
		executors: {},
		planStage: () => [contribution],
		destroy() {},
	});
	const exclusive = await plan(createRegistry([
		module("scene", { lane: "geometry", nodes: [createNode()] }),
		module("custom", {
			lane: "geometry",
			exclusive: true,
			nodes: [createNode("opaque-external")],
		}),
	]));
	assert.deepEqual(exclusive.nodes.map((node) => node.ownerId), ["custom"]);

	const composition = { namespace: "test", definition: {}, inputs: {} };
	await assert.rejects(
		plan(createRegistry([
			module("a", { lane: "postprocess", before: ["b"], composition }),
			module("b", { lane: "postprocess", after: ["a"], composition }),
		])),
		/has composed subgraphs from both/,
	);
}

async function testDeferredReflectionCompositeHasOneOwnerAndExecution() {
	let lightingExecutions = 0;
	let reflectionExecutions = 0;
	const deferredState = new WebGPUDeferredOpaqueStatePort();
	const deferred = new WebGPUDeferredFrameModule(
		{ async recordLightingPass() { lightingExecutions++; } },
		{},
		{ async recordMainPass() {} },
		deferredState,
	);
	const reflection = new WebGPUReflectionFrameModule(
		{},
		{ async composite() { reflectionExecutions++; } },
		{
			getFrameTargets: () => ({ planarReflectionMask: null }),
			getMSAATargets: () => null,
		},
	);
	const planningInput = {
		pass: PASS,
		context: {},
		state: {
			deferredActive: true,
			sceneTargetMode: "gbuffer",
			deferredGBufferLayout: "base",
		},
		messages: {
			getAll: (descriptor) => descriptor === WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE
				? [{ source: "reflection", needsPlanarReflectionComposite: true }]
				: [],
		},
	};
	const nodes = [
		...deferred.planStage(planningInput),
		...reflection.planStage(planningInput),
	].flatMap((contribution) => contribution.nodes ?? []);
	assert.equal(
		nodes.filter((node) => node.kind === "planar-reflection-composite").length,
		1,
	);
	const session = {
		context: {},
		commands: { encoder: {} },
		resources: {},
		targets: {
			frameTargets: { planarReflectionMask: null },
			msaaTargets: null,
		},
		configuration: {
			mrtSupported: true,
			samplePlan: { sampleCount: 1 },
		},
	};
	deferredState.publish({
			lightingEnabled: true,
			clearSceneColor: true,
			fallbackPackets: [],
	});
	await deferred.executors["deferred-lighting"]({}, session);
	assert.equal(lightingExecutions, 1);
	assert.equal(reflectionExecutions, 0);
	await reflection.executors["planar-reflection-composite"]({}, session);
	assert.equal(reflectionExecutions, 1);
}

async function testFinalOutputPublicationFeedsPresentationPlanning() {
	let presentedResource = null;
	const registry = createRegistry([{
		id: "post-process",
		executors: {},
		planStage(input) {
			return input.pass.stage === "postprocess" ? [{
				lane: "postprocess",
				finalOutput: {
					resource: "postprocess:final-color",
					colorDomain: "display-encoded",
				},
			}] : [];
		},
		destroy() {},
	}, {
		id: "presentation",
		executors: {},
		planStage(input) {
			if (input.finalization !== true) return [];
			presentedResource = input.finalColorResource;
			return [];
		},
		destroy() {},
	}]);
	const snapshot = new WebGPUFrameMessageSnapshot();
	await registry.planStage({
		pass: { ...PASS, stage: "postprocess" },
		context: {},
		state: {},
	}, snapshot);
	assert.deepEqual(registry.finalOutput, {
		resource: "postprocess:final-color",
		colorDomain: "display-encoded",
	});
	await registry.planStage({
		pass: { ...PASS, stage: "webgpu-present" },
		context: {},
		state: {},
		finalization: true,
	}, snapshot);
	assert.equal(presentedResource, "postprocess:final-color");
}

testRegistrationAndSealingValidation();
await testLaneOrderDoesNotDependOnRegistrationOrder();
await testStaticEdgesAndOwnerLocalIdentity();
await testExclusiveAndCompositionConflicts();
await testDeferredReflectionCompositeHasOneOwnerAndExecution();
await testFinalOutputPublicationFeedsPresentationPlanning();
console.log("WebGPU frame graph module registry tests passed");
