import assert from "node:assert/strict";
import {
	INTERACTION_TRANSIENT_STATE_KEY,
} from "../src/pipeline/types.ts";
import { resolvePostProcessState } from "../src/pipeline/PostProcessController.ts";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import { createWebGPUBuiltInPostProcessPasses } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";
import { WebGPUPostProcessGraph } from "../src/renderers/webgpu/WebGPUPostProcessGraph.ts";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import { FakeWebGPUBackend as FakeBackend } from "./helpers/test_fakes.mjs";

const capabilities = {
	ssao: true,
	ssgi: true,
	taa: true,
	ssr: true,
	volumetric: true,
	fog: true,
	"motion-blur": true,
	dof: true,
	bloom: true,
	tonemap: true,
	"color-filter": true,
	fxaa: true,
	"interaction-outline": true,
	gamma: true,
};

function createPostProcess(overrides = {}) {
	return resolvePostProcessState(overrides, capabilities, "webgpu");
}

function createDeps() {
	const executeCalls = [];
	const presentCalls = [];
	const historyUpdates = {
		taa: null,
		ssr: null,
		volumetric: null,
	};
	const flags = {
		taaHistoryValid: true,
		ssrHistoryValid: true,
		volumetricHistoryValid: true,
		motionHistoryValid: true,
	};
	const frameBinding = { id: "frame-binding" };
	const lightingState = { id: "lighting-state" };

	return {
		deps: {
			async executeRuntimePass(request) {
				executeCalls.push(request);
				return {
					ran: true,
					historyUpdated: true,
				};
			},
			getFrameBinding() {
				return frameBinding;
			},
			getLightingState() {
				return lightingState;
			},
			async presentToCanvas(source, applyGamma) {
				presentCalls.push({ source, applyGamma });
			},
			getTAAHistoryValid() {
				return flags.taaHistoryValid;
			},
			getSSRHistoryValid() {
				return flags.ssrHistoryValid;
			},
			getVolumetricHistoryValid() {
				return flags.volumetricHistoryValid;
			},
			getMotionHistoryValid() {
				return flags.motionHistoryValid;
			},
			setTAAHistoryUpdated(updated) {
				historyUpdates.taa = updated;
			},
			setSSRHistoryUpdated(updated) {
				historyUpdates.ssr = updated;
			},
			setVolumetricHistoryUpdated(updated) {
				historyUpdates.volumetric = updated;
			},
		},
		executeCalls,
		presentCalls,
		historyUpdates,
		flags,
		frameBinding,
		lightingState,
	};
}

function createPassContext() {
	return {
		backend: {},
		encoder: {},
		frameContext: {
			transient: new Map(),
		},
		targets: {
			sceneColor: { id: "sceneColor" },
		},
	};
}

function createCustomPass(id, runtime = null) {
	return {
		id,
		kind: "compute",
		dependsOn: [],
		precompileHints: runtime?.warmupHints,
		runtime: runtime ?? undefined,
		isEnabled() {
			return true;
		},
		async execute(ctx) {
			if (!runtime) {
				return;
			}
			await ctx.executeRuntimePass({
				passId: runtime.id,
				encoder: ctx.encoder,
				targets: ctx.targets,
				frameContext: ctx.frameContext,
			});
		},
	};
}

function testBuiltInPassGraphOrder() {
	const { deps } = createDeps();
	const passes = createWebGPUBuiltInPostProcessPasses(deps);
	const graph = new WebGPUPostProcessGraph(passes);
	const order = graph.getExecutionOrder(
		createPostProcess({
			ssao: { enabled: true },
			ssgi: { enabled: true },
			taa: { enabled: true },
			ssr: { enabled: true },
			volumetric: { enabled: true },
			fog: { enabled: true },
			"motion-blur": { enabled: true },
			dof: { enabled: true },
			bloom: { enabled: true },
			"color-filter": { enabled: true },
			fxaa: { enabled: true },
		}),
		() => {}
	);

	assert.deepEqual(
		order.map((pass) => pass.id),
		[
			"ssao",
			"ssgi",
			"taa",
			"ssr",
			"volumetric",
			"fog",
			"motion-blur",
			"dof",
			"bloom",
			"tonemap",
			"color-filter",
			"fxaa",
			"interaction-outline",
			"gamma",
		]
	);

	const sceneFogOrder = graph.getExecutionOrder(
		createPostProcess({
			fog: {
				enabled: true,
				options: {
					application: "scene",
				},
			},
		}),
		() => {}
	);
	assert.equal(sceneFogOrder.some((pass) => pass.id === "fog"), false);

	const fxaaOnlyOrder = graph.getExecutionOrder(
		createPostProcess({
			gamma: { enabled: false },
			"color-filter": { enabled: true },
			fxaa: { enabled: true },
		}),
		() => {}
	);
	assert.deepEqual(
		fxaaOnlyOrder.map((pass) => pass.id),
		["tonemap", "color-filter", "fxaa", "interaction-outline"]
	);

	const tonemapDisabledOrder = graph.getExecutionOrder(
		createPostProcess({
			tonemap: { enabled: false },
			fxaa: { enabled: true },
		}),
		() => {}
	);
	assert.equal(
		tonemapDisabledOrder.some((pass) => pass.id === "tonemap"),
		false
	);
}

async function testTemporalPassWiring() {
	const {
		deps,
		executeCalls,
		historyUpdates,
		flags,
		frameBinding,
		lightingState,
	} = createDeps();
	const passes = createWebGPUBuiltInPostProcessPasses(deps);
	const byId = new Map(passes.map((pass) => [pass.id, pass]));
	const context = createPassContext();

	flags.motionHistoryValid = false;
	flags.taaHistoryValid = true;
	await byId.get("taa").execute(context);
	assert.equal(executeCalls[0].passId, "taa");
	assert.equal(executeCalls[0].historyValid, false);
	assert.equal(historyUpdates.taa, true);

	flags.motionHistoryValid = true;
	flags.ssrHistoryValid = true;
	await byId.get("ssr").execute(context);
	assert.equal(executeCalls[1].passId, "ssr");
	assert.equal(executeCalls[1].historyValid, true);
	assert.equal(executeCalls[1].frameBinding, frameBinding);
	assert.equal(historyUpdates.ssr, true);

	flags.volumetricHistoryValid = true;
	await byId.get("volumetric").execute(context);
	assert.equal(executeCalls[2].passId, "volumetric");
	assert.equal(executeCalls[2].historyValid, true);
	assert.equal(executeCalls[2].frameBinding, frameBinding);
	assert.equal(executeCalls[2].lightingState, lightingState);
	assert.equal(historyUpdates.volumetric, true);
}

async function testInteractionAndGammaWiring() {
	const { deps, executeCalls, presentCalls } = createDeps();
	const passes = createWebGPUBuiltInPostProcessPasses(deps);
	const byId = new Map(passes.map((pass) => [pass.id, pass]));
	const context = createPassContext();

	await byId.get("interaction-outline").execute(context);
	assert.equal(executeCalls.length, 0);

	const interactionState = { selectedEntityIds: [42] };
	context.frameContext.transient.set(
		INTERACTION_TRANSIENT_STATE_KEY,
		interactionState
	);
	await byId.get("interaction-outline").execute(context);
	assert.equal(executeCalls.length, 1);
	assert.equal(executeCalls[0].passId, "interaction-outline");
	assert.equal(executeCalls[0].state, interactionState);

	await byId.get("tonemap").execute(context);
	assert.equal(executeCalls.length, 2);
	assert.equal(executeCalls[1].passId, "tonemap");

	await byId.get("gamma").execute(context);
	assert.equal(presentCalls.length, 1);
	assert.equal(presentCalls[0].source, context.targets.sceneColor);
	assert.equal(presentCalls[0].applyGamma, true);
}

async function testCustomRuntimePassRegistry() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const calls = {
		execute: 0,
		warmup: 0,
		invalidate: 0,
		shaderRuntimeChanged: 0,
	};
	const runtimePass = {
		id: "custom-registry",
		warmupHints: ["postprocess:custom-registry"],
		warmup(hint, context) {
			calls.warmup++;
			assert.equal(hint, "postprocess:custom-registry");
			assert.equal(context.compute, backend);
			return true;
		},
		execute(request, context) {
			calls.execute++;
			assert.equal(request.passId, "custom-registry");
			assert.equal(context.compute, backend);
			return { ran: true };
		},
		invalidateBindings(context) {
			calls.invalidate++;
			assert.equal(context.compute, backend);
		},
		onShaderRuntimeChanged(context) {
			calls.shaderRuntimeChanged++;
			assert.equal(context.compute, backend);
		},
	};
	const graph = new WebGPUPostProcessGraph();
	graph.registerPass(createCustomPass("custom-registry", runtimePass));
	runtime.registerRuntimePass(runtimePass);

	const context = {
		...createPassContext(),
		executeRuntimePass: (request) => runtime.executePass(request),
	};
	const executed = await graph.execute(context, createPostProcess(), () => {});
	assert.deepEqual(executed, ["custom-registry"]);
	assert.equal(calls.execute, 1);

	const warmup = await runtime.warmupHints([
		"postprocess:custom-registry",
		"postprocess:custom-registry",
	]);
	assert.equal(warmup.compiled, 1);
	assert.equal(warmup.failed, 0);
	assert.equal(calls.warmup, 1);

	runtime.invalidateBindings();
	assert.equal(calls.invalidate, 1);
	runtime.onShaderRuntimeChanged();
	assert.equal(calls.shaderRuntimeChanged, 1);

	graph.unregisterPass("custom-registry");
	runtime.unregisterRuntimePass("custom-registry");
	assert.equal(graph.hasPass("custom-registry"), false);
	const result = await runtime.executePass({
		passId: "custom-registry",
		encoder: context.encoder,
		targets: context.targets,
		frameContext: context.frameContext,
	});
	assert.deepEqual(result, { ran: false });
}

function testReservedAndDuplicateRegistrationGuards() {
	const { deps } = createDeps();
	const graph = new WebGPUPostProcessGraph(
		createWebGPUBuiltInPostProcessPasses(deps)
	);
	assert.throws(
		() => graph.registerPass(createCustomPass("ssao")),
		/built-in WebGPU post-process pass/
	);
	assert.throws(
		() => graph.unregisterPass("gamma"),
		/Cannot unregister built-in/
	);
	graph.registerPass(createCustomPass("custom-graph"));
	assert.throws(
		() => graph.registerPass(createCustomPass("custom-graph")),
		/already registered/
	);

	const runtime = new WebGPUPostProcessRuntime(new FakeBackend(), () => {});
	const customRuntime = {
		id: "custom-runtime",
		execute() {
			return { ran: true };
		},
	};
	runtime.registerRuntimePass(customRuntime);
	assert.throws(
		() => runtime.registerRuntimePass(customRuntime),
		/already registered/
	);
	assert.throws(
		() =>
			runtime.registerRuntimePass({
				id: "tonemap",
				execute() {
					return { ran: true };
				},
			}),
		/built-in WebGPU post-process runtime pass/
	);

	const backend = new WebGPUBackend();
	backend.postProcess.registerPass(createCustomPass("custom-backend"));
	assert.throws(
		() => backend.postProcess.registerPass(createCustomPass("custom-backend")),
		/already registered/
	);
	assert.throws(
		() => backend.postProcess.registerPass(createCustomPass("gamma")),
		/built-in WebGPU post-process pass/
	);
	assert.throws(
		() => backend.postProcess.unregisterPass("gamma"),
		/Cannot unregister built-in/
	);
	backend.postProcess.unregisterPass("custom-backend");
	backend.postProcess.registerPass(createCustomPass("custom-backend"));
}

async function run() {
	testBuiltInPassGraphOrder();
	await testTemporalPassWiring();
	await testInteractionAndGammaWiring();
	await testCustomRuntimePassRegistry();
	testReservedAndDuplicateRegistrationGuards();
	console.log("WebGPU built-in post-process pass factory tests passed");
}

await run();
