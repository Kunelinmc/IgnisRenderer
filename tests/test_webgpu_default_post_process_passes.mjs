import assert from "node:assert/strict";
import {
	INTERACTION_TRANSIENT_STATE_KEY,
} from "../src/pipeline/types.ts";
import { createWebGPUDefaultPostProcessPasses } from "../src/renderers/webgpu/WebGPUDefaultPostProcessPasses.ts";
import { WebGPUPostProcessGraph } from "../src/renderers/webgpu/WebGPUPostProcessGraph.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableToneMapping: true,
		enableSH: false,
		enableShadows: false,
		enableReflection: false,
		enableEnvironment: false,
		enableSSAO: true,
		enableSSGI: true,
		enableTAA: true,
		enableSSR: true,
		enableVolumetric: true,
		enableFog: true,
		enableMotionBlur: true,
		enableDOF: true,
		enableBloom: true,
		enableColorFilter: true,
		enableFXAA: true,
		enableClusteredLighting: false,
		warnings: [],
		ssrOptions: {},
		ssaoOptions: {},
		ssgiOptions: {},
		taaOptions: {},
		volumetricOptions: {},
		fogOptions: {
			application: "postprocess",
		},
		bloomOptions: {},
		motionBlurOptions: {},
		dofOptions: {},
		colorFilterOptions: {},
		...overrides,
	};
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

function testDefaultPassGraphOrder() {
	const { deps } = createDeps();
	const passes = createWebGPUDefaultPostProcessPasses(deps);
	const graph = new WebGPUPostProcessGraph(passes);
	const order = graph.getExecutionOrder(createFeatures(), () => {});

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
		createFeatures({
			fogOptions: {
				application: "scene",
			},
		}),
		() => {}
	);
	assert.equal(sceneFogOrder.some((pass) => pass.id === "fog"), false);

	const fxaaOnlyOrder = graph.getExecutionOrder(
		createFeatures({
			enableGamma: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
			enableColorFilter: true,
			enableFXAA: true,
		}),
		() => {}
	);
	assert.deepEqual(
		fxaaOnlyOrder.map((pass) => pass.id),
		["tonemap", "color-filter", "fxaa", "interaction-outline"]
	);

	const tonemapDisabledOrder = graph.getExecutionOrder(
		createFeatures({
			enableToneMapping: false,
			enableFXAA: true,
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
	const passes = createWebGPUDefaultPostProcessPasses(deps);
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
	const passes = createWebGPUDefaultPostProcessPasses(deps);
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

async function run() {
	testDefaultPassGraphOrder();
	await testTemporalPassWiring();
	await testInteractionAndGammaWiring();
	console.log("WebGPU default post-process pass factory tests passed");
}

await run();
