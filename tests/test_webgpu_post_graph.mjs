import assert from "node:assert/strict";
import { WebGPUPostProcessGraph } from "../src/renderers/webgpu/WebGPUPostProcessGraph.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableSH: false,
		enableShadows: false,
		enableReflection: false,
		enableSkybox: false,
		enableSSAO: true,
		enableSSGI: true,
		enableTAA: true,
		enableSSR: true,
		enableVolumetric: true,
		enableFog: true,
		enableMotionBlur: true,
		enableDOF: true,
		enableBloom: true,
		enableFXAA: true,
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
		...overrides,
	};
}

function createPass(id, dependsOn, key) {
	return {
		id,
		kind: "compute",
		dependsOn,
		isEnabled(features) {
			return !!features[key];
		},
		execute() {},
	};
}

function testPostGraphOrder() {
	const graph = new WebGPUPostProcessGraph([
		createPass("ssao", [], "enableSSAO"),
		createPass("ssgi", ["ssao"], "enableSSGI"),
		createPass("taa", ["ssgi", "ssao"], "enableTAA"),
		createPass("ssr", ["taa"], "enableSSR"),
		createPass("volumetric", ["ssr"], "enableVolumetric"),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(features) {
				return (
					features.enableFog &&
					(features.fogOptions?.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"], "enableMotionBlur"),
		createPass("dof", ["motion-blur"], "enableDOF"),
		createPass("bloom", ["dof"], "enableBloom"),
		createPass("fxaa", ["bloom"], "enableFXAA"),
		createPass("gamma", ["fxaa"], "enableGamma"),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createFeatures(), (key, message) => {
		warnings.push({ key, message });
	});

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
			"fxaa",
			"gamma",
		]
	);
	assert.equal(warnings.length, 0);
}

function testEnabledSubsetShrinksDependencyChain() {
	const graph = new WebGPUPostProcessGraph([
		createPass("ssao", [], "enableSSAO"),
		createPass("ssgi", ["ssao"], "enableSSGI"),
		createPass("taa", ["ssgi", "ssao"], "enableTAA"),
		createPass("ssr", ["taa"], "enableSSR"),
		createPass("volumetric", ["ssr"], "enableVolumetric"),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(features) {
				return (
					features.enableFog &&
					(features.fogOptions?.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"], "enableMotionBlur"),
		createPass("dof", ["motion-blur"], "enableDOF"),
		createPass("bloom", ["dof"], "enableBloom"),
		createPass("fxaa", ["bloom"], "enableFXAA"),
		createPass("gamma", ["fxaa"], "enableGamma"),
	]);

	const order = graph.getExecutionOrder(
		createFeatures({
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
			enableFXAA: false,
			enableGamma: true,
		}),
		() => {}
	);

	assert.deepEqual(
		order.map((pass) => pass.id),
		["gamma"]
	);
}

function testUnknownDependencySkipsPass() {
	const graph = new WebGPUPostProcessGraph([
		createPass("gamma", ["missing-pass"], "enableGamma"),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createFeatures(), (key, message) => {
		warnings.push({ key, message });
	});

	assert.deepEqual(
		order.map((pass) => pass.id),
		[]
	);
	assert.equal(warnings.length, 1);
	assert.ok(warnings[0].message.includes("unknown pass"));
}

function testCycleSkipsPassBranch() {
	const graph = new WebGPUPostProcessGraph([
		createPass("a", ["b"], "enableGamma"),
		createPass("b", ["a"], "enableGamma"),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createFeatures(), (key, message) => {
		warnings.push({ key, message });
	});

	assert.deepEqual(
		order.map((pass) => pass.id),
		[]
	);
	assert.ok(warnings.length >= 1);
}

function testFogSceneModeSkipsFogPass() {
	const graph = new WebGPUPostProcessGraph([
		createPass("volumetric", [], "enableVolumetric"),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(features) {
				return (
					features.enableFog &&
					(features.fogOptions?.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"], "enableMotionBlur"),
	]);
	const order = graph.getExecutionOrder(
		createFeatures({
			enableFog: true,
			fogOptions: {
				application: "scene",
			},
		}),
		() => {}
	);
	assert.deepEqual(order.map((pass) => pass.id), ["volumetric", "motion-blur"]);
}

function run() {
	testPostGraphOrder();
	testEnabledSubsetShrinksDependencyChain();
	testUnknownDependencySkipsPass();
	testCycleSkipsPassBranch();
	testFogSceneModeSkipsFogPass();
	console.log("WebGPU post graph tests passed");
}

run();
