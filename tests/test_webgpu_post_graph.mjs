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
		enableTAA: true,
		enableSSR: true,
		enableVolumetric: true,
		enableMotionBlur: true,
		enableDOF: true,
		enableBloom: true,
		enableFXAA: true,
		warnings: [],
		ssrOptions: {},
		ssaoOptions: {},
		taaOptions: {},
		volumetricOptions: {},
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
		createPass("taa", ["ssao"], "enableTAA"),
		createPass("ssr", ["taa"], "enableSSR"),
		createPass("volumetric", ["ssr"], "enableVolumetric"),
		createPass("motion-blur", ["volumetric"], "enableMotionBlur"),
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
			"taa",
			"ssr",
			"volumetric",
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
		createPass("taa", ["ssao"], "enableTAA"),
		createPass("ssr", ["taa"], "enableSSR"),
		createPass("volumetric", ["ssr"], "enableVolumetric"),
		createPass("motion-blur", ["volumetric"], "enableMotionBlur"),
		createPass("dof", ["motion-blur"], "enableDOF"),
		createPass("bloom", ["dof"], "enableBloom"),
		createPass("fxaa", ["bloom"], "enableFXAA"),
		createPass("gamma", ["fxaa"], "enableGamma"),
	]);

	const order = graph.getExecutionOrder(
		createFeatures({
			enableSSAO: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
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

function run() {
	testPostGraphOrder();
	testEnabledSubsetShrinksDependencyChain();
	testUnknownDependencySkipsPass();
	testCycleSkipsPassBranch();
	console.log("WebGPU post graph tests passed");
}

run();
