import assert from "node:assert/strict";
import { WebGLPostProcessGraph } from "../src/renderers/webgl/WebGLPostProcessGraph.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableSH: true,
		enableShadows: true,
		enableReflection: false,
		enableSkybox: true,
		enableSSAO: true,
		enableSSGI: false,
		enableTAA: true,
		enableSSR: false,
		enableVolumetric: false,
		enableMotionBlur: true,
		enableDOF: true,
		enableBloom: true,
		enableFXAA: true,
		enableClusteredLighting: true,
		warnings: [],
		ssrOptions: {},
		ssaoOptions: {},
		ssgiOptions: {},
		taaOptions: {},
		volumetricOptions: {},
		bloomOptions: {},
		motionBlurOptions: {},
		dofOptions: {},
		clusteredLightingOptions: {},
		...overrides,
	};
}

function createPass(id, dependsOn, key) {
	return {
		id,
		dependsOn,
		isEnabled(features) {
			return key ? !!features[key] : true;
		},
		execute() {},
	};
}

function testExecutionOrder() {
	const graph = new WebGLPostProcessGraph([
		createPass("ssao", [], "enableSSAO"),
		createPass("taa", ["ssao"], "enableTAA"),
		createPass("motion-blur", ["taa"], "enableMotionBlur"),
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
		["ssao", "taa", "motion-blur", "dof", "bloom", "fxaa", "gamma"]
	);
	assert.equal(warnings.length, 0);
}

function testUnknownDependencySkipsPass() {
	const graph = new WebGLPostProcessGraph([
		createPass("gamma", ["missing"], "enableGamma"),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createFeatures(), (key, message) => {
		warnings.push({ key, message });
	});
	assert.deepEqual(order.map((pass) => pass.id), []);
	assert.equal(warnings.length, 1);
	assert.ok(warnings[0].message.includes("unknown pass"));
}

function testCycleSkipsBranch() {
	const graph = new WebGLPostProcessGraph([
		createPass("a", ["b"], null),
		createPass("b", ["a"], null),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createFeatures(), (key, message) => {
		warnings.push({ key, message });
	});
	assert.deepEqual(order.map((pass) => pass.id), []);
	assert.ok(warnings.length >= 1);
}

function run() {
	testExecutionOrder();
	testUnknownDependencySkipsPass();
	testCycleSkipsBranch();
	console.log("WebGL post-process graph tests passed");
}

run();
