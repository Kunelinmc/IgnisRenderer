import assert from "node:assert/strict";
import { WebGLPostProcessGraph } from "../src/renderers/webgl/WebGLPostProcessGraph.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableToneMapping: true,
		enableSH: true,
		enableShadows: true,
		enableReflection: false,
		enableSkybox: true,
		enableSSAO: true,
		enableSSGI: false,
		enableTAA: true,
		enableSSR: false,
		enableVolumetric: true,
		enableFog: true,
		enableMotionBlur: true,
		enableDOF: true,
		enableBloom: true,
		enableColorFilter: true,
		enableFXAA: true,
		enableClusteredLighting: true,
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
		createPass("volumetric", ["taa"], "enableVolumetric"),
		{
			id: "fog",
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
		{
			id: "tonemap",
			dependsOn: ["bloom"],
			isEnabled(features) {
				return features.enableToneMapping !== false;
			},
			execute() {},
		},
		createPass("color-filter", ["tonemap"], "enableColorFilter"),
		createPass("fxaa", ["color-filter"], "enableFXAA"),
		createPass("gamma", ["tonemap"], "enableGamma"),
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
			"volumetric",
			"fog",
			"motion-blur",
			"dof",
			"bloom",
			"tonemap",
			"color-filter",
			"fxaa",
			"gamma",
		]
	);
	assert.equal(warnings.length, 0);
}

function testFogSceneModeSkipsFogPass() {
	const graph = new WebGLPostProcessGraph([
		createPass("volumetric", [], "enableVolumetric"),
		{
			id: "fog",
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
	testFogSceneModeSkipsFogPass();
	testUnknownDependencySkipsPass();
	testCycleSkipsBranch();
	console.log("WebGL post-process graph tests passed");
}

run();
