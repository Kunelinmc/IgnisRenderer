import assert from "node:assert/strict";
import { resolvePostProcessState } from "../src/pipeline/PostProcessController.ts";
import { WebGPUPostProcessGraph } from "../src/renderers/webgpu/WebGPUPostProcessGraph.ts";

const POST_PROCESS_CAPABILITIES = {
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

const ENABLED_POST_PROCESS_REQUEST = {
	ssao: { enabled: true },
	ssgi: { enabled: true },
	taa: { enabled: true },
	ssr: { enabled: true },
	volumetric: { enabled: true },
	fog: { enabled: true, options: { application: "postprocess" } },
	"motion-blur": { enabled: true },
	dof: { enabled: true },
	bloom: { enabled: true },
	tonemap: { enabled: true },
	"color-filter": { enabled: true },
	fxaa: { enabled: true },
	"interaction-outline": { enabled: true },
	gamma: { enabled: true },
};

function createPostProcess(overrides = {}) {
	return resolvePostProcessState(
		{
			...ENABLED_POST_PROCESS_REQUEST,
			...overrides,
		},
		POST_PROCESS_CAPABILITIES,
		"webgpu"
	);
}

function createPass(id, dependsOn, enabledId = id) {
	return {
		id,
		kind: "compute",
		dependsOn,
		isEnabled(postProcess) {
			return !!postProcess.enabled[enabledId];
		},
		execute() {},
	};
}

function testPostGraphOrder() {
	const graph = new WebGPUPostProcessGraph([
		createPass("ssao", []),
		createPass("ssgi", ["ssao"]),
		createPass("taa", ["ssgi", "ssao"]),
		createPass("ssr", ["taa"]),
		createPass("volumetric", ["ssr"]),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(postProcess) {
				return (
					postProcess.enabled.fog &&
					(postProcess.options.fog.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"]),
		createPass("dof", ["motion-blur"]),
		createPass("bloom", ["dof"]),
		createPass("color-filter", ["bloom"]),
		createPass("fxaa", ["color-filter"]),
		createPass("gamma", ["fxaa"]),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
			"color-filter",
			"fxaa",
			"gamma",
		]
	);
	assert.equal(warnings.length, 0);
}

function testEnabledSubsetShrinksDependencyChain() {
	const graph = new WebGPUPostProcessGraph([
		createPass("ssao", []),
		createPass("ssgi", ["ssao"]),
		createPass("taa", ["ssgi", "ssao"]),
		createPass("ssr", ["taa"]),
		createPass("volumetric", ["ssr"]),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(postProcess) {
				return (
					postProcess.enabled.fog &&
					(postProcess.options.fog.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"]),
		createPass("dof", ["motion-blur"]),
		createPass("bloom", ["dof"]),
		createPass("color-filter", ["bloom"]),
		createPass("fxaa", ["color-filter"]),
		createPass("gamma", ["fxaa"]),
	]);

	const order = graph.getExecutionOrder(
		createPostProcess({
			ssao: { enabled: false },
			ssgi: { enabled: false },
			taa: { enabled: false },
			ssr: { enabled: false },
			volumetric: { enabled: false },
			fog: { enabled: false },
			"motion-blur": { enabled: false },
			dof: { enabled: false },
			bloom: { enabled: false },
			"color-filter": { enabled: false },
			fxaa: { enabled: false },
			gamma: { enabled: true },
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
		createPass("gamma", ["missing-pass"]),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
		createPass("a", ["b"], "gamma"),
		createPass("b", ["a"], "gamma"),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
		createPass("volumetric", []),
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			isEnabled(postProcess) {
				return (
					postProcess.enabled.fog &&
					(postProcess.options.fog.application ?? "postprocess") !== "scene"
				);
			},
			execute() {},
		},
		createPass("motion-blur", ["fog"]),
	]);
	const order = graph.getExecutionOrder(
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
