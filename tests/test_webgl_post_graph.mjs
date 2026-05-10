import assert from "node:assert/strict";
import { resolvePostProcessState } from "../src/pipeline/PostProcess.ts";
import { WebGLPostProcessGraph } from "../src/renderers/webgl/WebGLPostProcessGraph.ts";

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
		"webgl"
	);
}

function createPass(id, dependsOn, enabledId = id) {
	return {
		id,
		dependsOn,
		isEnabled(postProcess) {
			return enabledId ? !!postProcess.enabled[enabledId] : true;
		},
		execute() {},
	};
}

function testExecutionOrder() {
	const graph = new WebGLPostProcessGraph([
		createPass("ssao", []),
		createPass("taa", ["ssao"]),
		createPass("volumetric", ["taa"]),
		{
			id: "fog",
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
		createPass("tonemap", ["bloom"]),
		createPass("color-filter", ["tonemap"]),
		createPass("fxaa", ["color-filter"]),
		createPass("gamma", ["tonemap"]),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
		createPass("volumetric", []),
		{
			id: "fog",
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

function testUnknownDependencySkipsPass() {
	const graph = new WebGLPostProcessGraph([
		createPass("gamma", ["missing"]),
	]);
	const warnings = [];
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
	const order = graph.getExecutionOrder(createPostProcess(), (key, message) => {
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
