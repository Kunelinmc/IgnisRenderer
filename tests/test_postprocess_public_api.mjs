import assert from "node:assert/strict";
import {
	DEFAULT_POST_PROCESS_CAPABILITIES,
	PostProcessController,
	resolvePostProcessState,
} from "../src/pipeline/PostProcess.ts";
import { ALL_POST_PROCESS_CAPABILITIES } from "./helpers/postprocess.mjs";

function testControllerDefaultsAndOptionsMerge() {
	const controller = new PostProcessController();
	let state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);

	assert.equal(state.enabled.tonemap, true);
	assert.equal(state.enabled.gamma, true);
	assert.equal(state.enabled["interaction-outline"], true);
	assert.equal(state.enabled.ssao, false);

	controller.setOptions("ssao", {
		radius: 4,
		intensity: 0.5,
	});
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, false);
	assert.equal(state.options.ssao.radius, 4);
	assert.equal(state.options.ssao.intensity, 0.5);

	controller.enable("ssao", {
		samples: 12,
		radius: 6,
	});
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, true);
	assert.equal(state.options.ssao.samples, 12);
	assert.equal(state.options.ssao.radius, 6);
	assert.equal(state.options.ssao.intensity, 0.5);

	controller.disable("tonemap");
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.tonemap, false);

	controller.reset("ssao");
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.ssao, false);
	assert.equal(state.options.ssao.radius, 8);
	assert.equal(state.options.ssao.intensity, 1);

	controller.reset();
	state = resolvePostProcessState(
		controller.getState(),
		ALL_POST_PROCESS_CAPABILITIES,
		"test"
	);
	assert.equal(state.enabled.tonemap, true);
	assert.equal(state.enabled.gamma, true);
	assert.equal(state.enabled["interaction-outline"], true);
	assert.equal(state.enabled.ssao, false);
}

function testUnsupportedExplicitEnableWarning() {
	const defaultState = resolvePostProcessState(
		{},
		DEFAULT_POST_PROCESS_CAPABILITIES,
		"software"
	);
	assert.equal(defaultState.enabled.tonemap, false);
	assert.equal(defaultState.enabled.gamma, false);
	assert.equal(defaultState.warnings.length, 0);

	const controller = new PostProcessController();
	controller.enable("ssr");

	const state = resolvePostProcessState(
		controller.getState(),
		{
			...ALL_POST_PROCESS_CAPABILITIES,
			ssr: false,
		},
		"software"
	);

	assert.equal(state.enabled.ssr, false);
	assert.ok(
		state.warnings.some(
			(warning) => warning.key === "software-postprocess-unsupported-ssr"
		)
	);
}

function run() {
	testControllerDefaultsAndOptionsMerge();
	testUnsupportedExplicitEnableWarning();
	console.log("Postprocess public API tests passed");
}

run();
