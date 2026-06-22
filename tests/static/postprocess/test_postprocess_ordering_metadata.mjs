import assert from "node:assert/strict";
import * as ordering from "../../../src/postprocess/ordering.ts";
import { isPostProcessPassStage } from "../../../src/postprocess/PostProcessGraphCompiler.ts";
import {
	TONE_MAPPING_PASS_ORDER,
} from "../../../src/postprocess/passes/ToneMappingPass.ts";
import { GAMMA_PASS_ORDER } from "../../../src/postprocess/passes/GammaPass.ts";

const {
	getBuiltinPostProcessOrder,
} = ordering;

function testBuiltinOrderIsDefinedOnPassModules() {
	assert.deepEqual(TONE_MAPPING_PASS_ORDER, {
		id: "tonemap",
		placement: "hdr",
		order: 600,
		incremental: {
			firstPass: "tonemap",
			grade: "light",
			inflationRadius: 0,
		},
	});
	assert.deepEqual(GAMMA_PASS_ORDER, {
		id: "gamma",
		placement: "present",
		order: 900,
		incremental: {
			firstPass: "gamma",
			grade: "light",
			inflationRadius: 0,
		},
	});
	assert.deepEqual(getBuiltinPostProcessOrder("tonemap"), {
		id: "tonemap",
		placement: "hdr",
		order: 600,
	});
	assert.equal(getBuiltinPostProcessOrder("ssao"), null);
	assert.equal(getBuiltinPostProcessOrder("custom"), null);
	assert.equal("BUILTIN_POST_PROCESS_ORDER" in ordering, false);
	assert.equal("defineBuiltinPostProcessOrder" in ordering, false);
}

function testPipelineStageHelperWorksWithoutPassBarrelImport() {
	assert.equal(isPostProcessPassStage("postprocess"), true);
	assert.equal(isPostProcessPassStage("ssao"), false);
	assert.equal(isPostProcessPassStage("tonemap"), true);
	assert.equal(isPostProcessPassStage("gamma"), true);
	assert.equal(isPostProcessPassStage("custom"), false);
}

function run() {
	testBuiltinOrderIsDefinedOnPassModules();
	testPipelineStageHelperWorksWithoutPassBarrelImport();
	console.log("Postprocess ordering metadata tests passed");
}

run();
