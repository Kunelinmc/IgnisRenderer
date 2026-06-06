import assert from "node:assert/strict";
import {
	BUILTIN_POST_PROCESS_ORDER,
	getBuiltinPostProcessOrder,
} from "../../../src/postprocess/ordering.ts";
import { isPostProcessPassStage } from "../../../src/postprocess/PostProcessPipeline.ts";

function testBuiltinOrderIsAvailableFromOrderingModule() {
	assert.deepEqual(
		BUILTIN_POST_PROCESS_ORDER.map((entry) => entry.id),
		[
			"ssao",
			"ssgi",
			"taa",
			"ssr",
			"ssrefraction",
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
	assert.deepEqual(getBuiltinPostProcessOrder("ssao"), {
		id: "ssao",
		placement: "spatial",
		order: 100,
	});
	assert.equal(getBuiltinPostProcessOrder("custom"), null);
}

function testPipelineStageHelperWorksWithoutPassBarrelImport() {
	assert.equal(isPostProcessPassStage("postprocess"), true);
	assert.equal(isPostProcessPassStage("ssao"), true);
	assert.equal(isPostProcessPassStage("gamma"), true);
	assert.equal(isPostProcessPassStage("custom"), false);
}

function run() {
	testBuiltinOrderIsAvailableFromOrderingModule();
	testPipelineStageHelperWorksWithoutPassBarrelImport();
	console.log("Postprocess ordering metadata tests passed");
}

run();
