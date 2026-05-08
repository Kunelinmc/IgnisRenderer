import assert from "node:assert/strict";
import {
	computePostProcessInflationRadius,
	resolvePostProcessGrade,
	scaleFullFrameFallbackAreaRatioForPostProcess,
} from "../src/pipeline/incremental.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: false,
		enableToneMapping: false,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableEnvironment: false,
		enableSSAO: false,
		enableSSGI: false,
		enableTAA: false,
		enableSSR: false,
		enableVolumetric: false,
		enableMotionBlur: false,
		enableDOF: false,
		enableBloom: false,
		enableColorFilter: false,
		enableFXAA: false,
		enableClusteredLighting: false,
		warnings: [],
		...overrides,
	};
}

function testResolvePostProcessGrade() {
	assert.equal(resolvePostProcessGrade(createFeatures()), "none");
	assert.equal(
		resolvePostProcessGrade(createFeatures({ enableFXAA: true })),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createFeatures({ enableToneMapping: true })),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createFeatures({ enableColorFilter: true })),
		"light"
	);
	assert.equal(
		resolvePostProcessGrade(createFeatures({ enableSSGI: true })),
		"standard"
	);
	assert.equal(
		resolvePostProcessGrade(createFeatures({ enableDOF: true })),
		"cinematic"
	);
}

function testComputePostProcessInflationRadius() {
	assert.equal(computePostProcessInflationRadius(createFeatures()), 0);
	assert.equal(
		computePostProcessInflationRadius(createFeatures({ enableGamma: true })),
		2
	);
	assert.equal(
		computePostProcessInflationRadius(
			createFeatures({ enableToneMapping: true })
		),
		2
	);
	assert.equal(
		computePostProcessInflationRadius(createFeatures({ enableSSGI: true })),
		12
	);
	assert.equal(
		computePostProcessInflationRadius(
			createFeatures({
				enableTAA: true,
				enableDOF: true,
			})
		),
		32
	);
}

function testScaleFullFrameFallbackAreaRatioForPostProcess() {
	const baseRatio = 0.3;
	const noneRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createFeatures()
	);
	const standardRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createFeatures({ enableBloom: true })
	);
	const cinematicRatio = scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		createFeatures({ enableTAA: true })
	);
	assert.equal(noneRatio, 0.3);
	assert.equal(standardRatio, 0.27);
	assert.equal(cinematicRatio, 0.24);
}

function run() {
	testResolvePostProcessGrade();
	testComputePostProcessInflationRadius();
	testScaleFullFrameFallbackAreaRatioForPostProcess();
	console.log("Incremental postfx grading tests passed");
}

run();
