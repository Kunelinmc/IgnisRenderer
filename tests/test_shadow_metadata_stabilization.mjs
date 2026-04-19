import assert from "node:assert/strict";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { createShadowRenderSet } from "../src/lights/ShadowMapping.ts";
import { updateShadowMapMetadata } from "../src/pipeline/ShadowMetadata.ts";

function createSceneBounds(radius) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function testShadowRadiusShrinkUsesHysteresis() {
	const light = new DirectionalLight();
	light.castShadow = true;
	light.shadow = {
		strategy: "single-map",
		size: 1024,
	};
	const renderSet = createShadowRenderSet(light.shadow);

	updateShadowMapMetadata(renderSet, light, createSceneBounds(100));
	const radiusAfterFirst = renderSet.slices[0].shadowMap.stabilizedBoundsRadius;
	updateShadowMapMetadata(renderSet, light, createSceneBounds(10));
	const radiusAfterShrink = renderSet.slices[0].shadowMap.stabilizedBoundsRadius;
	updateShadowMapMetadata(renderSet, light, createSceneBounds(120));
	const radiusAfterGrow = renderSet.slices[0].shadowMap.stabilizedBoundsRadius;

	assert.equal(radiusAfterFirst, 100);
	assert.ok(radiusAfterShrink > 10, "Shrink should be smoothed for stability");
	assert.ok(radiusAfterShrink < 100, "Smoothed radius should move toward target");
	assert.equal(radiusAfterGrow, 120);
}

function run() {
	testShadowRadiusShrinkUsesHysteresis();
	console.log("Shadow metadata stabilization tests passed");
}

run();
