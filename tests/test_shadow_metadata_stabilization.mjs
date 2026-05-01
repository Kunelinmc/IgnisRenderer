import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { createShadowRenderSet } from "../src/lights/shadows/ShadowMapping.ts";
import { updateShadowMapMetadata } from "../src/pipeline/ShadowMetadata.ts";

function createSceneBounds(radius) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function testShadowRadiusShrinkUsesHysteresis() {
	const scene = new Scene();
	const light = new DirectionalLight();
	scene.add(light);
	scene.shadows.bind(light, scene.shadows.createSingle({ size: 1024 }));
	const shadowConfig = scene.shadows.getLegacyShadowConfig(light);
	assert.ok(shadowConfig);
	const renderSet = createShadowRenderSet(shadowConfig);

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
