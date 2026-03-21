import assert from "node:assert/strict";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { ShadowMap } from "../src/lights/ShadowMapping.ts";
import { updateShadowMapMetadata } from "../src/pipeline/ShadowMetadata.ts";

function createSceneBounds(radius) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function createShadowLight(capturedRadii) {
	return {
		worldMatrix: Matrix4.identity(),
		shadow: {
			setupShadowCamera(ctx) {
				capturedRadii.push(ctx.sceneBounds.radius);
				return {
					view: Matrix4.identity(),
					projection: Matrix4.identity(),
					lightDir: { x: 0, y: -1, z: 0 },
				};
			},
		},
	};
}

function testShadowRadiusShrinkUsesHysteresis() {
	const capturedRadii = [];
	const light = createShadowLight(capturedRadii);
	const shadowMap = new ShadowMap(1024);

	updateShadowMapMetadata(shadowMap, light, createSceneBounds(100));
	updateShadowMapMetadata(shadowMap, light, createSceneBounds(10));
	updateShadowMapMetadata(shadowMap, light, createSceneBounds(120));

	assert.equal(capturedRadii[0], 100);
	assert.ok(capturedRadii[1] > 10, "Shrink should be smoothed for stability");
	assert.ok(capturedRadii[1] < 100, "Smoothed radius should move toward target");
	assert.equal(capturedRadii[2], 120);
}

function run() {
	testShadowRadiusShrinkUsesHysteresis();
	console.log("Shadow metadata stabilization tests passed");
}

run();
