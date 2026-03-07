import assert from "node:assert/strict";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { ShadowMap } from "../src/utils/ShadowMapping.ts";
import { sampleSoftwareShadow } from "../src/renderers/software/shadows/sampling.ts";

function createShadowFixture(overrides = {}, size = 4) {
	const shadowMap = new ShadowMap(size, {
		shadowBias: 0,
		shadowSlopeBias: 0,
		shadowTexelBias: 0,
		shadowMaxBias: 1,
		shadowNormalBias: 1,
		shadowNormalBiasMin: 0,
		shadowPCF: 0,
		shadowStrength: 1,
		shadowSamples: 1,
		shadowSearchSamples: 1,
		...overrides,
	});

	shadowMap.viewProjectionMatrix = Matrix4.identity();
	shadowMap.projectionMatrix = Matrix4.identity();
	shadowMap.latestLightDir = { x: 0, y: -1, z: 0 };

	const runtime = {
		size,
		depthBuffer: new Float32Array(size * size),
		transmissionBuffer: new Float32Array(size * size * 3),
	};
	runtime.depthBuffer.fill(1);
	runtime.transmissionBuffer.fill(1);

	return { shadowMap, runtime };
}

function testBasicShadowOcclusion() {
	const { shadowMap, runtime } = createShadowFixture();
	const centerIndex = 1 * runtime.size + 1;
	runtime.depthBuffer[centerIndex] = -1;

	const visibility = sampleSoftwareShadow(
		shadowMap,
		runtime,
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 }
	);

	assert.ok(
		visibility.r < 0.01 && visibility.g < 0.01 && visibility.b < 0.01,
		"Occluded point should be fully shadowed when strength=1"
	);
}

function testTransmissionTintOnLitSamples() {
	const { shadowMap, runtime } = createShadowFixture();
	const centerIndex = 1 * runtime.size + 1;
	const transmissionIndex = centerIndex * 3;
	runtime.transmissionBuffer[transmissionIndex] = 0.2;
	runtime.transmissionBuffer[transmissionIndex + 1] = 0.4;
	runtime.transmissionBuffer[transmissionIndex + 2] = 0.6;

	const visibility = sampleSoftwareShadow(
		shadowMap,
		runtime,
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 }
	);

	assert.ok(Math.abs(visibility.r - 0.2) < 1e-6);
	assert.ok(Math.abs(visibility.g - 0.4) < 1e-6);
	assert.ok(Math.abs(visibility.b - 0.6) < 1e-6);
}

function testNormalBiasAffectsSamplingPosition() {
	const { shadowMap, runtime } = createShadowFixture({
		shadowNormalBias: 1,
		shadowNormalBiasMin: 1,
	});

	const occludedIndex = 1 * runtime.size + 3;
	runtime.depthBuffer[occludedIndex] = -1;

	const withNormal = sampleSoftwareShadow(
		shadowMap,
		runtime,
		{ x: 0, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 }
	);
	const withoutNormal = sampleSoftwareShadow(
		shadowMap,
		runtime,
		{ x: 0, y: 0, z: 0 },
		null
	);

	assert.ok(
		withNormal.r < 0.01 && withoutNormal.r > 0.99,
		"Normal bias path should sample different texels than volume-bias path"
	);
}

function testPCSSPathHandlesBlockers() {
	const { shadowMap, runtime } = createShadowFixture({
		shadowRadius: 2,
		shadowSamples: 1,
		shadowSearchSamples: 1,
	});
	shadowMap.projectionMatrix = Matrix4.perspective(60, 1, 0.1, 10);
	const blockerIndex = 1 * runtime.size + 2;
	runtime.depthBuffer[blockerIndex] = 0;

	const visibility = sampleSoftwareShadow(
		shadowMap,
		runtime,
		{ x: 0, y: 0, z: 0.5 },
		null
	);

	assert.ok(
		visibility.r < 0.01 && visibility.g < 0.01 && visibility.b < 0.01,
		"PCSS branch should detect blockers and return shadowed visibility"
	);
}

function run() {
	testBasicShadowOcclusion();
	testTransmissionTintOnLitSamples();
	testNormalBiasAffectsSamplingPosition();
	testPCSSPathHandlesBlockers();
	console.log("Software shadow sampling tests passed");
}

run();
