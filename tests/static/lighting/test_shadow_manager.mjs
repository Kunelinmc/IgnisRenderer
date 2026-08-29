import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	linearizeShadowNdcDepth,
	resolveShadowDepthProjectionParams,
	resolveShadowFilterDiskSample,
	resolveShadowSearchDiskSample,
	SHADOW_SAMPLING_PRESETS,
} from "../../../src/lights/shadows/shadowSampling.ts";

function testBuiltInDefinitionLifecycle() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	const spot = scene.add(new SpotLight({ intensity: 1, range: 120 }));
	const single = scene.shadows.createSingle({ size: 1024 });
	const cascaded = scene.shadows.createCascaded({ size: 2048 });

	scene.shadows.bind(sun, single);
	assert.equal(scene.shadows.getBoundShadowMap(sun), single);
	scene.shadows.rebind(sun, cascaded);
	scene.shadows.bind(spot, cascaded);
	assert.equal(scene.shadows.getBoundShadowMap(sun), cascaded);
	assert.equal(scene.shadows.getBoundShadowMap(spot), cascaded);

	const revision = cascaded.revision;
	cascaded.update({ size: 1024 });
	assert.ok(cascaded.revision > revision);
	assert.equal(cascaded.snapshot().resolution, 1024);

	scene.shadows.unbindLight(sun);
	assert.equal(scene.shadows.getBoundShadowMap(sun), undefined);
	scene.shadows.destroy(cascaded);
	assert.equal(scene.shadows.getBoundShadowMap(spot), undefined);
}

function testBuiltInKinds() {
	const scene = new Scene();
	assert.equal(scene.shadows.createSingle().kind, "single");
	assert.equal(scene.shadows.createCascaded().kind, "cascaded");
}

function testSamplingAuthoringAndProjectionDepth() {
	const scene = new Scene();
	const definition = scene.shadows.createSingle({
		filterMode: "pcss",
		sampling: { quality: "high" },
		strength: 2,
	});
	assert.equal(definition.filterMode, "pcss");
	assert.equal(definition.sampling.quality, "high");
	assert.equal(definition.strength, 1);
	const revision = definition.revision;
	definition.update({
		filterMode: "pcf",
		sampling: { quality: "low" },
		strength: -1,
	});
	assert.equal(definition.revision, revision + 1);
	assert.deepEqual(definition.snapshot().sampling, { quality: "low" });
	assert.equal(definition.snapshot().filterMode, "pcf");
	assert.equal(definition.snapshot().strength, 0);
	assert.equal("samples" in definition.snapshot().sampling, false);
	assert.deepEqual(SHADOW_SAMPLING_PRESETS.medium, {
		pcfSamples: 3,
		pcssSearchSamples: 8,
		pcssFilterSamples: 5,
	});
	for (const sampleCount of [1, 3, 5, 7]) {
		const offsets = Array.from({ length: sampleCount }, (_, index) =>
			resolveShadowFilterDiskSample(index, sampleCount, 1, 0));
		const centroid = offsets.reduce(
			(sum, [x, y]) => [sum[0] + x / sampleCount, sum[1] + y / sampleCount],
			[0, 0],
		);
		assert.ok(Math.hypot(...centroid) < 1e-8);
		for (const [x, y] of offsets) assert.ok(Math.hypot(x, y) <= 1);
	}
	for (const sampleCount of [4, 8, 12]) {
		const offsets = Array.from({ length: sampleCount }, (_, index) =>
			resolveShadowSearchDiskSample(index, 1, 0));
		const centroid = offsets.reduce(
			(sum, [x, y]) => [sum[0] + x / sampleCount, sum[1] + y / sampleCount],
			[0, 0],
		);
		assert.ok(Math.hypot(...centroid) < 1e-8);
		for (const [x, y] of offsets) assert.ok(Math.hypot(x, y) <= 1);
	}

	const projection = Matrix4.perspective(60, 1, 0.1, 100);
	const params = resolveShadowDepthProjectionParams(projection);
	for (const distance of [0.1, 1, 25, 100]) {
		const clip = Matrix4.transformPoint(projection, { x: 0, y: 0, z: -distance });
		const reconstructed = linearizeShadowNdcDepth(clip.z / clip.w, params);
		assert.ok(Math.abs(reconstructed - distance) < 1e-3);
	}
}

testBuiltInDefinitionLifecycle();
testBuiltInKinds();
testSamplingAuthoringAndProjectionDepth();
console.log("shadow manager tests passed");
