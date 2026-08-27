import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	linearizeShadowNdcDepth,
	resolveShadowDepthProjectionParams,
	SHADOW_DISK_SAMPLES,
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

function testBuiltInKindsAndPagedSnapshot() {
	const scene = new Scene();
	assert.equal(scene.shadows.createSingle().kind, "single");
	assert.equal(scene.shadows.createCascaded().kind, "cascaded");
	const paged = scene.shadows.createPaged({ pageSize: 128, virtualResolution: 4096 });
	const settings = paged.snapshot().pagedSettings;
	assert.equal(paged.kind, "paged-shadow");
	assert.equal(settings?.pageSize, 128);
	assert.equal(settings?.virtualResolution, 4096);
	assert.equal("pageTableBase" in settings, false);
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
	assert.equal(SHADOW_DISK_SAMPLES.length, 12);
	for (const [x, y] of SHADOW_DISK_SAMPLES) {
		assert.ok(Number.isFinite(x) && Number.isFinite(y));
		assert.ok(Math.hypot(x, y) <= 1);
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
testBuiltInKindsAndPagedSnapshot();
testSamplingAuthoringAndProjectionDepth();
console.log("shadow manager tests passed");
