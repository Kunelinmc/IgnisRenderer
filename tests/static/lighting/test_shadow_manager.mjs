import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";

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
	assert.equal(scene.shadows.createVariance().kind, "variance");
	assert.equal(scene.shadows.createCascaded().kind, "cascaded");
	const paged = scene.shadows.createPaged({ pageSize: 128, virtualResolution: 4096 });
	const settings = paged.snapshot().pagedSettings;
	assert.equal(paged.kind, "paged-shadow");
	assert.equal(settings?.pageSize, 128);
	assert.equal(settings?.virtualResolution, 4096);
	assert.equal("pageTableBase" in settings, false);
}

testBuiltInDefinitionLifecycle();
testBuiltInKindsAndPagedSnapshot();
console.log("shadow manager tests passed");
