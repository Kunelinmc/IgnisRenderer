import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ShadowMapBase } from "../../../src/lights/shadows/ShadowMapBase.ts";
import { ShadowMapRegistry } from "../../../src/lights/shadows/ShadowMapRegistry.ts";
import { updateShadowMapMetadata } from "../../../src/pipeline/ShadowMetadata.ts";

function createSceneBounds(radius = 80) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function createBudgetCapabilities(maxDynamicShadowCost) {
	return {
		backendKey: "test",
		supportsFilterModes: ["pcf", "vsm"],
		supportsDirectionalCSM: true,
		supportsSpotCSM: true,
		supportsPointCSM: true,
		maxDynamicShadowCost,
	};
}

class ExternalCascadeShadowMap extends ShadowMapBase {
	kind = "external-cascade";

	resolveCascadeCount() {
		return 3;
	}

	toLegacyShadowConfig(_lightType, overrides = {}) {
		return this.createCSMLegacyConfig(overrides.cascadeCount ?? 3, {
			size: overrides.size,
			lambda: 0.5,
			blendRatio: 0.2,
			stabilize: true,
		});
	}
}

class ExternalSingleShadowMap extends ShadowMapBase {
	kind = "external-single";

	toLegacyShadowConfig(_lightType, overrides = {}) {
		return this.createSingleMapLegacyConfig(overrides.size);
	}
}

function testShadowManagerBindingLifecycle() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	const spot = scene.add(new SpotLight({ intensity: 1, range: 120 }));

	const single = scene.shadows.createSingle({ size: 1024 });
	const csm = scene.shadows.createCSM({
		size: 2048,
		cascadeCounts: {
			directional: 4,
			spot: 3,
			point: 2,
		},
	});

	scene.shadows.bind(sun, single);
	assert.equal(scene.shadows.getBoundShadowMap(sun), single);

	scene.shadows.rebind(sun, csm);
	assert.equal(scene.shadows.getBoundShadowMap(sun), csm);

	scene.shadows.bind(spot, csm);
	assert.equal(scene.shadows.getBoundShadowMap(spot), csm);

	scene.shadows.unbindLight(sun);
	assert.equal(scene.shadows.getBoundShadowMap(sun), undefined);
	assert.equal(scene.shadows.getBoundShadowMap(spot), csm);

	scene.shadows.destroy(csm);
	assert.equal(scene.shadows.getBoundShadowMap(spot), undefined);
	assert.equal(scene.shadows.getLegacyShadowConfig(spot), undefined);
}

function testSceneAcceptsExternalShadowMapRegistry() {
	const registry = new ShadowMapRegistry().register(
		"external-cascade",
		(options) => new ExternalCascadeShadowMap(options),
	);
	const scene = new Scene({ shadows: { registry } });
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	const shadowMap = scene.shadows.create("external-cascade", {
		size: 1024,
		priority: 4,
	});

	scene.shadows.bind(sun, shadowMap);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	assert.equal(frameState.records.length, 1);
	assert.equal(frameState.records[0].shadowMapKind, "external-cascade");
	assert.equal(frameState.records[0].priority, 4);

	const renderSet = frameState.get(sun);
	assert.ok(renderSet);
	assert.equal(renderSet?.resolvedConfig.strategy, "csm");
	assert.equal(renderSet?.resolvedConfig.cascadeCount, 3);
	assert.equal(renderSet?.slices.length, 3);

	updateShadowMapMetadata(renderSet, sun, createSceneBounds(60));
	assert.equal(renderSet?.effectiveStrategyType, "csm");
	assert.equal(renderSet?.slices.length, 3);
}

function testShadowManagerRegistersCustomMapType() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	scene.shadows.registerMapType(
		"external-single",
		(options) => new ExternalSingleShadowMap(options),
	);

	const shadowMap = scene.shadows.create("external-single", { size: 384 });
	scene.shadows.bind(sun, shadowMap);

	const config = scene.shadows.getLegacyShadowConfig(sun);
	assert.equal(config?.strategy, "single-map");
	assert.equal(config?.size, 384);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	assert.equal(frameState.records[0].shadowMapKind, "external-single");
}

function testVSMShadowMapUsesVSMFilterMetadataAndSingleMapRuntimeConfig() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const vsm = scene.shadows.createVSM({
		size: 1536,
		momentBias: 0.004,
		bleedReduction: 0.35,
		minVariance: 0.00008,
		sampling: {
			pcfRadius: 1.25,
			filterMode: "pcf",
		},
	});
	scene.shadows.bind(sun, vsm);

	const config = scene.shadows.getLegacyShadowConfig(sun);
	assert.ok(config);
	assert.equal(config?.strategy, "single-map");
	assert.equal(config?.size, 1536);
	assert.equal(vsm.filterMode, "vsm");
	assert.equal(config?.params?.shadowMomentBias, 0.004);
	assert.equal(config?.params?.shadowBleedReduction, 0.35);
	assert.equal(config?.params?.shadowMinVariance, 0.00008);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	assert.equal(frameState.records.length, 1);
	assert.equal(frameState.records[0].shadowMapKind, "vsm");
	assert.equal(frameState.records[0].filterMode, "vsm");
	assert.equal(frameState.records[0].renderSet.resolvedConfig.strategy, "single-map");
}

function testVSMShadowMapNormalizesParametersAndUpdatesSignature() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const vsm = scene.shadows.createVSM({
		momentBias: 0.001,
		bleedReduction: 0.2,
		minVariance: 0.00002,
	});
	scene.shadows.bind(sun, vsm);

	const frameA = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	const renderSetA = frameA.get(sun);
	assert.ok(renderSetA);
	const signatureA = renderSetA.configSignature;

	vsm.setVSMParameters({
		momentBias: -1,
		bleedReduction: 5,
		minVariance: 0,
	});
	assert.equal(vsm.momentBias, 0);
	assert.equal(vsm.bleedReduction, 1);
	assert.equal(vsm.minVariance, 1e-8);

	const frameB = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	const renderSetB = frameB.get(sun);
	assert.ok(renderSetB);
	assert.notEqual(renderSetB?.configSignature, signatureA);

	const nextConfig = scene.shadows.getLegacyShadowConfig(sun);
	assert.equal(nextConfig?.params?.shadowMomentBias, 0);
	assert.equal(nextConfig?.params?.shadowBleedReduction, 1);
	assert.equal(nextConfig?.params?.shadowMinVariance, 1e-8);
}

function testRenderSetSignatureUpdatesWhenShadowMapConfigChanges() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const single = scene.shadows.createSingle({ size: 1024 });
	scene.shadows.bind(sun, single);

	const frameA = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	const renderSetA = frameA.get(sun);
	assert.ok(renderSetA);
	const signatureA = renderSetA.configSignature;

	single.size = 512;

	const frameB = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
	});
	const renderSetB = frameB.get(sun);
	assert.ok(renderSetB);
	assert.equal(renderSetB?.size, 512);
	assert.notEqual(renderSetB?.configSignature, signatureA);
	assert.notStrictEqual(renderSetB, renderSetA);
}

function testPointCSMGeneratesCubeCascadeSlices() {
	const scene = new Scene();
	const point = scene.add(new PointLight({ range: 90 }));
	scene.shadows.bind(
		point,
		scene.shadows.createCSM({
			size: 1024,
			cascadeCounts: {
				point: 2,
			},
		}),
	);

	const frameState = scene.shadows.buildFrameState({
		lights: [point],
		enableShadows: true,
	});
	const renderSet = frameState.get(point);
	assert.ok(renderSet);

	updateShadowMapMetadata(renderSet, point, createSceneBounds(60));
	assert.equal(renderSet.effectiveStrategyType, "csm");
	assert.equal(renderSet.slices.length, 12);

	for (let cascadeIndex = 0; cascadeIndex < 2; cascadeIndex++) {
		const firstFace = renderSet.slices[cascadeIndex * 6];
		assert.ok(firstFace.splitFar > firstFace.splitNear);
		for (let face = 1; face < 6; face++) {
			const slice = renderSet.slices[cascadeIndex * 6 + face];
			assert.ok(Math.abs(slice.splitNear - firstFace.splitNear) < 1e-6);
			assert.ok(Math.abs(slice.splitFar - firstFace.splitFar) < 1e-6);
		}
		if (cascadeIndex > 0) {
			const previous = renderSet.slices[(cascadeIndex - 1) * 6];
			assert.ok(previous.splitFar <= firstFace.splitNear + 1e-6);
		}
	}
}

function testDynamicBudgetDegradesCascadeThenDisablesLowerScoreShadows() {
	const scene = new Scene();
	const highPriority = scene.add(new DirectionalLight({ intensity: 1 }));
	const lowPriority = scene.add(new DirectionalLight({ intensity: 100 }));

	scene.shadows.bind(
		highPriority,
		scene.shadows.createCSM({
			size: 1024,
			priority: 3,
			cascadeCounts: {
				directional: 4,
			},
		}),
	);
	scene.shadows.bind(lowPriority, scene.shadows.createSingle({ size: 1024 }));

	const frameState = scene.shadows.buildFrameState({
		lights: [highPriority, lowPriority],
		enableShadows: true,
		backendCapabilities: createBudgetCapabilities(2),
	});

	assert.equal(frameState.records.length, 1);
	assert.equal(frameState.has(lowPriority), false);
	const selected = frameState.get(highPriority);
	assert.ok(selected);
	assert.equal(selected?.resolvedConfig.strategy, "csm");
	assert.equal(selected?.resolvedConfig.cascadeCount, 2);
	assert.equal(selected?.size, 1024);
}

function testDynamicBudgetCanReduceResolutionAfterCascadeReduction() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 1 }));

	scene.shadows.bind(
		sun,
		scene.shadows.createCSM({
			size: 1024,
			priority: 2,
			cascadeCounts: {
				directional: 4,
			},
		}),
	);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
		backendCapabilities: createBudgetCapabilities(0.5),
	});

	const selected = frameState.get(sun);
	assert.ok(selected);
	assert.equal(selected?.resolvedConfig.strategy, "csm");
	assert.equal(selected?.resolvedConfig.cascadeCount, 2);
	assert.equal(selected?.size, 512);
}

function run() {
	testShadowManagerBindingLifecycle();
	testSceneAcceptsExternalShadowMapRegistry();
	testShadowManagerRegistersCustomMapType();
	testVSMShadowMapUsesVSMFilterMetadataAndSingleMapRuntimeConfig();
	testVSMShadowMapNormalizesParametersAndUpdatesSignature();
	testRenderSetSignatureUpdatesWhenShadowMapConfigChanges();
	testPointCSMGeneratesCubeCascadeSlices();
	testDynamicBudgetDegradesCascadeThenDisablesLowerScoreShadows();
	testDynamicBudgetCanReduceResolutionAfterCascadeReduction();
	console.log("Shadow manager tests passed");
}

run();
