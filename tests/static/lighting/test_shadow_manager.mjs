import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ShadowMapBase } from "../../../src/lights/shadows/ShadowMapBase.ts";
import { ShadowMapRegistry } from "../../../src/lights/shadows/ShadowMapRegistry.ts";
import { SingleShadowMap } from "../../../src/lights/shadows/SingleShadowMap.ts";
import { updateShadowMapMetadata } from "../../../src/pipeline/ShadowMetadata.ts";

function createSceneBounds(radius = 80) {
	return {
		center: { x: 0, y: 0, z: 0 },
		radius,
	};
}

function createCamera(overrides = {}) {
	const position = overrides.position ?? { x: 0, y: 4, z: 16 };
	const up = overrides.up ?? { x: 0, y: 1, z: 0 };
	const forward = overrides.forward ?? { x: 0, y: 0, z: -1 };
	return {
		near: overrides.near ?? 0.1,
		far: overrides.far ?? 100,
		fov: overrides.fov ?? 60,
		aspectRatio: overrides.aspectRatio ?? 16 / 9,
		position,
		up,
		getWorldPosition(target = { x: 0, y: 0, z: 0 }) {
			target.x = position.x;
			target.y = position.y;
			target.z = position.z;
			return target;
		},
		getWorldDirection(localDirection, target = { x: 0, y: 0, z: 0 }) {
			const useUpDirection =
				Math.abs(localDirection.x - up.x) <= 1e-6 &&
				Math.abs(localDirection.y - up.y) <= 1e-6 &&
				Math.abs(localDirection.z - up.z) <= 1e-6;
			const source = useUpDirection ? up : forward;
			target.x = source.x;
			target.y = source.y;
			target.z = source.z;
			return target;
		},
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

function createPagedCapabilities(supportsPagedShadows = true) {
	return {
		...createBudgetCapabilities(64),
		supportsPagedShadows,
		supportsPagedShadowRendering: supportsPagedShadows,
		maxPagedShadowPages: 2048,
		pagedShadowPageSizeRange: [64, 256],
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
	const csm = scene.shadows.createCascaded({
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

function testVarianceShadowMapUsesVSMFilterMetadataAndSingleMapRuntimeConfig() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const vsm = scene.shadows.createVariance({
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
	assert.equal(frameState.records[0].shadowMapKind, "variance");
	assert.equal(frameState.records[0].filterMode, "vsm");
	assert.equal(frameState.records[0].renderSet.resolvedConfig.strategy, "single-map");
}

function testVarianceShadowMapNormalizesParametersAndUpdatesSignature() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const vsm = scene.shadows.createVariance({
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

	vsm.setVarianceParameters({
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
		scene.shadows.createCascaded({
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
		scene.shadows.createCascaded({
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
		scene.shadows.createCascaded({
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

function testShadowLayoutsMirrorSingleAndCSMSlices() {
	const scene = new Scene();
	const singleLight = scene.add(new DirectionalLight({ intensity: 1 }));
	scene.shadows.bind(singleLight, scene.shadows.createSingle({ size: 1024 }));

	const singleFrame = scene.shadows.buildFrameState({
		lights: [singleLight],
		enableShadows: true,
	});
	const singleSet = singleFrame.get(singleLight);
	assert.ok(singleSet);
	updateShadowMapMetadata(singleSet, singleLight, createSceneBounds(40));
	assert.equal(singleSet.storageMode, "atlas");
	assert.equal(singleSet.layout.storageMode, "atlas");
	assert.equal(singleSet.layout.regions.length, 1);
	assert.equal(singleSet.layout.regions[0].kind, "single");
	assert.equal(
		singleSet.layout.regions[0].viewProjection,
		singleSet.slices[0].shadowMap.viewProjectionMatrix
	);

	const csmLight = scene.add(new DirectionalLight({ intensity: 1 }));
	scene.shadows.bind(
		csmLight,
		scene.shadows.createCascaded({
			size: 1024,
			cascadeCounts: { directional: 4 },
		})
	);
	const csmFrame = scene.shadows.buildFrameState({
		lights: [singleLight, csmLight],
		enableShadows: true,
	});
	const csmSet = csmFrame.get(csmLight);
	assert.ok(csmSet);
	updateShadowMapMetadata(csmSet, csmLight, createSceneBounds(80), {
		camera: createCamera(),
	});
	assert.equal(csmSet.storageMode, "atlas");
	assert.equal(csmSet.layout.regions.length, 4);
	for (let index = 0; index < csmSet.layout.regions.length; index++) {
		assert.equal(csmSet.layout.regions[index].kind, "cascade");
		assert.equal(csmSet.layout.regions[index].sourceSliceIndex, index);
		assert.equal(csmSet.layout.regions[index].splitNear, csmSet.slices[index].splitNear);
		assert.equal(csmSet.layout.regions[index].splitFar, csmSet.slices[index].splitFar);
		assert.equal(
			csmSet.layout.regions[index].viewProjection,
			csmSet.slices[index].shadowMap.viewProjectionMatrix
		);
	}
}

function testPagedShadowMapBuildsPagedRenderSetMetadata() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	const paged = scene.shadows.createPaged({
		size: 2048,
		virtualResolution: 8192,
		pageSize: 64,
		physicalPageCount: 512,
		clipmapLevels: 5,
		maxPagesPerFrame: 128,
		cacheFrames: 60,
		feedbackMode: "conservative",
		cascadeCounts: { directional: 4 },
	});
	scene.shadows.bind(sun, paged);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
		backendCapabilities: createPagedCapabilities(true),
	});
	assert.equal(frameState.records.length, 1);
	assert.equal(frameState.records[0].shadowMapKind, "paged-shadow");
	assert.equal(frameState.records[0].renderSet.storageMode, "paged");

	const renderSet = frameState.get(sun);
	assert.ok(renderSet);
	assert.equal(renderSet.storageMode, "paged");
	assert.equal(renderSet.layout.storageMode, "paged");
	assert.equal(renderSet.layout.paged?.virtualResolution, 8192);
	assert.equal(renderSet.layout.paged?.pageSize, 64);
	assert.equal(renderSet.layout.paged?.physicalPageCount, 512);
	assert.equal(renderSet.layout.paged?.maxPagesPerFrame, 128);

	updateShadowMapMetadata(renderSet, sun, createSceneBounds(100), {
		camera: createCamera(),
	});
	assert.equal(renderSet.resolvedConfig.strategy, "csm");
	assert.equal(renderSet.slices.length, 4);
	assert.equal(renderSet.layout.regions.length, 4);
	for (let index = 0; index < renderSet.layout.regions.length; index++) {
		const region = renderSet.layout.regions[index];
		assert.equal(region.kind, "paged-page");
		assert.equal(region.sourceSliceIndex, index);
		assert.equal(region.splitNear, renderSet.slices[index].splitNear);
		assert.equal(region.splitFar, renderSet.slices[index].splitFar);
		assert.equal(
			region.viewProjection,
			renderSet.slices[index].shadowMap.viewProjectionMatrix
		);
	}
}

function testPagedShadowMapFallsBackToAtlasWhenUnsupported() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight({ intensity: 2 }));
	scene.shadows.bind(
		sun,
		scene.shadows.createPaged({
			size: 1024,
			virtualResolution: 4096,
			pageSize: 128,
		})
	);

	const frameState = scene.shadows.buildFrameState({
		lights: [sun],
		enableShadows: true,
		backendCapabilities: createPagedCapabilities(false),
	});
	const renderSet = frameState.get(sun);
	assert.ok(renderSet);
	assert.equal(frameState.records[0].shadowMapKind, "paged-shadow");
	assert.equal(renderSet.storageMode, "atlas");
	assert.equal(renderSet.layout.storageMode, "atlas");
	assert.equal(renderSet.layout.paged, undefined);
	assert.equal(renderSet.resolvedConfig.strategy, "csm");
}

function testDefinitionAccessorsBatchRevisionAndSceneInvalidation() {
	const scene = new Scene();
	const sun = scene.add(new DirectionalLight());
	const shadow = scene.shadows.createSingle({ size: 512 });
	scene.shadows.bind(sun, shadow);
	const biasIdentity = shadow.bias;
	const samplingIdentity = shadow.sampling;

	let revision = shadow.revision;
	let sceneVersion = scene.version;
	shadow.size = 1024;
	assert.equal(shadow.revision, revision + 1);
	assert.equal(scene.version, sceneVersion + 1);

	revision = shadow.revision;
	sceneVersion = scene.version;
	shadow.bias.constant = 0.025;
	assert.equal(shadow.bias, biasIdentity);
	assert.equal(shadow.revision, revision + 1);
	assert.equal(scene.version, sceneVersion + 1);

	revision = shadow.revision;
	sceneVersion = scene.version;
	shadow.update({
		size: 2048,
		priority: 7,
		bias: { slope: 0.02, normal: 0.03 },
		sampling: { pcfRadius: 2, samples: 24 },
	});
	assert.equal(shadow.bias, biasIdentity);
	assert.equal(shadow.sampling, samplingIdentity);
	assert.equal(shadow.revision, revision + 1);
	assert.equal(scene.version, sceneVersion + 1);
	assert.equal(shadow.size, 2048);
	assert.equal(shadow.priority, 7);
	assert.equal(shadow.bias.slope, 0.02);
	assert.equal(shadow.sampling.samples, 24);
}

function testSharedDefinitionNotifiesEveryManagerAndSurvivesInactiveFrame() {
	const firstScene = new Scene();
	const secondScene = new Scene();
	const firstLight = firstScene.add(new DirectionalLight());
	const secondLight = secondScene.add(new DirectionalLight());
	const shared = new SingleShadowMap({ size: 512 });
	firstScene.shadows.bind(firstLight, shared);
	secondScene.shadows.bind(secondLight, shared);

	const firstVersion = firstScene.version;
	const secondVersion = secondScene.version;
	shared.sampling.strength = 0.75;
	assert.equal(firstScene.version, firstVersion + 1);
	assert.equal(secondScene.version, secondVersion + 1);

	firstScene.shadows.buildFrameState({ lights: [], enableShadows: true });
	assert.equal(firstScene.shadows.getBoundShadowMap(firstLight), shared);
	const restored = firstScene.shadows.buildFrameState({
		lights: [firstLight],
		enableShadows: true,
	});
	assert.equal(restored.records.length, 1);
}

function run() {
	testShadowManagerBindingLifecycle();
	testSceneAcceptsExternalShadowMapRegistry();
	testShadowManagerRegistersCustomMapType();
	testVarianceShadowMapUsesVSMFilterMetadataAndSingleMapRuntimeConfig();
	testVarianceShadowMapNormalizesParametersAndUpdatesSignature();
	testRenderSetSignatureUpdatesWhenShadowMapConfigChanges();
	testPointCSMGeneratesCubeCascadeSlices();
	testDynamicBudgetDegradesCascadeThenDisablesLowerScoreShadows();
	testDynamicBudgetCanReduceResolutionAfterCascadeReduction();
	testShadowLayoutsMirrorSingleAndCSMSlices();
	testPagedShadowMapBuildsPagedRenderSetMetadata();
	testPagedShadowMapFallsBackToAtlasWhenUnsupported();
	testDefinitionAccessorsBatchRevisionAndSceneInvalidation();
	testSharedDefinitionNotifiesEveryManagerAndSurvivesInactiveFrame();
	console.log("Shadow manager tests passed");
}

run();
