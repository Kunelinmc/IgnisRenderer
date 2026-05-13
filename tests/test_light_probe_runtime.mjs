import assert from "node:assert/strict";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { SH } from "../src/maths/SH.ts";
import {
	collectActiveLocalizedLightProbes,
	collectGlobalLightProbes,
	computeLightProbeMetric,
	computeLightProbeRawWeight,
	selectTopTwoLocalizedLightProbes,
} from "../src/pipeline/lightProbeRuntime.ts";

function createLocalizedProbe({
	id,
	shape = "sphere",
	radius = 2,
	halfExtents = { x: 2, y: 2, z: 2 },
	blendDistance = 0.2,
	priority = 0,
	position = { x: 0, y: 0, z: 0 },
} = {}) {
	const sh = SH.empty();
	sh[0] = { r: 8, g: 4, b: 2 };
	const probe = new LightProbe({
		sh,
		shape,
		radius,
		halfExtents,
		blendDistance,
		priority,
	});
	if (id) {
		probe.id = id;
	}
	probe.position.set(position.x, position.y, position.z);
	probe.updateWorldMatrix();
	probe.markRuntimeDirty();
	return probe;
}

function testConstructorAndLocalizedClone() {
	const probeFromSH = new LightProbe(null);
	assert.equal(probeFromSH.shape, "global");
	assert.equal("intensity" in probeFromSH, false);
	assert.equal(probeFromSH.sh.length, 16);

	const probe = new LightProbe({
		sh: SH.empty(),
		shape: "box",
		radius: -4,
		halfExtents: { x: -2, y: 0, z: Number.NaN },
		blendDistance: -1,
		priority: 3.9,
	});
	assert.equal(probe.shape, "box");
	assert.equal(probe.radius, 4);
	assert.ok(probe.halfExtents.x > 0);
	assert.ok(probe.halfExtents.y > 0);
	assert.ok(probe.halfExtents.z > 0);
	assert.equal(probe.blendDistance, 0);
	assert.equal(probe.priority, 3);

	const cloned = probe.clone(false);
	assert.equal(cloned.shape, "box");
	assert.equal(cloned.priority, 3);
	assert.notEqual(cloned.sh, probe.sh);
	cloned.sh[0].r = 99;
	assert.notEqual(cloned.sh[0].r, probe.sh[0].r);
}

function testCopyPreservesLocalizedState() {
	const source = createLocalizedProbe({
		shape: "box",
		radius: 5,
		halfExtents: { x: 3, y: 4, z: 5 },
		blendDistance: 0.15,
		priority: 12,
	});
	source.sh[5] = { r: 7, g: 3, b: 1 };

	const target = new LightProbe(SH.empty());
	target.copy(source);

	assert.equal(target.shape, "box");
	assert.equal(target.radius, 5);
	assert.deepEqual(
		{ x: target.halfExtents.x, y: target.halfExtents.y, z: target.halfExtents.z },
		{ x: 3, y: 4, z: 5 }
	);
	assert.equal(target.blendDistance, 0.15);
	assert.equal(target.priority, 12);
	assert.deepEqual(target.sh[5], { r: 7, g: 3, b: 1 });
}

function testMetricAndBlendCurve() {
	const boxProbe = createLocalizedProbe({
		shape: "box",
		halfExtents: { x: 2, y: 4, z: 6 },
	});
	const sphereProbe = createLocalizedProbe({
		shape: "sphere",
		radius: 4,
	});

	const boxMetric = computeLightProbeMetric({ x: 1, y: 0, z: 0 }, boxProbe);
	const sphereMetric = computeLightProbeMetric({ x: 2, y: 0, z: 0 }, sphereProbe);
	assert.ok(Math.abs(boxMetric - 0.5) < 1e-6);
	assert.ok(Math.abs(sphereMetric - 0.5) < 1e-6);

	let previous = Number.POSITIVE_INFINITY;
	for (let i = 0; i <= 40; i++) {
		const metric = 0.8 + i * 0.04;
		const weight = computeLightProbeRawWeight(metric, 0.2);
		assert.ok(weight <= previous + 1e-8);
		previous = weight;
	}

	const left = computeLightProbeRawWeight(0.9999, 0.2);
	const right = computeLightProbeRawWeight(1.0001, 0.2);
	assert.ok(Math.abs(left - right) < 1e-3);

	const cache = sphereProbe.getRuntimeCache();
	assert.ok(cache.effectiveBlendDistance >= 0.4);
}

function testPrioritySelectionAndTieBreak() {
	const lowPriority = createLocalizedProbe({
		id: "z-low",
		priority: 1,
		radius: 10,
		position: { x: 0, y: 0, z: 0 },
	});
	const highPriorityA = createLocalizedProbe({
		id: "b-high",
		priority: 10,
		radius: 2,
		position: { x: 0, y: 0, z: 0 },
	});
	const highPriorityB = createLocalizedProbe({
		id: "a-high",
		priority: 10,
		radius: 2,
		position: { x: 0, y: 0, z: 0 },
	});

	const probes = [lowPriority, highPriorityA, highPriorityB];
	const result = selectTopTwoLocalizedLightProbes(
		{ x: 0, y: 0, z: 0 },
		probes
	);

	assert.equal(result.priority, 10);
	assert.ok(result.firstIndex >= 0);
	assert.ok(result.secondIndex >= 0);
	assert.equal(probes[result.firstIndex].id, "a-high");
	assert.ok(Math.abs(result.firstWeight + result.secondWeight - 1) < 1e-6);
	assert.equal(result.coverage, 1);
	assert.notEqual(probes[result.firstIndex].id, lowPriority.id);
}

function testCollectionSeparatesGlobalAndLocalizedProbes() {
	const globalProbe = new LightProbe(SH.empty());
	globalProbe.id = "global";
	const farLocalized = Array.from({ length: 8 }, (_, index) =>
		createLocalizedProbe({
			id: `far-${index}`,
			radius: 2,
			position: { x: 100 + index, y: 0, z: 0 },
		})
	);
	const nearLocalized = createLocalizedProbe({
		id: "near",
		radius: 4,
		position: { x: 0, y: 0, z: 0 },
	});

	const globals = collectGlobalLightProbes([globalProbe, ...farLocalized, nearLocalized]);
	assert.equal(globals.length, 1);
	assert.equal(globals[0].id, "global");

	const collected = collectActiveLocalizedLightProbes(
		[globalProbe, ...farLocalized, nearLocalized],
		8,
		{ x: 0, y: 0, z: 0 }
	);
	assert.equal(collected.length, 8);
	assert.ok(collected.some((probe) => probe.id === "near"));
}

function run() {
	testConstructorAndLocalizedClone();
	testCopyPreservesLocalizedState();
	testMetricAndBlendCurve();
	testPrioritySelectionAndTieBreak();
	testCollectionSeparatesGlobalAndLocalizedProbes();
	console.log("Light probe runtime tests passed");
}

run();
