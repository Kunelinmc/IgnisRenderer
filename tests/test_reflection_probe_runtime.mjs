import assert from "node:assert/strict";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import {
	computeParallaxCorrectedDirection,
	computeProbeRawWeight,
	selectTopTwoReflectionProbes,
} from "../src/pipeline/reflectionProbeRuntime.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";

function testBlendCurveMonotonicAndContinuous() {
	const blendDistance = 0.2;
	const exponent = 1.0;
	let previous = Number.POSITIVE_INFINITY;
	for (let i = 0; i <= 40; i++) {
		const metric = 0.8 + i * 0.04;
		const weight = computeProbeRawWeight(metric, blendDistance, exponent);
		assert.ok(weight <= previous + 1e-8, "Blend weight must be monotonic");
		previous = weight;
	}

	const left = computeProbeRawWeight(0.9999, blendDistance, exponent);
	const right = computeProbeRawWeight(1.0001, blendDistance, exponent);
	assert.ok(
		Math.abs(left - right) < 1e-3,
		"Blend curve should stay continuous around the probe boundary"
	);

	const wBase = computeProbeRawWeight(1.1, blendDistance, 1.0);
	const wSharper = computeProbeRawWeight(1.1, blendDistance, 2.0);
	assert.ok(wSharper < wBase, "Higher blendExponent should sharpen the fade");
}

function testTopTwoTieBreakByProbeId() {
	const probeA = new ReflectionProbe({ shape: "sphere", radius: 2 });
	const probeB = new ReflectionProbe({ shape: "sphere", radius: 2 });
	const probes = [probeB, probeA];
	const result = selectTopTwoReflectionProbes({ x: 0, y: 0, z: 0 }, probes);
	assert.ok(result.firstIndex >= 0);

	const selected = probes[result.firstIndex];
	const expected =
		probeA.id.localeCompare(probeB.id) < 0 ? probeA.id : probeB.id;
	assert.equal(selected.id, expected, "Tie-break should be deterministic by probe id");
	assert.ok(Math.abs(result.firstWeight + result.secondWeight - 1) < 1e-6);
}

function testParallaxIntersectionAndFallback() {
	const boxProbe = new ReflectionProbe({
		shape: "box",
		halfExtents: { x: 1, y: 1, z: 1 },
		parallaxMode: "box",
	});
	boxProbe.worldMatrix = Matrix4.identity();
	boxProbe.markRuntimeDirty();
	const box = computeParallaxCorrectedDirection(
		{ x: 0, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 },
		boxProbe
	);
	assert.equal(box.valid, true);
	assert.ok(Math.abs(box.direction.x - 1) < 1e-6);
	assert.ok(Math.abs(box.direction.y) < 1e-6);
	assert.ok(Math.abs(box.direction.z) < 1e-6);

	const sphereProbe = new ReflectionProbe({
		shape: "sphere",
		radius: 1,
		parallaxMode: "sphere",
	});
	sphereProbe.worldMatrix = Matrix4.identity();
	sphereProbe.markRuntimeDirty();
	const sphere = computeParallaxCorrectedDirection(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		sphereProbe
	);
	assert.equal(sphere.valid, true);
	assert.ok(Math.abs(sphere.direction.z - 1) < 1e-6);

	const fallback = computeParallaxCorrectedDirection(
		{ x: 2, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 },
		sphereProbe
	);
	assert.equal(fallback.valid, false);
	assert.ok(Math.abs(fallback.direction.x - 1) < 1e-6);
}

function testRuntimeCacheDirtyBehavior() {
	const originalInverse3x3 = Matrix4.inverse3x3;
	let inverseCallCount = 0;
	Matrix4.inverse3x3 = function patchedInverse(matrix) {
		inverseCallCount++;
		return originalInverse3x3(matrix);
	};

	try {
		const probe = new ReflectionProbe({
			shape: "box",
			halfExtents: { x: 1, y: 2, z: 3 },
		});
		probe.worldMatrix = Matrix4.identity();
		probe.markRuntimeDirty();

		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 1);
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 1, "Cache should not recompute without changes");

		probe.worldMatrix = Matrix4.fromTranslation([1, 0, 0]);
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 2, "Matrix change should recompute cache");

		probe.markRuntimeDirty();
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 3, "Manual dirty flag should force recompute");
	} finally {
		Matrix4.inverse3x3 = originalInverse3x3;
	}
}

function run() {
	testBlendCurveMonotonicAndContinuous();
	testTopTwoTieBreakByProbeId();
	testParallaxIntersectionAndFallback();
	testRuntimeCacheDirtyBehavior();
	console.log("Reflection probe runtime tests passed");
}

run();
