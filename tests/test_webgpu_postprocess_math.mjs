import assert from "node:assert/strict";
import {
	buildHiZMinChain,
	clampHistoryToNeighborhoodYCoCg,
	computeHaltonJitterNDC,
	halton,
	isDepthHistoryCompatible,
	reprojectHistoryUv,
	rgbToYCoCg,
	traceSSRDepthHit,
} from "../src/maths/Misc";

function testHaltonSequenceAndJitterRange() {
	assert.equal(halton(1, 2), 0.5);
	assert.equal(halton(2, 2), 0.25);
	assert.equal(halton(3, 2), 0.75);
	assert.equal(halton(1, 3), 1 / 3);
	assert.equal(halton(2, 3), 2 / 3);

	const jA = computeHaltonJitterNDC(5, 1920, 1080, 1);
	const jB = computeHaltonJitterNDC(5, 1920, 1080, 1);
	const jCycleA = computeHaltonJitterNDC(0, 1920, 1080, 1, 16);
	const jCycleB = computeHaltonJitterNDC(16, 1920, 1080, 1, 16);
	assert.deepEqual(jA, jB);
	assert.notDeepEqual(jCycleA, jCycleB);
	assert.ok(Number.isFinite(jA[0]) && Number.isFinite(jA[1]));
	assert.ok(Math.abs(jA[0]) <= 1 / 1920);
	assert.ok(Math.abs(jA[1]) <= 1 / 1080);
}

function testTAAReprojectionAndDepthGate() {
	const uv = reprojectHistoryUv([0.4, 0.6], [0.2, -0.4]);
	assert.ok(Math.abs(uv[0] - 0.3) < 1e-6);
	assert.ok(Math.abs(uv[1] - 0.4) < 1e-6);

	assert.equal(isDepthHistoryCompatible(10, 10.1, 0.02), true);
	assert.equal(isDepthHistoryCompatible(10, 12, 0.02), false);
	assert.equal(isDepthHistoryCompatible(0, 12, 0.02), false);
}

function testTAANeighborhoodClamp() {
	const neighborhood = [
		[0.2, 0.2, 0.2],
		[0.3, 0.3, 0.3],
		[0.4, 0.4, 0.4],
		[0.5, 0.5, 0.5],
	];
	const clamped = clampHistoryToNeighborhoodYCoCg([2, 0, 0], neighborhood, 0);
	const clampedYCoCg = rgbToYCoCg(clamped);
	const neighborhoodYCoCg = neighborhood.map((rgb) => rgbToYCoCg(rgb));

	const minY = Math.min(...neighborhoodYCoCg.map((v) => v[0]));
	const maxY = Math.max(...neighborhoodYCoCg.map((v) => v[0]));
	const minCo = Math.min(...neighborhoodYCoCg.map((v) => v[1]));
	const maxCo = Math.max(...neighborhoodYCoCg.map((v) => v[1]));
	const minCg = Math.min(...neighborhoodYCoCg.map((v) => v[2]));
	const maxCg = Math.max(...neighborhoodYCoCg.map((v) => v[2]));

	assert.ok(clampedYCoCg[0] >= minY && clampedYCoCg[0] <= maxY);
	assert.ok(clampedYCoCg[1] >= minCo && clampedYCoCg[1] <= maxCo);
	assert.ok(clampedYCoCg[2] >= minCg && clampedYCoCg[2] <= maxCg);

	const meanNeighborhood = [
		[0.2, 0.2, 0.2],
		[0.8, 0.8, 0.8],
	];
	const toMean = clampHistoryToNeighborhoodYCoCg(
		[5, 5, 5],
		meanNeighborhood,
		0
	);
	assert.ok(Math.abs(toMean[0] - 0.5) < 1e-6);
	assert.ok(Math.abs(toMean[1] - 0.5) < 1e-6);
	assert.ok(Math.abs(toMean[2] - 0.5) < 1e-6);
}

function testHiZBuildAndSSRHitMiss() {
	const depth4x4 = new Float32Array([
		9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8,
	]);
	const chain = buildHiZMinChain(depth4x4, 4, 4);
	assert.equal(chain.length, 3);
	assert.equal(chain[1].width, 2);
	assert.equal(chain[1].height, 2);
	assert.deepEqual(Array.from(chain[1].data), [4, 2, 1, 3]);
	assert.equal(chain[2].width, 1);
	assert.equal(chain[2].height, 1);
	assert.equal(chain[2].data[0], 1);

	const flatDepth = {
		width: 8,
		height: 8,
		data: new Float32Array(64).fill(5),
	};
	const hit = traceSSRDepthHit(
		flatDepth,
		[0.2, 0.5],
		[0.08, 0],
		2,
		0.5,
		20,
		0.1
	);
	assert.equal(hit.hit, true);
	assert.ok(hit.hitUv);

	const miss = traceSSRDepthHit(
		flatDepth,
		[0.2, 0.5],
		[0.08, 0],
		2,
		0.5,
		4,
		0.1
	);
	assert.equal(miss.hit, false);
	assert.equal(miss.hitUv, null);
}

function run() {
	testHaltonSequenceAndJitterRange();
	testTAAReprojectionAndDepthGate();
	testTAANeighborhoodClamp();
	testHiZBuildAndSSRHitMiss();
	console.log("WebGPU postprocess math tests passed");
}

run();
