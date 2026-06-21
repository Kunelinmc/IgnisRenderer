import assert from "node:assert/strict";

import {
	cellularNoise2D,
	fractalBrownianMotion2D,
	generateNoiseMap2D,
	hashNoise2D,
	perlinNoise2D,
	ridgedNoise2D,
	simplexNoise2D,
	turbulenceNoise2D,
	valueNoise2D,
} from "../../../src/maths/Noise.ts";

function testHashNoiseIsDeterministicAndBounded() {
	const a = hashNoise2D(4.7, -2.2, 19);
	const b = hashNoise2D(4.1, -2.9, 19);
	const c = hashNoise2D(4.7, -2.2, 20);

	assert.equal(a, b, "Hash noise should use floored lattice coordinates");
	assert.notEqual(a, c, "Different seeds should decorrelate hash noise");
	assert.ok(a >= 0 && a < 1, `Hash noise out of range: ${a}`);
}

function testValueNoiseMatchesLatticeHashAtIntegerCoordinates() {
	const expected = hashNoise2D(3, 5, 7) * 2 - 1;
	const actual = valueNoise2D(3, 5, 7);

	assert.ok(Math.abs(actual - expected) < 1e-12);
}

function testPerlinNoiseIsZeroAtIntegerLatticeCoordinates() {
	assert.ok(Math.abs(perlinNoise2D(2, -4, 11)) < 1e-12);
	assert.ok(Math.abs(perlinNoise2D(-7, 8, 11)) < 1e-12);
}

function testNoiseFamiliesStayWithinDocumentedRanges() {
	const samples = [
		valueNoise2D(0.37, 9.11, 3),
		perlinNoise2D(0.37, 9.11, 3),
		simplexNoise2D(0.37, 9.11, 3),
		fractalBrownianMotion2D(0.37, 9.11, {
			seed: 3,
			octaves: 6,
			amplitude: 2,
		}),
	];

	for (const value of samples) {
		assert.ok(value >= -2 && value <= 2, `Signed noise out of range: ${value}`);
	}

	const turbulent = turbulenceNoise2D(0.37, 9.11, {
		seed: 3,
		octaves: 6,
	});
	const ridged = ridgedNoise2D(0.37, 9.11, {
		seed: 3,
		octaves: 6,
	});

	assert.ok(turbulent >= 0 && turbulent <= 1, "Turbulence should be normalized");
	assert.ok(ridged >= 0 && ridged <= 1, "Ridged noise should be normalized");
}

function testCellularNoiseReportsNearestCell() {
	const sample = cellularNoise2D(1.25, 2.75, {
		seed: 4,
		jitter: 0,
	});

	assert.equal(sample.cellX, 1);
	assert.equal(sample.cellY, 2);
	assert.ok(sample.distance > 0);
	assert.ok(sample.value >= 0 && sample.value <= 1);
}

function testNoiseMapGenerationUsesRowMajorOrder() {
	const data = generateNoiseMap2D(3, 2, {
		seed: 5,
		frequency: 0.25,
		normalize: true,
		sampler: valueNoise2D,
	});

	assert.equal(data.length, 6);
	for (const value of data) {
		assert.ok(value >= 0 && value <= 1, `Normalized map value out of range: ${value}`);
	}
	assert.equal(
		data[4],
		generateNoiseMap2D(3, 2, {
			seed: 5,
			frequency: 0.25,
			normalize: true,
			sampler: valueNoise2D,
		})[4]
	);
}

function run() {
	testHashNoiseIsDeterministicAndBounded();
	testValueNoiseMatchesLatticeHashAtIntegerCoordinates();
	testPerlinNoiseIsZeroAtIntegerLatticeCoordinates();
	testNoiseFamiliesStayWithinDocumentedRanges();
	testCellularNoiseReportsNearestCell();
	testNoiseMapGenerationUsesRowMajorOrder();
	console.log("Noise tests passed");
}

run();
