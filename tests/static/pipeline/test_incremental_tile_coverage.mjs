import assert from "node:assert/strict";
import {
	createIncrementalTileCoverage,
} from "../../../src/pipeline/incremental.ts";

function testPartialCoverageNormalizesAndComplementsTiles() {
	const coverage = createIncrementalTileCoverage(32, 3, 2, [4, 1, 1, -1, 6]);
	assert.equal(coverage.mode, "partial");
	assert.deepEqual(coverage.updatedTileRanges, [
		{ startTile: 1, endTileExclusive: 2 },
		{ startTile: 4, endTileExclusive: 5 },
	]);
	assert.deepEqual(coverage.reusableTileRanges, [
		{ startTile: 0, endTileExclusive: 1 },
		{ startTile: 2, endTileExclusive: 4 },
		{ startTile: 5, endTileExclusive: 6 },
	]);
}

function testFullAndUnchangedCoverageUseCompactRanges() {
	const full = createIncrementalTileCoverage(16, 3, 2, [], "full");
	assert.equal(full.mode, "full");
	assert.deepEqual(full.updatedTileRanges, [{ startTile: 0, endTileExclusive: 6 }]);
	assert.deepEqual(full.reusableTileRanges, []);

	const unchanged = createIncrementalTileCoverage(16, 3, 2, []);
	assert.equal(unchanged.mode, "unchanged");
	assert.deepEqual(unchanged.updatedTileRanges, []);
	assert.deepEqual(unchanged.reusableTileRanges, [
		{ startTile: 0, endTileExclusive: 6 },
	]);

	const duplicateOnly = createIncrementalTileCoverage(16, 2, 2, [0, 0, 0, 0]);
	assert.equal(duplicateOnly.mode, "partial");
}

testPartialCoverageNormalizesAndComplementsTiles();
testFullAndUnchangedCoverageUseCompactRanges();
console.log("Incremental tile coverage tests passed");
