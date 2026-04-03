import assert from "node:assert/strict";
import {
	buildDirtyTileCoverage,
	getDirtyTileCoverageAreaRatio,
	getDirtyRectsAreaRatio,
	inflateDirtyRects,
	mergeDirtyRects,
	tileCoverageToDirtyRects,
} from "../src/pipeline/incremental.ts";

function testOverlapMerge() {
	const merged = mergeDirtyRects(
		[
			{ x: 0, y: 0, width: 10, height: 10 },
			{ x: 5, y: 5, width: 10, height: 10 },
		],
		16,
		100,
		100
	);
	assert.equal(merged.length, 1);
	assert.deepEqual(merged[0], {
		x: 0,
		y: 0,
		width: 15,
		height: 15,
	});
}

function testRectCapMerge() {
	const merged = mergeDirtyRects(
		[
			{ x: 0, y: 0, width: 10, height: 10 },
			{ x: 20, y: 0, width: 10, height: 10 },
			{ x: 40, y: 0, width: 10, height: 10 },
			{ x: 60, y: 0, width: 10, height: 10 },
			{ x: 80, y: 0, width: 10, height: 10 },
		],
		2,
		200,
		100
	);
	assert.equal(merged.length, 2);
}

function testInflationAndClamp() {
	const inflated = inflateDirtyRects(
		[
			{ x: 10, y: 10, width: 10, height: 10 },
			{ x: -5, y: -5, width: 8, height: 8 },
		],
		2,
		100,
		100
	);
	assert.deepEqual(inflated[0], {
		x: 8,
		y: 8,
		width: 14,
		height: 14,
	});
	assert.deepEqual(inflated[1], {
		x: 0,
		y: 0,
		width: 5,
		height: 5,
	});
}

function testAreaRatio() {
	const ratio = getDirtyRectsAreaRatio(
		[
			{ x: 0, y: 0, width: 20, height: 10 },
			{ x: 50, y: 0, width: 10, height: 20 },
		],
		100,
		100
	);
	assert.equal(ratio, 0.04);
	assert.ok(ratio < 0.3);
}

function testTileCoverageRasterization() {
	const coverage = buildDirtyTileCoverage(
		[
			{ x: 3, y: 3, width: 10, height: 10 },
			{ x: 28, y: 4, width: 6, height: 9 },
		],
		64,
		32,
		16
	);
	assert.equal(coverage.tileColumns, 4);
	assert.equal(coverage.tileRows, 2);
	assert.deepEqual(coverage.dirtyTiles, [0, 1, 2]);

	const rects = tileCoverageToDirtyRects(coverage, 8, 64, 32);
	assert.deepEqual(rects, [{
		x: 0,
		y: 0,
		width: 48,
		height: 16,
	}]);
}

function testTileCoverageAreaRatio() {
	const coverage = buildDirtyTileCoverage(
		[
			{ x: 0, y: 0, width: 8, height: 8 },
			{ x: 24, y: 16, width: 8, height: 8 },
		],
		32,
		32,
		16
	);
	const ratio = getDirtyTileCoverageAreaRatio(coverage, 32, 32);
	assert.equal(ratio, 0.5);
}

function run() {
	testOverlapMerge();
	testRectCapMerge();
	testInflationAndClamp();
	testAreaRatio();
	testTileCoverageRasterization();
	testTileCoverageAreaRatio();
	console.log("Dirty rect merger tests passed");
}

run();
