import assert from "node:assert/strict";
import {
	getDirtyRectsAreaRatio,
	inflateDirtyRects,
	mergeDirtyRects,
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

function run() {
	testOverlapMerge();
	testRectCapMerge();
	testInflationAndClamp();
	testAreaRatio();
	console.log("Dirty rect merger tests passed");
}

run();
