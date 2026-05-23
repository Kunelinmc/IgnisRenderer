import assert from "node:assert/strict";

import { TemporalJitterState } from "../src/renderers/temporal/TemporalJitterState.ts";
import { computeHaltonJitterNDC } from "../src/maths/Misc.ts";

function testPerspectiveJitterTracksPreviousSample() {
	const state = new TemporalJitterState();
	const first = state.next({
		enabled: true,
		isOrthographic: false,
		width: 100,
		height: 50,
		jitterScale: 2,
	});
	const expectedFirst = computeHaltonJitterNDC(0, 100, 50, 2);
	assert.deepEqual(first, [expectedFirst[0], expectedFirst[1], 0, 0]);

	const second = state.next({
		enabled: true,
		isOrthographic: false,
		width: 100,
		height: 50,
		jitterScale: 2,
	});
	const expectedSecond = computeHaltonJitterNDC(1, 100, 50, 2);
	assert.deepEqual(second, [
		expectedSecond[0],
		expectedSecond[1],
		expectedFirst[0],
		expectedFirst[1],
	]);
}

function testOrthographicAndResetReturnZeroHistory() {
	const state = new TemporalJitterState();
	state.next({
		enabled: true,
		isOrthographic: false,
		width: 64,
		height: 64,
	});
	assert.deepEqual(
		state.next({
			enabled: true,
			isOrthographic: true,
			width: 64,
			height: 64,
		}),
		[0, 0, 0, 0]
	);

	const restarted = state.next({
		enabled: true,
		isOrthographic: false,
		width: 64,
		height: 64,
		reset: true,
	});
	const expected = computeHaltonJitterNDC(0, 64, 64, 1);
	assert.deepEqual(restarted, [expected[0], expected[1], 0, 0]);
}

testPerspectiveJitterTracksPreviousSample();
testOrthographicAndResetReturnZeroHistory();
console.log("Temporal jitter state tests passed");
