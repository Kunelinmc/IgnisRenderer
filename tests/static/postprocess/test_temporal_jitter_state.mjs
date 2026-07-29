import assert from "node:assert/strict";

import { TemporalJitterState } from "../../../src/backends/cross/TemporalJitterState.ts";
import { TemporalFrameState } from "../../../src/backends/cross/TemporalFrameState.ts";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { computeHaltonJitterNDC } from "../../../src/maths/Misc.ts";

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

function testFrameStateSnapshotUsesCurrentAndPreviousJitter() {
	const state = new TemporalJitterState();
	const first = state.nextFrameState({
		enabled: true,
		isOrthographic: false,
		width: 128,
		height: 64,
	});
	const expectedFirst = computeHaltonJitterNDC(0, 128, 64, 1);
	assert.deepEqual(first, {
		currentJitter: [expectedFirst[0], expectedFirst[1]],
		previousJitter: [0, 0],
	});

	const second = state.nextFrameState({
		enabled: true,
		isOrthographic: false,
		width: 128,
		height: 64,
	});
	const expectedSecond = computeHaltonJitterNDC(1, 128, 64, 1);
	assert.deepEqual(second, {
		currentJitter: [expectedSecond[0], expectedSecond[1]],
		previousJitter: [expectedFirst[0], expectedFirst[1]],
	});
}

function testTemporalFrameTransactionCommitsAndRollsBack() {
	const state = new TemporalFrameState();
	const requirements = {
		cameraJitter: { sequence: "halton-2-3", scale: 1 },
	};
	const firstMatrix = new Matrix4([
		[1, 0, 0, 1],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	]);
	const secondMatrix = new Matrix4([
		[1, 0, 0, 2],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	]);
	const camera = {
		type: CameraType.Perspective,
		viewProjectionMatrix: firstMatrix,
	};
	const first = state.beginFrame({
		camera,
		width: 100,
		height: 50,
		frameRequirements: requirements,
	});
	state.abortFrame();
	const retry = state.beginFrame({
		camera,
		width: 100,
		height: 50,
		frameRequirements: requirements,
	});
	assert.deepEqual(retry.jitterCurrentPrev, first.jitterCurrentPrev);
	assert.equal(retry.previousViewProjection, null);
	state.commitFrame();

	camera.viewProjectionMatrix = secondMatrix;
	const second = state.beginFrame({
		camera,
		width: 100,
		height: 50,
		frameRequirements: requirements,
	});
	assert.deepEqual(
		second.previousViewProjection?.elements,
		firstMatrix.elements,
	);
	state.abortFrame();
	const afterAbort = state.beginFrame({
		camera,
		width: 100,
		height: 50,
		frameRequirements: {},
	});
	assert.deepEqual(
		afterAbort.previousViewProjection?.elements,
		firstMatrix.elements,
	);
	assert.deepEqual(afterAbort.jitterCurrentPrev, [0, 0, 0, 0]);
	state.abortFrame();
}

testPerspectiveJitterTracksPreviousSample();
testOrthographicAndResetReturnZeroHistory();
testFrameStateSnapshotUsesCurrentAndPreviousJitter();
testTemporalFrameTransactionCommitsAndRollsBack();
console.log("Temporal jitter state tests passed");
