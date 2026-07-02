import assert from "node:assert/strict";
import { IrradianceProbeGrid } from "../../../src/lights/IrradianceProbeGrid.ts";
import { Matrix3 } from "../../../src/maths/Matrix3.ts";
import { SH } from "../../../src/maths/SH.ts";
import {
	computeIrradianceProbeGridRawWeight,
	sampleActiveIrradianceProbeGrid,
	sampleIrradianceProbeGrid,
	selectActiveIrradianceProbeGrid,
} from "../../../src/lights/runtime/irradianceProbeGridRuntime.ts";

function createSH(r, g = r, b = r) {
	const sh = SH.empty();
	sh[0] = { r, g, b };
	return sh;
}

function createGrid({
	id = "grid",
	dimensions = { x: 2, y: 2, z: 2 },
	halfExtents = { x: 1, y: 1, z: 1 },
	blendDistance = 0.1,
	priority = 0,
	position = { x: 0, y: 0, z: 0 },
} = {}) {
	const grid = new IrradianceProbeGrid({
		dimensions,
		halfExtents,
		blendDistance,
		priority,
	});
	grid.id = id;
	grid.position.set(position.x, position.y, position.z);
	grid.updateWorldMatrix();
	grid.markRuntimeDirty();
	return grid;
}

function testConstructorIndexingAndValidity() {
	assert.throws(
		() => new IrradianceProbeGrid(),
		/IrradianceProbeGridParams object/
	);
	assert.throws(
		() => new IrradianceProbeGrid({ dimensions: { x: 17, y: 16, z: 1 } }),
		/supports at most 256 cells/
	);

	const grid = new IrradianceProbeGrid({
		dimensions: { x: 2.9, y: 0, z: Number.NaN },
		halfExtents: { x: -2, y: 0, z: Number.NaN },
		blendDistance: -1,
		priority: 3.9,
		source: "capturedScene",
		captureResolution: { width: 32, height: 16 },
		includeMeshes: false,
	});
	assert.deepEqual(grid.dimensions, { x: 2, y: 1, z: 1 });
	assert.equal(grid.getCellIndex(1, 0, 0), 1);
	assert.equal(grid.priority, 3);
	assert.equal(grid.source, "capturedScene");
	assert.equal(grid.captureResolution.width, 32);
	assert.equal(grid.includeMeshes, false);
	assert.equal(grid.isCellValid(0), false);

	grid.setCellSH(1, createSH(8, 4, 2));
	assert.equal(grid.isCellValid({ x: 1, y: 0, z: 0 }), true);
	assert.deepEqual(grid.getCellSH(1)[0], { r: 8, g: 4, b: 2 });
	const textureRevision = grid.textureRevision;
	grid.clearCell(1);
	assert.equal(grid.isCellValid(1), false);
	assert.ok(grid.textureRevision > textureRevision);
}

function testMutableSHReferencesAdvanceRevisions() {
	const grid = new IrradianceProbeGrid({
		dimensions: { x: 2, y: 1, z: 1 },
	});
	grid.setCellSH(0, createSH(1));

	const firstTextureRevision = grid.textureRevision;
	const firstCaptureRevision = grid.captureRevision;
	grid.getCellSH(0)[0].r = 9;
	assert.equal(grid.getCellSH(0)[0].r, 9);
	assert.ok(grid.textureRevision > firstTextureRevision);
	assert.ok(grid.captureRevision > firstCaptureRevision);
	assert.equal(grid.getRuntimeCache().textureRevision, grid.textureRevision);

	const secondTextureRevision = grid.textureRevision;
	const secondCaptureRevision = grid.captureRevision;
	grid.sh[0][0].g = 6;
	assert.equal(grid.getCellSH(0)[0].g, 6);
	assert.ok(grid.textureRevision > secondTextureRevision);
	assert.ok(grid.captureRevision > secondCaptureRevision);
	assert.equal(grid.getRuntimeCache().textureRevision, grid.textureRevision);

	const thirdTextureRevision = grid.textureRevision;
	grid.sh[1] = createSH(4);
	assert.equal(grid.getCellSH(1)[0].r, 4);
	assert.equal(grid.isCellValid(1), true);
	assert.ok(grid.textureRevision > thirdTextureRevision);
	assert.equal(grid.getRuntimeCache().textureRevision, grid.textureRevision);
}

function testCloneAndRuntimeCache() {
	const grid = createGrid({
		dimensions: { x: 3, y: 2, z: 1 },
		halfExtents: { x: 3, y: 2, z: 1 },
		position: { x: 5, y: 0, z: 0 },
	});
	grid.setCellSH({ x: 2, y: 1, z: 0 }, createSH(12));
	grid.requestCapture({ x: 2, y: 1, z: 0 });

	const cache = grid.getRuntimeCache();
	assert.equal(cache.cellCount, 6);
	assert.ok(cache.worldToGrid3x3 instanceof Matrix3);
	assert.equal(cache.worldToGrid3x3.elements[0][0], 1);
	assert.equal(cache.cellWorldPositions[0].x, 2);
	assert.equal(cache.cellWorldPositions[5].x, 8);
	assert.equal(cache.validMask[grid.getCellIndex(2, 1, 0)], 1);

	const cloned = grid.clone(false);
	assert.equal(cloned.getCellSH({ x: 2, y: 1, z: 0 })[0].r, 12);
	assert.equal(
		cloned.getCellCaptureRequestToken(grid.getCellIndex(2, 1, 0)),
		grid.captureRequestToken
	);
	cloned.setCellSH(0, createSH(99));
	assert.notEqual(cloned.getCellSH(0)[0].r, grid.getCellSH(0)[0].r);
}

function testTrilinearSamplingAndInvalidNormalization() {
	const grid = createGrid();
	for (let z = 0; z < 2; z++) {
		for (let y = 0; y < 2; y++) {
			for (let x = 0; x < 2; x++) {
				const index = grid.getCellIndex(x, y, z);
				grid.setCellSH(index, createSH(index + 1));
			}
		}
	}

	const center = sampleIrradianceProbeGrid(grid, { x: 0, y: 0, z: 0 });
	assert.equal(center.coverage, 1);
	assert.ok(Math.abs(center.sh[0].r - 4.5) < 1e-6);

	for (let i = 1; i < 8; i++) {
		grid.clearCell(i);
	}
	const singleValid = sampleIrradianceProbeGrid(grid, { x: 0, y: 0, z: 0 });
	assert.equal(singleValid.coverage, 1);
	assert.equal(singleValid.sh[0].r, 1);

	grid.clearCell(0);
	const allInvalid = sampleIrradianceProbeGrid(grid, { x: 0, y: 0, z: 0 });
	assert.equal(allInvalid.coverage, 0);
	assert.equal(allInvalid.sh[0].r, 0);

	grid.setCellSH(0, createSH(5));
	const outside = sampleIrradianceProbeGrid(grid, { x: 4, y: 0, z: 0 });
	assert.equal(outside.coverage, 0);
}

function testCoverageCurveAndActiveSelection() {
	assert.equal(computeIrradianceProbeGridRawWeight(0.5, 0.1), 1);
	assert.equal(computeIrradianceProbeGridRawWeight(2, 0.1), 0);

	const low = createGrid({
		id: "low",
		priority: 1,
		position: { x: 0, y: 0, z: 0 },
	});
	const near = createGrid({
		id: "near",
		priority: 5,
		position: { x: 1, y: 0, z: 0 },
	});
	const far = createGrid({
		id: "far",
		priority: 5,
		position: { x: 10, y: 0, z: 0 },
	});
	const selected = selectActiveIrradianceProbeGrid(
		[low, far, near],
		{ x: 0, y: 0, z: 0 }
	);
	assert.equal(selected?.id, "near");

	near.setCellSH(0, createSH(7));
	const activeSample = sampleActiveIrradianceProbeGrid(
		[low, far, near],
		{ x: 1, y: 0, z: 0 },
		{ x: 0, y: 0, z: 0 }
	);
	assert.equal(activeSample.grid?.id, "near");
	assert.equal(activeSample.sh[0].r, 7);
}

function run() {
	testConstructorIndexingAndValidity();
	testMutableSHReferencesAdvanceRevisions();
	testCloneAndRuntimeCache();
	testTrilinearSamplingAndInvalidNormalization();
	testCoverageCurveAndActiveSelection();
	console.log("Irradiance probe grid runtime tests passed");
}

run();
