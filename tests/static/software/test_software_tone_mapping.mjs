import assert from "node:assert/strict";
import { SoftwareToneMappingImplementation } from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFrameContext(pixels) {
	return {
		attachments: {
			width: 3,
			height: 1,
			pixels,
		},
		features: {},
		postProcess: createResolvedPostProcess({
			tonemap: { enabled: true },
		}),
		incremental: {
			enabled: true,
			forceFullFrame: false,
			dirtyRects: [{ x: 1, y: 0, width: 1, height: 1 }],
		},
	};
}

function testToneMappingPreservesAlphaAndDirtyRect() {
	const pixels = new Uint8ClampedArray([
		255, 128, 64, 17,
		128, 160, 192, 99,
		64, 128, 255, 31,
	]);
	const before = Array.from(pixels);

	const implementation = new SoftwareToneMappingImplementation();
	implementation.execute({
		frameContext: createFrameContext(pixels),
	});

	assert.deepEqual(Array.from(pixels.slice(0, 4)), before.slice(0, 4));
	assert.deepEqual(Array.from(pixels.slice(8, 12)), before.slice(8, 12));
	assert.notDeepEqual(Array.from(pixels.slice(4, 7)), before.slice(4, 7));
	assert.equal(pixels[7], before[7]);
}

function run() {
	testToneMappingPreservesAlphaAndDirtyRect();
	console.log("Software tone mapping tests passed");
}

run();
