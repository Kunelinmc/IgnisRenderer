import assert from "node:assert/strict";
import { SoftwareToneMappingImplementation } from "../../../src/postprocess/index.ts";
import { applyACESToneMapping } from "../../../src/postprocess/passes/ToneMappingPass.ts";
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
	const pixels = new Float32Array([
		4, 0.5, 0.25, 0.1,
		2, 1.5, 1.25, 0.4,
		0.25, 0.5, 4, 0.2,
	]);
	const before = Array.from(pixels);

	const implementation = new SoftwareToneMappingImplementation();
	implementation.execute({
		frameContext: createFrameContext(pixels),
	}, {
		dirtyRects: [{ minX: 1, minY: 0, maxX: 1, maxY: 0 }],
		displayOutput: {
			requested: { mode: "sdr", exposure: 1, hdrHeadroom: 4 },
			activeDynamicRange: "sdr",
			colorSpace: "srgb",
		},
		resources: {
			color: { input: pixels, output: pixels },
		},
	});

	assert.deepEqual(Array.from(pixels.slice(0, 4)), before.slice(0, 4));
	assert.deepEqual(Array.from(pixels.slice(8, 12)), before.slice(8, 12));
	assert.notDeepEqual(Array.from(pixels.slice(4, 7)), before.slice(4, 7));
	assert.equal(pixels[7], before[7]);
	assert.ok(Math.abs(pixels[5] / pixels[4] - 0.75) < 1e-7);
	assert.ok(Math.abs(pixels[6] / pixels[4] - 0.625) < 1e-7);
	assert.ok(Math.abs(pixels[4] - 0.3941102756892233) < 1e-7);
}

function testSDRPreservesEmissiveColorRatiosUnderAttenuationAndExposure() {
	const colors = [
		[45 / 255 * 6, 205 / 255 * 6, 6],
		[5, 115 / 255 * 5, 45 / 255 * 5],
		[0, 3, 1],
	];
	for (const color of colors) {
		const sourcePeak = Math.max(...color);
		for (const attenuation of [1, 0.25, 0.01, 0.001]) {
			for (const exposure of [0.01, 1, 8]) {
				const input = color.map((channel) => channel * attenuation);
				const mapped = applyACESToneMapping(input, exposure);
				const mappedPeak = Math.max(...mapped);
				assert.ok(mappedPeak > 0 && mappedPeak <= 1);
				for (let channel = 0; channel < 3; channel++) {
					assert.ok(Math.abs(
						mapped[channel] / mappedPeak - color[channel] / sourcePeak,
					) < 1e-12, `channel ${channel} changed hue at ${attenuation}, ${exposure}`);
				}
			}
		}
	}
}

function testSDRRetainsGrayscaleCurveAndHandlesZeroAndNegativeChannels() {
	for (const [input, expected] of [
		[0, 0],
		[0.1, 0.1258396775638155],
		[0.5, 0.6163069544364508],
		[1, 0.8037974683544302],
		[4, 0.9734171097148379],
		[16, 1],
	]) {
		const mapped = applyACESToneMapping([input, input, input], 1);
		assert.ok(mapped.every((channel) => Math.abs(channel - expected) < 1e-12));
		const exposed = applyACESToneMapping([input / 4, input / 4, input / 4], 4);
		assert.deepEqual(exposed, mapped);
	}
	assert.deepEqual(applyACESToneMapping([4, 2, 1], 0), [0, 0, 0]);
	assert.deepEqual(applyACESToneMapping([-1, -2, 0], 1), [0, 0, 0]);
	const inputAndOutput = [-1, 4, 2];
	assert.equal(applyACESToneMapping(inputAndOutput, 1, inputAndOutput), inputAndOutput);
	assert.equal(inputAndOutput[0], 0);
	assert.ok(Math.abs(inputAndOutput[2] / inputAndOutput[1] - 0.5) < 1e-12);
}

function run() {
	testSDRPreservesEmissiveColorRatiosUnderAttenuationAndExposure();
	testSDRRetainsGrayscaleCurveAndHandlesZeroAndNegativeChannels();
	testToneMappingPreservesAlphaAndDirtyRect();
	console.log("Software tone mapping tests passed");
}

run();
