import assert from "node:assert/strict";

import {
	SoftwareColorFilterImplementation,
	SoftwareGammaImplementation,
} from "../../../src/postprocess/index.ts";
import { SoftwareFastApproximateAntiAliasingImplementation } from "../../../src/postprocess/passes/FastApproximateAntiAliasingPass.ts";
import { SoftwarePostProcessExecutor } from "../../../src/backends/software/SoftwarePostProcessExecutor.ts";
import { SoftwareParticlePass } from "../../../src/backends/software/passes/SoftwareParticlePass.ts";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { ParticleBlendMode } from "../../../src/particles/index.ts";

function createRequest(color, options = {}) {
	return {
		frameContext: {
			attachments: {
				pixels: new Uint8ClampedArray(color.length),
				width: color.length / 4,
				height: 1,
			},
		},
		options,
	};
}

function createResources(color) {
	return { color: { input: color, output: color } };
}

function testHDRDomainBuiltInsPreserveRadiance() {
	const colorFilterColor = new Float32Array([4, 2, 1, 1]);
	const colorFilter = new SoftwareColorFilterImplementation();
	const colorResult = colorFilter.execute(
		createRequest(colorFilterColor),
		{
			dirtyRects: [{ minX: 0, minY: 0, maxX: 0, maxY: 0 }],
			displayOutput: {
				requested: { mode: "hdr", exposure: 1, hdrHeadroom: 6 },
				activeDynamicRange: "hdr",
				colorSpace: "display-p3",
			},
			resources: createResources(colorFilterColor),
		},
	);
	assert.equal(colorResult.ran, true);
	assert.equal(colorFilterColor[0], 4);

	const fxaaColor = new Float32Array([
		4, 2, 1, 1,
		4, 2, 1, 1,
		4, 2, 1, 1,
	]);
	const fxaa = new SoftwareFastApproximateAntiAliasingImplementation();
	const fxaaResult = fxaa.execute(createRequest(fxaaColor), {
		attachments: { width: 3, height: 1 },
		canvasContext: null,
		dirtyRects: [{ minX: 0, minY: 0, maxX: 2, maxY: 0 }],
		resources: createResources(fxaaColor),
	});
	assert.equal(fxaaResult.ran, true);
	assert.equal(fxaaColor[0], 4);

	const gammaColor = new Float32Array([4, 2, 1, 1]);
	const gamma = new SoftwareGammaImplementation();
	const gammaResult = gamma.execute(createRequest(gammaColor), {
		dirtyRects: [{ minX: 0, minY: 0, maxX: 0, maxY: 0 }],
		displayOutput: {
			requested: { mode: "hdr", exposure: 1, hdrHeadroom: 6 },
			activeDynamicRange: "hdr",
			colorSpace: "display-p3",
		},
		resources: createResources(gammaColor),
	});
	assert.equal(gammaResult.ran, true);
	assert.ok(gammaColor[0] > 1);
}

function testAuthoritativeColorResourceAndDomainTracking() {
	const color = new Float32Array([4, 2, 1, 1]);
	const executor = new SoftwarePostProcessExecutor(() => color, () => ({
		requested: { mode: "hdr", exposure: 1, hdrHeadroom: 4 },
		activeDynamicRange: "hdr",
		colorSpace: "display-p3",
	}));
	const frame = {
		attachments: {
			color,
			pixels: new Uint8ClampedArray(4),
			depthBuffer: new Float32Array(1),
			normalBuffer: null,
			motionBuffer: null,
			width: 1,
			height: 1,
		},
		clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 1, maxYExclusive: 1 }],
	};
	executor.bindSoftwareFrame(frame);
	const frameContext = {
		attachments: {
			pixels: frame.attachments.pixels,
			depthBuffer: frame.attachments.depthBuffer,
			width: 1,
			height: 1,
		},
	};
	const bridge = executor.createGBufferBridge(frameContext);
	assert.equal(bridge.channels.color.format, "rgba32float");
	assert.equal(bridge.channels.color.handle.data, color);
	executor.bindSoftwareFrame(frame, "display-linear");
	assert.equal(executor.outputColorDomain, "display-linear");
	executor.unbindSoftwareFrame();
	executor.bindSoftwareFrame(frame);

	const tonePass = {
		colorContract: { input: "scene-linear-hdr", output: "display-linear" },
	};
	const request = {
		frameContext,
		pass: tonePass,
		passId: "custom",
		declaration: { color: { access: "read-write", output: "preserve" } },
		gBuffer: bridge,
		histories: {},
		transients: {},
	};
	const context = executor.createPassExecutionContext(request);
	assert.equal(context.resources.color.input, color);
	assert.notEqual(context.resources.color.input, frameContext.attachments.pixels);

	executor.completePass(request, { ran: false });
	assert.equal(executor.outputColorDomain, "scene-linear-hdr");
	executor.completePass(request, { ran: true });
	assert.equal(executor.outputColorDomain, "display-linear");
	const gammaRequest = {
		...request,
		pass: {
			colorContract: { input: "display-linear", output: "display-encoded" },
		},
	};
	executor.completePass(gammaRequest, { ran: true });
	assert.equal(executor.outputColorDomain, "display-encoded");
	executor.unbindSoftwareFrame();
	executor.bindSoftwareFrame(frame);
	assert.equal(executor.outputColorDomain, "scene-linear-hdr");
	executor.unbindSoftwareFrame();
}

function testAdditiveParticlesCanExceedOne() {
	const color = new Float32Array(3 * 3 * 4);
	const particles = Array.from({ length: 5 }, () => ({
		position: { x: 0, y: 0, z: -1 },
		size: 1,
		color: { r: 255, g: 255, b: 255, a: 1 },
		rotation: 0,
		depth: 1,
		uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
	}));
	new SoftwareParticlePass().render({
		frame: {
			attachments: {
				color,
				depthBuffer: new Float32Array(9).fill(Infinity),
				width: 3,
				height: 3,
			},
			camera: {
				type: CameraType.Perspective,
				fov: 60,
				viewMatrix: Matrix4.identity(),
				projectionMatrix: Matrix4.identity(),
			},
			features: { enableShadows: false },
			scene: { lights: [] },
			clipRegions: [{ minX: 0, minY: 0, maxXExclusive: 3, maxYExclusive: 3 }],
		},
		services: {
			particles: {
				batches: [{
					systemId: "hdr-additive",
					blendMode: ParticleBlendMode.Additive,
					texture: null,
					receiveShadows: false,
					castShadows: false,
					shadowDensity: 1,
					shadowSoftness: 0,
					particles,
				}],
				meshBatches: [],
			},
			shadow: { sampler: () => ({ r: 1, g: 1, b: 1 }) },
		},
	});
	const center = (1 * 3 + 1) << 2;
	assert.ok(color[center] > 4);
}

function run() {
	testHDRDomainBuiltInsPreserveRadiance();
	testAuthoritativeColorResourceAndDomainTracking();
	testAdditiveParticlesCanExceedOne();
	console.log("Software HDR pipeline tests passed");
}

run();
