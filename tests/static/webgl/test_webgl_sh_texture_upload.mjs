import assert from "node:assert/strict";
import {
	WebGLFrameServiceTestHarness as WebGLFrameExecutor,
} from "../../helpers/webgl-frame-services.mjs";
import { Logger } from "../../../src/foundation/Logger.ts";

function createFakeGL() {
	let textureId = 0;
	const texImage2DCalls = [];
	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		MAX_TEXTURE_IMAGE_UNITS: 0x8872,
		TEXTURE_2D: 0x0de1,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		CLAMP_TO_EDGE: 0x812f,
		NEAREST: 0x2600,
		RGBA: 0x1908,
		RGBA32F: 0x8814,
		FLOAT: 0x1406,
		getParameter(param) {
			if (
				param === this.MAX_TEXTURE_SIZE ||
				param === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			if (param === this.MAX_TEXTURE_IMAGE_UNITS) {
				return 16;
			}
			return 0;
		},
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		createTexture() {
			return { id: `tex-${++textureId}` };
		},
		deleteTexture() {},
		bindTexture() {},
		texParameteri() {},
		texImage2D(_target, _level, _internalFormat, width, height) {
			texImage2DCalls.push({ width, height });
		},
		texImage2DCalls,
	};
}

function testUploadSHAmbientCoefficients() {
	const warnings = [];
	const executor = new WebGLFrameExecutor(createFakeGL());
	Logger.configure({
		level: "warn",
		sink: {
			warn: (...args) => {
				warnings.push(args.map((arg) => String(arg)).join(" "));
			},
		},
		resetOnceKeys: true,
	});
	try {
		const coeffs = Array.from({ length: 16 }, (_, index) => ({
			r: index + 1,
			g: index + 2,
			b: index + 3,
		}));

		const uploaded = executor._uploadSHAmbientCoefficients(coeffs);
		assert.equal(uploaded, true);
		assert.ok(executor._shAmbientTexture);
		assert.equal(executor._shAmbientTextureWidth, 16);
		assert.equal(executor._shAmbientTextureHeight, 1);
		assert.equal(warnings.length, 0);
	} finally {
		Logger.reset();
	}
}

function testUploadLocalLightProbeCoefficients() {
	const executor = new WebGLFrameExecutor(createFakeGL());
	const probes = [
		{
			sh: Array.from({ length: 16 }, (_, index) => ({
				r: index + 1,
				g: index + 2,
				b: index + 3,
			})),
		},
		{
			sh: Array.from({ length: 16 }, (_, index) => ({
				r: index + 4,
				g: index + 5,
				b: index + 6,
			})),
		},
	];

	const uploaded = executor._uploadLocalLightProbeCoefficients(probes);
	assert.equal(uploaded, true);
	assert.ok(executor._localLightProbeSHTexture);
	assert.equal(executor._localLightProbeSHTextureWidth, 16);
	assert.equal(executor._localLightProbeSHTextureHeight, 2);
	assert.deepEqual(
		executor._gl.texImage2DCalls.at(-1),
		{ width: 16, height: 2 }
	);
}

function run() {
	testUploadSHAmbientCoefficients();
	testUploadLocalLightProbeCoefficients();
	console.log("WebGL SH texture upload tests passed");
}

run();
