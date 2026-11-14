import assert from "node:assert/strict";
import {
	WebGLFrameServiceTestHarness as WebGLFrameExecutor,
} from "../../helpers/webgl-frame-services.mjs";

function createFakeGL() {
	let textureId = 0;
	const texImage2DCalls = [];
	const createdTextures = [];
	const deletedTextures = [];
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
			const texture = { id: `tex-${++textureId}` };
			createdTextures.push(texture);
			return texture;
		},
		deleteTexture(texture) {
			deletedTextures.push(texture);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D(_target, _level, _internalFormat, width, height,
			_border, _format, _type, data) {
			texImage2DCalls.push({ width, height, data: Array.from(data) });
		},
		texImage2DCalls,
		createdTextures,
		deletedTextures,
	};
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
	probes[0].sh[0] = { r: 1024, g: -4, b: 0.5 };

	const uploaded = executor._probeSHTextures.uploadLocalLightProbeCoefficients(
		probes
	);
	assert.equal(uploaded, true);
	assert.ok(executor._probeSHTextures.localLightProbeSHTexture);
	assert.equal(
		executor._probeSHTextures.localLightProbeSHTextureWidth,
		16
	);
	assert.equal(
		executor._probeSHTextures.localLightProbeSHTextureHeight,
		2
	);
	const gl = executor._gl;
	const texture = executor._probeSHTextures.localLightProbeSHTexture;
	const firstUpload = gl.texImage2DCalls.at(-1);
	assert.equal(firstUpload.width, 16);
	assert.equal(firstUpload.height, 2);
	assert.equal(firstUpload.data.length, 128);
	assert.deepEqual(firstUpload.data.slice(0, 4), [1024, -4, 0.5, 0]);
	assert.deepEqual(firstUpload.data.slice(64, 68), [4, 5, 6, 0]);
	assert.equal(gl.createdTextures.length, 1);

	probes[1].sh[0].r = 42;
	assert.equal(executor._probeSHTextures.uploadLocalLightProbeCoefficients(probes), true);
	assert.equal(gl.createdTextures.length, 1, "Repeated uploads must reuse the local probe texture");
	assert.equal(gl.texImage2DCalls.at(-1).data[64], 42);
	executor._probeSHTextures.destroy();
	executor._probeSHTextures.destroy();
	assert.deepEqual(gl.deletedTextures, [texture], "The local probe texture must be deleted once");
}

function run() {
	testUploadLocalLightProbeCoefficients();
	console.log("WebGL SH texture upload tests passed");
}

run();
