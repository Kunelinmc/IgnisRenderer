import assert from "node:assert/strict";

import { WebGLIBLPrefilterRuntime } from "../../../src/backends/webgl/WebGLIBLPrefilterRuntime.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { IBLPrefilter } from "../../../src/lights/ibl/IBLPrefilter.ts";
import { captureIBLPrefilterSourceRevision } from "../../../src/lights/ibl/IBLPrefilterExecutor.ts";

function createFakeWebGL(options = {}) {
	let nextId = 0;
	const calls = {
		attachments: [],
		deletedFramebuffers: [],
		deletedTextures: [],
		draws: 0,
		pixelStore: [],
		viewports: [],
	};
	const packState = new Map();
	const gl = {
		NO_ERROR: 0,
		INVALID_OPERATION: 0x0502,
		TEXTURE_2D: 0x0de1,
		TEXTURE0: 0x84c0,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		TEXTURE_BASE_LEVEL: 0x813c,
		TEXTURE_MAX_LEVEL: 0x813d,
		LINEAR: 0x2601,
		NEAREST: 0x2600,
		LINEAR_MIPMAP_LINEAR: 0x2703,
		CLAMP_TO_EDGE: 0x812f,
		REPEAT: 0x2901,
		RGBA: 0x1908,
		RGBA8: 0x8058,
		RGBA16F: 0x881a,
		FLOAT: 0x1406,
		HALF_FLOAT: 0x140b,
		UNSIGNED_BYTE: 0x1401,
		FRAMEBUFFER: 0x8d40,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		COLOR_ATTACHMENT0: 0x8ce0,
		SCISSOR_TEST: 0x0c11,
		BLEND: 0x0be2,
		CULL_FACE: 0x0b44,
		DEPTH_TEST: 0x0b71,
		TRIANGLES: 0x0004,
		PIXEL_PACK_BUFFER: 0x88eb,
		PIXEL_PACK_BUFFER_BINDING: 0x88ed,
		PIXEL_UNPACK_BUFFER: 0x88ec,
		PIXEL_UNPACK_BUFFER_BINDING: 0x88ef,
		PACK_ALIGNMENT: 0x0d05,
		PACK_ROW_LENGTH: 0x0d02,
		PACK_SKIP_PIXELS: 0x0d04,
		PACK_SKIP_ROWS: 0x0d03,
		UNPACK_ALIGNMENT: 0x0cf5,
		UNPACK_ROW_LENGTH: 0x0cf2,
		UNPACK_IMAGE_HEIGHT: 0x806e,
		UNPACK_SKIP_PIXELS: 0x0cf4,
		UNPACK_SKIP_ROWS: 0x0cf3,
		UNPACK_SKIP_IMAGES: 0x806d,
		getExtension(name) {
			if (options.missingExtension === name) return null;
			if (
				name === "EXT_color_buffer_float" ||
				name === "OES_texture_float_linear"
			) {
				return {};
			}
			return null;
		},
		isContextLost() {
			return options.contextLost === true;
		},
		createTexture() {
			return { type: "texture", id: ++nextId };
		},
		createFramebuffer() {
			return { type: "framebuffer", id: ++nextId };
		},
		deleteTexture(texture) {
			calls.deletedTextures.push(texture);
		},
		deleteFramebuffer(framebuffer) {
			calls.deletedFramebuffers.push(framebuffer);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D() {},
		bindFramebuffer() {},
		drawBuffers() {},
		readBuffer() {},
		framebufferTexture2D(_target, _attachment, _textarget, _texture, level) {
			calls.attachments.push(level);
		},
		checkFramebufferStatus() {
			const level = calls.attachments.at(-1);
			return options.incompleteMip === level ? 0x8cd6 : this.FRAMEBUFFER_COMPLETE;
		},
		disable() {},
		depthMask() {},
		colorMask() {},
		useProgram() {},
		bindVertexArray() {},
		activeTexture() {},
		uniform1i() {},
		uniform1f() {},
		uniform2f() {},
		viewport(x, y, width, height) {
			calls.viewports.push([x, y, width, height]);
		},
		drawArrays() {
			calls.draws++;
		},
		readPixels(_x, _y, _width, _height, _format, _type, output) {
			output.fill(2.5);
		},
		getError() {
			return options.glError ?? this.NO_ERROR;
		},
		bindBuffer() {},
		pixelStorei(parameter, value) {
			packState.set(parameter, value);
			calls.pixelStore.push([parameter, value]);
		},
		getParameter(parameter) {
			if (parameter === this.PACK_ALIGNMENT) return 4;
			if (parameter === this.PACK_ROW_LENGTH) return 7;
			if (parameter === this.PACK_SKIP_PIXELS) return 2;
			if (parameter === this.PACK_SKIP_ROWS) return 3;
			if (parameter === this.PIXEL_PACK_BUFFER_BINDING) return null;
			return 0;
		},
	};
	return { gl, calls, packState };
}

function createRuntime(options = {}) {
	const { gl, calls, packState } = createFakeWebGL(options);
	const uniform = {};
	const runtime = new WebGLIBLPrefilterRuntime({
		gl,
		programs: {
			getIBLPrefilterProgram() {
				return {
					program: {},
					uniforms: {
						environmentMap: uniform,
						outputSize: uniform,
						sourceSize: uniform,
						roughness: uniform,
						sampleCount: uniform,
						sourceIsLinear: uniform,
						sourceMipLevelCount: uniform,
					},
				};
			},
		},
		getFullscreenVao: () => ({}),
	});
	return { runtime, calls, packState };
}

function createWorkPlan() {
	return {
		baseWidth: 4,
		baseHeight: 2,
		mipLevels: [
			{ level: 0, width: 4, height: 2, roughness: 0 },
			{ level: 1, width: 2, height: 1, roughness: 0.5 },
			{ level: 2, width: 1, height: 1, roughness: 1 },
		],
	};
}

function createExecutionRequest(texture, options = {}) {
	return {
		envMap: texture,
		plan: createWorkPlan(),
		sourceRevision: captureIBLPrefilterSourceRevision(texture),
		...options,
	};
}

async function testRendersAndReadsEveryMip() {
	const { runtime, calls, packState } = createRuntime();
	const texture = new Texture({
		data: new Float32Array(4 * 2 * 4).fill(4),
		width: 4,
		height: 2,
		colorSpace: "HDR",
	});
	const completed = [];
	const result = await runtime.execute(createExecutionRequest(texture, {
		onMipComplete: (level) => completed.push(level),
	}));
	assert.deepEqual(calls.attachments, [0, 1, 2]);
	assert.deepEqual(calls.viewports, [
		[0, 0, 4, 2],
		[0, 0, 2, 1],
		[0, 0, 1, 1],
	]);
	assert.equal(calls.draws, 3);
	assert.deepEqual(completed, [0, 1, 2]);
	assert.equal(result.length, 3);
	assert.equal(result[0].data[0], 2.5);
	assert.equal(result[0].data[3], 1);
	assert.equal(calls.deletedTextures.length, 2);
	assert.equal(calls.deletedFramebuffers.length, 1);
	assert.equal(packState.get(0x0d05), 4);
	assert.equal(packState.get(0x0d02), 7);
	assert.equal(packState.get(0x0d04), 2);
	assert.equal(packState.get(0x0d03), 3);
}

async function testFailureCleansResourcesAndRestoresState() {
	const { runtime, calls } = createRuntime({ incompleteMip: 1 });
	const texture = new Texture({
		data: new Uint8Array(4 * 2 * 4).fill(255),
		width: 4,
		height: 2,
		colorSpace: "sRGB",
	});
	await assert.rejects(
		runtime.execute(createExecutionRequest(texture)),
		/incomplete at mip 1/,
	);
	assert.equal(calls.deletedTextures.length, 2);
	assert.equal(calls.deletedFramebuffers.length, 1);
}

async function testAbortBetweenMipsCleansResources() {
	const { runtime, calls } = createRuntime();
	const texture = new Texture({
		data: new Float32Array(4 * 2 * 4).fill(1),
		width: 4,
		height: 2,
		colorSpace: "HDR",
	});
	const controller = new AbortController();
	await assert.rejects(
		runtime.execute(createExecutionRequest(texture, {
			signal: controller.signal,
			onMipComplete: () => controller.abort(),
		})),
		(error) => error instanceof Error && error.name === "AbortError",
	);
	assert.equal(calls.draws, 1);
	assert.equal(calls.deletedTextures.length, 2);
	assert.equal(calls.deletedFramebuffers.length, 1);
}

function testCapabilityPreflight() {
	const { runtime } = createRuntime({
		missingExtension: "EXT_color_buffer_float",
	});
	assert.deepEqual(runtime.getAvailability(), {
		state: "unsupported",
		acceptsRequests: false,
		reason: "WebGL IBL prefilter acceleration requires EXT_color_buffer_float.",
	});
}

function createQueuedBackendServices(id, calls) {
	return {
		frame: {
			destroy() {},
		},
		iblPrefilter: {
			getAvailability: () => ({
				state: "ready",
				acceptsRequests: true,
				reason: null,
			}),
			async execute(request) {
				calls.push(id);
				return request.plan.mipLevels.map((mip) => {
					request.onMipComplete?.(mip.level);
					return {
						...mip,
						data: new Float32Array(mip.width * mip.height * 4),
					};
				});
			},
		},
		restoreContextWorkBaseline() {},
		destroy() {},
	};
}

async function testExplicitLostRequestRestoresWithNewGeneration() {
	const calls = [];
	const backend = new WebGLBackend();
	backend._contextServices = createQueuedBackendServices("old", calls);
	backend._contextWorkQueue.bindContext();
	backend._contextLost = true;
	backend._contextWorkQueue.suspend();
	const request = new IBLPrefilter({ backend }).prefilter(
		new Texture({
			data: new Float32Array([1, 1, 1, 1]),
			width: 1,
			height: 1,
			colorSpace: "HDR",
		}),
		{ acceleration: "webgl", maxMipLevels: 1 },
	);
	let settled = false;
	void request.finally(() => {
		settled = true;
	});
	await Promise.resolve();
	assert.equal(settled, false);
	backend._contextServices = createQueuedBackendServices("restored", calls);
	backend._contextLost = false;
	backend._contextWorkQueue.bindContext();
	await request;
	assert.deepEqual(calls, ["restored"]);
	backend.destroy();
}

async function testAutoFallsBackAndChangedRetainedSourceRejects() {
	const calls = [];
	const backend = new WebGLBackend();
	backend._contextServices = createQueuedBackendServices("old", calls);
	backend._contextWorkQueue.bindContext();
	backend._contextLost = true;
	backend._contextWorkQueue.suspend();
	const autoResult = await new IBLPrefilter({ backend }).prefilter(
		new Texture({
			data: new Float32Array([1, 1, 1, 1]),
			width: 1,
			height: 1,
			colorSpace: "HDR",
		}),
		{ acceleration: "auto", maxMipLevels: 1 },
	);
	assert.equal(autoResult.mipmaps.length, 1);
	assert.equal(calls.length, 0);

	const source = new Texture({
		data: new Float32Array([1, 1, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	const retained = new IBLPrefilter({ backend }).prefilter(source, {
		acceleration: "webgl",
		maxMipLevels: 1,
	});
	source.markNeedsUpdate();
	backend._contextServices = createQueuedBackendServices("restored", calls);
	backend._contextLost = false;
	backend._contextWorkQueue.bindContext();
	await assert.rejects(retained, /source changed while waiting/);
	assert.equal(calls.length, 0);
	backend.destroy();
}

await testRendersAndReadsEveryMip();
await testFailureCleansResourcesAndRestoresState();
await testAbortBetweenMipsCleansResources();
testCapabilityPreflight();
await testExplicitLostRequestRestoresWithNewGeneration();
await testAutoFallsBackAndChangedRetainedSourceRejects();

console.log("WebGL IBL prefilter runtime tests passed");
