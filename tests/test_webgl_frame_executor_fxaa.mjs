import assert from "node:assert/strict";
import { WebGLFrameExecutor } from "../src/renderers/webgl/WebGLFrameExecutor.ts";

function createFXAATestGL() {
	const calls = [];
	return {
		calls,
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		FRAMEBUFFER: 0x8d40,
		COLOR_ATTACHMENT0: 0x8ce0,
		TEXTURE_2D: 0x0de1,
		TEXTURE0: 0x84c0,
		TRIANGLES: 0x0004,
		CULL_FACE: 0x0b44,
		DEPTH_TEST: 0x0b71,
		BLEND: 0x0be2,
		getParameter(parameter) {
			if (
				parameter === this.MAX_TEXTURE_SIZE ||
				parameter === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			return 0;
		},
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		deleteFramebuffer() {},
		deleteTexture() {},
		deleteRenderbuffer() {},
		bindFramebuffer(target, framebuffer) {
			calls.push({ name: "bindFramebuffer", target, framebuffer });
		},
		framebufferTexture2D(target, attachment, textarget, texture, level) {
			calls.push({
				name: "framebufferTexture2D",
				target,
				attachment,
				textarget,
				texture,
				level,
			});
		},
		drawBuffers(buffers) {
			calls.push({ name: "drawBuffers", buffers: [...buffers] });
		},
		viewport(x, y, width, height) {
			calls.push({ name: "viewport", x, y, width, height });
		},
		useProgram(program) {
			calls.push({ name: "useProgram", program });
		},
		bindVertexArray(vao) {
			calls.push({ name: "bindVertexArray", vao });
		},
		disable(cap) {
			calls.push({ name: "disable", cap });
		},
		activeTexture(unit) {
			calls.push({ name: "activeTexture", unit });
		},
		bindTexture(target, texture) {
			calls.push({ name: "bindTexture", target, texture });
		},
		uniform1i(location, value) {
			calls.push({ name: "uniform1i", location, value });
		},
		uniform2f(location, x, y) {
			calls.push({ name: "uniform2f", location, x, y });
		},
		drawArrays(mode, first, count) {
			calls.push({ name: "drawArrays", mode, first, count });
		},
	};
}

function createFrameTargetTestGL(options = {}) {
	const frameStatuses =
		options.frameStatuses ?? [0x8cd5, 0x8cd5];
	let statusIndex = 0;
	let textureId = 0;
	let framebufferId = 0;
	let renderbufferId = 0;

	const createdTextures = [];
	const deletedTextures = [];
	const createdFramebuffers = [];
	const deletedFramebuffers = [];
	const createdRenderbuffers = [];
	const deletedRenderbuffers = [];
	const texImage2DCalls = [];

	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		FRAMEBUFFER: 0x8d40,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		FRAMEBUFFER_INCOMPLETE_ATTACHMENT: 0x8cd6,
		RENDERBUFFER: 0x8d41,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		COLOR_ATTACHMENT2: 0x8ce2,
		DEPTH_ATTACHMENT: 0x8d00,
		DEPTH_COMPONENT24: 0x81a6,
		TEXTURE_2D: 0x0de1,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		LINEAR: 0x2601,
		CLAMP_TO_EDGE: 0x812f,
		RGBA: 0x1908,
		RGBA8: 0x8058,
		RGBA16F: 0x881a,
		HALF_FLOAT: 0x140b,
		UNSIGNED_BYTE: 0x1401,
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		getParameter(parameter) {
			if (
				parameter === this.MAX_TEXTURE_SIZE ||
				parameter === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			return 0;
		},
		getExtension(name) {
			if (name === "EXT_color_buffer_float") {
				return options.floatExtension ? {} : null;
			}
			return null;
		},
		createTexture() {
			const texture = { id: `tex-${++textureId}` };
			createdTextures.push(texture);
			return texture;
		},
		deleteTexture(texture) {
			deletedTextures.push(texture);
		},
		createFramebuffer() {
			const framebuffer = { id: `fbo-${++framebufferId}` };
			createdFramebuffers.push(framebuffer);
			return framebuffer;
		},
		deleteFramebuffer(framebuffer) {
			deletedFramebuffers.push(framebuffer);
		},
		createRenderbuffer() {
			const renderbuffer = { id: `rbo-${++renderbufferId}` };
			createdRenderbuffers.push(renderbuffer);
			return renderbuffer;
		},
		deleteRenderbuffer(renderbuffer) {
			deletedRenderbuffers.push(renderbuffer);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D(_target, _level, internalFormat, width, height, _border, _format, type) {
			texImage2DCalls.push({ internalFormat, type, width, height });
		},
		bindRenderbuffer() {},
		renderbufferStorage() {},
		bindFramebuffer() {},
		framebufferTexture2D() {},
		framebufferRenderbuffer() {},
		drawBuffers() {},
		checkFramebufferStatus() {
			const index = Math.min(statusIndex, frameStatuses.length - 1);
			statusIndex++;
			return frameStatuses[index];
		},
		createdTextures,
		deletedTextures,
		createdFramebuffers,
		deletedFramebuffers,
		createdRenderbuffers,
		deletedRenderbuffers,
		texImage2DCalls,
	};
}

function testFXAAPassUsesLatestPostSourceAndRebindsPostTarget() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	const sceneColor = { id: "scene-color" };
	const taaHistory = { id: "taa-history" };
	const postColor = { id: "post-color" };
	const postFramebuffer = { id: "post-fbo" };
	const fullscreenVao = { id: "fullscreen-vao" };

	executor._programs = {
		getFXAAProgram() {
			return {
				program: { id: "fxaa-program" },
				uniforms: {
					sourceMap: { id: "uSourceMap" },
					texelSize: { id: "uTexelSize" },
				},
			};
		},
	};
	executor._sceneColorTexture = sceneColor;
	executor._presentSourceTexture = taaHistory;
	executor._postColorTexture = postColor;
	executor._postFramebuffer = postFramebuffer;
	executor._fullscreenVao = fullscreenVao;
	executor._width = 1280;
	executor._height = 720;

	executor._applyFXAA();

	const attachmentWrite = gl.calls.find(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT0
	);
	assert.equal(attachmentWrite?.texture, postColor);

	const sourceBindCalls = gl.calls.filter(
		(call) => call.name === "bindTexture" && call.target === gl.TEXTURE_2D
	);
	const sourceBind = sourceBindCalls[sourceBindCalls.length - 1];
	assert.equal(sourceBind?.texture, taaHistory);

	assert.equal(executor._presentSourceTexture, postColor);
}

function testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension() {
	const warnings = [];
	const gl = createFrameTargetTestGL({ floatExtension: false });
	const executor = new WebGLFrameExecutor(gl, (key, message) =>
		warnings.push({ key, message })
	);

	executor._ensureFrameTargets(320, 180);

	assert.equal(
		gl.texImage2DCalls.some((call) => call.internalFormat === gl.RGBA16F),
		false
	);
	assert.ok(
		warnings.some(
			(warning) => warning.key === "webgl-motion-float-unsupported"
		)
	);
}

function testSceneFramebufferFailureCleansAllAllocatedTargets() {
	const gl = createFrameTargetTestGL({
		floatExtension: true,
		frameStatuses: [0x8cd6],
	});
	const executor = new WebGLFrameExecutor(gl, () => {});

	assert.throws(
		() => executor._ensureFrameTargets(256, 256),
		/WebGL scene framebuffer is incomplete/
	);
	assert.equal(gl.deletedTextures.length, gl.createdTextures.length);
	assert.equal(gl.deletedFramebuffers.length, gl.createdFramebuffers.length);
	assert.equal(gl.deletedRenderbuffers.length, gl.createdRenderbuffers.length);
}

function testEndFramePrunesStaleModelMatrixCache() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});

	executor._modelMatrixCache.set("keep", new Float32Array(16));
	executor._modelMatrixCache.set("drop", new Float32Array(16));
	executor._modelMatrixKeysThisFrame.add("keep");
	executor._presentedInFrame = true;
	executor.endFrame();

	assert.equal(executor._modelMatrixCache.has("keep"), true);
	assert.equal(executor._modelMatrixCache.has("drop"), false);
}

function testShadowSkinningWarningKeyIsStable() {
	const warnings = [];
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, (key, message) =>
		warnings.push({ key, message })
	);

	executor._drawShadowPacket(
		{ uniforms: { mvp: null } },
		{ meshInstance: { id: "mesh-a", skeleton: {} } },
		{}
	);

	assert.equal(warnings[0]?.key, "webgl-shadow-skinning-unsupported");
}

function run() {
	testFXAAPassUsesLatestPostSourceAndRebindsPostTarget();
	testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension();
	testSceneFramebufferFailureCleansAllAllocatedTargets();
	testEndFramePrunesStaleModelMatrixCache();
	testShadowSkinningWarningKeyIsStable();
	console.log("WebGL FXAA frame executor tests passed");
}

run();
