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

function run() {
	testFXAAPassUsesLatestPostSourceAndRebindsPostTarget();
	console.log("WebGL FXAA frame executor tests passed");
}

run();
