import assert from "node:assert/strict";
import { WebGLCustomRenderTargetRuntime } from "../../../src/backends/webgl/WebGLCustomRenderTargetRuntime.ts";
import {
	BufferUsage,
	PrimitiveTopology,
	TextureFormat,
} from "../../../src/backends/types.ts";
import {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../../../src/rendering/CustomRenderTargets.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFakeGL() {
	let nextId = 1;
	const calls = [];
	const gl = {
		calls,
		FRAMEBUFFER: 0x8d40,
		RENDERBUFFER: 0x8d41,
		TEXTURE_2D: 0x0de1,
		ARRAY_BUFFER: 0x8892,
		ELEMENT_ARRAY_BUFFER: 0x8893,
		STATIC_DRAW: 0x88e4,
		COLOR_ATTACHMENT0: 0x8ce0,
		DEPTH_ATTACHMENT: 0x8d00,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		DEPTH_COMPONENT24: 0x81a6,
		RGBA8: 0x8058,
		RGBA16F: 0x881a,
		RGBA: 0x1908,
		UNSIGNED_BYTE: 0x1401,
		HALF_FLOAT: 0x140b,
		FLOAT: 0x1406,
		UNSIGNED_SHORT: 0x1403,
		UNSIGNED_INT: 0x1405,
		LINEAR: 0x2601,
		NEAREST: 0x2600,
		CLAMP_TO_EDGE: 0x812f,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		POINTS: 0x0000,
		LINES: 0x0001,
		TRIANGLE_STRIP: 0x0005,
		TRIANGLES: 0x0004,
		CULL_FACE: 0x0b44,
		BACK: 0x0405,
		FRONT: 0x0404,
		CW: 0x0900,
		CCW: 0x0901,
		DEPTH_TEST: 0x0b71,
		BLEND: 0x0be2,
		ALWAYS: 0x0207,
		NEVER: 0x0200,
		EQUAL: 0x0202,
		LEQUAL: 0x0203,
		GREATER: 0x0204,
		GEQUAL: 0x0206,
		LESS: 0x0201,
		COLOR: 0x1800,
		DEPTH: 0x1801,
		createFramebuffer: () => ({ id: nextId++, kind: "framebuffer" }),
		deleteFramebuffer: (value) => calls.push(["deleteFramebuffer", value]),
		createTexture: () => ({ id: nextId++, kind: "texture" }),
		deleteTexture: (value) => calls.push(["deleteTexture", value]),
		createBuffer: () => ({ id: nextId++, kind: "buffer" }),
		deleteBuffer: (value) => calls.push(["deleteBuffer", value]),
		bindBuffer: (...args) => calls.push(["bindBuffer", ...args]),
		bufferData: (...args) => calls.push(["bufferData", ...args]),
		createRenderbuffer: () => ({ id: nextId++, kind: "renderbuffer" }),
		deleteRenderbuffer: (value) => calls.push(["deleteRenderbuffer", value]),
		createShader: (type) => ({ id: nextId++, kind: "shader", type }),
		shaderSource: (...args) => calls.push(["shaderSource", ...args]),
		compileShader: (...args) => calls.push(["compileShader", ...args]),
		getShaderParameter: () => true,
		getShaderInfoLog: () => "",
		deleteShader: (value) => calls.push(["deleteShader", value]),
		createProgram: () => ({ id: nextId++, kind: "program" }),
		attachShader: (...args) => calls.push(["attachShader", ...args]),
		linkProgram: (...args) => calls.push(["linkProgram", ...args]),
		getProgramParameter: () => true,
		getProgramInfoLog: () => "",
		deleteProgram: (value) => calls.push(["deleteProgram", value]),
		useProgram: (...args) => calls.push(["useProgram", ...args]),
		disable: (...args) => calls.push(["disable", ...args]),
		enable: (...args) => calls.push(["enable", ...args]),
		cullFace: (...args) => calls.push(["cullFace", ...args]),
		frontFace: (...args) => calls.push(["frontFace", ...args]),
		depthMask: (...args) => calls.push(["depthMask", ...args]),
		depthFunc: (...args) => calls.push(["depthFunc", ...args]),
		enableVertexAttribArray: (...args) => calls.push(["enableVertexAttribArray", ...args]),
		vertexAttribPointer: (...args) => calls.push(["vertexAttribPointer", ...args]),
		vertexAttribIPointer: (...args) => calls.push(["vertexAttribIPointer", ...args]),
		vertexAttribDivisor: (...args) => calls.push(["vertexAttribDivisor", ...args]),
		drawArrays: (...args) => calls.push(["drawArrays", ...args]),
		drawArraysInstanced: (...args) => calls.push(["drawArraysInstanced", ...args]),
		drawElements: (...args) => calls.push(["drawElements", ...args]),
		drawElementsInstanced: (...args) => calls.push(["drawElementsInstanced", ...args]),
		activeTexture: (...args) => calls.push(["activeTexture", ...args]),
		getUniformLocation: () => null,
		bindBufferBase: (...args) => calls.push(["bindBufferBase", ...args]),
		getUniformBlockIndex: () => 0xffffffff,
		uniformBlockBinding: (...args) => calls.push(["uniformBlockBinding", ...args]),
		bindTexture: (...args) => calls.push(["bindTexture", ...args]),
		texParameteri: (...args) => calls.push(["texParameteri", ...args]),
		texImage2D: (...args) => calls.push(["texImage2D", ...args]),
		bindRenderbuffer: (...args) => calls.push(["bindRenderbuffer", ...args]),
		renderbufferStorage: (...args) => calls.push(["renderbufferStorage", ...args]),
		bindFramebuffer: (...args) => calls.push(["bindFramebuffer", ...args]),
		framebufferTexture2D: (...args) => calls.push(["framebufferTexture2D", ...args]),
		framebufferRenderbuffer: (...args) => calls.push(["framebufferRenderbuffer", ...args]),
		drawBuffers: (...args) => calls.push(["drawBuffers", ...args]),
		checkFramebufferStatus: () => gl.FRAMEBUFFER_COMPLETE,
		readBuffer: (...args) => calls.push(["readBuffer", ...args]),
		readPixels: (_x, _y, width, height, _format, _type, out) => {
			calls.push(["readPixels", width, height]);
			for (let i = 0; i < width * height * 4; i++) {
				out[i] = i & 0xff;
			}
		},
		clearBufferfv: (...args) => calls.push(["clearBufferfv", ...args]),
		scissor: (...args) => calls.push(["scissor", ...args]),
	};
	return gl;
}

function createContext(passExecute) {
	return {
		backendProfile: {
			id: "webgl",
			capabilities: {},
			frameScheduling: "always",
			shadow: {},
			lighting: {},
		},
		camera: {},
		attachments: { width: 128, height: 64 },
		features: {},
		postProcess: createResolvedPostProcess("webgl"),
		renderTargets: new RenderTargetRegistrySnapshot([
			{
				id: "inspect",
				size: { mode: "canvas-scale", scale: 0.5 },
				color: [
					{ format: TextureFormat.RGBA8Unorm },
					{ format: TextureFormat.RGBA8Unorm },
				],
				depth: { format: TextureFormat.Depth32Float },
			},
		]),
		customRenderPasses: new CustomRenderPassRegistrySnapshot([
			{
				id: "inspect-pass",
				target: "inspect",
				execute: passExecute,
			},
		]),
		shadowMaps: new Map(),
		scene: {},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		incremental: { enabled: false, forceFullFrame: true, dirtyRects: [] },
		transient: new Map(),
	};
}

async function testWebGLCustomTargetExecutionAndReadback() {
	const gl = createFakeGL();
	const runtime = new WebGLCustomRenderTargetRuntime(gl);
	let observed = null;
	const context = createContext((passContext) => {
		observed = passContext;
		const vertexModule = passContext.resources.createShaderModule({
			stage: "vertex",
			language: "glsl",
			code: "#version 300 es\nlayout(location=0) in vec3 position;\nvoid main(){ gl_Position = vec4(position, 1.0); }",
		});
		const fragmentModule = passContext.resources.createShaderModule({
			stage: "fragment",
			language: "glsl",
			code: "#version 300 es\nprecision highp float;\nout vec4 color;\nvoid main(){ color = vec4(1.0); }",
		});
		const pipeline = passContext.resources.createRenderPipeline({
			label: "custom-webgl-test",
			vertex: {
				module: vertexModule,
				entryPoint: "main",
				buffers: [{
					arrayStride: 12,
					attributes: [{
						format: "float32x3",
						offset: 0,
						shaderLocation: 0,
					}],
				}],
			},
			fragment: {
				module: fragmentModule,
				entryPoint: "main",
				targets: [{ format: TextureFormat.RGBA8Unorm }],
			},
			primitive: { topology: PrimitiveTopology.TriangleList },
		});
		const vertexBuffer = passContext.resources.createBuffer({
			size: 36,
			usage: BufferUsage.Vertex,
			initialData: new Float32Array([
				-1, -1, 0,
				1, -1, 0,
				0, 1, 0,
			]),
		});
		passContext.encoder.beginRenderPass({
			colorAttachments: passContext.target.color.map((attachment) => ({
				view: attachment.texture,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 1, g: 0, b: 0, a: 1 },
			})),
			depthStencilAttachment: {
				view: passContext.target.depth.texture,
				depthLoadOp: "clear",
				depthStoreOp: "store",
				depthClearValue: 1,
			},
		});
		passContext.encoder.setPipeline(pipeline);
		passContext.encoder.setVertexBuffer(0, vertexBuffer);
		passContext.encoder.draw(3);
		passContext.encoder.endRenderPass();
	});

	runtime.sync(context);
	await runtime.executePass(
		{ stage: "inspect-pass", executor: "backend", enabled: true, dependsOn: [] },
		context
	);
	assert.equal(observed.backend, "webgl");
	assert.equal(observed.width, 64);
	assert.equal(observed.height, 32);
	assert.ok(gl.calls.some((call) => call[0] === "framebufferTexture2D"));
	assert.ok(gl.calls.some((call) => call[0] === "drawArrays"));

	await assert.rejects(() => runtime.readColor("inspect", 0), /successful frame/);
	runtime.markFrameCommitted();
	const readback = await runtime.readColor("inspect", 1);
	assert.equal(readback.width, 64);
	assert.equal(readback.height, 32);
	assert.equal(readback.bytes[1], 1);
	runtime.destroy();
	assert.ok(gl.calls.some((call) => call[0] === "deleteTexture"));
}

await testWebGLCustomTargetExecutionAndReadback();
console.log("WebGL custom render target tests passed");
