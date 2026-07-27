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

function createFakeGL(options = {}) {
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
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_DRAW_BUFFERS: 0x8824,
		MAX_COLOR_ATTACHMENTS: 0x8cdf,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		DEPTH_ATTACHMENT: 0x8d00,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		DEPTH_COMPONENT16: 0x81a5,
		DEPTH_COMPONENT24: 0x81a6,
		DEPTH_COMPONENT32F: 0x8cac,
		DEPTH_COMPONENT: 0x1902,
		R8: 0x8229,
		RG8: 0x822b,
		RGBA8: 0x8058,
		SRGB8_ALPHA8: 0x8c43,
		R16F: 0x822d,
		RG16F: 0x822f,
		RGBA16F: 0x881a,
		R32F: 0x822e,
		RG32F: 0x8230,
		RGBA32F: 0x8814,
		RED: 0x1903,
		RG: 0x8227,
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
		SCISSOR_TEST: 0x0c11,
		ALWAYS: 0x0207,
		NEVER: 0x0200,
		EQUAL: 0x0202,
		LEQUAL: 0x0203,
		GREATER: 0x0204,
		GEQUAL: 0x0206,
		LESS: 0x0201,
		COLOR: 0x1800,
		DEPTH: 0x1801,
		getExtension: (name) =>
			name === "EXT_color_buffer_float" && options.floatColorSupported !== false ?
				{}
			:	null,
		getParameter: (parameter) => {
			if (parameter === gl.MAX_TEXTURE_SIZE) return options.maxTextureSize ?? 4096;
			if (parameter === gl.MAX_DRAW_BUFFERS) return options.maxDrawBuffers ?? 4;
			if (parameter === gl.MAX_COLOR_ATTACHMENTS) {
				return options.maxColorAttachments ?? 4;
			}
			return 0;
		},
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
		viewport: (...args) => calls.push(["viewport", ...args]),
		colorMask: (...args) => calls.push(["colorMask", ...args]),
		bindVertexArray: (...args) => calls.push(["bindVertexArray", ...args]),
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
		checkFramebufferStatus: () =>
			options.framebufferStatus ?? gl.FRAMEBUFFER_COMPLETE,
		invalidateFramebuffer: (...args) => calls.push(["invalidateFramebuffer", ...args]),
		readBuffer: (...args) => calls.push(["readBuffer", ...args]),
		readPixels: (_x, _y, width, height, _format, _type, out) => {
			calls.push(["readPixels", width, height]);
			if (options.readPixelsThrows) {
				throw new Error("read-pixels-failed");
			}
			for (let i = 0; i < width * height * 4; i++) {
				out[i] = i & 0xff;
			}
		},
		clearBufferfv: (...args) => calls.push(["clearBufferfv", ...args]),
		scissor: (...args) => calls.push(["scissor", ...args]),
	};
	return gl;
}

function createContext(passExecute, targetDescriptor = {}) {
	const descriptor = {
		id: "inspect",
		size: { mode: "canvas-scale", scale: 0.5 },
		color: [
			{ format: TextureFormat.RGBA8Unorm },
			{ format: TextureFormat.RGBA8Unorm },
		],
		depth: { format: TextureFormat.Depth32Float },
		...targetDescriptor,
	};
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
		renderTargets: new RenderTargetRegistrySnapshot([descriptor]),
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
				storeOp: "discard",
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
		passContext.encoder.setScissorRect(0, 0, passContext.width, passContext.height);
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
	assert.ok(
		gl.calls.some((call) =>
			call[0] === "viewport" && call[3] === 64 && call[4] === 32
		)
	);
	assert.ok(
		gl.calls.some((call) => call[0] === "enable" && call[1] === gl.SCISSOR_TEST)
	);
	assert.ok(gl.calls.some((call) => call[0] === "invalidateFramebuffer"));
	assert.ok(
		gl.calls.some((call) =>
			call[0] === "framebufferTexture2D" && call[2] === gl.DEPTH_ATTACHMENT
		)
	);
	const restoredViewports = gl.calls.filter((call) => call[0] === "viewport");
	assert.deepEqual(restoredViewports.at(-1), ["viewport", 0, 0, 128, 64]);

	await assert.rejects(() => runtime.readColor("inspect", 0), /successful frame/);
	runtime.markFrameCommitted();
	const readback = await runtime.readColor("inspect", 1);
	assert.equal(readback.width, 64);
	assert.equal(readback.height, 32);
	assert.equal(readback.bytes[1], 1);
	assert.equal(readback.bytesPerPixel, 4);
	assert.equal(readback.origin, "bottom-left");
	await assert.rejects(() => runtime.readColor("inspect", 1, { width: 65 }), /between/);
	runtime.destroy();
	assert.ok(gl.calls.some((call) => call[0] === "deleteTexture"));
}

function testWebGLStrictFormatValidation() {
	const noFloatGL = createFakeGL({ floatColorSupported: false });
	const noFloatRuntime = new WebGLCustomRenderTargetRuntime(noFloatGL);
	const floatContext = createContext(() => {}, {
		color: [{ format: TextureFormat.RGBA16Float }],
		depth: null,
	});
	assert.throws(
		() => noFloatRuntime.sync(floatContext),
		/EXT_color_buffer_float/
	);

	const limitedGL = createFakeGL({ maxDrawBuffers: 1, maxColorAttachments: 1 });
	const limitedRuntime = new WebGLCustomRenderTargetRuntime(limitedGL);
	assert.throws(
		() => limitedRuntime.sync(createContext(() => {})),
		/color attachment limit 1/
	);

	const unsupportedGL = createFakeGL();
	const unsupportedRuntime = new WebGLCustomRenderTargetRuntime(unsupportedGL);
	const unsupportedContext = createContext(() => {}, {
		color: [{ format: TextureFormat.BGRA8Unorm }],
		depth: null,
	});
	assert.throws(
		() => unsupportedRuntime.sync(unsupportedContext),
		/unsupported/
	);
}

function testWebGLCoreFormatMappings() {
	const colorFormats = [
		[TextureFormat.R8Unorm, "R8"],
		[TextureFormat.RG8Unorm, "RG8"],
		[TextureFormat.RGBA8Unorm, "RGBA8"],
		[TextureFormat.RGBA8UnormSrgb, "SRGB8_ALPHA8"],
		[TextureFormat.R16Float, "R16F"],
		[TextureFormat.RG16Float, "RG16F"],
		[TextureFormat.RGBA16Float, "RGBA16F"],
		[TextureFormat.R32Float, "R32F"],
		[TextureFormat.RG32Float, "RG32F"],
		[TextureFormat.RGBA32Float, "RGBA32F"],
	];
	for (const [format, constantName] of colorFormats) {
		const gl = createFakeGL();
		const runtime = new WebGLCustomRenderTargetRuntime(gl);
		runtime.sync(createContext(() => {}, {
			color: [{ format }],
			depth: null,
		}));
		const allocation = gl.calls.find((call) => call[0] === "texImage2D");
		assert.equal(allocation[3], gl[constantName], format);
		runtime.destroy();
	}

	const depthFormats = [
		[TextureFormat.Depth16Unorm, "DEPTH_COMPONENT16"],
		[TextureFormat.Depth24Plus, "DEPTH_COMPONENT24"],
		[TextureFormat.Depth32Float, "DEPTH_COMPONENT32F"],
	];
	for (const [format, constantName] of depthFormats) {
		const gl = createFakeGL();
		const runtime = new WebGLCustomRenderTargetRuntime(gl);
		runtime.sync(createContext(() => {}, {
			color: [{ format: TextureFormat.RGBA8Unorm }],
			depth: { format },
		}));
		const allocations = gl.calls.filter((call) => call[0] === "texImage2D");
		assert.equal(allocations.at(-1)[3], gl[constantName], format);
		assert.ok(
			gl.calls.some((call) =>
				call[0] === "framebufferTexture2D" &&
				call[2] === gl.DEPTH_ATTACHMENT
			)
		);
		runtime.destroy();
	}
}

function testWebGLIncompleteFramebufferIsTransactional() {
	const gl = createFakeGL({ framebufferStatus: 0x8cd6 });
	const runtime = new WebGLCustomRenderTargetRuntime(gl);
	assert.throws(
		() => runtime.sync(createContext(() => {})),
		/incomplete/
	);
	assert.equal(
		gl.calls.filter((call) => call[0] === "deleteTexture").length,
		3
	);
	assert.equal(
		gl.calls.filter((call) => call[0] === "deleteFramebuffer").length,
		1
	);
}

async function testWebGLPassFailureRestoresState() {
	const gl = createFakeGL();
	const runtime = new WebGLCustomRenderTargetRuntime(gl);
	const context = createContext((passContext) => {
		passContext.encoder.beginRenderPass({
			colorAttachments: [{
				view: passContext.target.color[0].texture,
				loadOp: "clear",
				storeOp: "store",
			}],
		});
		throw new Error("custom-pass-failed");
	});
	runtime.sync(context);
	await assert.rejects(
		() => runtime.executePass(
			{ stage: "inspect-pass", executor: "backend", enabled: true, dependsOn: [] },
			context
		),
		/custom-pass-failed/
	);
	assert.ok(gl.calls.some((call) => call[0] === "deleteFramebuffer"));
	const viewports = gl.calls.filter((call) => call[0] === "viewport");
	assert.deepEqual(viewports.at(-1), ["viewport", 0, 0, 128, 64]);
	assert.ok(
		gl.calls.some((call) => call[0] === "bindVertexArray" && call[1] === null)
	);

	const unfinishedGL = createFakeGL();
	const unfinishedRuntime = new WebGLCustomRenderTargetRuntime(unfinishedGL);
	const unfinishedContext = createContext((passContext) => {
		passContext.encoder.beginRenderPass({
			colorAttachments: [{
				view: passContext.target.color[0].texture,
				loadOp: "load",
				storeOp: "store",
			}],
		});
	});
	unfinishedRuntime.sync(unfinishedContext);
	await assert.rejects(
		() => unfinishedRuntime.executePass(
			{ stage: "inspect-pass", executor: "backend", enabled: true, dependsOn: [] },
			unfinishedContext
		),
		/left a pass active/
	);
	const unfinishedViewports = unfinishedGL.calls.filter(
		(call) => call[0] === "viewport"
	);
	assert.deepEqual(unfinishedViewports.at(-1), ["viewport", 0, 0, 128, 64]);
}

async function testWebGLFloat16ReadbackLayout() {
	const gl = createFakeGL();
	const runtime = new WebGLCustomRenderTargetRuntime(gl);
	const context = createContext(() => {}, {
		color: [{ format: TextureFormat.RGBA16Float }],
		depth: null,
	});
	runtime.sync(context);
	runtime.markFrameCommitted();
	const readback = await runtime.readColor("inspect", 0, { width: 1, height: 1 });
	assert.equal(readback.format, TextureFormat.RGBA16Float);
	assert.equal(readback.bytesPerPixel, 8);
	assert.equal(readback.bytes.length, 8);
	assert.equal(readback.toRGBAFloat32().length, 4);

	const failingGL = createFakeGL({ readPixelsThrows: true });
	const failingRuntime = new WebGLCustomRenderTargetRuntime(failingGL);
	failingRuntime.sync(context);
	failingRuntime.markFrameCommitted();
	await assert.rejects(
		() => failingRuntime.readColor("inspect", 0, { width: 1, height: 1 }),
		/read-pixels-failed/
	);
	const framebufferBinds = failingGL.calls.filter(
		(call) => call[0] === "bindFramebuffer"
	);
	assert.equal(framebufferBinds.at(-1)[2], null);
}

await testWebGLCustomTargetExecutionAndReadback();
testWebGLStrictFormatValidation();
testWebGLCoreFormatMappings();
testWebGLIncompleteFramebufferIsTransactional();
await testWebGLPassFailureRestoresState();
await testWebGLFloat16ReadbackLayout();
console.log("WebGL custom render target tests passed");
