import assert from "node:assert/strict";
import { WebGLFrameExecutor } from "../../../src/renderers/webgl/WebGLFrameExecutor.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import {
	FastApproximateAntiAliasingPass,
	ScreenSpaceAmbientOcclusionPass,
	TemporalAntiAliasingPass,
	ToneMappingPass,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFXAATestGL() {
	const calls = [];
	return {
		calls,
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		FRAMEBUFFER: 0x8d40,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		COLOR_ATTACHMENT2: 0x8ce2,
		NONE: 0,
		MAX_DRAW_BUFFERS: 0x8824,
		TEXTURE_2D: 0x0de1,
		TEXTURE0: 0x84c0,
		TEXTURE1: 0x84c1,
		TEXTURE2: 0x84c2,
		TEXTURE3: 0x84c3,
		TRIANGLES: 0x0004,
		CULL_FACE: 0x0b44,
		DEPTH_TEST: 0x0b71,
		BLEND: 0x0be2,
		SCISSOR_TEST: 0x0c11,
		LESS: 0x0201,
		LEQUAL: 0x0203,
		ZERO: 0,
		ONE: 1,
		SRC_ALPHA: 0x0302,
		ONE_MINUS_SRC_ALPHA: 0x0303,
		getParameter(parameter) {
			if (
				parameter === this.MAX_TEXTURE_SIZE ||
				parameter === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			if (parameter === this.MAX_DRAW_BUFFERS) {
				return 4;
			}
			return 0;
		},
		getExtension() {
			return null;
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
		enable(cap) {
			calls.push({ name: "enable", cap });
		},
		depthMask(flag) {
			calls.push({ name: "depthMask", flag });
		},
		depthFunc(func) {
			calls.push({ name: "depthFunc", func });
		},
		colorMask(r, g, b, a) {
			calls.push({ name: "colorMask", r, g, b, a });
		},
		blendFuncSeparate(srcRgb, dstRgb, srcAlpha, dstAlpha) {
			calls.push({
				name: "blendFuncSeparate",
				srcRgb,
				dstRgb,
				srcAlpha,
				dstAlpha,
			});
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
		uniform1f(location, value) {
			calls.push({ name: "uniform1f", location, value });
		},
		uniform2f(location, x, y) {
			calls.push({ name: "uniform2f", location, x, y });
		},
		uniform3f(location, x, y, z) {
			calls.push({ name: "uniform3f", location, x, y, z });
		},
		uniform4f(location, x, y, z, w) {
			calls.push({ name: "uniform4f", location, x, y, z, w });
		},
		uniform4fv(location, values) {
			calls.push({ name: "uniform4fv", location, values: Array.from(values) });
		},
		uniformMatrix4fv(location, transpose, values) {
			calls.push({
				name: "uniformMatrix4fv",
				location,
				transpose,
				values: Array.from(values),
			});
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

function captureWarnMessages(run) {
	const warnings = [];
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
		run();
	} finally {
		Logger.reset();
	}
	return warnings;
}

function testFXAAPassUsesLatestPostSourceAndRebindsPostTarget() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const sceneColor = { id: "scene-color" };
	const taaHistory = { id: "taa-history" };
	const postColor = { id: "post-color" };
	const postFramebuffer = { id: "post-fbo" };
	const fullscreenVao = { id: "fullscreen-vao" };

	executor._programs = {
		tryGetFXAAProgram() {
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

	const frameContext = {
		postProcess: createResolvedPostProcess(
			{ fxaa: { enabled: true } },
			"webgl"
		),
		transient: new Map(),
	};
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });
	const request = {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "fxaa",
		options: frameContext.postProcess.getOptions("fxaa"),
		startPassId: null,
	};
	const context = executor.getPassExecutionContext({
		...request,
		implementation: pass.getImplementation("webgl"),
	});
	const result = pass.getImplementation("webgl").execute(
		request,
		context
	);
	assert.deepEqual(result, { ran: true });

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

function testFXAAPassSkipsWhileProgramPending() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const sceneColor = { id: "scene-color" };
	const sourceColor = { id: "source-color" };
	const postColor = { id: "post-color" };
	let tryCalls = 0;

	executor._programs = {
		tryGetFXAAProgram() {
			tryCalls++;
			return null;
		},
	};
	executor._sceneColorTexture = sceneColor;
	executor._presentSourceTexture = sourceColor;
	executor._postColorTexture = postColor;
	executor._postFramebuffer = { id: "post-fbo" };
	executor._fullscreenVao = { id: "fullscreen-vao" };
	executor._width = 1280;
	executor._height = 720;

	const frameContext = {
		postProcess: createResolvedPostProcess(
			{ fxaa: { enabled: true } },
			"webgl"
		),
		transient: new Map(),
	};
	const pass = new FastApproximateAntiAliasingPass({ enabled: true });
	const request = {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "fxaa",
		options: frameContext.postProcess.getOptions("fxaa"),
		startPassId: null,
	};
	const context = executor.getPassExecutionContext({
		...request,
		implementation: pass.getImplementation("webgl"),
	});
	const result = pass.getImplementation("webgl").execute(request, context);

	assert.deepEqual(result, { ran: false });
	assert.equal(tryCalls, 1);
	assert.equal(executor._presentSourceTexture, sourceColor);
	assert.equal(
		gl.calls.some((call) => call.name === "framebufferTexture2D"),
		false
	);
}

function testToneMappingPassUsesLatestPostSourceAndRebindsPostTarget() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const sceneColor = { id: "scene-color" };
	const bloomColor = { id: "bloom-color" };
	const postColor = { id: "post-color" };
	const postFramebuffer = { id: "post-fbo" };
	const fullscreenVao = { id: "fullscreen-vao" };

	executor._programs = {
		tryGetToneMappingProgram() {
			return {
				program: { id: "tonemap-program" },
				uniforms: {
					sourceMap: { id: "uSourceMap" },
				},
			};
		},
	};
	executor._sceneColorTexture = sceneColor;
	executor._presentSourceTexture = bloomColor;
	executor._postColorTexture = postColor;
	executor._postFramebuffer = postFramebuffer;
	executor._fullscreenVao = fullscreenVao;
	executor._width = 1280;
	executor._height = 720;

	const frameContext = {
		postProcess: createResolvedPostProcess({
			tonemap: { enabled: true },
		}),
		transient: new Map(),
	};
	const pass = new ToneMappingPass({ enabled: true });
	const request = {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "tonemap",
		options: {},
		startPassId: null,
	};
	const result = pass
		.getImplementation("webgl")
		.execute(request, executor.getPassExecutionContext({
			...request,
			implementation: pass.getImplementation("webgl"),
		}));
	assert.deepEqual(result, { ran: true });

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
	assert.equal(sourceBind?.texture, bloomColor);

	assert.equal(executor._presentSourceTexture, postColor);
}

function testExecutePostProcessPassLeavesFXAAToPassImplementation() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const events = [];

	executor._applyToneMapping = () => {
		events.push("tonemap");
	};
	executor._applyInteractionOutline = () => {
		events.push("interaction-outline");
	};
	executor._present = () => {
		events.push("gamma");
	};

	const frameContext = {
		postProcess: createResolvedPostProcess({
			gamma: { enabled: true },
			fxaa: { enabled: true },
		}),
		transient: new Map(),
	};
	const pass = new TemporalAntiAliasingPass({ enabled: true });
	const request = {
		frameContext,
		histories: {},
	};

	const toneMapResult = executor.executePostProcessPass("tonemap", request);
	const fxaaResult = executor.executePostProcessPass("fxaa", request);
	const outlineResult = executor.executePostProcessPass(
		"interaction-outline",
		request
	);
	const gammaResult = executor.executePostProcessPass("gamma", request);

	assert.deepEqual(toneMapResult, { ran: false });
	assert.deepEqual(fxaaResult, { ran: false });
	assert.deepEqual(outlineResult, { ran: false });
	assert.deepEqual(gammaResult, { ran: false });
	assert.deepEqual(events, []);
}

function testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension() {
	const gl = createFrameTargetTestGL({ floatExtension: false });
	let executor;
	const warnings = captureWarnMessages(() => {
		executor = new WebGLFrameExecutor(gl);
		executor._ensureFrameTargets(320, 180, 1);
	});

	assert.equal(
		gl.texImage2DCalls.some((call) => call.internalFormat === gl.RGBA16F),
		false
	);
	const bridge = executor.createGBufferBridge({
		attachments: { width: 320, height: 180 },
	});
	assert.equal(bridge.channels.color.format, "rgba8unorm");
	assert.equal(bridge.channels.depth.format, "rgba8unorm");
	assert.equal(bridge.channels.motion.format, "rgba8unorm");
	assert.equal(bridge.channels.normal.format, "rgba8unorm");
	assert.equal(bridge.channels.normal.encoding, "encoded-world-normal");
	assert.ok(
		warnings.some(
			(warning) => warning.includes("[webgl-hdr-float-unsupported]")
		)
	);
}

function testFrameTargetsCreateOITResourcesWithFloatExtension() {
	const gl = createFrameTargetTestGL({ floatExtension: true });
	const executor = new WebGLFrameExecutor(gl);

	executor._ensureFrameTargets(320, 180, 1);

	assert.ok(executor._oitFramebuffer);
	assert.ok(executor._oitAccumTexture);
	assert.ok(executor._oitRevealTexture);
	const bridge = executor.createGBufferBridge({
		attachments: { width: 320, height: 180 },
	});
	assert.equal(bridge.channels.color.format, "rgba16float");
	assert.equal(bridge.channels.depth.format, "rgba16float");
	assert.equal(bridge.channels.motion.format, "rgba16float");
	assert.equal(bridge.channels.normal.format, "rgba8unorm");
}

function testPostProcessResourceFormatFollowsFloatExtension() {
	const fallbackGL = createFrameTargetTestGL({ floatExtension: false });
	let fallbackResource;
	const warnings = captureWarnMessages(() => {
		const executor = new WebGLFrameExecutor(fallbackGL);
		fallbackResource = executor.createPostProcessResource({
			id: "hdr-history",
			width: 16,
			height: 8,
			format: "rgba16float",
			usage: ["sampled", "render-target"],
		});
	});

	assert.equal(fallbackResource.format, "rgba8unorm");
	assert.equal(
		fallbackGL.texImage2DCalls.at(-1).internalFormat,
		fallbackGL.RGBA8
	);
	assert.ok(
		warnings.some((warning) =>
			warning.includes("[webgl-hdr-float-unsupported]")
		)
	);

	const hdrGL = createFrameTargetTestGL({ floatExtension: true });
	const executor = new WebGLFrameExecutor(hdrGL);
	const hdrResource = executor.createPostProcessResource({
		id: "hdr-history",
		width: 16,
		height: 8,
		format: "rgba16float",
		usage: ["sampled", "render-target"],
	});

	assert.equal(hdrResource.format, "rgba16float");
	assert.equal(hdrGL.texImage2DCalls.at(-1).internalFormat, hdrGL.RGBA16F);
}

function testConfigureOITWarnsWithoutRuntimeTargets() {
	const gl = createFrameTargetTestGL({ floatExtension: false });
	const executor = new WebGLFrameExecutor(gl);
	const warnings = captureWarnMessages(() => {
		executor._configureOIT({
			features: {
				enableOIT: true,
			},
		});
	});

	assert.equal(executor._oitActive, false);
	assert.ok(
		warnings.some((warning) =>
			warning.includes("[webgl-oit-disabled-runtime]")
		)
	);
}

function testOITTransparentAndParticleExecutionOrder() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const events = [];

	executor._oitActive = true;
	executor._oitFramebuffer = { id: "oit-fbo" };
	executor._sceneFramebuffer = { id: "scene-fbo" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._postFramebuffer = { id: "post-fbo" };
	executor._postColorTexture = { id: "post-color" };
	executor._oitAccumTexture = { id: "oit-accum" };
	executor._oitRevealTexture = { id: "oit-reveal" };
	executor._fullscreenVao = { id: "fullscreen-vao" };

	executor._clearOITTargets = () => {
		events.push("clear");
	};
	executor._resolveOITComposition = () => {
		events.push("resolve");
	};
	executor._renderPackets = (_context, packets, _transparent, options = {}) => {
		events.push(
			`packets:${options.blendMode ?? "legacy"}:${options.oitPassMode ?? 0}:${packets.length}`
		);
	};
	executor._renderParticles = (_context, options = {}) => {
		const blendModes = options.includeBlendModes ?? [];
		const label =
			blendModes.length === 0 ? "all" : blendModes.map(String).join(",");
		events.push(`particles:${label}:${options.oitPassMode ?? 0}`);
	};

	const context = {
		scene: {
			transparentPackets: [
				{ id: "oit-packet", material: {} },
				{ id: "legacy-packet", material: { transmissionFactor: 1 } },
			],
			particleSystems: [{ id: "ps-0" }],
		},
	};

	executor._renderOITTransparentPass(context);
	executor._renderOITParticlePass(context);

	assert.deepEqual(events, [
		"clear",
		"packets:oit-accum:1:1",
		"packets:oit-reveal:2:1",
		"particles:alpha:1",
		"particles:alpha:2",
		"resolve",
		"packets:legacy:0:1",
		"particles:additive:0",
	]);
}

function testOITTransparentResolvesImmediatelyWithoutParticles() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const events = [];

	executor._oitActive = true;
	executor._oitFramebuffer = { id: "oit-fbo" };
	executor._sceneFramebuffer = { id: "scene-fbo" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._postFramebuffer = { id: "post-fbo" };
	executor._postColorTexture = { id: "post-color" };
	executor._oitAccumTexture = { id: "oit-accum" };
	executor._oitRevealTexture = { id: "oit-reveal" };
	executor._fullscreenVao = { id: "fullscreen-vao" };

	executor._clearOITTargets = () => {
		events.push("clear");
	};
	executor._resolveOITComposition = () => {
		events.push("resolve");
	};
	executor._renderPackets = (_context, packets, _transparent, options = {}) => {
		events.push(
			`packets:${options.blendMode ?? "legacy"}:${options.oitPassMode ?? 0}:${packets.length}`
		);
	};

	const context = {
		scene: {
			transparentPackets: [
				{ id: "oit-packet", material: {} },
				{ id: "legacy-packet", material: { transmissionFactor: 1 } },
			],
			particleSystems: [],
		},
	};

	executor._renderOITTransparentPass(context);

	assert.deepEqual(events, [
		"clear",
		"packets:oit-accum:1:1",
		"packets:oit-reveal:2:1",
		"resolve",
		"packets:legacy:0:1",
	]);
}

function testMainOpaqueRunsEarlyZPrepassBeforeColorPass() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const events = [];
	const packet = { id: "opaque-0", material: {} };
	const context = {
		scene: {
			opaquePackets: [packet],
		},
	};

	executor._renderEarlyZPrepass = (_context, packets) => {
		events.push(`prepass:${packets.length}`);
		return new Set(["opaque-0"]);
	};
	executor._renderPackets = (_context, packets, transparent, options = {}) => {
		events.push(
			`color:${packets.length}:${transparent}:${options.earlyZPacketIds?.has("opaque-0")}`
		);
	};

	executor.executePass({ stage: "main-opaque" }, context);

	assert.deepEqual(events, [
		"prepass:1",
		"color:1:false:true",
	]);
}

function testMainOpaqueCanDisableEarlyZPrepass() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, undefined, undefined, {
		enableEarlyZPrepass: false,
	});
	const events = [];
	const packet = { id: "opaque-0", material: {} };
	const context = {
		scene: {
			opaquePackets: [packet],
		},
	};

	executor._renderPackets = (_context, packets, transparent, options = {}) => {
		events.push(
			`color:${packets.length}:${transparent}:${options.earlyZPacketIds?.size ?? 0}`
		);
	};

	executor.executePass({ stage: "main-opaque" }, context);

	assert.deepEqual(events, [
		"color:1:false:0",
	]);
	assert.equal(executor._enableEarlyZPrepass, false);
}

function testSceneFramebufferFailureCleansAllAllocatedTargets() {
	const gl = createFrameTargetTestGL({
		floatExtension: true,
		frameStatuses: [0x8cd6],
	});
	const executor = new WebGLFrameExecutor(gl);

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
	const executor = new WebGLFrameExecutor(gl);

	executor._modelMatrixCache.set("keep", new Float32Array(16));
	executor._modelMatrixCache.set("drop", new Float32Array(16));
	executor._modelMatrixKeysThisFrame.add("keep");
	executor._presentedInFrame = true;
	executor.endFrame();

	assert.equal(executor._modelMatrixCache.has("keep"), true);
	assert.equal(executor._modelMatrixCache.has("drop"), false);
}

function testShadowSkinningWarningKeyIsStable() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const warnings = captureWarnMessages(() => {
		executor._drawShadowPacket(
			{ uniforms: { mvp: null } },
			{ meshInstance: { id: "mesh-a", skeleton: {} } },
			{}
		);
	});

	assert.ok(
		warnings.some(
			(warning) => warning.includes("[webgl-shadow-skinning-unsupported]")
		)
	);
}

function testTransparentRenderPacketsConfiguresBlendAndDepthState() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	executor._sceneFramebuffer = { id: "scene-fbo" };
	executor._sceneNormalTexture = { id: "normal" };
	executor._programs = {
		getSceneProgram() {
			return {
				program: { id: "scene-program" },
				uniforms: {},
			};
		},
	};
	executor._bindGlobalUniforms = () => {};
	executor._drawPacket = () => {};

	executor._renderPackets(
		{
			camera: {
				viewProjectionMatrix: [
					[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
		},
		[{ material: {} }],
		true
	);

	const drawBuffersCall = gl.calls.find((call) => call.name === "drawBuffers");
	assert.deepEqual(drawBuffersCall?.buffers, [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
	]);
	const depthMaskCalls = gl.calls.filter((call) => call.name === "depthMask");
	assert.equal(depthMaskCalls[0]?.flag, false);
	assert.equal(depthMaskCalls[depthMaskCalls.length - 1]?.flag, true);
	const blendCall = gl.calls.find((call) => call.name === "blendFuncSeparate");
	assert.equal(blendCall?.srcRgb, gl.SRC_ALPHA);
	assert.equal(blendCall?.dstRgb, gl.ONE_MINUS_SRC_ALPHA);
	assert.equal(blendCall?.srcAlpha, gl.ONE);
	assert.equal(blendCall?.dstAlpha, gl.ONE_MINUS_SRC_ALPHA);
	assert.ok(
		gl.calls.some((call) => call.name === "enable" && call.cap === gl.BLEND)
	);
	assert.ok(
		gl.calls.some((call) => call.name === "disable" && call.cap === gl.BLEND)
	);
}

function testTAAPassDetachesMotionAttachmentAndSanitizesOptions() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	executor._programs = {
		tryGetTAAProgram() {
			return {
				program: { id: "taa-program" },
				uniforms: {
					sceneColor: "uSceneColor",
					historyMap: "uHistoryMap",
					motionMap: "uMotionMap",
					motionHistory: "uMotionHistory",
					texelSize: "uTexelSize",
					historyWeight: "uHistoryWeight",
					depthThreshold: "uDepthThreshold",
					motionFactor: "uMotionFactor",
					varianceClampGamma: "uVarianceClampGamma",
					sharpen: "uSharpen",
					historyValid: "uHistoryValid",
				},
			};
		},
	};
	executor._postFramebuffer = { id: "post-fbo" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._sceneMotionTexture = { id: "scene-motion" };
	executor._postColorTexture = { id: "post-color" };
	executor._fullscreenVao = { id: "fullscreen-vao" };
	executor._width = 1920;
	executor._height = 1080;

	const frameContext = {
		postProcess: createResolvedPostProcess(
			{
				taa: {
					enabled: true,
					options: {
						historyWeight: Number.POSITIVE_INFINITY,
						disocclusionDepthThreshold: Number.NaN,
						motionFactor: 1e9,
						varianceClampGamma: -5,
						sharpen: 4,
					},
				},
			},
			"webgl"
		),
		transient: new Map(),
	};
	const pass = new TemporalAntiAliasingPass({ enabled: true });
	const request = {
		frameContext,
		postProcess: frameContext.postProcess,
		gBuffer: {},
		histories: {
			taa: {
				valid: false,
				read: { resource: { id: "history-a" } },
				write: { resource: { id: "history-b" } },
			},
			motion: {
				valid: false,
				read: { resource: { id: "motion-a" } },
				write: { resource: { id: "motion-b" } },
			},
		},
		pass,
		passId: "taa",
		options: frameContext.postProcess.getOptions("taa"),
		startPassId: null,
	};
	const context = executor.getPassExecutionContext({
		...request,
		implementation: pass.getImplementation("webgl"),
	});
	const result = pass.getImplementation("webgl").execute(
		request,
		context
	);
	assert.deepEqual(result, { ran: true, updatedHistoryIds: ["taa", "motion"] });

	const attachment0Writes = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT0
	);
	assert.equal(attachment0Writes[attachment0Writes.length - 1]?.texture.id, "post-color");

	const attachment1Writes = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT1
	);
	assert.equal(attachment1Writes[attachment1Writes.length - 1]?.texture, null);
	assert.equal(attachment1Writes[0]?.texture.id, "history-b");

	const attachment2Writes = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT2
	);
	assert.equal(attachment2Writes[0]?.texture.id, "motion-b");
	assert.equal(attachment2Writes[attachment2Writes.length - 1]?.texture, null);

	const drawBuffersCalls = gl.calls.filter((call) => call.name === "drawBuffers");
	assert.deepEqual(drawBuffersCalls[0]?.buffers, [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
	]);
	assert.deepEqual(drawBuffersCalls[drawBuffersCalls.length - 1]?.buffers, [
		gl.COLOR_ATTACHMENT0,
	]);
	assert.equal(executor._presentSourceTexture.id, "post-color");

	const uniform1fCalls = gl.calls.filter((call) => call.name === "uniform1f");
	const uniformMap = new Map(uniform1fCalls.map((call) => [call.location, call.value]));
	assert.equal(uniformMap.get("uHistoryWeight"), 0.9);
	assert.equal(uniformMap.get("uDepthThreshold"), 0.02);
	assert.equal(uniformMap.get("uMotionFactor"), 512);
	assert.equal(uniformMap.get("uVarianceClampGamma"), 0);
	assert.equal(uniformMap.get("uSharpen"), 2);
	assert.equal(uniformMap.get("uHistoryValid"), 0);
}

function testSSAOPassDetachesSecondaryAttachmentForDownsampleTargets() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	executor._programs = {
		tryGetSSAORawProgram() {
			return { program: { id: "ssao-raw" }, uniforms: {} };
		},
		tryGetSSAOBlurProgram() {
			return { program: { id: "ssao-blur" }, uniforms: {} };
		},
		tryGetSSAOCombineProgram() {
			return { program: { id: "ssao-combine" }, uniforms: {} };
		},
	};
	executor._postFramebuffer = { id: "post-fbo" };
	executor._postColorTexture = { id: "post-color" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._sceneMotionTexture = { id: "scene-motion" };
	executor._sceneNormalTexture = { id: "scene-normal" };
	executor._ssaoRawTexture = { id: "ssao-raw" };
	executor._ssaoBlurTexture = { id: "ssao-blur" };
	executor._fullscreenVao = { id: "fullscreen-vao" };
	executor._width = 1280;
	executor._height = 720;
	executor._targetSSAODownsample = 2;

	const identity = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const context = {
		camera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1280 / 720,
			viewMatrix: { elements: identity },
			getWorldPosition() {
				return { x: 0, y: 0, z: 5 };
			},
		},
		postProcess: createResolvedPostProcess({
			ssao: { enabled: true },
		}),
	};

	const pass = new ScreenSpaceAmbientOcclusionPass({ enabled: true });
	const request = {
		frameContext: context,
		postProcess: context.postProcess,
		gBuffer: {},
		histories: {},
		pass,
		passId: "ssao",
		options: context.postProcess.getOptions("ssao"),
		startPassId: null,
	};
	const passContext = executor.getPassExecutionContext({
		...request,
		implementation: pass.getImplementation("webgl"),
	});
	const result =
		pass.getImplementation("webgl").execute(
			request,
			passContext
		);
	assert.equal(result.ran, true);

	const detachCalls = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT1 &&
			call.texture === null
	);
	assert.ok(detachCalls.length > 0);
}

function testGlobalUniformsBindLightProbeIBLTextures() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const envTexture = { id: "env-specular" };
	const brdfTexture = { id: "brdf-lut" };
	const envProbeMap = {
		mipmaps: [new Float32Array(4), new Float32Array(4), new Float32Array(4)],
	};
	let envSpecularMapCalls = 0;
	let fallbackMapCalls = 0;

	executor._textures = {
		getEnvironmentSpecularTexture(texture) {
			if (texture === envProbeMap) {
				envSpecularMapCalls++;
				return { texture: envTexture, isLinear: true };
			}
			assert.equal(texture, null);
			fallbackMapCalls++;
			return { texture: null, isLinear: true };
		},
		getBRDFLUTTexture(texture) {
			assert.ok(texture);
			return { texture: brdfTexture, isLinear: true };
		},
	};
	executor._lightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		envSpecularMap: envProbeMap,
	};
	const sceneProgram = {
		uniforms: {
			envSpecularMap: "uEnvSpecularMap",
			hasEnvSpecularMap: "uHasEnvSpecularMap",
			envSpecularMapIsLinear: "uEnvSpecularMapIsLinear",
			envSpecularMaxMipLevel: "uEnvSpecularMaxMipLevel",
			brdfLUT: "uBrdfLUT",
		},
	};
	const context = {
		features: {
			enableLighting: true,
			enableShadows: false,
		},
		postProcess: createResolvedPostProcess(),
	};

	executor._bindGlobalUniforms(sceneProgram, context);

	const uniform1i = new Map(
		gl.calls
			.filter((call) => call.name === "uniform1i")
			.map((call) => [call.location, call.value])
	);
	assert.equal(uniform1i.get("uEnvSpecularMap"), 2);
	assert.equal(uniform1i.get("uHasEnvSpecularMap"), 1);
	assert.equal(uniform1i.get("uEnvSpecularMapIsLinear"), 1);
	assert.equal(uniform1i.get("uBrdfLUT"), 3);

	const uniform1f = new Map(
		gl.calls
			.filter((call) => call.name === "uniform1f")
			.map((call) => [call.location, call.value])
	);
	assert.equal(uniform1f.get("uEnvSpecularMaxMipLevel"), 2);

	const activeTextureUnits = gl.calls
		.filter((call) => call.name === "activeTexture")
		.map((call) => call.unit);
	assert.ok(activeTextureUnits.includes(gl.TEXTURE2));
	assert.ok(activeTextureUnits.includes(gl.TEXTURE3));
	assert.equal(envSpecularMapCalls, 1);
	assert.equal(fallbackMapCalls, 1);
}

function testGlobalUniformsSanitizeNonFiniteCameraAndLightValues() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);

	executor._lightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [
			{
				direction: [Number.NaN, 1, 0],
				color: [Number.POSITIVE_INFINITY, 1, 0.5],
			},
		],
		directionalShadows: [
			{
				viewProjectionMatrix: [
					[1, 0, 0, 0],
					[0, Number.NaN, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
		],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
	};

	const sceneProgram = {
		uniforms: {
			viewProjection: "uViewProjection",
			viewMatrix: "uViewMatrix",
			cameraPosition: "uCameraPosition",
			dirLightDirection: "uDirLightDirection",
			dirLightColor: "uDirLightColor",
			dirShadowViewProjection: "uDirShadowViewProjection",
		},
	};
	const context = {
		camera: {
			viewProjectionMatrix: {
				elements: [
					[Number.NaN, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
			viewMatrix: {
				elements: [
					[1, 0, 0, 0],
					[0, Number.POSITIVE_INFINITY, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
			getWorldPosition() {
				return { x: Number.NaN, y: Number.POSITIVE_INFINITY, z: 3 };
			},
		},
		features: {
			enableLighting: true,
			enableShadows: true,
		},
		postProcess: createResolvedPostProcess(),
	};

	const warnings = captureWarnMessages(() => {
		executor._bindGlobalUniforms(sceneProgram, context);
	});

	const matrixUploads = gl.calls.filter(
		(call) =>
			call.name === "uniformMatrix4fv" &&
			(call.location === "uViewProjection" ||
				call.location === "uViewMatrix" ||
				call.location === "uDirShadowViewProjection")
	);
	for (const upload of matrixUploads) {
		assert.equal(upload.values.every(Number.isFinite), true);
	}

	const vecUploads = gl.calls.filter(
		(call) =>
			(call.name === "uniform3f" && call.location === "uCameraPosition") ||
			(call.name === "uniform4fv" &&
				(call.location === "uDirLightDirection" ||
					call.location === "uDirLightColor"))
	);
	for (const upload of vecUploads) {
		const values =
			upload.name === "uniform3f" ? [upload.x, upload.y, upload.z] : upload.values;
		assert.equal(values.every(Number.isFinite), true);
	}

	assert.ok(
		warnings.some(
			(warning) => warning.includes("[webgl-camera-view-projection-invalid]")
		)
	);
}

async function testWarmupCollectsPostProcessHintsFromPlanOrder() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl);
	const calls = [];

	executor._programs = {
		getSceneProgram() {
			calls.push("scene");
			return { program: { id: "scene" }, uniforms: {} };
		},
		warmupSSAORawProgram() {
			calls.push("ssao-raw");
			return {
				label: "ssao-raw",
				isComplete: () => true,
				finalize: () => {},
			};
		},
		warmupSSAOBlurProgram() {
			calls.push("ssao-blur");
			return {
				label: "ssao-blur",
				isComplete: () => true,
				finalize: () => {},
			};
		},
		warmupSSAOCombineProgram() {
			calls.push("ssao-combine");
			return {
				label: "ssao-combine",
				isComplete: () => true,
				finalize: () => {},
			};
		},
		getBloomProgram() {
			calls.push("bloom");
			return { program: { id: "bloom" }, uniforms: {} };
		},
		warmupBloomProgram() {
			calls.push("bloom");
			return {
				label: "bloom",
				isComplete: () => true,
				finalize: () => {},
			};
		},
		getFXAAProgram() {
			calls.push("fxaa");
			return { program: { id: "fxaa" }, uniforms: {} };
		},
		warmupFXAAProgram() {
			calls.push("fxaa");
			return {
				label: "fxaa",
				isComplete: () => true,
				finalize: () => {},
			};
		},
		getPresentProgram() {
			calls.push("present");
			return { program: { id: "present" }, uniforms: {} };
		},
	};

	await executor.warmup(
		{
			features: {
				enableOIT: false,
			},
			postProcess: createResolvedPostProcess({
				ssao: { enabled: true },
				fxaa: { enabled: true },
				gamma: { enabled: true },
				bloom: { enabled: true },
			}),
		},
		{
			materials: [],
			shaderMaterials: [],
			enableEnvironment: false,
			enableShadows: false,
			enableParticles: false,
			includePostProcess: true,
			postProcessPasses: ["ssao", "bloom", "fxaa", "custom-pass"],
			sceneTargetMode: "mrt",
		}
	);

	assert.deepEqual(calls, [
		"scene",
		"ssao-raw",
		"ssao-blur",
		"ssao-combine",
		"bloom",
		"fxaa",
		"present",
	]);
}

async function run() {
	testFXAAPassUsesLatestPostSourceAndRebindsPostTarget();
	testFXAAPassSkipsWhileProgramPending();
	testToneMappingPassUsesLatestPostSourceAndRebindsPostTarget();
	testExecutePostProcessPassLeavesFXAAToPassImplementation();
	testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension();
	testFrameTargetsCreateOITResourcesWithFloatExtension();
	testPostProcessResourceFormatFollowsFloatExtension();
	testConfigureOITWarnsWithoutRuntimeTargets();
	testOITTransparentAndParticleExecutionOrder();
	testOITTransparentResolvesImmediatelyWithoutParticles();
	testMainOpaqueRunsEarlyZPrepassBeforeColorPass();
	testMainOpaqueCanDisableEarlyZPrepass();
	testSceneFramebufferFailureCleansAllAllocatedTargets();
	testEndFramePrunesStaleModelMatrixCache();
	testShadowSkinningWarningKeyIsStable();
	testTransparentRenderPacketsConfiguresBlendAndDepthState();
	testTAAPassDetachesMotionAttachmentAndSanitizesOptions();
	testSSAOPassDetachesSecondaryAttachmentForDownsampleTargets();
	testGlobalUniformsBindLightProbeIBLTextures();
	testGlobalUniformsSanitizeNonFiniteCameraAndLightValues();
	await testWarmupCollectsPostProcessHintsFromPlanOrder();
	console.log("WebGL FXAA frame executor tests passed");
}

await run();
