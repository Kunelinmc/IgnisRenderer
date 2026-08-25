import assert from "node:assert/strict";
import { Logger } from "../../../src/foundation/Logger.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { WEBGL_AUXILIARY_RASTER_EXTENSION } from "../../../src/backends/webgl/WebGLAuxiliaryRaster.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../../src/pipeline/types.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function installFrameServices(backend, frame) {
	frame.scene ??= {
		prepareSceneProgramSources: async () => {},
	};
	backend._contextServices = {
		frame,
		auxiliaryRaster: {
			hasExtension: () => false,
			execute: async (_generation, _signal, task) => task({}),
			destroy() {},
		},
		restoreContextWorkBaseline() {
			frame.restoreContextWorkBaseline?.();
		},
		destroy() {
			frame.destroy?.();
		},
	};
}

function createFakeWebGL2Context(options = {}) {
	const debugInfo = options.debugRendererInfo;
	const supportedExtensions = options.supportedExtensions ?? [
		"EXT_color_buffer_float",
		"OES_texture_half_float_linear",
	];
	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		MAX_TEXTURE_IMAGE_UNITS: 0x8872,
		MAX_VERTEX_TEXTURE_IMAGE_UNITS: 0x8b4c,
		MAX_COMBINED_TEXTURE_IMAGE_UNITS: 0x8b4d,
		MAX_DRAW_BUFFERS: 0x8824,
		MAX_COLOR_ATTACHMENTS: 0x8cdf,
		VENDOR: 0x1f00,
		RENDERER: 0x1f01,
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		VALIDATE_STATUS: 0x8b83,
		FRAMEBUFFER: 0x8d40,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		RENDERBUFFER: 0x8d41,
		ARRAY_BUFFER: 0x8892,
		ELEMENT_ARRAY_BUFFER: 0x8893,
		STATIC_DRAW: 0x88e4,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		COLOR_ATTACHMENT2: 0x8ce2,
		NONE: 0,
		DEPTH_ATTACHMENT: 0x8d00,
		DEPTH_COMPONENT24: 0x81a6,
		TEXTURE_2D: 0x0de1,
		TEXTURE0: 0x84c0,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		LINEAR: 0x2601,
		NEAREST: 0x2600,
		NEAREST_MIPMAP_NEAREST: 0x2700,
		LINEAR_MIPMAP_LINEAR: 0x2703,
		CLAMP_TO_EDGE: 0x812f,
		REPEAT: 0x2901,
		MIRRORED_REPEAT: 0x8370,
		RGBA: 0x1908,
		UNSIGNED_BYTE: 0x1401,
		FLOAT: 0x1406,
		HALF_FLOAT: 0x140b,
		RGBA16F: 0x881a,
		UNSIGNED_SHORT: 0x1403,
		UNSIGNED_INT: 0x1405,
		COLOR_BUFFER_BIT: 0x4000,
		DEPTH_BUFFER_BIT: 0x0100,
		BLEND: 0x0be2,
		CULL_FACE: 0x0b44,
		DEPTH_TEST: 0x0b71,
		LESS: 0x0201,
		LEQUAL: 0x0203,
		BACK: 0x0405,
		FRONT: 0x0404,
		CCW: 0x0901,
		TRIANGLES: 0x0004,
		LINES: 0x0001,
		POINTS: 0x0000,
		SRC_ALPHA: 0x0302,
		ONE_MINUS_SRC_ALPHA: 0x0303,
		ZERO: 0,
		ONE: 1,
		getParameter(param) {
			if (param === this.MAX_TEXTURE_SIZE) return 4096;
			if (param === this.MAX_RENDERBUFFER_SIZE) return 4096;
			if (param === this.MAX_TEXTURE_IMAGE_UNITS) return 16;
			if (param === this.MAX_VERTEX_TEXTURE_IMAGE_UNITS) return 8;
			if (param === this.MAX_COMBINED_TEXTURE_IMAGE_UNITS) return 24;
			if (param === this.MAX_DRAW_BUFFERS) return 4;
			if (param === this.MAX_COLOR_ATTACHMENTS) return 4;
			if (param === this.VENDOR) return options.vendor ?? "Masked Vendor";
			if (param === this.RENDERER) return options.renderer ?? "Masked Renderer";
			if (debugInfo && param === debugInfo.UNMASKED_VENDOR_WEBGL) {
				return debugInfo.vendor;
			}
			if (debugInfo && param === debugInfo.UNMASKED_RENDERER_WEBGL) {
				return debugInfo.renderer;
			}
			return 0;
		},
		getExtension(name) {
			if (name === "WEBGL_debug_renderer_info" && debugInfo) {
				return {
					UNMASKED_VENDOR_WEBGL: debugInfo.UNMASKED_VENDOR_WEBGL,
					UNMASKED_RENDERER_WEBGL: debugInfo.UNMASKED_RENDERER_WEBGL,
				};
			}
			if (supportedExtensions.includes(name)) return {};
			return null;
		},
		getSupportedExtensions() {
			return supportedExtensions;
		},
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		createFramebuffer() {
			return {};
		},
		deleteFramebuffer() {},
		createTexture() {
			return {};
		},
		deleteTexture() {},
		createRenderbuffer() {
			return {};
		},
		deleteRenderbuffer() {},
		createShader() {
			return {};
		},
		deleteShader() {},
		shaderSource() {},
		compileShader() {},
		getShaderParameter() {
			return true;
		},
		getShaderInfoLog() {
			return "";
		},
		createProgram() {
			return {};
		},
		deleteProgram() {},
		attachShader() {},
		linkProgram() {},
		getProgramParameter() {
			return true;
		},
		getProgramInfoLog() {
			return "";
		},
		getUniformLocation() {
			return {};
		},
		createBuffer() {
			return {};
		},
		deleteBuffer() {},
		bindBuffer() {},
		bufferData() {},
		enableVertexAttribArray() {},
		vertexAttribPointer() {},
		bindVertexArray() {},
		bindFramebuffer() {},
		framebufferTexture2D() {},
		framebufferRenderbuffer() {},
		checkFramebufferStatus() {
			return options.framebufferComplete === false ? 0x8cd6 : this.FRAMEBUFFER_COMPLETE;
		},
		bindRenderbuffer() {},
		renderbufferStorage() {},
		bindTexture() {},
		activeTexture() {},
		texParameteri() {},
		texImage2D() {},
		pixelStorei() {},
		generateMipmap() {},
		viewport() {},
		clearColor() {},
		clearDepth() {},
		clear() {},
		enable() {},
		disable() {},
		depthMask() {},
		depthFunc() {},
		colorMask() {},
		frontFace() {},
		cullFace() {},
		blendFuncSeparate() {},
		useProgram() {},
		uniform1i() {},
		uniform1f() {},
		uniform3f() {},
		uniform4f() {},
		uniform4fv() {},
		uniformMatrix4fv() {},
		uniformMatrix3fv() {},
		drawArrays() {},
		drawElements() {},
	};
}

function createFakeCanvas(gl) {
	const listeners = new Map();
	return {
		width: 640,
		height: 360,
		contextOptions: null,
		getContext(type, options) {
			this.contextOptions = options;
			return type === "webgl2" ? gl : null;
		},
		addEventListener(name, listener) {
			let list = listeners.get(name);
			if (!list) {
				list = [];
				listeners.set(name, list);
			}
			list.push(listener);
		},
		removeEventListener(name, listener) {
			const list = listeners.get(name);
			if (!list) return;
			const index = list.indexOf(listener);
			if (index >= 0) list.splice(index, 1);
		},
		dispatch(name, event = {}) {
			const list = listeners.get(name);
			if (!list) return;
			for (const listener of list) {
				listener(event);
			}
		},
	};
}

function createWebGLSession(options, canvas, handlers = {}) {
	const backend = new WebGLBackend(options);
	backend.attach({
		surface: { canvas },
		events: {
			emit(event) {
				if (event.type === "device-lost") {
					handlers.onDeviceLost?.(event.info);
			} else if (event.type === "device-restored") {
					handlers.onDeviceRestored?.();
			}
			},
		},
	});
	return backend;
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

async function testInitRequiresWebGL2() {
	const canvas = createFakeCanvas(null);
	const backend = createWebGLSession({}, canvas);
	await assert.rejects(
		backend.initialize(),
		/WebGL2 context|requires WebGL2|acquire WebGL2/
	);
}

async function testInitAndPassRouting() {
	const warnings = [];
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	const backend = createWebGLSession({}, canvas);
	await backend.initialize();

	assert.equal(backend.profile.id, "webgl");
	assert.equal(backend.profile.frameScheduling, "on-demand");
	assert.equal(backend.isEarlyZPrepassEnabled(), true);
	assert.equal(backend._frameServices._enableEarlyZPrepass, true);
	assert.deepEqual(canvas.contextOptions, {
		alpha: true,
		antialias: false,
		depth: true,
		stencil: false,
		premultipliedAlpha: true,
		preserveDrawingBuffer: false,
		powerPreference: "high-performance",
	});
	assert.equal("passExecutors" in backend, false);
	assert.deepEqual(backend.profile.capabilities, {
		displayHDR: true,
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		meshParticles: false,
		oit: true,
		occlusionCulling: false,
		postProcess: true,
		renderTargets: true,
		renderTargetReadback: true,
	});
	assert.equal("postProcessCapabilities" in backend, false);

	const calls = [];
	installFrameServices(backend, {
		resize(width, height) {
			calls.push(["resize", width, height]);
		},
		destroy() {
			calls.push(["destroy"]);
		},
	});
	backend._frameGraphRuntime = {
		beginFrame(context) {
			calls.push(["begin", context]);
		},
		executePass(pass, context) {
			calls.push(["pass", pass.stage, context]);
		},
		endFrame(context) {
			calls.push(["end", context]);
		},
		abortFrame() {
			calls.push(["abort"]);
		},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};

	backend.resize({ width: 800, height: 600 });
	await backend.beginFrame({ frameId: 1 });
	await backend.executePass({ stage: "main-opaque" }, { frameId: 1 });
	await backend.executePass(
		{ stage: "particle-sim" },
		{ transient: new Map([["pipeline:particle-delta-time-seconds", 0.016]]) }
	);
	await backend.executePass({ stage: "particles" }, { frameId: 1 });
	await backend.executePass({ stage: "shadow" }, { frameId: 1 });
	await backend.executePass({ stage: "shadow" }, { frameId: 1 });
	await backend.endFrame();
	backend.destroy();

	assert.deepEqual(calls, [
		["resize", 800, 600],
		["begin", { frameId: 1 }],
		["pass", "main-opaque", { frameId: 1 }],
		["pass", "particles", { frameId: 1 }],
		["pass", "shadow", { frameId: 1 }],
		["pass", "shadow", { frameId: 1 }],
		["end", { frameId: 1 }],
		["destroy"],
	]);
	assert.equal(
		warnings.filter((warning) =>
			warning.includes("does not support pass \"shadow\"")
		).length,
		0
	);
}

async function testDebugInfoUsesWebGLDebugRendererExtension() {
	const canvas = createFakeCanvas(
		createFakeWebGL2Context({
			debugRendererInfo: {
				UNMASKED_VENDOR_WEBGL: 0x9245,
				UNMASKED_RENDERER_WEBGL: 0x9246,
				vendor: "GPU Vendor",
				renderer: "GPU Renderer",
			},
			supportedExtensions: [
				"EXT_beta",
				"EXT_alpha",
				"EXT_color_buffer_float",
				"OES_texture_half_float_linear",
			],
		})
	);
	const backend = createWebGLSession({}, canvas);

	assert.equal(backend.getDebugInfo().available, false);
	await backend.initialize();

	const debugInfo = backend.getDebugInfo();
	assert.equal(debugInfo.backend, "webgl");
	assert.equal(debugInfo.api, "webgl2");
	assert.equal(debugInfo.available, true);
	assert.equal(debugInfo.device.vendor, "GPU Vendor");
	assert.equal(debugInfo.device.renderer, "GPU Renderer");
	assert.equal(debugInfo.device.raw.unmaskedVendor, "GPU Vendor");
	assert.equal(debugInfo.device.raw.unmaskedRenderer, "GPU Renderer");
	assert.deepEqual(debugInfo.features, [
		"EXT_alpha",
		"EXT_beta",
		"EXT_color_buffer_float",
		"OES_texture_half_float_linear",
	]);
	assert.equal(debugInfo.limits.MAX_TEXTURE_SIZE, 4096);
	assert.equal(debugInfo.limits.MAX_TEXTURE_IMAGE_UNITS, 16);
	assert.equal(debugInfo.limits.MAX_DRAW_BUFFERS, 4);
}

async function testDebugInfoFallsBackToMaskedWebGLStrings() {
	const canvas = createFakeCanvas(
		createFakeWebGL2Context({
			vendor: "Masked Test Vendor",
			renderer: "Masked Test Renderer",
		})
	);
	const backend = createWebGLSession({}, canvas);
	await backend.initialize();

	const debugInfo = backend.getDebugInfo();
	assert.equal(debugInfo.available, true);
	assert.equal(debugInfo.device.vendor, "Masked Test Vendor");
	assert.equal(debugInfo.device.renderer, "Masked Test Renderer");
	assert.equal(debugInfo.device.raw.vendor, "Masked Test Vendor");
	assert.equal(debugInfo.device.raw.renderer, "Masked Test Renderer");
}

async function testEarlyZPrepassOptionCanDisable() {
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	const backend = createWebGLSession(
		{ enableEarlyZPrepass: false },
		canvas
	);
	await backend.initialize();

	assert.equal(backend.isEarlyZPrepassEnabled(), false);
	assert.equal(backend._frameServices._enableEarlyZPrepass, false);
}

async function testStrictHDRCapabilityBoundary() {
	for (const [options, code] of [
		[
			{ supportedExtensions: ["OES_texture_half_float_linear"] },
			"hdr-float-color-buffer-unavailable",
		],
		[
			{ supportedExtensions: ["EXT_color_buffer_float"], framebufferComplete: false },
			"hdr-float-color-buffer-unavailable",
		],
		[
			{ supportedExtensions: ["EXT_color_buffer_float"] },
			"hdr-float-linear-filtering-unavailable",
		],
	]) {
		const backend = createWebGLSession(
			{},
			createFakeCanvas(createFakeWebGL2Context(options)),
		);
		await assert.rejects(backend.initialize(), (error) => error?.code === code);
		assert.equal(backend._gl, null);
		assert.equal(backend._contextServices, null);
		assert.equal(backend.getDebugInfo().available, false);
	}
}

async function testRestoreCapabilityLossStaysLost() {
	const supportedExtensions = [
		"EXT_color_buffer_float",
		"OES_texture_half_float_linear",
	];
	const canvas = createFakeCanvas(createFakeWebGL2Context({ supportedExtensions }));
	let restored = false;
	const backend = createWebGLSession({}, canvas, {
		onDeviceRestored() {
			restored = true;
		},
	});
	await backend.initialize();
	canvas.dispatch("webglcontextlost", { preventDefault() {} });
	supportedExtensions.splice(0, supportedExtensions.length);
	captureWarnMessages(() => canvas.dispatch("webglcontextrestored", {}));
	assert.equal(backend._contextLost, true);
	assert.equal(restored, false);
}

async function testContextLostAndRestored() {
	const deviceLostInfos = [];
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	const backend = createWebGLSession({}, canvas, {
		onDeviceLost(info) {
			deviceLostInfos.push(info);
		},
	});
	await backend.initialize();

	const originalServices = backend._frameServices;
	let prevented = false;
	const warnings = captureWarnMessages(() => {
		canvas.dispatch("webglcontextlost", {
			preventDefault() {
				prevented = true;
			},
		});
		assert.equal(prevented, true);
		assert.equal(backend._contextLost, true);
		assert.equal(deviceLostInfos.length, 1);
		assert.equal(deviceLostInfos[0].reason, "context-lost");
		backend.resize({ width: 777, height: 333 });

		canvas.dispatch("webglcontextrestored", {});
		assert.equal(backend._contextLost, false);
		assert.notStrictEqual(backend._frameServices, originalServices);
		assert.equal(backend._frameServices._session.width, 777);
		assert.equal(backend._frameServices._session.height, 333);
	});

	assert.equal(
		warnings.some((warning) => warning.includes("context was lost")),
		true
	);
	assert.equal(
		warnings.some((warning) => warning.includes("context was restored")),
		true
	);
}

async function testAuxiliaryRasterExtensionPersistsAcrossContextRestore() {
	const requiredExtensions = [
		"EXT_color_buffer_float",
		"OES_texture_float_linear",
	];
	const canvas = createFakeCanvas(
		createFakeWebGL2Context({ supportedExtensions: requiredExtensions }),
	);
	const backend = createWebGLSession({}, canvas);
	const facade = backend.extensions.getBackendExtension(
		WEBGL_AUXILIARY_RASTER_EXTENSION,
	);
	assert.ok(facade);
	assert.equal(facade.getAvailability().state, "temporarily-unavailable");

	await backend.initialize();
	assert.equal(facade.getAvailability().state, "ready");
	backend.onDeviceLost({ reason: "test" });
	assert.equal(facade.getAvailability().state, "temporarily-unavailable");
	backend.restore();
	assert.strictEqual(
		backend.extensions.getBackendExtension(
			WEBGL_AUXILIARY_RASTER_EXTENSION,
		),
		facade,
	);
	assert.equal(facade.getAvailability().state, "ready");
}

async function testPublicLifecycleMethods() {
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	const backend = createWebGLSession({}, canvas);
	await backend.initialize();

	const originalServices = backend._frameServices;
	const warnings = captureWarnMessages(() => {
		backend.onDeviceLost({
			reason: "manual-test",
			message: "manual loss",
		});
		assert.equal(backend._contextLost, true);

		backend.restore();
		assert.equal(backend._contextLost, false);
		assert.notStrictEqual(backend._frameServices, originalServices);
	});

	assert.equal(
		warnings.some((warning) => warning.includes("manual loss")),
		true
	);
}

function testParticleDeltaTimeIsClampedToSafeMaximum() {
	const backend = createWebGLSession({}, {});
	const transient = new Map([
		[PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 1000],
	]);
	const deltaTimeSeconds = backend._resolveParticleDeltaTime({ transient });
	assert.equal(deltaTimeSeconds, 0.5);
}

function createDependencyContext() {
	return {
		features: {
			enableLighting: true,
			enableSH: true,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: true,
			enableClusteredLighting: true,
			warnings: [],
			clusteredLightingOptions: {},
		},
		postProcess: createResolvedPostProcess({
			"motion-blur": { enabled: true },
			dof: { enabled: true },
		}),
		scene: {
			environment: { backgroundEnabled: false },
			particleSystems: [{}],
			shadowCasterPackets: [],
			reflectivePackets: [],
			transparentPackets: [],
			decalPackets: [],
		},
		transient: new Map(),
		framePlan: {
			stageOrder: [
				{ id: "particle-sim", kind: "backend-pass", dependsOn: [] },
				{ id: "main-opaque", kind: "backend-pass", dependsOn: [] },
				{ id: "particles", kind: "backend-pass", dependsOn: ["main-opaque"] },
				{ id: "postprocess", kind: "renderer", dependsOn: ["particles"] },
			],
			backendPasses: [
				{
					stage: "particle-sim",
					executor: "backend",
					enabled: true,
					dependsOn: [],
				},
				{
					stage: "main-opaque",
					executor: "backend",
					enabled: true,
					dependsOn: [],
				},
				{
					stage: "particles",
					executor: "backend",
					enabled: true,
					dependsOn: ["main-opaque"],
				},
			],
		},
	};
}

async function testBackendPlanOmitsRendererOwnedPostProcessStage() {
	const backend = createWebGLSession({}, {});
	const context = createDependencyContext();
	installFrameServices(backend, {
		resize() {},
		destroy() {},
	});
	backend._frameGraphRuntime = {
		beginFrame() {},
		executePass() {},
		endFrame() {},
		abortFrame() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};
	backend._contextWorkQueue.bindContext();

	await backend.beginFrame(context);
	assert.equal(backend._plannedPasses.has("postprocess"), false);
	assert.equal(backend._plannedPassOrder.has("postprocess"), false);
	await backend.abortFrame();
	backend.destroy();
}

async function testEndFramePromiseAndReadbackScheduling() {
	const backend = createWebGLSession({}, {});
	const context = createDependencyContext();
	let resolveEnd;
	const endGate = new Promise((resolve) => {
		resolveEnd = resolve;
	});
	let readbackCalls = 0;
	installFrameServices(backend, {
		readCustomRenderTargetColor: async () => {
			readbackCalls++;
			return { data: new Uint8Array([1, 2, 3, 4]), width: 1, height: 1 };
		},
		restoreContextWorkBaseline() {},
		commitTemporalFrame() {},
		destroy() {},
	});
	backend._frameGraphRuntime = {
		beginFrame() {},
		executePass() {},
		endFrame: () => endGate,
		abortFrame() {},
		commitGraphAnalysis() {},
		abortGraphAnalysis() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};
	backend._postProcessRuntime.commitFrame = () => undefined;
	backend._contextWorkQueue.bindContext();

	await backend.beginFrame(context);
	await assert.rejects(
		backend.readRenderTargetColor("scene"),
		(error) => error?.code === "active-frame",
	);
	const ending = backend.endFrame();
	await Promise.resolve();
	assert.equal(backend._activeContext, context);
	resolveEnd();
	await ending;
	assert.equal(backend._activeContext, null);
	const result = await backend.readRenderTargetColor("scene");
	assert.deepEqual(Array.from(result.data), [1, 2, 3, 4]);
	assert.equal(readbackCalls, 1);
	backend.destroy();
}

async function testWarmupAndResizeUseContextWorkScheduling() {
	const backend = createWebGLSession({}, {});
	const context = createDependencyContext();
	context.scene.opaquePackets = [];
	const log = [];
	installFrameServices(backend, {
		resize(width, height) {
			log.push(`resize:${width}x${height}`);
		},
		warmupCoordinator: {
			warmup(_context, _plan, _options, _postProcessPlan, signal) {
				assert.equal(signal instanceof AbortSignal, true);
				log.push("warmup");
				return {
					phase: "webgl-programs",
					total: 1,
					compiled: 1,
					skipped: 0,
					failed: 0,
					errors: [],
				};
			},
		},
		restoreContextWorkBaseline() {
			log.push("baseline");
		},
		commitTemporalFrame() {},
		destroy() {},
	});
	backend._frameGraphRuntime = {
		beginFrame() {
			log.push("begin");
		},
		executePass() {},
		endFrame() {
			log.push("end");
		},
		abortFrame() {},
		commitGraphAnalysis() {},
		abortGraphAnalysis() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};
	backend._postProcessRuntime.commitFrame = () => undefined;
	backend._contextWorkQueue.bindContext();

	await backend.beginFrame(context);
	const warmup = backend.warmup(context, {
		includeCorePasses: false,
		includePostProcess: false,
		includeShadowPass: false,
		includeParticles: false,
	});
	backend.resize({ width: 320, height: 180 });
	backend.resize({ width: 640, height: 360 });
	await Promise.resolve();
	assert.equal(log.includes("warmup"), false);
	assert.equal(log.some((entry) => entry.startsWith("resize:")), false);
	await backend.endFrame();
	// Frame-end settlement must not wait for unrelated idle maintenance.
	assert.equal(log.filter((entry) => entry.startsWith("resize:")).length, 0);
	let guard = 0;
	while (!log.some((entry) => entry.startsWith("resize:")) && guard++ < 100) {
		await Promise.resolve();
	}
	assert.equal(log.filter((entry) => entry.startsWith("resize:")).length, 1);
	assert.ok(log.includes("resize:640x360"));
	await warmup;
	assert.ok(log.indexOf("end") < log.indexOf("resize:640x360"));
	assert.ok(log.indexOf("resize:640x360") < log.indexOf("warmup"));
	backend.destroy();
}

async function run() {
	await testInitRequiresWebGL2();
	await testInitAndPassRouting();
	await testDebugInfoUsesWebGLDebugRendererExtension();
	await testDebugInfoFallsBackToMaskedWebGLStrings();
	await testEarlyZPrepassOptionCanDisable();
	await testStrictHDRCapabilityBoundary();
	await testRestoreCapabilityLossStaysLost();
	await testContextLostAndRestored();
	await testAuxiliaryRasterExtensionPersistsAcrossContextRestore();
	await testPublicLifecycleMethods();
	testParticleDeltaTimeIsClampedToSafeMaximum();
	await testBackendPlanOmitsRendererOwnedPostProcessStage();
	await testEndFramePromiseAndReadbackScheduling();
	await testWarmupAndResizeUseContextWorkScheduling();
	console.log("WebGL backend v2 tests passed");
}

await run();
