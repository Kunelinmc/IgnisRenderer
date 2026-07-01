import assert from "node:assert/strict";
import { Logger } from "../../../src/foundation/Logger.ts";
import { WebGLBackend } from "../../../src/renderers/WebGLBackend.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../../src/pipeline/types.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFakeWebGL2Context(options = {}) {
	const debugInfo = options.debugRendererInfo;
	const supportedExtensions = options.supportedExtensions ?? [];
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
			return this.FRAMEBUFFER_COMPLETE;
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
		getContext(type) {
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
	assert.equal(backend._frameExecutor._enableEarlyZPrepass, true);
	assert.equal("passExecutors" in backend, false);
	assert.deepEqual(backend.profile.capabilities, {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		oit: true,
		occlusionCulling: false,
		postProcess: true,
		customRenderTargets: true,
		customRenderPasses: true,
		renderTargetReadback: true,
	});
	assert.equal("postProcessCapabilities" in backend, false);

	const calls = [];
	backend._frameExecutor = {
		resize(width, height) {
			calls.push(["resize", width, height]);
		},
		destroy() {
			calls.push(["destroy"]);
		},
	};
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
	backend.beginFrame({ frameId: 1 });
	backend.executePass({ stage: "main-opaque" }, { frameId: 1 });
	backend.executePass(
		{ stage: "particle-sim" },
		{ transient: new Map([["pipeline:particle-delta-time-seconds", 0.016]]) }
	);
	backend.executePass({ stage: "particles" }, { frameId: 1 });
	backend.executePass({ stage: "shadow" }, { frameId: 1 });
	backend.executePass({ stage: "shadow" }, { frameId: 1 });
	backend.endFrame();
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
			supportedExtensions: ["EXT_beta", "EXT_alpha"],
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
	assert.deepEqual(debugInfo.features, ["EXT_alpha", "EXT_beta"]);
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
	assert.equal(backend._frameExecutor._enableEarlyZPrepass, false);
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

	const originalExecutor = backend._frameExecutor;
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

		canvas.dispatch("webglcontextrestored", {});
		assert.equal(backend._contextLost, false);
		assert.notStrictEqual(backend._frameExecutor, originalExecutor);
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

async function testPublicLifecycleMethods() {
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	const backend = createWebGLSession({}, canvas);
	await backend.initialize();

	const originalExecutor = backend._frameExecutor;
	const warnings = captureWarnMessages(() => {
		backend.onDeviceLost({
			reason: "manual-test",
			message: "manual loss",
		});
		assert.equal(backend._contextLost, true);

		backend.restore();
		assert.equal(backend._contextLost, false);
		assert.notStrictEqual(backend._frameExecutor, originalExecutor);
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

function testBackendPlanOmitsRendererOwnedPostProcessStage() {
	const backend = createWebGLSession({}, {});
	const context = createDependencyContext();
	backend._frameExecutor = {
		resize() {},
		destroy() {},
	};
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

	backend.beginFrame(context);
	assert.equal(backend._plannedPasses.has("postprocess"), false);
	assert.equal(backend._plannedPassOrder.has("postprocess"), false);
}

async function run() {
	await testInitRequiresWebGL2();
	await testInitAndPassRouting();
	await testDebugInfoUsesWebGLDebugRendererExtension();
	await testDebugInfoFallsBackToMaskedWebGLStrings();
	await testEarlyZPrepassOptionCanDisable();
	await testContextLostAndRestored();
	await testPublicLifecycleMethods();
	testParticleDeltaTimeIsClampedToSafeMaximum();
	testBackendPlanOmitsRendererOwnedPostProcessStage();
	console.log("WebGL backend v2 tests passed");
}

await run();
