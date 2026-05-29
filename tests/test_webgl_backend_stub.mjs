import assert from "node:assert/strict";
import { Logger } from "../src/foundation/Logger.ts";
import { WebGLBackend } from "../src/renderers/WebGLBackend.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../src/pipeline/types.ts";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

function createFakeWebGL2Context() {
	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
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
			return 0;
		},
		getExtension() {
			return null;
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

function createRendererBridge(overrides = {}) {
	const warnings = [];
	return {
		warnings,
		bridge: {
			canvas: { width: 1, height: 1 },
			...overrides,
		},
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

async function testInitRequiresWebGL2() {
	const backend = new WebGLBackend();
	const { bridge } = createRendererBridge();
	backend.setRenderer(bridge);
	const canvas = createFakeCanvas(null);
	await assert.rejects(
		backend.init(canvas),
		/WebGL2 context|requires WebGL2|acquire WebGL2/
	);
}

async function testInitAndPassRouting() {
	const backend = new WebGLBackend();
	const { bridge, warnings } = createRendererBridge();
	backend.setRenderer(bridge);
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	await backend.init(canvas);

	assert.equal(backend.type, "webgl");
	assert.equal(backend.frameScheduling, "on-demand");
	assert.deepEqual(backend.passExecutors, {
		"animation-sim": "shared",
		"particle-sim": "backend",
	});
	assert.deepEqual(backend.capabilities, {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		clusteredLighting: true,
		oit: true,
	});
	assert.equal("postProcessCapabilities" in backend, false);

	const calls = [];
	backend._frameExecutor = {
		beginFrame(context) {
			calls.push(["begin", context]);
		},
		executePass(pass, context) {
			calls.push(["pass", pass.stage, context]);
		},
		endFrame() {
			calls.push(["end"]);
		},
		resize(width, height) {
			calls.push(["resize", width, height]);
		},
		destroy() {
			calls.push(["destroy"]);
		},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};

	backend.resize(800, 600);
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
		["end"],
		["destroy"],
	]);
	assert.equal(
		warnings.filter((warning) =>
			warning.includes("does not support pass \"shadow\"")
		).length,
		0
	);
}

async function testContextLostAndRestored() {
	const backend = new WebGLBackend();
	const deviceLostInfos = [];
	const { bridge } = createRendererBridge({
		onDeviceLost(info) {
			deviceLostInfos.push(info);
		},
	});
	backend.setRenderer(bridge);
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	await backend.init(canvas);

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
	const backend = new WebGLBackend();
	const { bridge } = createRendererBridge();
	backend.setRenderer(bridge);
	const canvas = createFakeCanvas(createFakeWebGL2Context());
	await backend.init(canvas);

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
	const backend = new WebGLBackend();
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
		},
		transient: new Map(),
	};
}

function testBackendPlanOmitsRendererOwnedPostProcessStage() {
	const backend = new WebGLBackend();
	const context = createDependencyContext();
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		endFrame() {},
		resize() {},
		destroy() {},
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
	await testContextLostAndRestored();
	await testPublicLifecycleMethods();
	testParticleDeltaTimeIsClampedToSafeMaximum();
	testBackendPlanOmitsRendererOwnedPostProcessStage();
	console.log("WebGL backend v2 tests passed");
}

await run();
