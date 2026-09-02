import assert from "node:assert/strict";

import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGLShadowRuntime } from "../../../src/backends/webgl/WebGLShadowRuntime.ts";

function createGL() {
	let handle = 0;
	const deletedTextures = [];
	const deletedFramebuffers = [];
	const viewports = [];
	return {
		FRAMEBUFFER: 0x8d40,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		DEPTH_ATTACHMENT: 0x8d00,
		COLOR_ATTACHMENT0: 0x8ce0,
		DEPTH_COMPONENT24: 0x81a6,
		DEPTH_COMPONENT: 0x1902,
		UNSIGNED_INT: 0x1405,
		RGBA8: 0x8058,
		RGBA: 0x1908,
		UNSIGNED_BYTE: 0x1401,
		R32F: 0x822e,
		RED: 0x1903,
		FLOAT: 0x1406,
		TEXTURE_2D: 0x0de1,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		NEAREST: 0x2600,
		CLAMP_TO_EDGE: 0x812f,
		NONE: 0,
		TRIANGLES: 0x0004,
		BLEND: 0x0be2,
		DEPTH_TEST: 0x0b71,
		CULL_FACE: 0x0b44,
		SCISSOR_TEST: 0x0c11,
		DEPTH_BUFFER_BIT: 0x0100,
		COLOR_BUFFER_BIT: 0x4000,
		ZERO: 0,
		ONE: 1,
		SRC_COLOR: 0x0300,
		deletedTextures,
		deletedFramebuffers,
		viewports,
		createTexture() {
			return { id: `texture-${++handle}` };
		},
		createFramebuffer() {
			return { id: `framebuffer-${++handle}` };
		},
		deleteTexture(texture) {
			deletedTextures.push(texture);
		},
		deleteFramebuffer(framebuffer) {
			deletedFramebuffers.push(framebuffer);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D() {},
		bindFramebuffer() {},
		framebufferTexture2D() {},
		drawBuffers() {},
		readBuffer() {},
		checkFramebufferStatus() {
			return this.FRAMEBUFFER_COMPLETE;
		},
		useProgram() {},
		disable() {},
		enable() {},
		depthMask() {},
		colorMask() {},
		clearDepth() {},
		clear() {},
		viewport(x, y, width, height) {
			viewports.push({ x, y, width, height });
		},
		scissor() {},
		blendFuncSeparate() {},
		clearColor() {},
		bindVertexArray() {},
		uniformMatrix4fv() {},
		uniform3f() {},
		drawElements() {},
	};
}

function createHost(gl) {
	let depthProgramAvailable = true;
	return {
		gl,
		programCompiler: {
			createSlot(descriptor) {
				const depth = descriptor.label === "WebGLShadowDepthProgram";
				const resource = depth ? {
					program: { id: "depth-program" },
					uniforms: { mvp: null },
				} : {
					program: { id: "transmittance-program" },
					uniforms: { mvp: "mvp", transmittance: "transmittance" },
				};
				return {
					label: descriptor.label,
					get: () => resource,
					tryGet: () => depth && !depthProgramAvailable ? null : resource,
					warmup: () => ({
						label: descriptor.label,
						isComplete: () => true,
						finalize() {},
					}),
					invalidate() {},
					destroy() {},
				};
			},
		},
		geometry: {
			getGeometry() {
				return null;
			},
		},
		maxTextureSize: 4096,
		getSceneFramebuffer() {
			return { id: "scene-framebuffer" };
		},
		getWidth() {
			return 320;
		},
		getHeight() {
			return 180;
		},
		setDepthProgramAvailable(value) {
			depthProgramAvailable = value;
		},
	};
}

function createContext() {
	return {
		features: {},
		shadowPlan: { hasRasterWork: true },
		scene: {
			lights: [],
			shadowCasterSubmissions: [],
			shadowTransmitterSubmissions: [{}],
			particleSystems: [],
			sceneBounds: null,
			camera: null,
		},
		shadowMaps: new Map(),
		transient: new Map(),
	};
}

function testShadowWarmupContributesDepthAndTransmittancePrograms() {
	const runtime = new WebGLShadowRuntime(createHost(createGL()));
	const disabled = runtime.collectWarmupTasks({
		context: {},
		plan: { enableShadows: false },
		postProcessPlan: null,
	});
	assert.deepEqual(disabled, []);
	const enabled = runtime.collectWarmupTasks({
		context: {},
		plan: { enableShadows: true },
		postProcessPlan: null,
	});
	assert.equal(enabled.length, 1);
	const handles = enabled[0].run();
	assert.deepEqual(
		handles.map((handle) => handle.label),
		[
			"WebGLShadowDepthProgram",
			"WebGLShadowDepthProgram_static:1",
			"WebGLShadowDepthProgram_skin4:0",
			"WebGLShadowDepthProgram_skin4:1",
			"WebGLShadowDepthProgram_skin8:0",
			"WebGLShadowDepthProgram_skin8:1",
			"WebGLShadowTransmittanceProgram",
			"WebGLShadowTransmittanceProgram_static:1",
			"WebGLShadowTransmittanceProgram_skin4:0",
			"WebGLShadowTransmittanceProgram_skin4:1",
			"WebGLShadowTransmittanceProgram_skin8:0",
			"WebGLShadowTransmittanceProgram_skin8:1",
		],
	);
	runtime.destroy();
}

function createShadow(overrides = {}) {
	return {
		enabled: true,
		shadowMap: {},
		shadowMapBaseSize: 64,
		shadowMapSize: 64,
		strategyType: "single",
		cascadeCount: 1,
		cascadeViewProjectionMatrices: [],
		cascadeSplits: [],
		viewProjectionMatrix: Matrix4.identity(),
		atlasTileSize: 0,
		...overrides,
	};
}

function createLightState(shadows = [createShadow()], spotShadows = []) {
	return {
		directionalShadows: shadows,
		spotShadows,
	};
}

function testRuntimeLifecycleAndStableSamplingState() {
	const gl = createGL();
	const host = createHost(gl);
	const runtime = new WebGLShadowRuntime(host);
	const context = createContext();
	const lights = createLightState();
	const sampling = runtime.getSamplingState();

	runtime.beginFrame(context);
	runtime.prepareFrame(context, lights);
	assert.equal(runtime.getSamplingState(), sampling);
	assert.equal(sampling.enabled, true);
	assert.equal(sampling.atlasTileSize, 64);
	assert.equal(lights.directionalShadows[0].atlasTileSize, 64);
	assert.throws(
		() => runtime.prepareFrame(context, lights),
		/expected phase=begun, actual=prepared/,
	);
	runtime.renderPreparedFrame(context);
	assert.throws(
		() => runtime.renderPreparedFrame(context),
		/expected phase=prepared, actual=rendered/,
	);
	runtime.abortFrame();
	assert.equal(sampling.enabled, false);
	assert.deepEqual(runtime.describeGraphResources(), { resources: [], bindings: [] });
	runtime.destroy();
	assert.ok(gl.deletedTextures.length >= 2);
	assert.ok(gl.deletedFramebuffers.length >= 1);
}

function testContextMismatchAndProgramFallback() {
	const gl = createGL();
	const host = createHost(gl);
	const runtime = new WebGLShadowRuntime(host);
	const context = createContext();
	const otherContext = createContext();
	runtime.beginFrame(context);
	assert.throws(
		() => runtime.prepareFrame(otherContext, createLightState()),
		/context different from beginFrame/,
	);
	runtime.abortFrame();

	host.setDepthProgramAvailable(false);
	runtime.beginFrame(context);
	runtime.prepareFrame(context, createLightState());
	const sampling = runtime.getSamplingState();
	assert.equal(sampling.enabled, false);
	assert.ok(sampling.atlasTexture);
	assert.equal(runtime.describeGraphResources().resources.length, 2);
	runtime.renderPreparedFrame(context);
	runtime.destroy();
}

function testCSMSpotPlanAndParticleResourceCatalog() {
	const gl = createGL();
	const host = createHost(gl);
	const runtime = new WebGLShadowRuntime(host);
	const context = createContext();
	context.scene.shadowTransmitterSubmissions = [];
	runtime.beginFrame(context);
	context.scene.particleSystems.push({});
	const csm = createShadow({
		strategyType: "csm",
		cascadeCount: 4,
		shadowMapSize: 32,
		cascadeViewProjectionMatrices: [
			Matrix4.identity(),
			Matrix4.identity(),
			Matrix4.identity(),
			Matrix4.identity(),
		],
		cascadeSplits: [
			[0, 1, 0, 0],
			[1, 2, 1, 0],
			[2, 3, 0, 1],
			[3, 4, 1, 1],
		],
	});
	const spot = createShadow();
	runtime.prepareFrame(context, createLightState([csm], [spot]));
	const catalog = runtime.describeGraphResources();
	const particle = catalog.resources.find((entry) =>
		entry.id === "shadow:particle-volume");
	assert.ok(particle);
	assert.equal(particle.format, "r32float");
	assert.equal(particle.width, 512);
	assert.equal(particle.height, 1024);
	assert.equal(
		catalog.bindings.find((entry) =>
			entry.resourceId === "shadow:particle-volume").physicalId,
		"webgl:slot:shadow:particle-volume",
	);
	runtime.renderPreparedFrame(context);
	assert.ok(gl.viewports.some((viewport) =>
		viewport.x === 32 && viewport.y === 32 &&
		viewport.width === 32 && viewport.height === 32));
	assert.ok(gl.viewports.some((viewport) =>
		viewport.x === 0 && viewport.y === 64 &&
		viewport.width === 64 && viewport.height === 64));
	runtime.destroy();
	assert.ok(gl.deletedTextures.length >= 3);
}

function run() {
	testRuntimeLifecycleAndStableSamplingState();
	testShadowWarmupContributesDepthAndTransmittancePrograms();
	testContextMismatchAndProgramFallback();
	testCSMSpotPlanAndParticleResourceCatalog();
	console.log("WebGL shadow runtime tests passed");
}

run();
