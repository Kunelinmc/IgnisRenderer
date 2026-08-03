import assert from "node:assert/strict";import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";import { Matrix4 } from "../../../src/maths/Matrix4.ts";import { drawWebGLPacket } from "../../../src/backends/webgl/WebGLScenePass.ts";import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../../src/pipeline/types.ts";import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";import { createScenePassCaptureGL, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

function testSceneProgramDrawBuffersMatchFragmentOutputCount() {
	const gl = createScenePassCaptureGL();
	const material = new ShaderMaterial({
		uniformBindings: [],
	});
	const sceneProgram = {
		program: {},
		uniforms: {},
		targetMode: "single",
	};

	const packet = {
		id: "test-pkt",
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
		meshInstance: {
			id: "test-mesh",
			skeleton: null,
		},
	};

	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture() {
				return { texture: null, isLinear: false };
			},
		},
		_activeDrawBuffers: [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2],
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
		_setCullMode() {},
	};

	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, sceneProgram, packet, false, {});

	// Should have changed draw buffers to [COLOR_ATTACHMENT0], then drawn, then restored back to original
	assert.deepEqual(gl.calls.drawBuffers, [
		[gl.COLOR_ATTACHMENT0],
		[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2],
	]);

	host._activeDrawBuffers = [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
		gl.COLOR_ATTACHMENT3,
		gl.COLOR_ATTACHMENT4,
	];
	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 3,
	}, packet, false, {});
	assert.deepEqual(gl.calls.drawBuffers, [
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		],
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
			gl.COLOR_ATTACHMENT3,
			gl.COLOR_ATTACHMENT4,
		],
	]);

	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 5,
	}, packet, false, {});
	assert.deepEqual(gl.calls.drawBuffers, []);

	const drawError = new Error("draw failed");
	const originalDrawElements = gl.drawElements;
	gl.drawElements = () => {
		throw drawError;
	};
	assert.throws(
		() => drawWebGLPacket(host, {
			program: {},
			uniforms: {},
			targetMode: "mrt",
			colorOutputCount: 3,
		}, packet, false, {}),
		(error) => error === drawError,
	);
	assert.deepEqual(gl.calls.drawBuffers, [
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		],
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
			gl.COLOR_ATTACHMENT3,
			gl.COLOR_ATTACHMENT4,
		],
	]);
	gl.drawElements = originalDrawElements;
}

function testWebGLBackendParticleDeltaTimeClamp() {
	const backend = new WebGLBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	const transient = new Map([
		[PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 1000],
	]);
	const deltaTimeSeconds = backend._resolveParticleDeltaTime({ transient });
	assert.equal(deltaTimeSeconds, 0.5);
}

async function testWebGLBackendWarmupDelegatesToCoordinator() {
	const backend = new WebGLBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	backend._contextServices = {
		frame: {
			warmupCoordinator: {
				warmup() {
					return {
						phase: "webgl-programs",
						total: 3,
						compiled: 2,
						skipped: 1,
						failed: 0,
						errors: [],
					};
				},
			},
		},
		restoreContextWorkBaseline() {},
	};
	backend._contextWorkQueue.bindContext();
	const report = await backend.warmup({
		viewCamera: {},
		attachments: { width: 1, height: 1 },
		features: {
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess(),
		shadowMaps: new Map(),
		scene: {
			environment: null,
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: Matrix4.identity(),
		transient: new Map(),
	});
	assert.equal(report.total, 3);
	assert.equal(report.compiled, 2);
	assert.equal(report.skipped, 1);
	assert.equal(report.failed, 0);
}

await runWebGLBackendFile([
	testSceneProgramDrawBuffersMatchFragmentOutputCount,
	testWebGLBackendParticleDeltaTimeClamp,
	testWebGLBackendWarmupDelegatesToCoordinator,
], "WebGL backend integration tests");
