import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { Logger } from "../../../src/foundation/Logger.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { drawWebGLPacket } from "../../../src/backends/webgl/WebGLScenePass.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";
import { createScenePassCaptureGL, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { WebGLEnvironmentRenderer } from "../../../src/backends/webgl/WebGLEnvironmentRenderer.ts";
import { WebGLSceneRuntime } from "../../../src/backends/webgl/WebGLSceneRuntime.ts";
import { WebGLFullscreenRenderer } from "../../../src/backends/webgl/WebGLFullscreenRenderer.ts";
import { WebGLTransparencyWarmupContributor } from "../../../src/backends/webgl/WebGLTransparencyRuntime.ts";

function testEnvironmentRendererOwnsProgramAndDrawLifecycle() {
	const gl = createScenePassCaptureGL();
	gl.drawArrays = () => {
		gl.calls.drawArrays = (gl.calls.drawArrays ?? 0) + 1;
	};
	let slotDestroyed = false;
	const uniform = {};
	const program = {
		program: {},
		uniforms: {
			environmentMap: uniform,
			environmentBasisRight: uniform,
			environmentBasisUp: uniform,
			environmentBasisBackward: uniform,
			environmentIsOrthographic: uniform,
			environmentBackgroundTint: uniform,
			environmentBackgroundExposure: uniform,
			environmentBackgroundStrength: uniform,
		},
	};
	const renderer = new WebGLEnvironmentRenderer({
		gl,
		programCompiler: {
			createSlot(descriptor) {
				return {
					label: descriptor.label,
					get: () => program,
					tryGet: () => program,
					warmup: () => ({
						label: descriptor.label,
						isComplete: () => true,
						finalize() {},
					}),
					invalidate() {},
					destroy() {
						slotDestroyed = true;
					},
				};
			},
		},
		targets: { _sceneFramebuffer: {} },
		textures: {
			getEnvironmentTexture() {
				return { texture: {} };
			},
		},
		getFullscreenVao: () => ({}),
		getWidth: () => 64,
		getHeight: () => 32,
	});
	const context = {
		viewCamera: {
			type: CameraType.Perspective,
			fov: 60,
			aspectRatio: 2,
			viewMatrix: Matrix4.identity(),
		},
		scene: {
			environment: {
				backgroundEnabled: true,
				backgroundTexture: {},
				backgroundTintLinear: { r: 1, g: 0.5, b: 0.25 },
				backgroundExposure: 1,
				backgroundStrength: 1,
			},
		},
	};
	const rendered = renderer.render(context);
	assert.equal(rendered, true);
	assert.equal(gl.calls.drawArrays, 1);
	assert.deepEqual(gl.calls.depthMask, [false, true]);
	assert.equal(renderer.collectWarmupTasks({
		context: {},
		plan: { enableEnvironment: true },
		postProcessPlan: null,
	}).length, 1);
	const dirtyRects = [
		{ x: 2, y: 3, width: 4, height: 5 },
		{ x: 40, y: 12, width: 6, height: 4 },
	];
	const draws = [];
	let scissorEnabled = false;
	const enable = gl.enable;
	const disable = gl.disable;
	gl.enable = (capability) => {
		if (capability === gl.SCISSOR_TEST) scissorEnabled = true;
		enable(capability);
	};
	gl.disable = (capability) => {
		if (capability === gl.SCISSOR_TEST) scissorEnabled = false;
		disable(capability);
	};
	gl.drawArrays = () => draws.push(scissorEnabled ? gl.calls.scissor.at(-1) : "full");
	const scene = new WebGLSceneRuntime({
		gl, modelMatrixCache: new Map(), modelMatrixKeysThisFrame: new Set(),
		getWidth: () => 64, getHeight: () => 32,
		isIncrementalPartial: () => true,
		resolveDirtyRects: () => dirtyRects,
		setScissorRect: (x, y, width, height, viewportHeight) =>
			gl.scissor(x, viewportHeight - y - height, width, height),
		renderEnvironment: (frame) => renderer.render(frame),
	});
	scene.renderEnvironment({ ...context, features: { enableEnvironment: true } });
	assert.deepEqual(draws, [
		{ x: 2, y: 24, width: 4, height: 5 },
		{ x: 40, y: 16, width: 6, height: 4 },
	], "environment draws must stay inside the cleared dirty regions");
	assert.equal(scissorEnabled, false);
	renderer.destroy();
	assert.equal(slotDestroyed, true);
}

async function testTransparencyWarmupContributorSelectsRuntimePrograms() {
	const warmed = [];
	const contributor = new WebGLTransparencyWarmupContributor({
		warmupCopyProgram() {
			warmed.push("copy");
		},
		warmupOITResolveProgram() {
			warmed.push("oit-resolve");
		},
	});
	const tasks = contributor.collectWarmupTasks({
		context: { features: { enableOIT: true } },
		plan: { materials: [{ transmissionFactor: 1 }] },
		postProcessPlan: null,
	});

	assert.deepEqual(
		tasks.map(({ label, priority }) => ({ label, priority })),
		[
			{ label: "WebGLCopyProgram", priority: "core" },
			{ label: "WebGLOITResolveProgram", priority: "optional" },
		],
	);
	for (const task of tasks) await task.run();
	assert.deepEqual(warmed, ["copy", "oit-resolve"]);

	const emptyTasks = contributor.collectWarmupTasks({
		context: { features: { enableOIT: false } },
		plan: { materials: [] },
		postProcessPlan: null,
	});
	assert.deepEqual(emptyTasks, []);
}

function testSceneProgramDrawBuffersMatchFragmentOutputCount() {
	const gl = createScenePassCaptureGL();
	const material = new ShaderMaterial({
		uniformBindings: [],
	});
	const sceneProgram = {
		program: {},
		uniforms: {},
		targetMode: "single",
		samplerLayout: {
			units: {},
			activeSamplerNames: [],
			required: 0,
			available: 16,
		},
	};

	const packet = createTestDrawPacket({
		id: "test-pkt",
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
		meshInstance: {
			id: "test-mesh",
			skeleton: null,
		},
	});

	const deps = {
		gl,
		targets: {
			_sceneFramebuffer: null,
			_sceneNormalTexture: null,
			_materialGBufferEnabled: false,
		},
		drawState: {
			oitPassMode: 0,
			activeDrawBuffers: [
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
			],
		},
		scenePrograms: {},
		geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		textures: {
			getBaseColorTexture() {
				return { texture: null };
			},
		},
		animationPayloads: null,
		modelMatrixCache: new Map(),
		modelMatrixKeysThisFrame: new Set(),
		getWidth: () => 64,
		getHeight: () => 64,
		isIncrementalPartial: () => false,
		resolveDirtyRects: () => [{ x: 0, y: 0, width: 64, height: 64 }],
		resolvePacketsForRect: (_context, packets) => packets,
		setScissorRect: () => {},
		bindGlobalUniforms: () => {},
		bindAnimationPayload: () => true,
		getLightState: () => null,
		getShadowSamplingState: () => ({
			enabled: false,
			transmittanceAvailable: false,
		}),
	};

	gl.calls.drawBuffers = [];
	drawWebGLPacket(deps, sceneProgram, packet, false, {});

	// Should have changed draw buffers to [COLOR_ATTACHMENT0], then drawn, then restored back to original
	assert.deepEqual(gl.calls.drawBuffers, [
		[gl.COLOR_ATTACHMENT0],
		[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2],
	]);

	deps.drawState.activeDrawBuffers = [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
		gl.COLOR_ATTACHMENT3,
		gl.COLOR_ATTACHMENT4,
	];
	gl.calls.drawBuffers = [];
	drawWebGLPacket(deps, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 3,
		samplerLayout: { units: {}, activeSamplerNames: [], required: 0, available: 16 },
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
	drawWebGLPacket(deps, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 5,
		samplerLayout: { units: {}, activeSamplerNames: [], required: 0, available: 16 },
	}, packet, false, {});
	assert.deepEqual(gl.calls.drawBuffers, []);

	const drawError = new Error("draw failed");
	const originalDrawElements = gl.drawElements;
	gl.drawElements = () => {
		throw drawError;
	};
	assert.throws(
		() => drawWebGLPacket(deps, {
			program: {},
			uniforms: {},
			targetMode: "mrt",
			colorOutputCount: 3,
			samplerLayout: { units: {}, activeSamplerNames: [], required: 0, available: 16 },
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

function testFullscreenPresentationRepaintsIncrementalCanvas() {
	const warnings = [];
	Logger.configure({ resetOnceKeys: true, sink: { warn: (...args) => warnings.push(args.join(" ")) } });
	let domain = "display-encoded";
	const gl = createScenePassCaptureGL();
	let drawCount = 0;
	let dirtyRectResolveCount = 0;
	let scissorSetCount = 0;
	let presentedCount = 0;
	gl.createVertexArray = () => ({});
	gl.deleteVertexArray = () => {};
	gl.viewport = () => {};
	gl.drawArrays = () => drawCount++;
	const program = {
		program: {},
		uniforms: {
			sourceMap: null,
			hdrEnabled: null,
		},
	};
	const renderer = new WebGLFullscreenRenderer({
		gl,
		targets: {
			_presentSourceTexture: {},
			_sceneColorTexture: null,
		},
		programCompiler: {
			createSlot: () => ({
				get: () => program,
				tryGet: () => program,
				warmup: () => ({ poll: () => program }),
				destroy: () => {},
			}),
		},
		getWidth: () => 320,
		getHeight: () => 180,
		isIncrementalPartial: () => true,
		resolveDirtyRects: () => {
			dirtyRectResolveCount++;
			return [{ x: 96, y: 32, width: 128, height: 128 }];
		},
		setScissorRect: () => scissorSetCount++,
		getColorDomain: () => domain,
		markPresented: () => presentedCount++,
	});

	assert.equal(renderer.present({ incremental: {} }), true);
	assert.equal(drawCount, 1);
	assert.equal(dirtyRectResolveCount, 0);
	assert.equal(scissorSetCount, 0);
	assert.equal(presentedCount, 1);
	assert.equal(gl.calls.disable.includes(gl.SCISSOR_TEST), true);
	assert.equal(warnings.length, 0, "encoded presentation must not warn");
	for (const nextDomain of ["scene-linear-hdr", "display-linear", "display-encoded"]) {
		domain = nextDomain;
		assert.equal(renderer.present(null), true);
		assert.equal(renderer.present(null), true);
	}
	assert.equal(warnings.length, 2);
	assert.match(warnings[0], /presentation-color-domain-incomplete-scene-linear-hdr/);
	assert.match(warnings[0], /tone mapping.*display encoding/);
	assert.match(warnings[1], /presentation-color-domain-incomplete-display-linear/);
	assert.match(warnings[1], /may not match expectations/);
	const shader = readFileSync(new URL("../../../src/shaders/webgl/utility/presentFragment.glsl", import.meta.url), "utf8");
	assert.doesNotMatch(shader, /acesFitted|hdrSoftShoulder|linearToSrgb|linearSrgbToDisplayP3|uExposure/,
		"presentation must not apply missing color conversions");
	Logger.reset();
	renderer.destroy();
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
			shadowCasterSubmissions: [],
			shadowTransmitterSubmissions: [],
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
	testEnvironmentRendererOwnsProgramAndDrawLifecycle,
	testTransparencyWarmupContributorSelectsRuntimePrograms,
	testSceneProgramDrawBuffersMatchFragmentOutputCount,
	testFullscreenPresentationRepaintsIncrementalCanvas,
	testWebGLBackendWarmupDelegatesToCoordinator,
], "WebGL backend integration tests");
