import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { FrameCoordinator } from "../../../src/rendering/FrameCoordinator.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { installNoopPostProcessAdapter } from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class TargetBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "webgpu";
		this.capabilities = {
			...this.capabilities,
			renderTargets: true,
			renderTargetReadback: true,
		};
		this.frameScheduling = "always";
		this.contexts = [];
		this.executedPasses = [];
		this.readRequests = [];
		installNoopPostProcessAdapter(this, "webgpu");
	}
	getAttachments({ width, height }) { return { width, height }; }
	beginFrame(context) { this.contexts.push(context); }
	executePass(pass) { this.executedPasses.push(pass.stage); }
	readRenderTargetColor(id, attachmentIndex = 0, options = {}) {
		this.readRequests.push({ id, attachmentIndex, options });
		const format = attachmentIndex === 1 ?
			TextureFormat.RGBA16Float : TextureFormat.RGBA8Unorm;
		const bytesPerPixel = attachmentIndex === 1 ? 8 : 4;
		return Promise.resolve({
			bytes: new Uint8Array((options.width ?? 1) * (options.height ?? 1) * bytesPerPixel),
			width: options.width ?? 1,
			height: options.height ?? 1,
			format,
			bytesPerPixel,
			bytesPerRow: (options.width ?? 1) * bytesPerPixel,
			origin: "top-left",
			toFloat32: () => new Float32Array(1),
			toRGBAFloat32: () => new Float32Array(4),
			toNormalizedRGBA8Float32: () => new Float32Array(4),
		});
	}
}

function createRenderer() {
	const backend = new TargetBackend();
	const canvas = {
		width: 320,
		height: 180,
		getBoundingClientRect: () => ({ width: 320, height: 180 }),
	};
	const renderer = new Renderer(canvas, backend, new Camera());
	renderer.features.enableReflection = false;
	renderer.features.enableEnvironment = false;
	return { renderer, backend };
}

async function testHandleJobsAndCommittedReadback() {
	const { renderer, backend } = createRenderer();
	const target = renderer.renderTargets.create({
		size: { mode: "fixed", width: 64, height: 32 },
		color: [
			{ format: TextureFormat.RGBA8Unorm },
			{ format: TextureFormat.RGBA16Float },
		],
		depth: { format: TextureFormat.Depth32Float },
	});
	const recurring = target.registerJob({ kind: "custom-pass", execute() {} });
	const ticket = target.enqueueJob({
		kind: "custom-pass",
		execute() {},
		readback: { attachmentIndex: 1, width: 2, height: 2 },
	});
	await renderer.renderFrame(16);
	const completion = await ticket.done;
	const context = backend.contexts.at(-1);
	assert.ok(context.renderTargets.has(target.id));
	assert.equal(context.renderTargetJobs.size, 1);
	assert.ok(backend.executedPasses.includes("render-target-views"));
	assert.equal(completion.targetId, target.id);
	assert.equal(completion.generation, 1);
	assert.equal(completion.readback.format, TextureFormat.RGBA16Float);
	assert.equal(completion.readback.origin, "top-left");
	assert.equal(backend.readRequests[0].id, target.id);
	assert.equal(backend.readRequests[0].attachmentIndex, 1);
	recurring.destroy();
	target.destroy();
	assert.throws(() => target.enqueueJob({ kind: "custom-pass", execute() {} }), /destroyed/);
}

async function testCancellationAndSceneValidation() {
	const { renderer } = createRenderer();
	const customTarget = renderer.renderTargets.create({
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA8Unorm }],
	});
	assert.throws(
		() => customTarget.enqueueJob({ kind: "scene-view", camera: renderer.camera }),
		/require one rgba16float/,
	);
	const ticket = customTarget.enqueueJob({ kind: "custom-pass", execute() {} });
	ticket.cancel();
	await assert.rejects(ticket.done, /cancelled/);
	const sceneTarget = renderer.renderTargets.create({
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA16Float }],
		depth: { format: TextureFormat.Depth32Float },
	});
	const sceneTicket = sceneTarget.enqueueJob({
		kind: "scene-view",
		camera: renderer.camera,
	});
	sceneTicket.cancel();
	await assert.rejects(sceneTicket.done, /cancelled/);
}

async function testProbeFaceCameraMatrices() {
	const coordinator = new FrameCoordinator(new TargetBackend());
	const jobs = [];
	const handle = {
		enqueueJob(descriptor) {
			jobs.push(descriptor);
			return { done: Promise.resolve({ readback: null }) };
		},
	};
	const delegate = { renderTargets: { createInternal: () => handle } };
	const directions = [
		{ x: 1, y: 0, z: 0 }, { x: -1, y: 0, z: 0 },
		{ x: 0, y: 1, z: 0 }, { x: 0, y: -1, z: 0 },
		{ x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: -1 },
	];
	const upVectors = [
		{ x: 0, y: -1, z: 0 }, { x: 0, y: -1, z: 0 },
		{ x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: -1 },
		{ x: 0, y: -1, z: 0 }, { x: 0, y: -1, z: 0 },
	];
	const position = { x: 3, y: 4, z: 5 };
	for (const faceIndex of [0, 1, 2, 3, 4, 5, 99, NaN]) {
		await coordinator._enqueueProbeFace(delegate, {
			faceIndex, faceSize: 4, captureWorldPosition: position, captureFar: 50,
			includeEnvironment: true, includeMeshes: true, includeTransparent: true,
			includeParticles: false, includeShadows: false, targetId: "probe-test",
		});
		const { camera } = jobs.at(-1);
		const resolvedFace = Number.isNaN(faceIndex) ? 0 : Math.min(5, faceIndex);
		const direction = directions[resolvedFace];
		const up = upVectors[resolvedFace];
		const pointInFront = {
			x: position.x + direction.x,
			y: position.y + direction.y,
			z: position.z + direction.z,
		};
		const centered = Matrix4.transformPoint(camera.viewProjectionMatrix, pointInFront);
		assert.ok(Math.abs(centered.x) < 1e-6 && Math.abs(centered.y) < 1e-6);
		assert.ok(centered.z > -1 && centered.z < 1);
		const above = Matrix4.transformPoint(camera.viewMatrix, {
			x: pointInFront.x + up.x, y: pointInFront.y + up.y, z: pointInFront.z + up.z,
		});
		assert.ok(Math.abs(above.x) < 1e-6 && Math.abs(above.y - 1) < 1e-6);
		assert.ok(Math.abs(above.z + 1) < 1e-6);
		assert.equal(camera.fov, 90);
		assert.equal(camera.aspectRatio, 1);
		assert.equal(camera.near, 0.1);
		assert.equal(camera.far, 50);
		assert.ok(camera.isPointInFrustum(pointInFront));
		const world = Matrix4.transformPoint(camera.inverseViewMatrix, { x: 0, y: 0, z: -1 });
		for (const axis of ["x", "y", "z"]) {
			assert.ok(Math.abs(world[axis] - pointInFront[axis]) < 1e-6);
		}
	}
	assert.notEqual(jobs[0].camera.inverseViewMatrix, jobs[1].camera.inverseViewMatrix);
}

function testDescriptorValidation() {
	const { renderer } = createRenderer();
	const multisampled = renderer.renderTargets.create({
		size: { mode: "fixed", width: 4, height: 4 },
		color: [{ format: TextureFormat.RGBA8Unorm }],
		sampleCount: 4.9,
	});
	assert.equal(multisampled.descriptor.sampleCount, 4);
	assert.throws(
		() => renderer.renderTargets.create({
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.RGBA8Unorm }],
			sampleCount: Number.NaN,
		}),
		/sampleCount must be a finite number/,
	);
	assert.throws(
		() => renderer.renderTargets.create({
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.Depth32Float }],
		}),
		/renderable color format/,
	);
}

async function run() {
	const originalWindow = globalThis.window;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		testDescriptorValidation();
		await testCancellationAndSceneValidation();
		await testHandleJobsAndCommittedReadback();
		await testProbeFaceCameraMatrices();
		console.log("Renderer render target handle tests passed");
	} finally {
		globalThis.window = originalWindow;
	}
}

await run();
