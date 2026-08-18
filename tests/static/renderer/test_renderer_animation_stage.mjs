import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Node } from "../../../src/core/Node.ts";
import { Material } from "../../../src/materials/Material.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { AnimationClip } from "../../../src/animation/AnimationClip.ts";
import { KeyframeTrack } from "../../../src/animation/KeyframeTrack.ts";
import { Skeleton } from "../../../src/animation/Skeleton.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class StubBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
		};
		installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.frameScheduling = "on-demand";
		this.beginFrameCount = 0;
		this.sharedStages = [];
		this.executedStages = [];
		this.mainOpaqueCenters = [];
		this.mainOpaquePackets = [];
	}

	resize() {}

	getAttachments({ width, height }) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		};
	}

	beginFrame() {
		this.beginFrameCount++;
	}

	executeSharedPass(pass) {
		this.sharedStages.push(pass.stage);
	}

	executePass(pass, context) {
		this.executedStages.push(pass.stage);
		if (pass.stage === "main-opaque") {
			const packet = context.scene.opaquePackets[0];
			if (packet) {
				this.mainOpaqueCenters.push(packet.worldBounds.center.x);
			}
			this.mainOpaquePackets.push(
				context.scene.opaquePackets.map((candidate) => ({
					meshInstanceId: candidate.meshInstance.id,
					centerX: candidate.worldBounds.center.x,
					deformationRevision: candidate.deformationRevision,
				}))
			);
		}
	}

	endFrame() {}
}

function createTriangleMesh() {
	return MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new StubBackend();
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const camera = new Camera();
		camera.position.set(0, 0, 5);
		const renderer = new Renderer({ backend, canvas, camera });
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const meshInstance = renderer.scene.add(
			new MeshInstance({
				mesh: createTriangleMesh(),
				name: "animatedMesh",
			})
		);
		const joint = renderer.scene.add(new Node({ name: "animatedJoint" }));
		const skinnedMesh = createTriangleMesh();
		const skinnedGeometry = skinnedMesh.primitives[0].geometry;
		skinnedGeometry.joints0 = new Uint16Array(12);
		skinnedGeometry.weights0 = new Float32Array([
			1, 0, 0, 0,
			1, 0, 0, 0,
			1, 0, 0, 0,
		]);
		const skinnedInstance = renderer.scene.add(
			new MeshInstance({
				mesh: skinnedMesh,
				name: "skinnedMesh",
				skeleton: new Skeleton({
					joints: [joint],
					inverseBindMatrices: [Matrix4.identity()],
				}),
			})
		);

		const clip = new AnimationClip({
			name: "move",
			duration: 1,
			tracks: [
				new KeyframeTrack({
					binding: {
						targetType: "node",
						targetPath: "/animated",
						property: "translation",
					},
					times: [0, 1],
					values: [0, 0, 0, 2, 0, 0],
					valueSize: 3,
					interpolation: "linear",
				}),
				new KeyframeTrack({
					binding: {
						targetType: "node",
						targetPath: "/joint",
						property: "translation",
					},
					times: [0, 1],
					values: [0, 0, 0, 2, 0, 0],
					valueSize: 3,
					interpolation: "linear",
				}),
			],
		});
		const mixer = renderer.animationSystem.createMixer(renderer.scene.root);
		mixer.addClip(clip);
		mixer.bindNode("/animated", meshInstance);
		mixer.bindNode("/joint", joint);
		mixer.bindMorph("/skinned", skinnedInstance);
		mixer.clipAction("move").play();

		await renderer.renderFrame(0);
		await renderer.renderFrame(16);
		await renderer.renderFrame(516);

		assert.equal(backend.beginFrameCount, 3);
		assert.ok(backend.mainOpaqueCenters.length >= 3);
		assert.ok(
			backend.mainOpaqueCenters[2] > backend.mainOpaqueCenters[1] + 0.9
		);
		const skinnedAt16 = backend.mainOpaquePackets[1].find(
			(packet) => packet.meshInstanceId === skinnedInstance.id
		);
		const skinnedAt516 = backend.mainOpaquePackets[2].find(
			(packet) => packet.meshInstanceId === skinnedInstance.id
		);
		assert.ok(skinnedAt16);
		assert.ok(skinnedAt516);
		assert.ok(skinnedAt516.centerX > skinnedAt16.centerX + 0.9);
		assert.notEqual(
			skinnedAt516.deformationRevision,
			skinnedAt16.deformationRevision
		);
		assert.equal(backend.executedStages.includes("animation-sim"), false);
		assert.equal(backend.sharedStages.includes("animation-sim"), false);

		camera.setRotationFromEuler(0, 0.05, 0);
		const cameraChangedFrame = await renderer.renderFrame(600);
		assert.equal(cameraChangedFrame.incremental.plan.forceFullFrame, true);
		const animationAfterCameraFrame = await renderer.renderFrame(700);
		assert.equal(
			animationAfterCameraFrame.incremental.plan.forceFullFrame,
			false
		);

		renderer.animationAutoRender = false;
		await renderer.renderFrame(1016);
		assert.equal(backend.beginFrameCount, 5);

		console.log("Renderer animation stage tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
