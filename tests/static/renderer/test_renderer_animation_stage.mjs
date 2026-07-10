import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Material } from "../../../src/materials/Material.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { AnimationClip } from "../../../src/animation/AnimationClip.ts";
import { KeyframeTrack } from "../../../src/animation/KeyframeTrack.ts";
import { Renderer } from "../../../src/renderers/Renderer.ts";
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
		const renderer = new Renderer(backend, canvas, camera);
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
			],
		});
		const mixer = renderer.animationSystem.createMixer(renderer.scene.root);
		mixer.addClip(clip);
		mixer.bindNode("/animated", meshInstance);
		mixer.clipAction("move").play();

		await renderer.renderFrame(0);
		await renderer.renderFrame(16);
		await renderer.renderFrame(516);

		assert.equal(backend.beginFrameCount, 3);
		assert.ok(backend.mainOpaqueCenters.length >= 3);
		assert.ok(
			backend.mainOpaqueCenters[2] > backend.mainOpaqueCenters[1] + 0.9
		);
		assert.equal(backend.executedStages.includes("animation-sim"), false);
		assert.equal(backend.sharedStages.includes("animation-sim"), false);

		renderer.animationAutoRender = false;
		await renderer.renderFrame(1016);
		assert.equal(backend.beginFrameCount, 3);

		console.log("Renderer animation stage tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
