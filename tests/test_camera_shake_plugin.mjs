import assert from "node:assert/strict";
import {
	Camera,
	CameraShakePlugin,
	Matrix4,
	OrbitCamera,
	Renderer,
	Vector3,
} from "../src/index.ts";

class StubBackend {
	constructor(frameScheduling = "always") {
		this.type = "stub";
		this.frameScheduling = frameScheduling;
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			skybox: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			bloom: false,
		};
		this.beginFrameCount = 0;
		this.beginSnapshots = [];
	}

	async init() {}

	resize() {}

	getAttachments(width, height) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		};
	}

	beginFrame(context) {
		this.beginFrameCount++;
		const camera = context.camera;
		const snapshot = {
			position: {
				x: camera.position.x,
				y: camera.position.y,
				z: camera.position.z,
			},
			quaternion: {
				x: camera.quaternion.x,
				y: camera.quaternion.y,
				z: camera.quaternion.z,
				w: camera.quaternion.w,
			},
		};

		if (camera instanceof OrbitCamera) {
			snapshot.target = {
				x: camera.target.x,
				y: camera.target.y,
				z: camera.target.z,
			};
			snapshot.up = {
				x: camera.up.x,
				y: camera.up.y,
				z: camera.up.z,
			};
		}

		this.beginSnapshots.push(snapshot);
	}

	executePass() {}

	endFrame() {}
}

const TEST_CANVAS = {
	width: 320,
	height: 180,
	getBoundingClientRect() {
		return { width: 320, height: 180 };
	},
};

function distance3(a, b) {
	return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function assertVectorClose(actual, expected, epsilon = 1e-7) {
	assert.ok(
		distance3(actual, expected) <= epsilon,
		`expected vector ${JSON.stringify(actual)} to equal ${JSON.stringify(expected)}`
	);
}

function assertQuaternionClose(actual, expected, epsilon = 1e-7) {
	const delta = Math.hypot(
		actual.x - expected.x,
		actual.y - expected.y,
		actual.z - expected.z,
		actual.w - expected.w
	);
	assert.ok(
		delta <= epsilon,
		`expected quaternion ${JSON.stringify(actual)} to equal ${JSON.stringify(expected)}`
	);
}

function useMockBrowserRuntime() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;
	globalThis.window = { devicePixelRatio: 1 };
	globalThis.requestAnimationFrame = () => 0;
	return () => {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	};
}

async function testShakeAppliedBeforeFrameAndRestoredAfterFrame() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("always");
		const camera = new Camera();
		camera.position.set(0, 2, 6);
		camera.updateMatrices();

		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.features.enableGamma = false;
		renderer.features.enableReflection = false;
		renderer.features.enableSkybox = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultIntensity: 1,
			defaultDurationSeconds: 0.25,
			defaultFrequencyHz: 14,
			defaultPositionAmplitude: { x: 0.5, y: 0.4, z: 0.3 },
			defaultRotationAmplitude: { x: 0.03, y: 0.03, z: 0.02 },
		});
		plugin.attach(renderer);

		const basePosition = {
			x: camera.position.x,
			y: camera.position.y,
			z: camera.position.z,
		};
		const baseQuaternion = {
			x: camera.quaternion.x,
			y: camera.quaternion.y,
			z: camera.quaternion.z,
			w: camera.quaternion.w,
		};

		plugin.trigger();
		await renderer.renderScene(16);

		assert.equal(backend.beginFrameCount, 1);
		const beginSnapshot = backend.beginSnapshots[0];
		assert.ok(
			distance3(beginSnapshot.position, basePosition) > 1e-5,
			"shake should be visible in beginFrame camera position"
		);

		assertVectorClose(camera.position, basePosition);
		assertQuaternionClose(camera.quaternion, baseQuaternion);

		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

async function testOnDemandSchedulingStaysAwakeWhileShakeIsActive() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("on-demand");
		const camera = new Camera();
		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.features.enableGamma = false;
		renderer.features.enableReflection = false;
		renderer.features.enableSkybox = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin();
		plugin.attach(renderer);
		plugin.trigger({
			intensity: 1,
			durationSeconds: 0.08,
			frequencyHz: 18,
			positionAmplitude: { x: 0.4, y: 0.3, z: 0.2 },
		});

		await renderer.renderScene(16);
		await renderer.renderScene(32);
		assert.equal(backend.beginFrameCount, 2);

		await renderer.renderScene(220);
		assert.equal(
			backend.beginFrameCount,
			2,
			"renderer should stop rendering once shake has ended"
		);

		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

async function testOrbitCameraShakeDoesNotDriftPoseBetweenFrames() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("always");
		const camera = new OrbitCamera(new Vector3(0, 0, 0), 240);
		camera.theta = 0.35;
		camera.phi = 1.1;
		camera.updatePosition();

		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.features.enableGamma = false;
		renderer.features.enableReflection = false;
		renderer.features.enableSkybox = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultIntensity: 1,
			defaultDurationSeconds: 0.18,
			defaultFrequencyHz: 20,
			defaultPositionAmplitude: { x: 0.3, y: 0.25, z: 0.2 },
			defaultRotationAmplitude: { x: 0.015, y: 0.015, z: 0.01 },
		});
		plugin.attach(renderer);

		const basePosition = {
			x: camera.position.x,
			y: camera.position.y,
			z: camera.position.z,
		};
		const baseTarget = {
			x: camera.target.x,
			y: camera.target.y,
			z: camera.target.z,
		};
		const baseUp = {
			x: camera.up.x,
			y: camera.up.y,
			z: camera.up.z,
		};

		plugin.trigger();
		for (let t = 0; t <= 224; t += 16) {
			await renderer.renderScene(t);
			assertVectorClose(camera.position, basePosition);
			assertVectorClose(camera.target, baseTarget);
			assertVectorClose(camera.up, baseUp);
		}

		const observedShake = backend.beginSnapshots.some((snapshot) => {
			if (!snapshot.target) return false;
			return (
				distance3(snapshot.position, basePosition) > 1e-5 ||
				distance3(snapshot.target, baseTarget) > 1e-5
			);
		});
		assert.equal(observedShake, true);

		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

await testShakeAppliedBeforeFrameAndRestoredAfterFrame();
await testOnDemandSchedulingStaysAwakeWhileShakeIsActive();
await testOrbitCameraShakeDoesNotDriftPoseBetweenFrames();
console.log("Camera shake plugin tests passed");
