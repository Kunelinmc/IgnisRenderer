import assert from "node:assert/strict";
import {
	Camera,
	CameraShakePlugin,
	Matrix4,
	OrbitCamera,
	perlinNoise1D,
	Quaternion,
	Renderer,
	Vector3,
} from "../../../src/index.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class StubBackend extends TestRenderBackend {
	constructor(frameScheduling = "always") {
		super();
		this.type = "stub";
		this.frameScheduling = frameScheduling;
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
			bloom: false,
		};
		installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.beginFrameCount = 0;
		this.beginSnapshots = [];
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

	beginFrame(context) {
		this.beginFrameCount++;
		const camera = context.viewCamera;
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
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
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
		await renderer.renderFrame(16);

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
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin();
		plugin.attach(renderer);
		plugin.trigger({
			intensity: 1,
			durationSeconds: 0.08,
			frequencyHz: 18,
			positionAmplitude: { x: 0.4, y: 0.3, z: 0.2 },
		});

		await renderer.renderFrame(16);
		await renderer.renderFrame(32);
		assert.equal(backend.beginFrameCount, 2);

		await renderer.renderFrame(220);
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

async function testStackedShakesKeepIndependentEnvelopes() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("always");
		const camera = new Camera();
		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultFalloffExponent: 2,
		});
		plugin.attach(renderer);
		const firstPositionAmplitude = { x: 1, y: 0, z: 0 };
		const firstRotationAmplitude = { x: 1, y: 0, z: 0 };
		plugin.trigger({
			intensity: 0.5,
			durationSeconds: 1,
			frequencyHz: 1,
			positionAmplitude: firstPositionAmplitude,
			rotationAmplitude: firstRotationAmplitude,
		});
		firstPositionAmplitude.x = 99;
		firstRotationAmplitude.x = 99;

		await renderer.renderFrame(1);
		await renderer.renderFrame(101);

		plugin.trigger({
			intensity: 0.25,
			durationSeconds: 0.5,
			frequencyHz: 2,
			positionAmplitude: { x: 2, y: 0, z: 0 },
			rotationAmplitude: { x: 0, y: 0, z: 0 },
		});
		await renderer.renderFrame(101);

		const firstElapsedSeconds = 0.1;
		const firstEnvelope = Math.pow(1 - firstElapsedSeconds / 1, 2);
		const firstPhase = firstElapsedSeconds * 1 * Math.PI * 2;
		const firstContribution =
			1 * 0.5 * firstEnvelope *
			perlinNoise1D(firstPhase + 0.713, 101) * 2;
		const secondContribution =
			2 * 0.25 * perlinNoise1D(0.713, 101) * 2;
		const expectedRotationX =
			1 * 0.5 * firstEnvelope *
			perlinNoise1D(firstPhase + 1.371, 401) * 2;
		const expectedQuaternion = new Quaternion()
			.fromEuler(expectedRotationX, 0, 0)
			.normalize();
		const stackedSnapshot = backend.beginSnapshots.at(-1);

		assert.ok(
			Math.abs(
				stackedSnapshot.position.x -
				(firstContribution + secondContribution)
			) <= 1e-7,
			"each stacked shake should retain its own elapsed envelope and frequency"
		);
		assertQuaternionClose(stackedSnapshot.quaternion, expectedQuaternion);

		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

async function testTraumaAccumulatesDecaysAndUsesExponent() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("always");
		const camera = new Camera();
		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultFrequencyHz: 1,
			defaultTraumaDecayRatePerSecond: 1,
			defaultTraumaExponent: 2,
			defaultPositionAmplitude: { x: 1, y: 0, z: 0 },
			defaultRotationAmplitude: { x: 0, y: 0, z: 0 },
		});
		plugin.attach(renderer);

		plugin.addTrauma(0.4);
		plugin.addTrauma(0.8);
		assert.equal(plugin.trauma, 1, "Trauma should accumulate and clamp to 1");

		await renderer.renderFrame(1);
		const initialExpected = perlinNoise1D(0.713, 101) * 2;
		assert.ok(
			Math.abs(backend.beginSnapshots[0].position.x - initialExpected) <= 1e-7,
			"full Trauma should use a gain of 1"
		);

		await renderer.renderFrame(101);
		const elapsedSeconds = 0.1;
		const expectedTrauma = 0.9;
		const expectedGain = expectedTrauma ** 2;
		const expectedPositionX =
			expectedGain *
			perlinNoise1D(elapsedSeconds * Math.PI * 2 + 0.713, 101) * 2;
		assert.ok(Math.abs(plugin.trauma - expectedTrauma) <= 1e-12);
		assert.ok(
			Math.abs(
				backend.beginSnapshots.at(-1).position.x - expectedPositionX
			) <= 1e-7,
			"Trauma should decay linearly and map through the configured exponent"
		);

		plugin.stop();
		assert.equal(plugin.trauma, 0);
		assert.equal(plugin.isActive, false);
		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

async function testTraumaKeepsOnDemandRendererAwakeUntilDecayCompletes() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("on-demand");
		const camera = new Camera();
		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultTraumaDecayRatePerSecond: 2,
		});
		plugin.attach(renderer);
		plugin.setTrauma(0.1);

		await renderer.renderFrame(1);
		assert.equal(backend.beginFrameCount, 1);
		assert.equal(plugin.isActive, true);

		await renderer.renderFrame(101);
		assert.equal(plugin.trauma, 0);
		assert.equal(plugin.isActive, false);
		assert.equal(
			backend.beginFrameCount,
			1,
			"on-demand rendering should become clean once Trauma reaches zero"
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
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
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
			await renderer.renderFrame(t);
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

async function testOrbitRotationShakeRotatesAroundPivot() {
	const restoreRuntime = useMockBrowserRuntime();
	try {
		const backend = new StubBackend("always");
		const camera = new OrbitCamera(new Vector3(4, -2, 3), 180);
		camera.theta = -0.42;
		camera.phi = 1.24;
		camera.updatePosition();

		const renderer = new Renderer(backend, TEST_CANVAS, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;

		const plugin = new CameraShakePlugin({
			defaultIntensity: 1,
			defaultDurationSeconds: 0.24,
			defaultFrequencyHz: 17,
			defaultPositionAmplitude: { x: 0, y: 0, z: 0 },
			defaultRotationAmplitude: { x: 0.025, y: 0.03, z: 0.02 },
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
		const baseDistance = distance3(basePosition, baseTarget);

		plugin.trigger({
			positionAmplitude: { x: 0, y: 0, z: 0 },
			rotationAmplitude: { x: 0.025, y: 0.03, z: 0.02 },
		});
		await renderer.renderFrame(16);

		assert.equal(backend.beginFrameCount, 1);
		const beginSnapshot = backend.beginSnapshots[0];
		assert.ok(beginSnapshot.target, "orbit snapshot should include target");
		assert.ok(
			distance3(beginSnapshot.position, basePosition) > 1e-5,
			"rotation-only orbit shake should move camera position around pivot"
		);
		assert.ok(
			distance3(beginSnapshot.target, baseTarget) <= 1e-5,
			"rotation-only orbit shake should keep pivot target stable"
		);
		assert.ok(
			Math.abs(distance3(beginSnapshot.position, beginSnapshot.target) - baseDistance) <=
				1e-4,
			"rotation-only orbit shake should preserve orbit radius"
		);

		assertVectorClose(camera.position, basePosition);
		assertVectorClose(camera.target, baseTarget);

		plugin.detach();
	}
	finally {
		restoreRuntime();
	}
}

await testShakeAppliedBeforeFrameAndRestoredAfterFrame();
await testOnDemandSchedulingStaysAwakeWhileShakeIsActive();
await testStackedShakesKeepIndependentEnvelopes();
await testTraumaAccumulatesDecaysAndUsesExponent();
await testTraumaKeepsOnDemandRendererAwakeUntilDecayCompletes();
await testOrbitCameraShakeDoesNotDriftPoseBetweenFrames();
await testOrbitRotationShakeRotatesAroundPivot();
console.log("Camera shake plugin tests passed");
