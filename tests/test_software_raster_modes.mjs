import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Logger } from "../src/foundation/Logger.ts";
import { BasicMaterial, AlphaMode } from "../src/materials/index.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SoftwareBackend } from "../src/renderers/SoftwareBackend.ts";
import { FakeWorker } from "./helpers/test_fakes.mjs";

const WIDTH = 64;
const HEIGHT = 64;

function createZeroSH() {
	return Array.from({ length: 9 }, () => ({ r: 0, g: 0, b: 0 }));
}

function createRendererBridge(camera, warnings) {
	return {
		canvas: { width: WIDTH, height: HEIGHT },
		camera,
		scene: {
			getLights() {
				return [];
			},
		},
		features: {
			enableShadows: false,
		},
		logger: {
			warn(message) {
				warnings.push(String(message));
			},
		},
	};
}

async function captureWarnMessagesAsync(run) {
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
		await run();
	} finally {
		Logger.reset();
	}
	return warnings;
}

function createContext(backend, camera, packetsByStage = {}) {
	const attachments = backend.getAttachments(WIDTH, HEIGHT);
	const zeroSH = createZeroSH();

	return {
		camera,
		attachments,
		features: {
			enableLighting: false,
			enableGamma: false,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			ssrOptions: {},
			volumetricOptions: {},
			ssaoOptions: {},
			taaOptions: {},
			warnings: [],
		},
		shadowMaps: new Map(),
		scene: {
			sceneBounds: {
				center: { x: 0, y: 0, z: 0 },
				radius: 100,
			},
			lights: [],
			particleSystems: [],
			hasActiveAnimations: false,
			camera,
			skybox: null,
			meshInstances: [],
			shadowMaps: new Map(),
			opaquePackets: packetsByStage.opaquePackets ?? [],
			transparentPackets: packetsByStage.transparentPackets ?? [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
		shCoeffs: zeroSH,
		shAmbientCoeffs: zeroSH,
		worldMatrix: Matrix4.identity(),
		transient: new Map(),
	};
}

function createTrianglePacket(id, color, options = {}) {
	const zOffset = options.zOffset ?? 0;
	const alphaMode = options.alphaMode ?? AlphaMode.Opaque;
	const opacity = options.opacity ?? 1;
	const material = new BasicMaterial({
		name: `material-${id}`,
		diffuse: color,
		doubleSided: true,
		alphaMode,
		opacity,
	});

	const geometry = {
		positions: new Float32Array([
			-0.6,
			-0.6,
			0,
			0.6,
			-0.6,
			0,
			0.0,
			0.6,
			0,
		]),
		normals: new Float32Array([
			0,
			0,
			1,
			0,
			0,
			1,
			0,
			0,
			1,
		]),
		indices: new Uint32Array([0, 1, 2]),
	};

	const primitive = {
		id: `primitive-${id}`,
		geometry,
		material,
		boundingSphere: {
			center: { x: 0, y: 0, z: 0 },
			radius: 2,
		},
		boundingBox: {
			min: { x: -1, y: -1, z: -1 },
			max: { x: 1, y: 1, z: 1 },
		},
		visible: true,
		castShadows: false,
		receiveShadows: false,
	};

	const worldMatrix = Matrix4.fromTranslation([0, 0, zOffset]);
	const mesh = {
		boundingSphere: {
			center: { x: 0, y: 0, z: 0 },
			radius: 2,
		},
		primitives: [primitive],
	};
	const meshInstance = {
		id: `mesh-instance-${id}`,
		mesh,
		worldMatrix,
		skeleton: null,
		visible: true,
	};

	return {
		id: `packet-${id}`,
		meshInstance,
		mesh,
		primitive,
		material,
		geometry,
		worldMatrix,
		normalMatrix: Matrix4.identity(),
		worldBounds: {
			center: { x: 0, y: 0, z: zOffset },
			radius: 2,
		},
		sortDepth: 0,
		pipelineKey: `packet-${id}`,
		passFlags: 0,
	};
}

function createMockScheduler(options = {}) {
	const pools = new Map();
	const registerCalls = [];
	const unregisterCalls = [];
	const scheduleCalls = [];

	const scheduler = {
		hasPool(poolId) {
			return pools.has(poolId);
		},
		registerPool(poolOptions) {
			registerCalls.push(poolOptions.id);
			pools.set(poolOptions.id, poolOptions);
		},
		async schedule(poolId, payload) {
			scheduleCalls.push({ poolId, payload });
			if (options.throwOnSchedule) {
				throw new Error("mock schedule failure");
			}
			return computeBinsFromPayload(payload, !!options.reverseTriangleOrder);
		},
		unregisterPool(poolId) {
			unregisterCalls.push(poolId);
			pools.delete(poolId);
		},
		get registerCalls() {
			return registerCalls;
		},
		get unregisterCalls() {
			return unregisterCalls;
		},
		get scheduleCalls() {
			return scheduleCalls;
		},
	};

	return scheduler;
}

function computeBinsFromPayload(payload, reverseTriangleOrder) {
	if (!payload || payload.type !== "bin-main-pass") {
		throw new Error("Unsupported payload in mock scheduler");
	}

	const tileSize = Math.max(1, Math.floor(payload.tileSize));
	const tileColumns = Math.max(1, Math.ceil(payload.width / tileSize));
	const maxTileX = tileColumns - 1;
	const maxTileY = Math.max(0, Math.ceil(payload.height / tileSize) - 1);
	const bins = new Map();

	const start = Math.max(0, Math.floor(payload.startIndex));
	const end = Math.min(
		payload.triangleBounds.length,
		Math.max(start, Math.floor(payload.endIndex))
	);
	for (let index = start; index < end; index++) {
		const bounds = payload.triangleBounds[index];
		if (!bounds) continue;
		if (bounds.maxTileX < bounds.minTileX || bounds.maxTileY < bounds.minTileY) {
			continue;
		}

		const minTileX = Math.max(0, Math.min(maxTileX, bounds.minTileX));
		const minTileY = Math.max(0, Math.min(maxTileY, bounds.minTileY));
		const maxTileXClamped = Math.max(0, Math.min(maxTileX, bounds.maxTileX));
		const maxTileYClamped = Math.max(0, Math.min(maxTileY, bounds.maxTileY));

		for (let tileY = minTileY; tileY <= maxTileYClamped; tileY++) {
			for (let tileX = minTileX; tileX <= maxTileXClamped; tileX++) {
				const tileIndex = tileY * tileColumns + tileX;
				let bucket = bins.get(tileIndex);
				if (!bucket) {
					bucket = [];
					bins.set(tileIndex, bucket);
				}
				bucket.push(index);
			}
		}
	}

	const entries = [...bins.entries()]
		.sort((left, right) => left[0] - right[0])
		.map(([tileIndex, triangleIndices]) => ({
			tileIndex,
			triangleIndices:
				reverseTriangleOrder ? triangleIndices.reverse() : triangleIndices,
		}));

	return {
		type: "bin-main-pass",
		bins: entries,
	};
}

function createCamera() {
	const camera = new Camera();
	camera.position.set(0, 0, 4);
	camera.aspectRatio = WIDTH / HEIGHT;
	camera.near = 0.1;
	camera.far = 100;
	camera.updateMatrices();
	return camera;
}

async function renderPass(backend, stage, packetsByStage, warnings) {
	const camera = createCamera();
	backend.setRenderer(createRendererBridge(camera, warnings));
	const context = createContext(backend, camera, packetsByStage);
	backend.beginFrame(context);
	await backend.executePass(
		{
			stage,
			executor: "backend",
			enabled: true,
		},
		context
	);
	return context.attachments;
}

async function testDefaultRasterMode() {
	const backend = new SoftwareBackend();
	assert.equal(backend.requestedRasterMode, "scanline");
	assert.equal(backend.activeRasterMode, "scanline");
}

async function testTileModeFallsBackWhenWorkerUnavailable() {
	const originalWorker = globalThis.Worker;
	globalThis.Worker = undefined;

	try {
		const backend = new SoftwareBackend({
			rasterMode: "tile",
		});
		const warnings = await captureWarnMessagesAsync(async () => {
			await renderPass(backend, "main-opaque", { opaquePackets: [] }, []);
		});
		assert.equal(backend.requestedRasterMode, "tile");
		assert.equal(backend.activeRasterMode, "scanline");
		assert.ok(
			warnings.some((warning) =>
				warning.includes("software-raster-worker-unavailable")
			)
		);
	} finally {
		globalThis.Worker = originalWorker;
	}
}

async function testTileModeMatchesScanlineAndPreservesOrder() {
	const originalWorker = globalThis.Worker;
	globalThis.Worker = FakeWorker;

	try {
		const redPacket = createTrianglePacket(
			"transparent-red",
			{ r: 255, g: 0, b: 0 },
			{
				alphaMode: AlphaMode.Blend,
				opacity: 0.65,
			}
		);
		const greenPacket = createTrianglePacket(
			"transparent-green",
			{ r: 0, g: 255, b: 0 },
			{
				alphaMode: AlphaMode.Blend,
				opacity: 0.65,
			}
		);
		const packets = {
			transparentPackets: [redPacket, greenPacket],
		};

		const scanlineWarnings = [];
		const scanlineBackend = new SoftwareBackend({
			rasterMode: "scanline",
		});
		const scanlineAttachments = await renderPass(
			scanlineBackend,
			"main-transparent",
			packets,
			scanlineWarnings
		);

		const scheduler = createMockScheduler({
			reverseTriangleOrder: true,
		});
		const tileWarnings = [];
		const tileBackend = new SoftwareBackend({
			rasterMode: "tile",
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-raster-test-tile-order",
				scheduler,
			},
		});
		const tileAttachments = await renderPass(
			tileBackend,
			"main-transparent",
			packets,
			tileWarnings
		);

		assert.equal(tileBackend.activeRasterMode, "tile");
		assert.equal(scheduler.registerCalls.length, 1);
		assert.ok(scheduler.scheduleCalls.length > 0);
		assert.equal(tileWarnings.length, 0);

		assert.deepEqual(
			Array.from(tileAttachments.pixels),
			Array.from(scanlineAttachments.pixels)
		);

		const depthTolerance = 1e-6;
		for (let i = 0; i < tileAttachments.depthBuffer.length; i++) {
			const left = tileAttachments.depthBuffer[i];
			const right = scanlineAttachments.depthBuffer[i];
			if (!Number.isFinite(left) && !Number.isFinite(right)) continue;
			assert.ok(Math.abs(left - right) <= depthTolerance);
		}

		tileBackend.destroy();
		assert.deepEqual(scheduler.unregisterCalls, [
			"software-raster-test-tile-order",
		]);
	} finally {
		globalThis.Worker = originalWorker;
	}
}

async function testTileModeFallsBackOnWorkerTaskError() {
	const originalWorker = globalThis.Worker;
	globalThis.Worker = class FakeWorker {};

	try {
		const scheduler = createMockScheduler({
			throwOnSchedule: true,
		});
		const packet = createTrianglePacket("opaque-blue", {
			r: 32,
			g: 64,
			b: 255,
		});
		const backend = new SoftwareBackend({
			rasterMode: "tile",
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-raster-test-fallback",
				scheduler,
			},
		});

		const warnings = await captureWarnMessagesAsync(async () => {
			await renderPass(
				backend,
				"main-opaque",
				{
					opaquePackets: [packet],
				},
				[]
			);
		});

		assert.equal(backend.activeRasterMode, "scanline");
		assert.ok(
			warnings.some((warning) =>
				warning.includes("software-raster-worker-task-failed")
			)
		);
	} finally {
		globalThis.Worker = originalWorker;
	}
}

async function run() {
	await testDefaultRasterMode();
	await testTileModeFallsBackWhenWorkerUnavailable();
	await testTileModeMatchesScanlineAndPreservesOrder();
	await testTileModeFallsBackOnWorkerTaskError();
	console.log("Software raster mode tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
