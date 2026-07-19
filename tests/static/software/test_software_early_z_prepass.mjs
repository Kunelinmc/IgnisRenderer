import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { BasicMaterial, AlphaMode } from "../../../src/materials/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const WIDTH = 64;
const HEIGHT = 64;
const DEPTH_TOLERANCE = 1e-6;

function createZeroSH() {
	return Array.from({ length: 9 }, () => ({ r: 0, g: 0, b: 0 }));
}

function createRendererBridge() {
	return {
		canvas: { width: WIDTH, height: HEIGHT },
	};
}

function createSoftwareSession(options = {}) {
	const backend = new SoftwareBackend(options);
	backend.attach({
		surface: createRendererBridge(),
		events: { emit: () => {} },
	});
	return backend;
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

function createContext(backend, camera, packetsByStage = {}, incremental = {}) {
	const attachments = backend.getAttachments({ width: WIDTH, height: HEIGHT });
	const zeroSH = createZeroSH();

	return {
		viewCamera: camera,
		attachments,
		features: {
			enableLighting: false,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess({
			gamma: { enabled: false },
			tonemap: { enabled: false },
		}),
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
			environment: null,
			meshInstances: [],
			shadowMaps: new Map(),
			opaquePackets: packetsByStage.opaquePackets ?? [],
			transparentPackets: packetsByStage.transparentPackets ?? [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
			spatialIndex: null,
		},
		shCoeffs: zeroSH,
		shAmbientCoeffs: zeroSH,
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: incremental.enabled ?? false,
			forceFullFrame: incremental.forceFullFrame ?? true,
			dirtyRects: incremental.dirtyRects ?? [],
			dirtyTileSize: 0,
			dirtyTileColumns: 0,
			dirtyTileRows: 0,
			dirtyTiles: [],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: incremental.temporalHistoryReset ?? false,
		},
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
		depthWrite: options.depthWrite ?? true,
		map: options.map ?? null,
	});
	if (typeof options.alphaCutoff === "number") {
		material.alphaCutoff = options.alphaCutoff;
	}

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

function createMockScheduler() {
	const pools = new Map();
	return {
		hasPool(poolId) {
			return pools.has(poolId);
		},
		registerPool(poolOptions) {
			pools.set(poolOptions.id, poolOptions);
		},
		async schedule(_poolId, payload) {
			return computeBinsFromPayload(payload);
		},
		unregisterPool(poolId) {
			pools.delete(poolId);
		},
	};
}

function computeBinsFromPayload(payload) {
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
			triangleIndices,
		}));

	return {
		type: "bin-main-pass",
		bins: entries,
	};
}

function copyAttachments(attachments) {
	return {
		pixels: new Uint8ClampedArray(attachments.pixels),
		depthBuffer: new Float32Array(attachments.depthBuffer),
	};
}

function assertAttachmentEqual(left, right) {
	assert.deepEqual(Array.from(left.pixels), Array.from(right.pixels));
	for (let i = 0; i < left.depthBuffer.length; i++) {
		const a = left.depthBuffer[i];
		const b = right.depthBuffer[i];
		if (!Number.isFinite(a) && !Number.isFinite(b)) continue;
		assert.ok(Math.abs(a - b) <= DEPTH_TOLERANCE);
	}
}

async function renderOpaqueFrame(
	backend,
	camera,
	opaquePackets,
	incremental = {}
) {
	const context = createContext(
		backend,
		camera,
		{
			opaquePackets,
			transparentPackets: [],
		},
		incremental
	);
	backend.beginFrame(context);
	await backend.executePass(
		{
			stage: "main-opaque",
			executor: "backend",
			enabled: true,
		},
		context
	);
	return context.attachments;
}

async function testScanlinePrepassParity() {
	const packets = [
		createTrianglePacket("opaque-far-blue", { r: 0, g: 0, b: 255 }, { zOffset: -0.2 }),
		createTrianglePacket("opaque-near-red", { r: 255, g: 0, b: 0 }, { zOffset: 0.25 }),
	];

	const cameraA = createCamera();
	const backendA = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: false,
	});
	const withoutPrepass = copyAttachments(
		await renderOpaqueFrame(backendA, cameraA, packets)
	);

	const cameraB = createCamera();
	const backendB = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: true,
	});
	const withPrepass = copyAttachments(
		await renderOpaqueFrame(backendB, cameraB, packets)
	);

	assertAttachmentEqual(withoutPrepass, withPrepass);
}

async function testTilePrepassParity() {
	const originalWorker = globalThis.Worker;
	globalThis.Worker = class FakeWorker {};

	try {
		const packets = [
			createTrianglePacket("tile-far-green", { r: 0, g: 255, b: 0 }, { zOffset: -0.25 }),
			createTrianglePacket("tile-near-red", { r: 255, g: 0, b: 0 }, { zOffset: 0.2 }),
		];

		const backendA = createSoftwareSession({
			rasterMode: "tile",
			enableEarlyZPrepass: false,
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-early-z-tile-off",
				scheduler: createMockScheduler(),
			},
		});
		const withoutPrepass = copyAttachments(
			await renderOpaqueFrame(backendA, createCamera(), packets)
		);
		assert.equal(backendA.activeRasterMode, "tile");

		const backendB = createSoftwareSession({
			rasterMode: "tile",
			enableEarlyZPrepass: true,
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-early-z-tile-on",
				scheduler: createMockScheduler(),
			},
		});
		const withPrepass = copyAttachments(
			await renderOpaqueFrame(backendB, createCamera(), packets)
		);
		assert.equal(backendB.activeRasterMode, "tile");

		assertAttachmentEqual(withoutPrepass, withPrepass);
		backendA.destroy();
		backendB.destroy();
	} finally {
		globalThis.Worker = originalWorker;
	}
}

async function testMaskPacketParity() {
	const maskMap = new Texture(
		new Uint8ClampedArray([255, 255, 255, 0]),
		1,
		1
	);
	const packets = [
		createTrianglePacket("mask-far-blue", { r: 0, g: 0, b: 255 }, { zOffset: -0.2 }),
		createTrianglePacket("mask-front-transparent", { r: 255, g: 0, b: 0 }, {
			zOffset: 0.2,
			alphaMode: AlphaMode.Mask,
			alphaCutoff: 0.5,
			map: maskMap,
			opacity: 1,
		}),
	];

	const backendA = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: false,
	});
	const withoutPrepass = copyAttachments(
		await renderOpaqueFrame(backendA, createCamera(), packets)
	);

	const backendB = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: true,
	});
	const withPrepass = copyAttachments(
		await renderOpaqueFrame(backendB, createCamera(), packets)
	);

	assertAttachmentEqual(withoutPrepass, withPrepass);
}

function createDepthWriteDisabledPackets() {
	return [
		createTrianglePacket("depth-read-near-red", { r: 255, g: 0, b: 0 }, {
			zOffset: 0.2,
			depthWrite: false,
		}),
		createTrianglePacket("depth-write-far-blue", { r: 0, g: 0, b: 255 }, {
			zOffset: -0.2,
		}),
	];
}

async function testDepthWriteDisabledScanlinePrepassParity() {
	const backendA = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: false,
	});
	const withoutPrepass = copyAttachments(
		await renderOpaqueFrame(
			backendA,
			createCamera(),
			createDepthWriteDisabledPackets()
		)
	);

	const backendB = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass: true,
	});
	const withPrepass = copyAttachments(
		await renderOpaqueFrame(
			backendB,
			createCamera(),
			createDepthWriteDisabledPackets()
		)
	);

	assertAttachmentEqual(withoutPrepass, withPrepass);
}

async function testDepthWriteDisabledTilePrepassParity() {
	const originalWorker = globalThis.Worker;
	globalThis.Worker = class FakeWorker {};

	try {
		const backendA = createSoftwareSession({
			rasterMode: "tile",
			enableEarlyZPrepass: false,
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-depth-write-tile-off",
				scheduler: createMockScheduler(),
			},
		});
		const withoutPrepass = copyAttachments(
			await renderOpaqueFrame(
				backendA,
				createCamera(),
				createDepthWriteDisabledPackets()
			)
		);

		const backendB = createSoftwareSession({
			rasterMode: "tile",
			enableEarlyZPrepass: true,
			tile: {
				tileSize: 16,
				workerCount: 2,
				poolId: "software-depth-write-tile-on",
				scheduler: createMockScheduler(),
			},
		});
		const withPrepass = copyAttachments(
			await renderOpaqueFrame(
				backendB,
				createCamera(),
				createDepthWriteDisabledPackets()
			)
		);

		assertAttachmentEqual(withoutPrepass, withPrepass);
		backendA.destroy();
		backendB.destroy();
	} finally {
		globalThis.Worker = originalWorker;
	}
}

async function runIncrementalScenario(enableEarlyZPrepass) {
	const camera = createCamera();
	const backend = createSoftwareSession({
		rasterMode: "scanline",
		enableEarlyZPrepass,
	});

	const frameOnePackets = [
		createTrianglePacket("inc-far-blue", { r: 0, g: 0, b: 255 }, { zOffset: -0.2 }),
	];
	await renderOpaqueFrame(backend, camera, frameOnePackets, {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [],
	});

	const frameTwoPackets = [
		createTrianglePacket("inc-far-blue", { r: 0, g: 0, b: 255 }, { zOffset: -0.2 }),
		createTrianglePacket("inc-near-red", { r: 255, g: 0, b: 0 }, { zOffset: 0.2 }),
	];
	const finalAttachments = await renderOpaqueFrame(backend, camera, frameTwoPackets, {
		enabled: true,
		forceFullFrame: false,
		dirtyRects: [{ x: 16, y: 16, width: 32, height: 32 }],
		temporalHistoryReset: false,
	});

	return copyAttachments(finalAttachments);
}

async function testIncrementalParity() {
	const withoutPrepass = await runIncrementalScenario(false);
	const withPrepass = await runIncrementalScenario(true);
	assertAttachmentEqual(withoutPrepass, withPrepass);
}

async function run() {
	await testScanlinePrepassParity();
	await testTilePrepassParity();
	await testMaskPacketParity();
	await testDepthWriteDisabledScanlinePrepassParity();
	await testDepthWriteDisabledTilePrepassParity();
	await testIncrementalParity();
	console.log("Software early Z pre-pass tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
