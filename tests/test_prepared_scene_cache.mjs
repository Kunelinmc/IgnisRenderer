import assert from "node:assert/strict";
import { Material } from "../src/materials/Material.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { PreparedSceneBuilder } from "../src/pipeline/PreparedSceneBuilder.ts";
import { PreparedSceneCache } from "../src/pipeline/PreparedSceneCache.ts";
import { DEFAULT_INCREMENTAL_RENDERING_OPTIONS } from "../src/pipeline/incremental.ts";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableSkybox: false,
		enableSSAO: false,
		enableSSGI: false,
		enableTAA: false,
		enableSSR: false,
		enableVolumetric: false,
		enableMotionBlur: false,
		enableDOF: false,
		enableBloom: false,
		enableFXAA: true,
		enableClusteredLighting: false,
		warnings: [],
		...overrides,
	};
}

function createCamera() {
	return {
		viewProjectionMatrix: Matrix4.identity(),
		getWorldDirection(direction, out) {
			out.x = direction.x;
			out.y = direction.y;
			out.z = direction.z;
			return out;
		},
	};
}

function createPacket(id, centerX, radius = 0.1) {
	return {
		id,
		meshInstance: {
			id: `mesh-${id}`,
			visible: true,
		},
		mesh: {
			id: `asset-${id}`,
		},
		primitive: {
			id: `primitive-${id}`,
			visible: true,
		},
		material: new Material(),
		geometry: {
			id: `geometry-${id}`,
		},
		worldMatrix: Matrix4.fromTranslation([centerX, 0, 0]),
		normalMatrix: Matrix4.identity(),
		worldBounds: {
			center: {
				x: centerX,
				y: 0,
				z: 0,
			},
			radius,
		},
		sortDepth: 0,
		pipelineKey: "default:pipeline",
		passFlags: 0,
	};
}

function createFrame(camera, packets) {
	return {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		lights: [],
		particleSystems: [],
		hasActiveAnimations: false,
		camera,
		skybox: null,
		meshInstances: packets.map((packet) => packet.meshInstance),
		shadowMaps: new Map(),
		opaquePackets: packets,
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
	};
}

function testPacketDiffLifecycle() {
	const camera = createCamera();
	const packetA0 = createPacket("A", 0.0, 0.08);
	const packetA1 = createPacket("A", 0.0, 0.08);
	const packetA2 = createPacket("A", 0.45, 0.08);
	const frames = [
		createFrame(camera, [packetA0]),
		createFrame(camera, [packetA1]),
		createFrame(camera, [packetA2]),
		createFrame(camera, []),
	];
	let frameIndex = 0;

	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => {
		const resolved = frames[Math.min(frameIndex, frames.length - 1)];
		frameIndex++;
		return resolved;
	};

	try {
		const buildInput = {
			renderer: {},
			viewportWidth: 320,
			viewportHeight: 180,
			features: createFeatures(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);
		assert.equal(first.dirtyRects.length, 1);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.equal(second.dirtyRects.length, 0);

		const third = cache.build(buildInput);
		assert.equal(third.forceFullFrame, false);
		assert.ok(third.dirtyRects.length > 0);
		assert.ok(third.packetRects.has("A"));

		const fourth = cache.build(buildInput);
		assert.equal(fourth.forceFullFrame, false);
		assert.ok(fourth.dirtyRects.length > 0);
		assert.equal(fourth.packetRects.size, 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testAreaFallbackToFullFrame() {
	const camera = createCamera();
	const packetLarge0 = createPacket("L", 0, 2.0);
	const packetLarge1 = createPacket("L", 0.2, 2.0);
	const frames = [
		createFrame(camera, [packetLarge0]),
		createFrame(camera, [packetLarge1]),
	];
	let frameIndex = 0;

	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => {
		const resolved = frames[Math.min(frameIndex, frames.length - 1)];
		frameIndex++;
		return resolved;
	};

	try {
		const buildInput = {
			renderer: {},
			viewportWidth: 256,
			viewportHeight: 256,
			features: createFeatures(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
				fullFrameFallbackAreaRatio: 0.3,
			},
		};

		cache.build(buildInput);
		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, true);
		assert.equal(second.dirtyRects.length, 1);
		assert.equal(second.dirtyAreaRatio, 1);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function run() {
	testPacketDiffLifecycle();
	testAreaFallbackToFullFrame();
	console.log("Prepared scene cache tests passed");
}

run();
