import assert from "node:assert/strict";
import { Material } from "../../../src/materials/Material.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import {
	PreparedSceneBuilder,
	PreparedScenePacketCache,
} from "../../../src/pipeline/PreparedSceneBuilder.ts";
import { PreparedSceneCache } from "../../../src/pipeline/PreparedSceneCache.ts";
import { DEFAULT_INCREMENTAL_RENDERING_OPTIONS } from "../../../src/pipeline/incremental.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createFeatures(overrides = {}) {
	return {
		enableLighting: true,
		enableGamma: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableEnvironment: false,
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
		enableOcclusionCulling: false,
		clusteredLightingOptions: {},
		occlusionCullingOptions: {},
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

function createPacket(id, centerX, radius = 0.1, deformationRevision = 0) {
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
		deformationRevision,
		sortDepth: 0,
		pipelineKey: "default:pipeline",
		passFlags: 0,
	};
}

function rectContainsRect(container, target) {
	return (
		container.x <= target.x &&
		container.y <= target.y &&
		container.x + container.width >= target.x + target.width &&
		container.y + container.height >= target.y + target.height
	);
}

function createDecalPacket(id, centerX, radius = 0.1, overrides = {}) {
	const material = overrides.material ?? new Material();
	const worldMatrix = Matrix4.fromTranslation([centerX, 0, 0]);
	return {
		id,
		decal: {
			id,
			name: id,
			visible: true,
		},
		material,
		worldMatrix,
		inverseWorldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
		worldBounds: {
			center: {
				x: centerX,
				y: 0,
				z: 0,
			},
			radius,
		},
		receiverLayerMask: overrides.receiverLayerMask ?? 1,
		priority: overrides.priority ?? 0,
		opacity: overrides.opacity ?? 1,
		edgeFade: overrides.edgeFade ?? 0,
		channelBlendModes: overrides.channelBlendModes ?? {},
		sceneOrder: overrides.sceneOrder ?? 0,
	};
}

function createFrame(camera, packets, decalPackets = []) {
	return {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		lights: [],
		particleSystems: [],
		hasActiveAnimations: false,
		camera,
		environment: null,
		meshInstances: packets.map((packet) => packet.meshInstance),
		shadowMaps: new Map(),
		opaquePackets: packets,
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		decalPackets,
		occlusion: null,
		spatialIndex: null,
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);
		assert.equal(first.dirtyRects.length, 1);
		assert.ok(first.dirtyTiles.length > 0);
		assert.ok(first.frame.spatialIndex);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.equal(second.dirtyRects.length, 0);
		assert.equal(second.dirtyTiles.length, 0);

		const third = cache.build(buildInput);
		assert.equal(third.forceFullFrame, false);
		assert.ok(third.dirtyRects.length > 0);
		assert.ok(third.packetRects.has("A"));
		assert.ok(third.dirtyTiles.length > 0);
		assert.ok(third.frame.spatialIndex);
		const queryRect = third.packetRects.get("A");
		assert.ok(queryRect);
		const spatialHits = third.frame.spatialIndex.queryOpaquePackets(queryRect);
		assert.equal(spatialHits.length, 1);
		assert.equal(spatialHits[0].id, "A");

		const fourth = cache.build(buildInput);
		assert.equal(fourth.forceFullFrame, false);
		assert.ok(fourth.dirtyRects.length > 0);
		assert.equal(fourth.packetRects.size, 0);
		assert.ok(fourth.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testBackendDirtyRectsJoinPreparedCoverage() {
	const camera = createCamera();
	const packet = createPacket("backend-dirty", 0, 0.08);
	const frame = createFrame(camera, [packet]);
	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => frame;

	try {
		const buildInput = {
			viewportWidth: 320,
			viewportHeight: 180,
			features: createFeatures(),
			postProcess: createResolvedPostProcess(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
				fullFrameFallbackAreaRatio: 1,
			},
		};
		cache.build(buildInput);
		const stable = cache.build(buildInput);
		assert.equal(stable.dirtyRects.length, 0);

		const backendRect = { x: 64, y: 32, width: 8, height: 8 };
		const targeted = cache.build({
			...buildInput,
			additionalDirtyRects: [backendRect],
		});
		assert.equal(targeted.forceFullFrame, false);
		assert.ok(targeted.dirtyAreaRatio > 0);
		assert.ok(targeted.dirtyAreaRatio < 1);
		assert.ok(
			targeted.dirtyRects.some((rect) => rectContainsRect(rect, backendRect)),
		);

		const unbounded = cache.build({
			...buildInput,
			forceFullFrame: true,
		});
		assert.equal(unbounded.forceFullFrame, true);
		assert.equal(unbounded.dirtyAreaRatio, 1);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testDecalDiffLifecycle() {
	const camera = createCamera();
	const decalA0 = createDecalPacket("decal-A", 0.0, 0.08);
	const decalA1 = createDecalPacket("decal-A", 0.0, 0.08);
	const decalA2 = createDecalPacket("decal-A", 0.45, 0.08);
	const frames = [
		createFrame(camera, [], [decalA0]),
		createFrame(camera, [], [decalA1]),
		createFrame(camera, [], [decalA2]),
		createFrame(camera, [], []),
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
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
		assert.equal(second.dirtyTiles.length, 0);

		const third = cache.build(buildInput);
		assert.equal(third.forceFullFrame, false);
		assert.ok(third.dirtyRects.length > 0);
		assert.ok(third.dirtyTiles.length > 0);
		assert.equal(third.packetRects.has("decal-A"), false);

		const fourth = cache.build(buildInput);
		assert.equal(fourth.forceFullFrame, false);
		assert.ok(fourth.dirtyRects.length > 0);
		assert.ok(fourth.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testDecalStateDiffDetectsBlendAndOpacityChanges() {
	const camera = createCamera();
	const decalBase = createDecalPacket("decal-state", 0.0, 0.08, {
		opacity: 0.5,
		channelBlendModes: {
			baseColor: "lerp",
		},
	});
	const decalChanged = createDecalPacket("decal-state", 0.0, 0.08, {
		opacity: 0.75,
		channelBlendModes: {
			baseColor: "multiply",
		},
	});
	const frames = [
		createFrame(camera, [], [decalBase]),
		createFrame(camera, [], [decalChanged]),
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.ok(second.dirtyRects.length > 0);
		assert.ok(second.dirtyTiles.length > 0);
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
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

function testOcclusionHideRevealDirtyRects() {
	const camera = createCamera();
	const packet = createPacket("O", 0.0, 0.08);
	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	let visible = true;

	PreparedSceneBuilder.build = (_renderer, options = {}) => {
		const candidate = {
			packetId: packet.id,
			packet,
			eligible: true,
			signatureA: 1,
			signatureB: 2,
		};
		const providerVisible =
			options.occlusionVisibilityProvider?.isPacketVisible(candidate) ?? true;
		const framePackets = visible && providerVisible ? [packet] : [];
		return createFrame(camera, framePackets);
	};

	try {
		const buildInput = {
			renderer: {},
			viewportWidth: 320,
			viewportHeight: 180,
			features: createFeatures({
				enableOcclusionCulling: true,
			}),
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
			occlusionVisibilityProvider: {
				sourceFrameIndex: 0,
				isPacketVisible() {
					return visible;
				},
			},
			occlusionCullingOptions: {},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);
		assert.equal(first.packetRects.has("O"), true);

		visible = false;
		const hidden = cache.build(buildInput);
		assert.equal(hidden.forceFullFrame, false);
		assert.ok(hidden.dirtyRects.length > 0);
		assert.equal(hidden.packetRects.has("O"), false);

		visible = true;
		const revealed = cache.build(buildInput);
		assert.equal(revealed.forceFullFrame, false);
		assert.ok(revealed.dirtyRects.length > 0);
		assert.equal(revealed.packetRects.has("O"), true);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testMatrixDiffDetectsSmallFloatChanges() {
	const camera = createCamera();
	const packetBase = createPacket("S", 0, 0.08);
	const packetSmallDelta = createPacket("S", 0, 0.08);
	packetSmallDelta.worldMatrix.elements[0][3] = 0.00001;

	const frames = [
		createFrame(camera, [packetBase]),
		createFrame(camera, [packetSmallDelta]),
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.ok(second.dirtyRects.length > 0);
		assert.ok(second.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testMaterialDiffDetectsSmallFloatChanges() {
	const camera = createCamera();
	const packetBase = createPacket("M", 0, 0.08);
	const packetSmallDelta = createPacket("M", 0, 0.08);
	packetSmallDelta.material.opacity = 0.50001;

	const frames = [
		createFrame(camera, [packetBase]),
		createFrame(camera, [packetSmallDelta]),
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.ok(second.dirtyRects.length > 0);
		assert.ok(second.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testMaterialDiffDetectsDepthWriteChanges() {
	const camera = createCamera();
	const packetBase = createPacket("D", 0, 0.08);
	const packetDepthRead = createPacket("D", 0, 0.08);
	packetDepthRead.material.depthWrite = false;

	const frames = [
		createFrame(camera, [packetBase]),
		createFrame(camera, [packetDepthRead]),
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
			postProcess: createResolvedPostProcess({
				fxaa: { enabled: true },
			}),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
			},
		};

		const first = cache.build(buildInput);
		assert.equal(first.forceFullFrame, true);

		const second = cache.build(buildInput);
		assert.equal(second.forceFullFrame, false);
		assert.ok(second.dirtyRects.length > 0);
		assert.ok(second.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testDeformationRevisionAndBoundsDirtyPreviousAndCurrentCoverage() {
	const camera = createCamera();
	const previousPacket = createPacket("skin", 0, 0.08, 1);
	previousPacket.worldBounds.center.x = -0.45;
	const currentPacket = createPacket("skin", 0, 0.08, 2);
	currentPacket.worldBounds.center.x = 0.45;
	const unchangedPacket = createPacket("skin", 0, 0.08, 2);
	unchangedPacket.worldBounds.center.x = 0.45;
	const revisionOnlyPacket = createPacket("skin", 0, 0.08, 3);
	revisionOnlyPacket.worldBounds.center.x = 0.45;
	const boundsOnlyPacket = createPacket("skin", 0, 0.08, 3);
	boundsOnlyPacket.worldBounds.center.x = 0.3;
	const deformationRemovedPacket = createPacket("skin", 0, 0.08, 0);
	const frames = [
		createFrame(camera, [previousPacket]),
		createFrame(camera, [currentPacket]),
		createFrame(camera, [unchangedPacket]),
		createFrame(camera, [revisionOnlyPacket]),
		createFrame(camera, [boundsOnlyPacket]),
		createFrame(camera, [deformationRemovedPacket]),
	];
	let frameIndex = 0;
	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => frames[frameIndex++];

	try {
		const buildInput = {
			viewportWidth: 320,
			viewportHeight: 180,
			features: createFeatures(),
			postProcess: createResolvedPostProcess(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
				fullFrameFallbackAreaRatio: 1,
			},
		};

		const first = cache.build(buildInput);
		const previousRect = first.packetRects.get("skin");
		assert.ok(previousRect);
		const moved = cache.build(buildInput);
		const currentRect = moved.packetRects.get("skin");
		assert.ok(currentRect);
		assert.ok(moved.dirtyTiles.length > 0);
		assert.ok(moved.dirtyRects.some((rect) => rectContainsRect(rect, previousRect)));
		assert.ok(moved.dirtyRects.some((rect) => rectContainsRect(rect, currentRect)));

		const unchanged = cache.build(buildInput);
		assert.equal(unchanged.dirtyTiles.length, 0);

		const revisionOnly = cache.build(buildInput);
		assert.ok(revisionOnly.dirtyTiles.length > 0);

		const boundsOnly = cache.build(buildInput);
		assert.ok(boundsOnly.dirtyTiles.length > 0);

		const deformationRemoved = cache.build(buildInput);
		assert.ok(deformationRemoved.dirtyTiles.length > 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testCameraMatrixChangeForcesFullFrameAndRebasesPacketRects() {
	const initialCamera = createCamera();
	const rotatedCamera = createCamera();
	rotatedCamera.viewProjectionMatrix = Matrix4.fromTranslation([0.35, 0, 0]);
	const initialPacket = createPacket("camera-skin", 0, 0.08, 1);
	const rotatedPacket = createPacket("camera-skin", 0, 0.08, 1);
	const stablePacket = createPacket("camera-skin", 0, 0.08, 1);
	const animatedPacket = createPacket("camera-skin", 0, 0.08, 2);
	animatedPacket.worldBounds.center.x = 0.15;
	const frames = [
		createFrame(initialCamera, [initialPacket]),
		createFrame(rotatedCamera, [rotatedPacket]),
		createFrame(rotatedCamera, [stablePacket]),
		createFrame(rotatedCamera, [animatedPacket]),
	];
	let frameIndex = 0;
	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => frames[frameIndex++];

	try {
		const buildInput = {
			viewportWidth: 320,
			viewportHeight: 180,
			features: createFeatures(),
			postProcess: createResolvedPostProcess(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled: true,
				fullFrameFallbackAreaRatio: 1,
			},
		};

		cache.build(buildInput);
		const cameraChanged = cache.build(buildInput);
		assert.equal(cameraChanged.forceFullFrame, true);
		assert.equal(cameraChanged.dirtyAreaRatio, 1);

		const stable = cache.build(buildInput);
		assert.equal(stable.forceFullFrame, false);
		assert.equal(stable.dirtyTiles.length, 0);

		const animated = cache.build(buildInput);
		assert.equal(animated.forceFullFrame, false);
		assert.ok(animated.dirtyTiles.length > 0);
		const currentRect = animated.packetRects.get("camera-skin");
		assert.ok(currentRect);
		assert.ok(animated.dirtyRects.some((rect) => rectContainsRect(rect, currentRect)));
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testPreparedPacketCacheReusesViewLocalPackets() {
	const material = new Material();
	const mesh = MeshAsset.fromFaces([{
		material,
		vertices: [
			{ x: 0, y: 0, z: -2 },
			{ x: 1, y: 0, z: -2 },
			{ x: 0, y: 1, z: -2 },
		],
	}]);
	const scene = new Scene();
	const camera = scene.add(new Camera());
	const instance = scene.add(new MeshInstance({ mesh }));
	scene.updateWorldMatrices();
	camera.updateMatrices();
	const packets = new PreparedScenePacketCache();
	const build = (view = camera) => {
		packets.beginFrame();
		const frame = PreparedSceneBuilder.build(
			{ scene, camera: view, hasActiveAnimations: false },
			{ packetCache: packets },
		);
		packets.endFrame();
		return frame.opaquePackets[0];
	};
	const first = build();
	const firstNormal = first.normalMatrix;
	const second = build();
	assert.equal(second, first);
	assert.equal(second.normalMatrix, firstNormal);

	instance.scale.x = 2;
	scene.updateWorldMatrices();
	const transformed = build();
	assert.equal(transformed, first);
	assert.notEqual(transformed.normalMatrix, firstNormal);

	const secondary = new Camera();
	secondary.updateMatrices();
	assert.notEqual(build(secondary), first);
}

function run() {
	testPacketDiffLifecycle();
	testBackendDirtyRectsJoinPreparedCoverage();
	testDecalDiffLifecycle();
	testDecalStateDiffDetectsBlendAndOpacityChanges();
	testAreaFallbackToFullFrame();
	testOcclusionHideRevealDirtyRects();
	testMatrixDiffDetectsSmallFloatChanges();
	testMaterialDiffDetectsSmallFloatChanges();
	testMaterialDiffDetectsDepthWriteChanges();
	testDeformationRevisionAndBoundsDirtyPreviousAndCurrentCoverage();
	testCameraMatrixChangeForcesFullFrameAndRebasesPacketRects();
	testPreparedPacketCacheReusesViewLocalPackets();
	console.log("Prepared scene cache tests passed");
}

run();
