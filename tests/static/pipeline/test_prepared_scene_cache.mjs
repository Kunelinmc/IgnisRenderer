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
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

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
	return createTestDrawPacket({
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
		geometryId: `geometry-${id}`,
		geometry: { positions: new Float32Array(0), indices: new Uint32Array(0) },
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
	});
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
		meshInstances: [],
		shadowMaps: new Map(),
		opaquePackets: packets,
		transparentPackets: [],
		shadowCasterSubmissions: [],
		shadowTransmitterSubmissions: [],
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
		assert.equal(spatialHits[0].submission.id, "A");

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
			packetId: packet.submission.id,
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
	packetSmallDelta.submission.instance.worldMatrix.elements[0][3] = 0.00001;

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
	packetSmallDelta.submission.material.effective.opacity = 0.50001;

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
	packetDepthRead.submission.material.effective.depthWrite = false;

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
	previousPacket.submission.worldBounds.center.x = -0.45;
	const currentPacket = createPacket("skin", 0, 0.08, 2);
	currentPacket.submission.worldBounds.center.x = 0.45;
	const unchangedPacket = createPacket("skin", 0, 0.08, 2);
	unchangedPacket.submission.worldBounds.center.x = 0.45;
	const revisionOnlyPacket = createPacket("skin", 0, 0.08, 3);
	revisionOnlyPacket.submission.worldBounds.center.x = 0.45;
	const boundsOnlyPacket = createPacket("skin", 0, 0.08, 3);
	boundsOnlyPacket.submission.worldBounds.center.x = 0.3;
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

function testFullFrameBuildSkipsDiffAndRebasesState() {
	for (const reason of ["camera", "forced", "first"]) {
		const camera = createCamera();
		let frame = createFrame(camera, [createPacket("old", -0.6, 0.05)], [
			createDecalPacket("old-decal", -0.6, 0.05),
		]);
		const cache = new PreparedSceneCache();
		const input = {
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
		const originalBuild = PreparedSceneBuilder.build;
		PreparedSceneBuilder.build = () => frame;
		try {
			if (reason !== "first") cache.build(input);
			if (reason === "camera") {
				camera.viewProjectionMatrix = Matrix4.fromTranslation([0.2, 0, 0]);
			}
			input.forceFullFrame = reason === "forced";
			input.incrementalOptions.enabled = true;
			frame = createFrame(camera, [
				createPacket("old", -0.2, 0.05),
				createPacket("new", 0.4, 0.05),
			], [
				createDecalPacket("old-decal", -0.2, 0.05),
				createDecalPacket("new-decal", 0.4, 0.05),
			]);
			frame.transparentPackets = [createPacket("transparent", 0.1, 0.05)];
			// Count work on the previous baseline, without replacing the diff logic.
			let previousStateReads = 0;
			for (const states of [cache._packetStateById, cache._decalStateById]) {
				for (const method of ["get", "entries"]) {
					const original = states[method].bind(states);
					states[method] = (...args) => {
						previousStateReads++;
						return original(...args);
					};
				}
			}
			let additionalRectReads = 0;
			input.additionalDirtyRects = [{
				get x() { additionalRectReads++; return 0; },
				y: 0, width: 1, height: 1,
			}];

			const full = cache.build(input);
			assert.equal(previousStateReads, 0, `${reason}: skip previous-state diff work`);
			assert.equal(additionalRectReads, 0, `${reason}: skip dirty-candidate collection`);
			assert.equal(full.forceFullFrame, true);
			assert.equal(full.dirtyAreaRatio, 1);
			const viewport = { x: 0, y: 0, width: 320, height: 180 };
			assert.deepEqual(full.dirtyRects, [viewport]);
			assert.equal(full.dirtyTiles.length, full.dirtyTileColumns * full.dirtyTileRows);
			assert.deepEqual([...full.packetRects.keys()], ["old", "new", "transparent"]);
			assert.deepEqual(
				full.frame.spatialIndex.queryOpaquePackets(viewport).map((p) => p.submission.id),
				["old", "new"],
			);
			assert.deepEqual(
				full.frame.spatialIndex.queryTransparentPackets(viewport).map((p) => p.submission.id),
				["transparent"],
			);

			input.forceFullFrame = false;
			input.incrementalOptions.enabled = true;
			input.additionalDirtyRects = [];
			const stable = cache.build(input);
			assert.equal(stable.forceFullFrame, false, `${reason}: advance the frame baseline`);
			assert.deepEqual(stable.dirtyTiles, [], `${reason}: rebase packets and decals`);

			frame.decalPackets = [];
			const removedDecals = cache.build(input);
			assert.equal(removedDecals.forceFullFrame, false);
			assert.ok(removedDecals.dirtyTiles.length > 0, `${reason}: retain decal coverage`);
			assert.deepEqual(cache.build(input).dirtyTiles, []);

			frame = createFrame(camera, []);
			const removedPackets = cache.build(input);
			assert.equal(removedPackets.forceFullFrame, false);
			for (const rect of full.packetRects.values()) {
				assert.ok(
					removedPackets.dirtyRects.some((dirty) => rectContainsRect(dirty, rect)),
					`${reason}: retain current-view coverage for removed packets`,
				);
			}
		} finally {
			PreparedSceneBuilder.build = originalBuild;
		}
	}
}

function testCameraMatrixChangeForcesFullFrameAndRebasesPacketRects() {
	const initialCamera = createCamera();
	const rotatedCamera = createCamera();
	rotatedCamera.viewProjectionMatrix = Matrix4.fromTranslation([0.35, 0, 0]);
	const initialPacket = createPacket("camera-skin", 0, 0.08, 1);
	const rotatedPacket = createPacket("camera-skin", 0, 0.08, 1);
	const stablePacket = createPacket("camera-skin", 0, 0.08, 2);
	const animatedPacket = createPacket("camera-skin", 0, 0.08, 2);
	stablePacket.submission.worldBounds.center.x = 0.15;
	animatedPacket.submission.worldBounds.center.x = 0.15;
	const frames = [
		createFrame(initialCamera, [initialPacket]),
		createFrame(rotatedCamera, [rotatedPacket]),
		createFrame(rotatedCamera, [animatedPacket]),
		createFrame(rotatedCamera, [stablePacket]),
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

		// Exercise the full-frame baseline before a stable build can refresh it.
		const animated = cache.build(buildInput);
		assert.equal(animated.forceFullFrame, false);
		assert.ok(animated.dirtyTiles.length > 0);
		const previousRect = cameraChanged.packetRects.get("camera-skin");
		assert.ok(previousRect);
		assert.ok(animated.dirtyRects.some((rect) => rectContainsRect(rect, previousRect)));
		const currentRect = animated.packetRects.get("camera-skin");
		assert.ok(currentRect);
		assert.ok(animated.dirtyRects.some((rect) => rectContainsRect(rect, currentRect)));

		const stable = cache.build(buildInput);
		assert.equal(stable.forceFullFrame, false);
		assert.equal(stable.dirtyTiles.length, 0);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testMainViewReusesCameraIndependentPreparedState() {
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

	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	const originalBuildView = PreparedSceneBuilder.buildView;
	let fullBuildCount = 0;
	let viewBuildCount = 0;
	PreparedSceneBuilder.build = (...args) => {
		fullBuildCount++;
		return originalBuild.call(PreparedSceneBuilder, ...args);
	};
	PreparedSceneBuilder.buildView = (...args) => {
		viewBuildCount++;
		return originalBuildView.call(PreparedSceneBuilder, ...args);
	};

	try {
		const buildInput = {
			source: {
				scene,
				camera,
				hasActiveAnimations: false,
				deformationStates: null,
			},
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

		const initial = cache.build(buildInput);
		assert.equal(fullBuildCount, 1);
		assert.equal(viewBuildCount, 0);
		assert.ok(initial.frame.submissions.length > 0);

		camera.position.x = 0.25;
		scene.updateWorldMatrices();
		camera.updateMatrices();
		const cameraChanged = cache.build(buildInput);
		assert.equal(fullBuildCount, 1);
		assert.equal(viewBuildCount, 1);
		assert.equal(cameraChanged.forceFullFrame, true);
		assert.equal(cameraChanged.dirtyAreaRatio, 1);
		assert.equal(cameraChanged.frame.submissions, initial.frame.submissions);
		assert.equal(cameraChanged.frame.lights, initial.frame.lights);
		assert.equal(cameraChanged.frame.environment, initial.frame.environment);
		assert.equal(
			cameraChanged.frame.shadowCasterSubmissions,
			initial.frame.shadowCasterSubmissions,
		);
		assert.equal(
			cameraChanged.frame.shadowTransmitterSubmissions,
			initial.frame.shadowTransmitterSubmissions,
		);
		assert.notEqual(cameraChanged.frame.opaquePackets, initial.frame.opaquePackets);
		assert.equal(
			cameraChanged.frame.opaquePackets[0].submission,
			initial.frame.opaquePackets[0].submission,
		);

		instance.position.x = 0.5;
		scene.updateWorldMatrices();
		camera.updateMatrices();
		cache.build(buildInput);
		assert.equal(fullBuildCount, 2);
		assert.equal(viewBuildCount, 1);
	} finally {
		PreparedSceneBuilder.build = originalBuild;
		PreparedSceneBuilder.buildView = originalBuildView;
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
	const firstSubmission = first.submission;
	const firstNormal = first.submission.instance.normalMatrix;
	const second = build();
	assert.equal(second, first);
	assert.equal(second.submission, firstSubmission);
	assert.equal(second.submission.instance.normalMatrix, firstNormal);

	instance.scale.x = 2;
	scene.updateWorldMatrices();
	const transformed = build();
	assert.notEqual(transformed, first);
	assert.notEqual(transformed.submission, firstSubmission);
	assert.notEqual(transformed.submission.instance.normalMatrix, firstNormal);

	const secondary = new Camera();
	secondary.updateMatrices();
	const secondaryPacket = build(secondary);
	assert.notEqual(secondaryPacket, transformed);
	assert.equal(secondaryPacket.submission, transformed.submission);

	mesh.primitives[0].topology = "line-list";
	const topologyChanged = build();
	assert.notEqual(topologyChanged.submission, transformed.submission);
	assert.equal(topologyChanged.submission.geometry.topology, "line-list");
}

function testPreparedSubmissionsShareMaterialRevisionScans() {
	const materials = [new Material(), new Material()];
	const revisionReads = [0, 0];
	const revisionGetter = Object.getOwnPropertyDescriptor(Material.prototype, "revision").get;
	for (const [index, material] of materials.entries()) {
		Object.defineProperty(material, "revision", {
			get() {
				revisionReads[index]++;
				return revisionGetter.call(this);
			},
		});
	}
	const mesh = MeshAsset.fromFaces(materials.map((material) => ({
		material,
		vertices: [
			{ x: 0, y: 0, z: -2 },
			{ x: 1, y: 0, z: -2 },
			{ x: 0, y: 1, z: -2 },
		],
	})));
	const scene = new Scene();
	const camera = scene.add(new Camera());
	scene.add(new MeshInstance({ mesh }));
	scene.add(new MeshInstance({ mesh }));
	scene.updateWorldMatrices();
	camera.updateMatrices();
	const packets = new PreparedScenePacketCache();
	const build = (cached = true) => {
		if (cached) packets.beginFrame();
		try {
			return PreparedSceneBuilder.build(
				{ scene, camera, hasActiveAnimations: false },
				cached ? { packetCache: packets } : {},
			).submissions;
		} finally {
			if (cached) packets.endFrame();
		}
	};
	const first = build();
	assert.equal(first.length, 4);
	assert.deepEqual(revisionReads, [1, 1], "shared materials are scanned once on cache misses");
	const unchanged = build();
	assert.deepEqual(revisionReads, [2, 2], "each new frame refreshes its material observations");
	for (const [index, submission] of unchanged.entries()) {
		assert.equal(submission, first[index]);
	}

	materials[0].opacity = 0.5;
	const changed = build();
	assert.deepEqual(revisionReads, [3, 3], "rebuilding changed submissions reuses the scan");
	for (const [index, submission] of changed.entries()) {
		if (submission.material.effective === materials[0]) {
			assert.notEqual(submission, first[index]);
			assert.ok(submission.material.revision > first[index].material.revision);
		} else {
			assert.equal(submission, first[index]);
		}
	}

	packets.clear();
	materials[1].depthWrite = false;
	const reset = build();
	assert.deepEqual(revisionReads, [4, 4], "resize-style cache clearing keeps scans deduplicated");
	for (const [index, submission] of reset.entries()) {
		if (submission.material.effective === materials[1]) {
			assert.ok(submission.material.revision > changed[index].material.revision);
		} else {
			assert.equal(submission.material.revision, changed[index].material.revision);
		}
	}

	materials[0].depthWrite = false;
	const uncached = build(false);
	assert.equal(uncached.length, first.length);
	for (const [index, submission] of uncached.entries()) {
		if (submission.material.effective === materials[0]) {
			assert.ok(submission.material.revision > reset[index].material.revision);
		} else {
			assert.equal(submission.material.revision, reset[index].material.revision);
		}
	}
}

function testDirtySignaturesAreSharedOnlyWithinOneBuild() {
	for (const enabled of [true, false]) {
		const sharedMaterial = new Material();
		const otherMaterial = new Material();
		const texture = { version: 1 };
		sharedMaterial.map = texture;
		const sharedMatrix = Matrix4.identity();
		const otherMatrix = Matrix4.identity();
		const packets = [createPacket("shared-a", -0.5), createPacket("shared-b", 0)];
		const otherPacket = createPacket("other", 0.5);
		for (const packet of packets) {
			packet.submission.material.effective = sharedMaterial;
			packet.submission.instance.worldMatrix = sharedMatrix;
		}
		otherPacket.submission.material.effective = otherMaterial;
		otherPacket.submission.instance.worldMatrix = otherMatrix;
		const decals = [
			createDecalPacket("decal-a", -0.5, 0.1, { material: sharedMaterial, opacity: 0.2 }),
			createDecalPacket("decal-b", 0.5, 0.1, {
				material: sharedMaterial, opacity: 0.8,
				channelBlendModes: { baseColor: "multiply" },
			}),
		];
		for (const decal of decals) decal.worldMatrix = sharedMatrix;
		const frame = createFrame(createCamera(), [...packets, otherPacket], decals);
		const materialReads = [0, 0];
		const matrixReads = [0, 0];
		const opacity = [sharedMaterial.opacity, otherMaterial.opacity];
		const elements = [sharedMatrix.elements, otherMatrix.elements];
		for (const [index, material] of [sharedMaterial, otherMaterial].entries()) {
			Object.defineProperty(material, "opacity", {
				get() { materialReads[index]++; return opacity[index]; },
			});
		}
		for (const [index, matrix] of [sharedMatrix, otherMatrix].entries()) {
			Object.defineProperty(matrix, "elements", {
				get() { matrixReads[index]++; return elements[index]; },
			});
		}
		const cache = new PreparedSceneCache();
		const input = {
			viewportWidth: 320, viewportHeight: 180,
			features: createFeatures(), postProcess: createResolvedPostProcess(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
				enabled, fullFrameFallbackAreaRatio: 1,
			},
		};
		const originalBuild = PreparedSceneBuilder.build;
		PreparedSceneBuilder.build = () => frame;
		const build = () => {
			materialReads.fill(0);
			matrixReads.fill(0);
			const result = cache.build(input);
			const hashes = input.incrementalOptions.enabled ? [1, 1] : [0, 0];
			assert.deepEqual(materialReads, hashes, "hash only when incremental state is needed");
			assert.deepEqual(matrixReads, hashes, "hash only when incremental state is needed");
			assert.equal(result.packetRects.size, input.incrementalOptions.enabled ? 3 : 0);
			return result;
		};
		try {
			build();
			const stable = build();
			if (enabled) assert.equal(stable.dirtyRects.length, 0);
			for (const [index, change] of [
				() => { opacity[0] -= 1e-10; },
				() => { texture.version++; },
				() => { elements[0][0][3] += 1e-10; },
				() => { decals[0].opacity = 0.4; },
				() => { decals[1].channelBlendModes.baseColor = "lerp"; },
				() => { otherMaterial.depthWrite = false; },
			].entries()) {
				change();
				const changed = build();
				assert.ok(changed.dirtyRects.length > 0, "observe changes in the next build");
				if (enabled && index === 3) {
					assert.ok(changed.dirtyRects.every((rect) => rect.x + rect.width <= 160),
						"changing the left decal must not dirty the right decal's shared material");
				}
				if (enabled) assert.equal(build().dirtyRects.length, 0);
			}
			// Re-enabling must rebase before comparing against incremental history.
			input.incrementalOptions.enabled = true;
			if (!enabled) assert.equal(build().forceFullFrame, true);
			assert.equal(build().dirtyRects.length, 0);
			cache.reset();
			assert.equal(build().forceFullFrame, true);
		} finally {
			PreparedSceneBuilder.build = originalBuild;
		}
	}
}

function createSubmissionReuseFixture(count = 3) {
	const scene = new Scene();
	const camera = scene.add(new Camera());
	const instances = Array.from({ length: count }, () => scene.add(new MeshInstance({
		mesh: MeshAsset.fromFaces([{
			material: new Material(),
			vertices: [{ x: 0, y: 0, z: -2 }, { x: 1, y: 0, z: -2 }, { x: 0, y: 1, z: -2 }],
		}]),
	})));
	scene.updateWorldMatrices();
	camera.updateMatrices();
	return { scene, camera, instances, source: { scene, camera, hasActiveAnimations: false } };
}

function testDecalOnlyMemoDoesNotMixOtherDecals() {
	const material = new Material();
	const left = createDecalPacket("left", -0.5, 0.1, { material, opacity: 0.2 });
	const right = createDecalPacket("right", 0.5, 0.1, { material, opacity: 0.8 });
	right.worldMatrix = left.worldMatrix;
	const frame = createFrame(createCamera(), [], [left, right]);
	const cache = new PreparedSceneCache();
	const originalBuild = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => frame;
	try {
		const input = {
			viewportWidth: 320, viewportHeight: 180,
			features: createFeatures(), postProcess: createResolvedPostProcess(),
			incrementalOptions: {
				...DEFAULT_INCREMENTAL_RENDERING_OPTIONS, enabled: true, fullFrameFallbackAreaRatio: 1,
			},
		};
		cache.build(input);
		assert.equal(cache.build(input).dirtyRects.length, 0);
		left.opacity = 0.4;
		const changed = cache.build(input);
		assert.ok(changed.dirtyRects.length > 0);
		assert.ok(changed.dirtyRects.every((rect) => rect.x + rect.width <= 160),
			"decal-local mixing must not corrupt the cached base used by the right decal");
	} finally {
		PreparedSceneBuilder.build = originalBuild;
	}
}

function testFallbackConsumesSuccessfulValidationOnce() {
	for (const changedIndex of [0, 49, 99]) {
		const fixture = createSubmissionReuseFixture(100);
		const cache = new PreparedSceneCache();
		const input = {
			source: fixture.source, viewportWidth: 320, viewportHeight: 180,
			features: createFeatures(), postProcess: createResolvedPostProcess(),
			incrementalOptions: { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS, enabled: false },
		};
		const first = cache.build(input);
		fixture.instances[changedIndex].position.x = 0.01;
		fixture.scene.updateWorldMatrices();
		const original = PreparedScenePacketCache.prototype._isSignatureCurrent;
		let checks = 0;
		PreparedScenePacketCache.prototype._isSignatureCurrent = function (...args) {
			checks++;
			return original.apply(this, args);
		};
		try {
			const next = cache.build(input);
			assert.equal(checks, 101, "fallback must not revalidate the successful prefix");
			for (let index = 0; index < 100; index++) {
				assert.equal(
					next.frame.submissions[index] === first.frame.submissions[index],
					index !== changedIndex,
				);
			}
			checks = 0;
			cache.build(input);
			assert.equal(checks, 100, "the next preparation must validate again");
		} finally {
			PreparedScenePacketCache.prototype._isSignatureCurrent = original;
		}
	}
}

function testSubmissionValidationLifetimeAndMutations() {
	const { scene, camera, instances, source } = createSubmissionReuseFixture();
	const packets = new PreparedScenePacketCache(2);
	const build = () => PreparedSceneBuilder.build(source, { packetCache: packets });
	const validate = (frame) => packets.canReuseSubmissions(
		scene.getMeshInstances(), new Map(frame.submissions.map((s) => [s.id, s])),
		source.deformationStates ?? null,
	);
	packets.beginFrame();
	let frame = build();
	packets.endFrame();
	assert.equal(packets.getDebugStats().entries, 3, "active entries exceed the soft limit safely");

	const original = packets._isSignatureCurrent;
	let checks = 0;
	packets._isSignatureCurrent = function (...args) {
		checks++;
		return original.apply(this, args);
	};
	packets.beginFrame();
	assert.equal(validate(frame), true);
	assert.equal(checks, 3);
	const firstReuse = build();
	assert.equal(checks, 3, "consume successful checks once");
	assert.equal(packets.getDebugStats().frameHits, 6, "preserve hit accounting");
	build();
	assert.equal(checks, 6, "consumed checks cannot be used twice");
	packets.endFrame();

	packets.beginFrame();
	assert.equal(validate(firstReuse), true);
	packets.endFrame();
	const primitive = instances[0].mesh.primitives[0];
	primitive.topology = "line-list";
	const afterEnd = build();
	assert.equal(afterEnd.submissions[0].geometry.topology, "line-list");
	assert.notEqual(afterEnd.submissions[0], firstReuse.submissions[0]);

	// Replacing a signature must also invalidate an unconsumed validation marker.
	packets.beginFrame();
	assert.equal(validate(afterEnd), true);
	packets.storeSubmission(instances[0], primitive, null, afterEnd.submissions[0]);
	checks = 0;
	packets.getReusableSubmission(instances[0], primitive, null);
	assert.equal(checks, 1);
	packets.endFrame();

	const changes = [
		() => { primitive.castShadows = !primitive.castShadows; },
		() => { primitive.receiveShadows = !primitive.receiveShadows; },
		() => { instances[0].renderLayers = 4; },
		() => { primitive.material.opacity = 0.4; },
		() => { primitive.visible = false; },
		() => { primitive.visible = true; },
		() => { instances[0].visible = false; },
		() => { instances[0].visible = true; },
		() => { scene.remove(instances[2]); },
		() => { scene.add(instances[2]); },
		() => {
			source.deformationStates = new Map([[`${instances[0].id}:${primitive.id}`, {
				mode: "morph", revision: 1, jointPayloadKey: null,
				morphPayloadKey: "morph-payload", localBounds: primitive.boundingSphere,
			}]]);
		},
		() => { source.deformationStates.values().next().value.revision++; },
		() => { source.deformationStates.values().next().value.morphPayloadKey = "new-payload"; },
		() => { source.deformationStates = null; },
	];
	for (const change of changes) {
		packets.beginFrame();
		frame = build();
		assert.equal(validate(frame), true);
		packets.endFrame();
		change();
		packets.beginFrame();
		assert.equal(validate(frame), false, "a new preparation must observe authoring changes");
		const actual = build();
		const expected = PreparedSceneBuilder.build(source);
		assert.deepEqual(
			actual.submissions.map((s) => [s.id, s.passFlags, s.instance.renderLayers, s.deformation]),
			expected.submissions.map((s) => [s.id, s.passFlags, s.instance.renderLayers, s.deformation]),
		);
		packets.endFrame();
	}

	packets.beginFrame();
	frame = build();
	assert.equal(validate(frame), true);
	packets.clear();
	primitive.topology = "triangle-list";
	packets.beginFrame();
	const reset = build();
	assert.notEqual(reset.submissions[0], frame.submissions[0]);
	packets.endFrame();

	// Inactive entries can be evicted; validated active entries must stay available.
	scene.remove(instances[0]);
	packets.beginFrame();
	build();
	packets.endFrame();
	assert.equal(packets.getDebugStats().entries, 2);
	scene.add(instances[0]);
	packets.beginFrame();
	assert.equal(validate(reset), false);
	const restored = build();
	assert.notEqual(restored.submissions.find((s) => s.id === reset.submissions[0].id), reset.submissions[0]);
	packets.endFrame();
	const otherCamera = new Camera();
	otherCamera.position.z = 0.5;
	otherCamera.updateWorldMatrix();
	otherCamera.updateMatrices();
	packets.beginFrame();
	const otherView = PreparedSceneBuilder.build({ ...source, camera: otherCamera }, { packetCache: packets });
	assert.equal(otherView.submissions[0], restored.submissions[0]);
	assert.notEqual(otherView.opaquePackets[0], restored.opaquePackets[0]);
	packets.endFrame();
}

function testRepeatedValidationDiscardsEarlierSuccesses() {
	const { scene, instances, source } = createSubmissionReuseFixture();
	const packets = new PreparedScenePacketCache();
	packets.beginFrame();
	const frame = PreparedSceneBuilder.build(source, { packetCache: packets });
	const prepared = new Map(frame.submissions.map((s) => [s.id, s]));
	assert.equal(packets.canReuseSubmissions(scene.getMeshInstances(), prepared, null), true);
	instances[0].mesh.primitives[0].topology = "line-list";
	instances[2].renderLayers = 8;
	assert.equal(packets.canReuseSubmissions(scene.getMeshInstances(), prepared, null), false);
	const changed = PreparedSceneBuilder.build(source, { packetCache: packets });
	assert.equal(changed.submissions[0].geometry.topology, "line-list");
	assert.equal(changed.submissions[2].instance.renderLayers, 8);
	packets.endFrame();
}

function testDisabledIncrementalRebasesBeforePartialUpdates() {
	const camera = createCamera();
	const packet = createPacket("mesh", -0.6, 0.05);
	const decal = createDecalPacket("decal", -0.5, 0.05);
	let frame = createFrame(camera, [packet], [decal]);
	const cache = new PreparedSceneCache();
	const input = {
		viewportWidth: 320, viewportHeight: 180, features: createFeatures(),
		postProcess: createResolvedPostProcess(),
		incrementalOptions: { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS,
			enabled: true, fullFrameFallbackAreaRatio: 1 },
	};
	const original = PreparedSceneBuilder.build;
	PreparedSceneBuilder.build = () => frame;
	try {
		cache.build(input);
		assert.deepEqual(cache.build(input).dirtyRects, []);
		input.incrementalOptions.enabled = false;
		for (const x of [-0.2, 0.4]) {
			frame = createFrame(camera, [createPacket("mesh", x, 0.05)],
				[createDecalPacket("decal", -x, 0.05)]);
			const disabled = cache.build(input);
			assert.equal(disabled.forceFullFrame, true);
			assert.equal(disabled.packetRects.size, 0, "disabled builds skip projected coverage");
			assert.equal(disabled.frame.spatialIndex, null, "full frames need no tile index");
			assert.equal(disabled.dirtyTiles.length,
				disabled.dirtyTileRows * disabled.dirtyTileColumns);
		}
		input.incrementalOptions.enabled = true;
		const rebased = cache.build(input);
		assert.equal(rebased.forceFullFrame, true, "re-enable cannot use stale incremental history");
		assert.ok(rebased.packetRects.size > 0);
		assert.ok(rebased.frame.spatialIndex);
		assert.deepEqual(cache.build(input).dirtyRects, []);
		frame = createFrame(camera, [], frame.decalPackets);
		const removed = cache.build(input);
		assert.equal(removed.forceFullFrame, false);
		for (const rect of rebased.packetRects.values()) {
			assert.ok(removed.dirtyRects.some(dirty => rectContainsRect(dirty, rect)),
				"removed packets retain the re-enabled frame coverage");
		}
		assert.ok(removed.dirtyTiles.length > 0, "removed mesh coverage is dirty");
		frame = createFrame(camera, [], []);
		const removedDecal = cache.build(input);
		assert.equal(removedDecal.forceFullFrame, false);
		assert.ok(removedDecal.dirtyTiles.length > 0,
			"removed decal retains independent re-enabled coverage");
	} finally {
		PreparedSceneBuilder.build = original;
	}
}

function testOpaqueSortKeepsCompatibleGroupsContiguous() {
	const camera = createCamera();
	const material = new Material();
	const otherMaterial = new Material();
	material.name = otherMaterial.name = "same-name";
	const geometryA = {};
	const geometryB = {};
	const make = (name, depth, geometry, effective = material) => {
		const packet = createPacket(name, 0, 0.05);
		packet.sortDepth = depth;
		packet.submission.material = { ...packet.submission.material,
			effective, pipelineKey: "same-pipeline" };
		packet.submission.geometry = { ...packet.submission.geometry, resourceKey: geometry };
		return packet;
	};
	const a = make("a", 3, geometryA);
	const z = make("z", 1, geometryA);
	const m = make("m", 2, geometryB);
	const duplicateName = make("q", 2, geometryA, otherMaterial);
	const transparent = [make("near", 1, geometryA), make("far", 3, geometryB)];
	for (const [input, size] of [[[a, m, z], 3], [[a, m, z, duplicateName], 4],
		[[z, a, duplicateName, m], 4], [[duplicateName, m, a, z], 4]]) {
		PreparedSceneBuilder._finalizeViewPackets(input, transparent, camera, [], {});
		const nearIndex = input.indexOf(z);
		assert.equal(input[nearIndex + 1], a,
			"compatible geometry must be contiguous and front-to-back regardless of instance IDs");
		assert.equal(input.length, size, "all material and geometry groups remain present");
		assert.deepEqual(transparent.map(packet => packet.sortDepth), [3, 1],
			"transparent ordering remains back-to-front");
	}
}

function run() {
	testOpaqueSortKeepsCompatibleGroupsContiguous();
	testDisabledIncrementalRebasesBeforePartialUpdates();
	testDecalOnlyMemoDoesNotMixOtherDecals();
	testRepeatedValidationDiscardsEarlierSuccesses();
	testFallbackConsumesSuccessfulValidationOnce();
	testSubmissionValidationLifetimeAndMutations();
	testDirtySignaturesAreSharedOnlyWithinOneBuild();
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
	testFullFrameBuildSkipsDiffAndRebasesState();
	testCameraMatrixChangeForcesFullFrameAndRebasesPacketRects();
	testMainViewReusesCameraIndependentPreparedState();
	testPreparedPacketCacheReusesViewLocalPackets();
	testPreparedSubmissionsShareMaterialRevisionScans();
	console.log("Prepared scene cache tests passed");
}

run();
