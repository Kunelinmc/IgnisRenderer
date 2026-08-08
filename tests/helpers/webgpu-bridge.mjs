import assert from "node:assert/strict";
import { Matrix4 } from "../../src/maths/Matrix4.ts";
import { SH } from "../../src/maths/SH.ts";
import { Texture } from "../../src/core/Texture.ts";
import { CubeTexture } from "../../src/core/CubeTexture.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../src/meshes/MeshInstance.ts";
import { resolveFeatureState } from "../../src/pipeline/FeatureResolver.ts";
import { WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT } from "../../src/backends/webgpu/bufferLayouts.ts";
import { createResolvedPostProcess } from "./postprocess.mjs";
import { EMPTY_SHADOW_FRAME_PLAN } from "../../src/lights/shadows/ShadowFramePlan.ts";

function nearlyEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);
}

function createMainFrameOptions(options = {}) {
	return {
		scopeKey: "main",
		sceneTargetMode: "mrt",
		...options,
	};
}


function createModel(materials) {
	const mesh = MeshAsset.fromFaces(
		materials.map((material, index) => ({
			material,
			vertices: [
				{
					x: index,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: index + 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: index,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		}))
	);
	return new MeshInstance({ mesh });
}

function createPacket(model) {
	model.updateWorldMatrix(model.parent?.worldMatrix);
	const primitive = model.mesh.primitives[0];
	const worldMatrix = model.worldMatrix;
	return {
		id: `${model.id}:${primitive.id}`,
		meshInstance: model,
		mesh: model.mesh,
		primitive,
		material: primitive.material,
		geometry: primitive.geometry,
		worldMatrix,
		normalMatrix: Matrix4.normalMatrix(worldMatrix),
		worldBounds: primitive.boundingSphere,
		sortDepth: 1,
		pipelineKey: "test",
		passFlags: 0,
	};
}

function createFrame(packet) {
	const cameraPosition = { x: 0, y: 0, z: 5 };
	const frame = {
		sceneBounds: packet.mesh.boundingSphere,
		lights: [],
		camera: {
			viewProjectionMatrix: Matrix4.identity(),
			viewMatrix: Matrix4.identity(),
			position: cameraPosition,
			getWorldPosition() {
				return this.position;
			},
			fov: 60,
			aspectRatio: 1,
			type: "perspective",
		},
		environment: createEnvironmentSnapshot(null, null),
		meshInstances: [packet.meshInstance],
		shadowMaps: new Map(),
		opaquePackets: [packet],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		decalPackets: [],
	};
	frame.shadowPlan = EMPTY_SHADOW_FRAME_PLAN;
	return frame;
}

function createFrameContext(frame, features, options = {}) {
	const width = options.width ?? 1;
	const height = options.height ?? 1;
	return {
		viewCamera: frame.camera,
		attachments: { width, height },
		features,
		postProcess: options.postProcess ?? createResolvedPostProcess(),
		shadowPlan: frame.shadowPlan,
		scene: options.scene ?? frame,
		shCoeffs: options.shCoeffs ?? SH.empty(),
		shAmbientCoeffs: options.shAmbientCoeffs ?? SH.empty(),
		worldMatrix: options.worldMatrix ?? Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: Math.max(width, height),
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset: options.temporalHistoryReset === true,
		},
		transient: options.transient ?? new Map(),
	};
}

function createFrameContextWithFeatures(
	frame,
	featureRequest,
	capabilities,
	options = {}
) {
	return createFrameContext(
		frame,
		resolveFeatureState(featureRequest, capabilities, "webgpu"),
		options
	);
}

function createPreparedFrameResources(options = {}) {
	const scopeKey = options.scopeKey ?? "test";
	const sceneTargetMode = options.sceneTargetMode ?? "single";
	return {
		scopeKey,
		sceneTargetMode,
		frameBinding: { bindGroup: { label: `${scopeKey}:frame` } },
		decalFrameBinding: { bindGroup: { label: `${scopeKey}:decal-frame` } },
		environmentBinding: null,
		clusteredSceneBinding: null,
		lightingState: {},
		featureData: { get: () => undefined },
		featureState: {},
		environmentState: {},
		jointMatrixMap: new Map(),
		morphWeightMap: new Map(),
	};
}

function createFrameScopeAdapter(resources) {
	return {
		prepare: (context, options) => resources.prepareFrame(context, options),
		updateParticleShadowVolumes() {},
		destroy() {},
	};
}


function createTinyTexture(mips = 1) {
	const texture = new Texture({
		data: new Float32Array([1, 1, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
	texture.mipmaps = Array.from(
		{ length: mips },
		() => new Float32Array([1, 1, 1, 1])
	);
	return texture;
}

function createTinyCubeTexture(mips = 1, value = 1) {
	const createFace = () => new Float32Array([value, value, value, 1]);
	const faceMipmaps = [];
	for (let level = 1; level < mips; level++) {
		faceMipmaps.push(
			Array.from({ length: 6 }, () => createFace())
		);
	}
	return new CubeTexture({
		faces: Array.from({ length: 6 }, () => createFace()),
		faceMipmaps,
		size: 1,
		colorSpace: "HDR",
	});
}

function createEnvironmentSnapshot(
	backgroundTexture = null,
	iblTexture = null
) {
	return {
		backgroundEnabled: true,
		lightingEnabled: true,
		backgroundTexture,
		iblTexture,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};
}


function createWebGPUFrameContextForTemporalTest(
	frame,
	features,
	postProcess,
	width,
	height,
	temporalHistoryReset = false
) {
	return {
		viewCamera: frame.camera,
		attachments: { width, height },
		features,
		postProcess,
		shadowPlan: frame.shadowPlan,
		scene: frame,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width, height }],
			dirtyTileSize: Math.max(width, height),
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			postProcessStartPass: null,
			reasonMask: 0,
			temporalHistoryReset,
		},
		transient: new Map(),
	};
}

function readLatestFrameCameraUniformField(backend, field) {
	const frameWrites = backend.writeCalls.filter(
		(call) =>
			call[0] === "writeBuffer" &&
			call[1]?.label === "WebGPUFrameCameraUniforms"
	);
	assert.ok(frameWrites.length > 0);
	const data = frameWrites[frameWrites.length - 1][2];
	const offset = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteOffsetOf(field) / 4;
	const length = WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteSizeOf(field) / 4;
	return Array.from(data.slice(offset, offset + length));
}

function assertArrayNearlyEqual(actual, expected, epsilon = 1e-6) {
	assert.equal(actual.length, expected.length);
	for (let i = 0; i < actual.length; i++) {
		nearlyEqual(actual[i], expected[i], epsilon);
	}
}

export {
	nearlyEqual,
	createMainFrameOptions,
	createModel,
	createPacket,
	createFrame,
	createFrameContext,
	createFrameContextWithFeatures,
	createPreparedFrameResources,
	createFrameScopeAdapter,
	createTinyTexture,
	createTinyCubeTexture,
	createEnvironmentSnapshot,
	createWebGPUFrameContextForTemporalTest,
	readLatestFrameCameraUniformField,
	assertArrayNearlyEqual,
};
