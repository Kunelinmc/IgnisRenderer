import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { compileWGSLMath } from "../../helpers/wgsl-math.mjs";
import { WebGPUFrameCameraState } from "../../../src/backends/webgpu/WebGPUFrameCameraState.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
	packFrameCameraUniformData,
	packFrameEnvironmentUniformData,
	packFrameLightUniformData,
	packFrameShadowUniformData,
	packFlatMaterialUniformData,
	packMaterialCommonUniformData,
	packObjectUniformData,
	packPBRMaterialUniformData,
	packPhongMaterialUniformData,
} from "../../../src/backends/webgpu/packing.ts";
import {
	MAX_AREA_LIGHTS,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../../src/backends/webgpu/constants.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT,
	WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT,
	WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
	WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT,
	WEBGPU_OBJECT_UNIFORM_LAYOUT,
	WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT,
} from "../../../src/backends/webgpu/bufferLayouts.ts";

function matrix(base) {
	return new Matrix4([
		[base + 0, base + 1, base + 2, base + 3],
		[base + 4, base + 5, base + 6, base + 7],
		[base + 8, base + 9, base + 10, base + 11],
		[base + 12, base + 13, base + 14, base + 15],
	]);
}

function readVec(layout, data, path, length) {
	const offset = layout.byteOffsetOf(path) >> 2;
	const floats = data instanceof ArrayBuffer ? new Float32Array(data) : data;
	return Array.from(floats.slice(offset, offset + length));
}

function readU32Vec(layout, data, path, length) {
	const offset = layout.byteOffsetOf(path) >> 2;
	return Array.from(
		new Uint32Array(data.buffer, data.byteOffset, data.length).slice(
			offset,
			offset + length
		)
	);
}

function createTextureSlot(transformA, transformB) {
	return {
		map: null,
		transformA,
		transformB,
	};
}

function createCommonMaterialData() {
	const textureSlots = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => null);
	textureSlots[0] = createTextureSlot([101, 102, 103, 104], [201, 202, 203, 204]);
	textureSlots[1] = createTextureSlot([105, 106, 107, 108], [205, 206, 207, 208]);
	textureSlots[16] = createTextureSlot([49, 50, 51, 52], [53, 54, 55, 56]);

	return {
		baseColorFactor: [1, 2, 3, 4],
		emissiveFactor: [5, 6, 7, 8],
		materialParams: [57, 58, 59, 60],
		renderParams: [63, 0, 0, 0],
		textureSlots,
	};
}

function createPBRMaterialData() {
	return {
		surfaceParams0: [9, 10, 11, 12],
		surfaceParams1: [13, 14, 15, 16],
		surfaceParams2: [17, 18, 19, 20],
		surfaceParams3: [21, 22, 23, 24],
		specularColorFactor: [25, 26, 27, 28],
		sheenColorClearcoatNormalScale: [37, 38, 39, 40],
		attenuationColor: [41, 42, 43, 44],
		anisotropyParams: [45, 46, 47, 48],
		pbrMasks: [61, 62, 0, 0],
	};
}

function testSeparatedMaterialUniformPacking() {
	const objectData = packObjectUniformData(
		matrix(1),
		[
			[301, 302, 303],
			[304, 305, 306],
			[307, 308, 309],
		],
		matrix(401),
		7,
		true,
		false,
	);
	assert.equal(objectData.length * 4, WEBGPU_OBJECT_UNIFORM_BYTE_SIZE);
	assert.deepEqual(readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "modelMatrix", 16), [
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16,
	]);
	assert.deepEqual(readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "normalMatrix", 16), [
		301, 304, 307, 0,
		302, 305, 308, 0,
		303, 306, 309, 0,
		0, 0, 0, 1,
	]);
	assert.deepEqual(
		readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "instanceData", 4),
		[7, 1, 0, 0],
	);

	const commonData = packMaterialCommonUniformData(createCommonMaterialData());
	assert.equal(commonData.length * 4, WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, "baseColorFactor", 4),
		[1, 2, 3, 4],
	);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformA", 16], 4), [
		49, 50, 51, 52,
	]);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformA", 1], 4), [
		105, 106, 107, 108,
	]);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformB", 0], 4), [
		201, 202, 203, 204,
	]);

	const pbrData = packPBRMaterialUniformData(createPBRMaterialData());
	assert.equal(pbrData.length * 4, WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readU32Vec(WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT, pbrData, "pbrMasks", 4),
		[61, 62, 0, 0],
	);

	const legacy = { ambientShininess: [29, 30, 31, 32], specular: [33, 34, 35, 0] };
	const phongData = packPhongMaterialUniformData(legacy);
	const flatData = packFlatMaterialUniformData(legacy);
	assert.equal(phongData.length * 4, WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.equal(flatData.length * 4, WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readVec(WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT, phongData, "ambientShininess", 4),
		legacy.ambientShininess,
	);
	assert.deepEqual(
		readVec(WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT, flatData, "specular", 4),
		legacy.specular,
	);
}

function createReflectionProbe(index) {
	return {
		id: `probe-${index}`,
		worldToProbeMatrix: matrix(1000 + index * 100),
		probeToWorldMatrix: matrix(2000 + index * 100),
		invHalfExtents: [1 + index, 2 + index, 3 + index],
		radiusInv: 4 + index,
		captureWorldPosition: [5 + index, 6 + index, 7 + index],
		shape: index % 2,
		parallaxMode: index,
		blendDistance: 8 + index,
		blendExponent: 9 + index,
		layer: 10 + index,
	};
}

function createFrameInput() {
	return {
		viewProjectionMatrix: matrix(1),
		prevViewProjectionMatrix: matrix(101),
		inverseViewMatrix: matrix(201),
		inverseProjectionMatrix: matrix(301),
		jitterNdc: [43, 44],
		prevJitterNdc: [45, 46],
		ambientColor: [15, 16, 17],
		shAmbientCoeffs: null,
		localLightProbeCount: 0,
		localLightProbes: [],
		irradianceProbeGrid: {
			id: "grid-0",
			worldToGridMatrix: matrix(3000),
			dimensions: [3, 4, 5],
			invHalfExtents: [0.25, 0.5, 0.75],
			blendDistance: 12,
			cellCount: 60,
			textureRevision: 1,
			sh: [],
			validMask: new Uint8Array(60),
		},
		directionalLights: [
			{
				direction: [18, 19, 20],
				color: [21, 22, 23],
			},
		],
		directionalShadows: [],
		pointLights: [
			{
				position: [24, 25, 26],
				range: 27,
				color: [28, 29, 30],
			},
		],
		spotLights: [
			{
				position: [31, 32, 33],
				range: 34,
				direction: [35, 36, 37],
				outerCos: 38,
				color: [39, 40, 41],
				innerCos: 42,
			},
		],
		spotShadows: [],
		areaLights: [
			{
				position: [47, 48, 49],
				range: 50,
				right: [51, 52, 53],
				width: 54,
				up: [55, 56, 57],
				height: 58,
				normal: [59, 60, 61],
				areaScale: 62,
				color: [63, 64, 65],
			},
		],
		reflectionProbeCount: 2,
		reflectionProbes: [
			createReflectionProbe(0),
			createReflectionProbe(1),
		],
		enableLighting: true,
		enableShadows: true,
		shAmbientEnabled: false,
		enableClusteredLighting: true,
		envSpecularIsLinear: false,
		hasEnvSpecular: true,
		hasBRDFLUT: true,
		envSpecularMaxMipLevel: 5,
	};
}

function testCameraCacheAndInverseFailureRecovery() {
	const state = new WebGPUFrameCameraState();
	const view = Matrix4.identity();
	const projection = Matrix4.identity();
	const viewProjection = Matrix4.identity();
	const inverse = Matrix4.inverse;
	let calls = 0;
	Matrix4.inverse = (...args) => { calls++; return inverse(...args); };
	try {
		state.prepare(view, projection, viewProjection);
		assert.equal(calls, 3);
		const inverseView = state.inverseViewMatrix;
		const inverseProjection = state.inverseProjectionMatrix;
		state.prepare(view.clone(), projection.clone(), viewProjection.clone());
		assert.equal(calls, 3, "unchanged numeric values reuse inverse outputs");
		view.elements[0][3] = 7;
		state.prepare(view, projection, viewProjection);
		assert.equal(calls, 4, "in-place view mutation recomputes only that inverse");
		assert.equal(state.inverseViewMatrix, inverseView);
		assert.equal(inverseView.elements[0][3], -7);
		projection.elements[0][0] = 2;
		state.prepare(view, projection, viewProjection);
		assert.equal(calls, 5);
		assert.equal(state.inverseProjectionMatrix, inverseProjection);
		assert.equal(inverseProjection.elements[0][0], 0.5);
		viewProjection.elements[1][1] = 0;
		assert.throws(() => state.prepare(view, projection, viewProjection), /view-projection.*singular/i);
	} finally {
		Matrix4.inverse = inverse;
	}
	const recoverable = new WebGPUFrameCameraState();
	const identity = Matrix4.identity();
	recoverable.prepare(identity, identity, identity);
	const overflowInverse = new Matrix4([
		[1, 1e38, 0, 0], [0, 1, 1e38, 0], [0, 0, 1, 1e38], [0, 0, 0, 1],
	]);
	assert.throws(() => recoverable.prepare(overflowInverse, identity, identity), /view inverse.*finite/i);
	recoverable.prepare(identity, identity, identity);
	assert.deepEqual(recoverable.inverseViewMatrix.elements, identity.elements,
		"a rejected inverse must not poison the previous value cache");
}

function testPackedCameraProjectionRoundTrip() {
	const near = 0.3;
	const far = 100;
	const left = -0.7, right = 1.3, bottom = -0.4, top = 0.8;
	const projections = [
		new Matrix4([
			[2 * near / (right - left), 0, (right + left) / (right - left), 0],
			[0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
			[0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
			[0, 0, -1, 0],
		]),
		new Matrix4([
			[0.2, 0, 0, -0.4], [0, 0.25, 0, -0.5],
			[0, 0, -2 / (far - near), -(far + near) / (far - near)], [0, 0, 0, 1],
		]),
	];
	const transform = (packed, offset, point) => Array.from({ length: 4 }, (_, row) =>
		point.reduce((sum, value, column) => sum + packed[offset + column * 4 + row] * value, 0));
	for (const projection of projections) {
		const state = new WebGPUFrameCameraState();
		const view = new Matrix4([[1, 0, 0, -3], [0, 1, 0, 2], [0, 0, 1, -5], [0, 0, 0, 1]]);
		state.prepare(view, projection, projection);
		const jitter = [0.003, -0.004];
		const packed = new Float32Array(packFrameCameraUniformData({
			viewProjectionMatrix: projection, prevViewProjectionMatrix: projection,
			inverseViewMatrix: state.inverseViewMatrix,
			inverseProjectionMatrix: state.inverseProjectionMatrix,
			jitterNdc: jitter, prevJitterNdc: [0, 0],
		}));
		for (const point of [[1.2, -0.7, -2, 1], [-1, 1.1, -17, 1]]) {
			const clip = transform(packed, 0, point);
			const uv = [(clip[0] / clip[3] + jitter[0]) * 0.5 + 0.5,
				0.5 - (clip[1] / clip[3] + jitter[1]) * 0.5];
			const hardwareDepth = clip[2] / clip[3] * 0.5 + 0.5;
			const reconstructed = transform(packed, 48, [uv[0] * 2 - 1 - jitter[0],
				1 - uv[1] * 2 - jitter[1], hardwareDepth * 2 - 1, 1]);
			const viewPoint = reconstructed.map((value) => value / reconstructed[3]);
			for (let component = 0; component < 3; component++) {
				assert.ok(Math.abs(viewPoint[component] - point[component]) < 1e-4);
			}
			const worldPoint = transform(packed, 32, viewPoint);
			assert.ok(Math.abs(worldPoint[0] - (point[0] + 3)) < 1e-4);
			assert.ok(Math.abs(worldPoint[1] - (point[1] - 2)) < 1e-4);
			assert.ok(Math.abs(worldPoint[2] - (point[2] + 5)) < 1e-4);
		}
	}
}

function testCameraMathWithObliqueClipping() {
	const source = readFileSync(new URL(
		"../../../src/shaders/webgpu/common/frameCamera.wgsl", import.meta.url,
	), "utf8");
	const cameraMath = compileWGSLMath(source, [
		"cameraWorldRayDirection", "cameraWorldRayOrigin", "cameraReconstructViewPosition",
	]);
	const columns = (matrix) => [0, 1, 2, 3].map((column) =>
		matrix.elements.map((row) => Math.fround(row[column])));
	const assertVector = (actual, expected, label) => {
		assert.ok(actual.every(Number.isFinite), `${label}: expected finite values, got ${actual}`);
		for (let i = 0; i < expected.length; i++) {
			const tolerance = Math.max(1, Math.abs(expected[i])) * 1e-5;
			assert.ok(Math.abs(actual[i] - expected[i]) < tolerance,
				`${label}[${i}]: expected ${expected[i]}, got ${actual[i]}`);
		}
	};
	const identity = columns(Matrix4.identity());
	const mirrorInverseView = columns(new Matrix4([
		[1, 0, 0, 0], [0, -1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1],
	]));
	const samples = [
		{ uv: [0.5, 0.6], point: [0, -0.4618802154, -4], ray: [0, -0.1147078669, -0.9933992678] },
		{ uv: [0.5, 0.5], point: [0, 0, -4], ray: [0, 0, -1] },
		{ uv: [0.5, 0.4], point: [0, 0.4618802154, -4], ray: [0, 0.1147078669, -0.9933992678] },
	];
	for (const oblique of [false, true]) {
		const projection = Matrix4.perspective(60, 1, 0.1, 1000);
		if (oblique) {
			// A horizontal mirror one unit below a camera looking along -Z.
			projection.applyObliqueClipping({ normal: { x: 0, y: -1, z: 0 }, constant: -1 });
		}
		const inverseProjection = columns(Matrix4.inverse(projection));
		for (const jitter of [[0, 0], [0.02, -0.04]]) {
			for (const { uv, point, ray } of samples) {
				const rasterUv = [uv[0] + jitter[0] * 0.5, uv[1] - jitter[1] * 0.5];
				const label = `oblique=${oblique}, uv=${uv}, jitter=${jitter}`;
				assertVector(cameraMath.cameraWorldRayDirection(
					identity, inverseProjection, jitter, rasterUv,
				), ray, `${label}: forward ray`);
				assertVector(cameraMath.cameraReconstructViewPosition(
					inverseProjection, jitter, rasterUv, 4,
				), point, `${label}: linear depth`);
				assertVector(cameraMath.cameraWorldRayOrigin(
					mirrorInverseView, inverseProjection, jitter, rasterUv,
				), [0, -1, 0], `${label}: mirrored world origin`);
				assertVector(cameraMath.cameraWorldRayDirection(
					mirrorInverseView, inverseProjection, jitter, rasterUv,
				), [ray[0], -ray[1], ray[2]], `${label}: mirrored world ray`);
			}
		}
	}
	const orthoInverse = columns(Matrix4.inverse(Matrix4.ortho(-2, 6, -3, 5, 0.1, 100)));
	for (const [uv, origin] of [[[0.5, 0.5], [2, 1, 0]], [[0.25, 0.75], [0, -1, 0]]]) {
		assertVector(cameraMath.cameraWorldRayOrigin(identity, orthoInverse, [0, 0], uv),
			origin, `asymmetric orthographic origin at ${uv}`);
		assertVector(cameraMath.cameraWorldRayDirection(identity, orthoInverse, [0, 0], uv),
			[0, 0, -1], `orthographic ray at ${uv}`);
		assertVector(cameraMath.cameraReconstructViewPosition(orthoInverse, [0, 0], uv, 4),
			[origin[0], origin[1], -4], `orthographic linear depth at ${uv}`);
	}
	const distantInverse = columns(Matrix4.inverse(Matrix4.perspective(60, 1, 0.001, 5000)));
	for (const [depth, expected] of [
		[1000, [288.6751346, -115.4700538, -1000]],
		[4000, [1154.7005384, -461.8802154, -4000]],
		[5000, [1443.3756730, -577.3502692, -5000]],
	]) {
		assertVector(cameraMath.cameraReconstructViewPosition(
			distantInverse, [0, 0], [0.75, 0.6], depth,
		), expected, `distant linear depth ${depth} with a small near plane`);
	}
}

function testFrameUniformPacking() {
	assert.equal(WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE, 272);
	const input = createFrameInput();
	input.directionalShadows = [{
		enabled: true,
		strategyType: "single-map",
		cascadeCount: 1,
		cascadeBlendRatio: 0,
		cascadeViewProjectionMatrices: [null, null, null, null],
		cascadeSplits: [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
		depthProjectionParams: [[-1, -2, -1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
		viewProjectionMatrix: matrix(500),
		depthBias: 0.01,
		slopeBias: 0.02,
		normalBias: 0.03,
		normalBiasMin: 0.04,
		filterMode: "pcss",
		samplingQuality: "high",
		shadowStrength: 0.9,
		shadowMapBaseSize: 1024,
		shadowMapSize: 1024,
		atlasTileSize: 1024,
	}];
	const cameraData = packFrameCameraUniformData(input);
	const lightData = packFrameLightUniformData(input);
	const shadowData = packFrameShadowUniformData(input);
	const environmentData = packFrameEnvironmentUniformData(input);
	assert.equal(
		WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		cameraData.byteLength + lightData.byteLength + shadowData.byteLength + environmentData.byteLength,
		12768
	);
	assert.deepEqual(
		readVec(
			WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
			shadowData,
			["directionalShadows", 0, "paramsD"],
			4,
		),
		[1, 2, 0, 0],
	);
	assert.deepEqual(
		readVec(
			WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
			shadowData,
			["directionalShadows", 0, "depthProjectionParams", 0],
			4,
		),
		[-1, -2, -1, 0],
	);
	assert.ok(cameraData instanceof ArrayBuffer);
	assert.equal(cameraData.byteLength, 272);
	assert.equal(lightData.byteLength, 1712);
	assert.equal(environmentData.byteLength, 4240);
	for (const [field, offset] of Object.entries({
		viewProjection: 0, prevViewProjection: 64, inverseView: 128,
		inverseProjection: 192, jitterNdc: 256, prevJitterNdc: 264,
	})) {
		assert.equal(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteOffsetOf(field), offset);
	}
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "inverseView", 4), [201, 205, 209, 213]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "inverseProjection", 4), [301, 305, 309, 313]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "jitterNdc", 2), [43, 44]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "prevJitterNdc", 2), [45, 46]);
	assert.deepEqual(Array.from(new Uint32Array(lightData, 16, 8)), [1, 1, 1, 1, 1, 1, 0, 0]);
	assert.deepEqual(Array.from(new Uint32Array(shadowData, 0, 4)), [1, 0, 0, 0]);
	assert.deepEqual(Array.from(new Uint32Array(environmentData, 0, 4)), [0, 1, 1, 0]);
	assert.deepEqual(Array.from(new Uint32Array(environmentData, 20, 3)), [0, 0, 0]);
	assert.equal(new DataView(environmentData).getFloat32(16, true), 5);
	const writer = WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.createWriter();
	const reused = packFrameLightUniformData(input, writer);
	input.pointLights = [];
	input.enableLighting = false;
	assert.equal(packFrameLightUniformData(input, writer), reused);
	assert.deepEqual(Array.from(new Uint32Array(reused, 16, 8)), [1, 0, 1, 1, 0, 1, 0, 0]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, reused, ["pointLights", 0, "positionRange"], 4), [0, 0, 0, 0]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["directionalLights", 0, "direction"], 4), [
		18, 19, 20, 0,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["pointLights", 0, "positionRange"], 4), [
		24, 25, 26, 27,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["spotLights", 0, "colorInner"], 4), [
		39, 40, 41, 42,
	]);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "localLightProbeCounts", 4),
		[0, 2, 0, 0]
	);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["areaLights", 0, "positionRange"], 4), [
		47, 48, 49, 50,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["areaLights", 0, "normalAreaScale"], 4), [
		59, 60, 61, 62,
	]);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 0, "worldToProbeRow0"], 4),
		[1000, 1001, 1002, 1003]
	);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 1, "probeToWorldRow2"], 4),
		[2108, 2109, 2110, 2111]
	);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 1, "dataC"], 4), [
		1, 9, 10, 11,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridWorldToGridRow1", 4), [
		3004, 3005, 3006, 3007,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataA", 4), [
		3, 4, 5, 60,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataB", 4), [
		0.25, 0.5, 0.75, 12,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataC", 4), [
		1, WEBGPU_SH_COEFFICIENT_COUNT, 60, 0,
	]);
	assert.equal(
		readVec(WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT, shadowData, ["directionalShadows", 0, "paramsA"], 4)[0],
		1
	);
}

testSeparatedMaterialUniformPacking();
testCameraCacheAndInverseFailureRecovery();
testPackedCameraProjectionRoundTrip();
testCameraMathWithObliqueClipping();
testFrameUniformPacking();
console.log("WebGPU packing tests passed");
