import assert from "node:assert/strict";import { Material } from "../../../src/materials/Material.ts";import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";import { Matrix4 } from "../../../src/maths/Matrix4.ts";import { WebGLGeometryRegistry } from "../../../src/backends/webgl/WebGLGeometryRegistry.ts";import { drawWebGLPacket } from "../../../src/backends/webgl/WebGLScenePass.ts";import { createGeometryTestGL, createRetryGeometryTestGL, createGeometryCaptureGL, createScenePassCaptureGL, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

function testGeometryRegistryRejectsOutOfRangeIndices() {
	const warnings = [];
	const registry = new WebGLGeometryRegistry(createGeometryTestGL(), (k, m) =>
		warnings.push({ key: k, message: m })
	);

	const primitive = {
		id: "p0",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: null,
			uv0: null,
			indices: new Uint32Array([0, 1, 9]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-0",
		primitive,
	};

	assert.equal(registry.getGeometry(packet), null);
	assert.ok(
		warnings.some((warning) => warning.key === "webgl-geometry-index-range-p0")
	);
}

function testGeometryRegistryRetriesAfterUploadAllocationFailure() {
	const warnings = [];
	const registry = new WebGLGeometryRegistry(createRetryGeometryTestGL(), (k, m) =>
		warnings.push({ key: k, message: m })
	);

	const primitive = {
		id: "p-retry",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: null,
			uv0: null,
			indices: new Uint32Array([0, 1, 2]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-retry",
		primitive,
	};

	assert.equal(registry.getGeometry(packet), null);
	const retried = registry.getGeometry(packet);
	assert.ok(retried);
	assert.equal(retried?.indexCount, 3);
	assert.ok(
		warnings.some(
			(warning) => warning.key === "webgl-geometry-upload-failed-p-retry"
		)
	);
}

function testGeometryRegistryUploadsUV1Attribute() {
	const gl = createGeometryCaptureGL();
	const registry = new WebGLGeometryRegistry(gl, () => {});
	const primitive = {
		id: "p-uv1",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: new Float32Array([
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
			]),
			uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
			uv1: new Float32Array([0.25, 0.5, 0.75, 0.5, 0.25, 0.9]),
			uv2: new Float32Array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75]),
			uv3: new Float32Array([0.875, 0.75, 0.625, 0.5, 0.375, 0.25]),
			tangents: new Float32Array([
				1, 0, 0, 1,
				0, 1, 0, -1,
				0, 0, 1, 1,
			]),
			indices: new Uint32Array([0, 1, 2]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-uv1",
		primitive,
	};

	const handle = registry.getGeometry(packet);
	assert.ok(handle);
	assert.ok(gl.calls.vertexData instanceof Float32Array);
	assert.equal(gl.calls.vertexData.length, 54);
	assert.equal(gl.calls.vertexData[8], 0.25);
	assert.equal(gl.calls.vertexData[9], 0.5);
	assert.equal(gl.calls.vertexData[10], 0.125);
	assert.equal(gl.calls.vertexData[11], 0.25);
	assert.equal(gl.calls.vertexData[12], 0.875);
	assert.equal(gl.calls.vertexData[13], 0.75);
	assert.deepEqual(Array.from(gl.calls.vertexData.slice(14, 18)), [1, 0, 0, 1]);
	assert.equal(gl.calls.vertexData[26], 0.75);
	assert.equal(gl.calls.vertexData[27], 0.5);
	assert.equal(gl.calls.vertexData[28], 0.375);
	assert.equal(gl.calls.vertexData[29], 0.5);
	assert.equal(gl.calls.vertexData[30], 0.625);
	assert.equal(gl.calls.vertexData[31], 0.5);
	assert.deepEqual(Array.from(gl.calls.vertexData.slice(32, 36)), [0, 1, 0, -1]);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 3 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 32
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 4 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 40
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 5 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 48
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 6 &&
				call.size === 4 &&
				call.stride === 72 &&
				call.offset === 56
		)
	);
}

function testGeometryRegistryUploadsSkinAndMorphResources() {
	const gl = createGeometryCaptureGL();
	gl.MAX_TEXTURE_SIZE = 0x0d33;
	gl.TEXTURE_2D = 0x0de1;
	gl.TEXTURE_MIN_FILTER = 0x2801;
	gl.TEXTURE_MAG_FILTER = 0x2800;
	gl.TEXTURE_WRAP_S = 0x2802;
	gl.TEXTURE_WRAP_T = 0x2803;
	gl.NEAREST = 0x2600;
	gl.CLAMP_TO_EDGE = 0x812f;
	gl.RGBA32F = 0x8814;
	gl.RGBA = 0x1908;
	gl.getParameter = () => 64;
	const textureUploads = [];
	let textureId = 0;
	gl.createTexture = () => ({ id: ++textureId });
	gl.deleteTexture = () => {};
	gl.bindTexture = () => {};
	gl.texParameteri = () => {};
	gl.texImage2D = (...args) => {
		textureUploads.push({
			width: args[3],
			height: args[4],
			data: new Float32Array(args[8]),
		});
	};
	const registry = new WebGLGeometryRegistry(gl, () => {});
	const positionDelta = new Float32Array([
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	]);
	const normalDelta = new Float32Array([
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9,
	]);
	const morphTargets = Array.from({ length: 10 }, (_, index) => index === 0 ? {
		positions: positionDelta,
		normals: normalDelta,
	} : {});
	const primitive = {
		id: "p-deformation",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			indices: new Uint32Array([0, 1, 2]),
			joints0: new Uint16Array([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]),
			weights0: new Float32Array([0.5, 0.5, 0, 0, 1, 0, 0, 0, 0.25, 0.75, 0, 0]),
			joints1: new Uint16Array(12),
			weights1: new Float32Array(12),
			morphTargets,
		},
		topology: "triangle-list",
	};
	const handle = registry.getGeometry({ id: "packet-deformation", primitive });
	assert.ok(handle);
	assert.equal(handle.skinProfile, "skin8");
	assert.equal(handle.morphTargetCount, 8);
	assert.equal(handle.morphSemanticMask, 3);
	assert.ok(handle.skinBuffer);
	assert.ok(handle.morphPositionTexture);
	assert.ok(handle.morphNormalTexture);
	for (const location of [7, 8, 9, 10]) {
		assert.ok(gl.calls.attributePointers.some((call) => call.index === location));
	}
	assert.equal(textureUploads.length, 2);
	assert.deepEqual(Array.from(textureUploads[0].data.slice(0, 4)), [1, 2, 3, 0]);
	assert.ok(Math.abs(textureUploads[1].data[0] - 0.1) < 1e-6);
	registry.destroy();
}

function testDrawWebGLPacketBindsPBRTexturesAndUVSets() {
	const gl = createScenePassCaptureGL();
	const material = new PBRMaterial();
	const baseMap = { id: "base-map", linear: false };
	const normalMap = { id: "normal-map", linear: true };
	const metallicRoughnessMap = { id: "mr-map", linear: true };
	const emissiveMap = { id: "emissive-map", linear: false };
	const occlusionMap = { id: "occlusion-map", linear: true };
	const iridescenceMap = { id: "iridescence-map", linear: true };
	const iridescenceThicknessMap = {
		id: "iridescence-thickness-map",
		linear: true,
	};
	const anisotropyMap = { id: "anisotropy-map", linear: true };
	baseMap.repeat = { x: 0.5, y: 1.5 };
	baseMap.offset = { x: 0.25, y: -0.125 };
	baseMap.rotation = Math.PI / 6;
	normalMap.repeat = { x: 2, y: 0.5 };
	normalMap.offset = { x: -0.2, y: 0.3 };
	normalMap.rotation = -Math.PI / 4;
	metallicRoughnessMap.repeat = { x: 1.25, y: 0.75 };
	metallicRoughnessMap.offset = { x: 0.1, y: 0.2 };
	metallicRoughnessMap.rotation = Math.PI / 8;
	emissiveMap.repeat = { x: 0.8, y: 0.9 };
	emissiveMap.offset = { x: -0.05, y: 0.15 };
	emissiveMap.rotation = 0;
	occlusionMap.repeat = { x: 1.1, y: 1.2 };
	occlusionMap.offset = { x: 0.05, y: -0.1 };
	occlusionMap.rotation = Math.PI / 3;
	iridescenceMap.repeat = { x: 0.7, y: 1.4 };
	iridescenceMap.offset = { x: 0.12, y: -0.07 };
	iridescenceMap.rotation = Math.PI / 5;
	iridescenceThicknessMap.repeat = { x: 1.3, y: 0.6 };
	iridescenceThicknessMap.offset = { x: -0.11, y: 0.09 };
	iridescenceThicknessMap.rotation = -Math.PI / 7;
	anisotropyMap.repeat = { x: 0.9, y: 1.1 };
	anisotropyMap.offset = { x: 0.07, y: -0.03 };
	anisotropyMap.rotation = Math.PI / 9;
	material.map = baseMap;
	material.albedoMapUV = 2;
	material.normalMap = normalMap;
	material.normalMapUV = 3;
	material.normalScale = 0.35;
	material.metallicRoughnessMap = metallicRoughnessMap;
	material.metallicRoughnessMapUV = 2;
	material.emissiveMap = emissiveMap;
	material.emissiveMapUV = 3;
	material.occlusionMap = occlusionMap;
	material.occlusionMapUV = 2;
	material.occlusionStrength = 0.4;
	material.iridescenceFactor = 0.8;
	material.iridescenceMap = iridescenceMap;
	material.iridescenceMapUV = 3;
	material.iridescenceThicknessMap = iridescenceThicknessMap;
	material.iridescenceThicknessMapUV = 2;
	material.anisotropyStrength = 0.6;
	material.anisotropyRotation = Math.PI / 4;
	material.anisotropyMap = anisotropyMap;
	material.anisotropyMapUV = 3;
	const textureTable = new Map([
		[baseMap, { texture: { id: "base" }, isLinear: false }],
		[normalMap, { texture: { id: "normal" }, isLinear: true }],
		[metallicRoughnessMap, { texture: { id: "mr" }, isLinear: true }],
		[emissiveMap, { texture: { id: "emissive" }, isLinear: false }],
		[occlusionMap, { texture: { id: "occlusion" }, isLinear: true }],
		[iridescenceMap, { texture: { id: "iridescence" }, isLinear: true }],
		[
			iridescenceThicknessMap,
			{ texture: { id: "iridescence-thickness" }, isLinear: true },
		],
		[anisotropyMap, { texture: { id: "anisotropy" }, isLinear: true }],
	]);
	const sceneProgram = {
		program: {},
		samplerLayout: {
			units: {
				uBaseMap: 0,
				uNormalMap: 1,
				uMetallicRoughnessMap: 2,
				uEmissiveMap: 3,
				uOcclusionMap: 4,
				uIridescenceMap: 5,
				uIridescenceThicknessMap: 6,
				uAnisotropyMap: 7,
			},
			activeSamplerNames: [
				"uBaseMap",
				"uNormalMap",
				"uMetallicRoughnessMap",
				"uEmissiveMap",
				"uOcclusionMap",
				"uIridescenceMap",
				"uIridescenceThicknessMap",
				"uAnisotropyMap",
			],
			required: 8,
			available: 16,
		},
		uniforms: {
			model: null,
			normalMatrix: null,
			prevModel: null,
			shadingModel: null,
			baseColor: null,
			emissive: null,
			pbr: null,
			transmissionVolume: null,
			iridescence: "uIridescence",
			attenuationColor: null,
			anisotropy: "uAnisotropy",
			phong: null,
			alpha: null,
			baseMap: "uBaseMap",
			hasBaseMap: "uHasBaseMap",
			baseMapIsLinear: "uBaseMapIsLinear",
			baseMapUV: "uBaseMapUV",
			baseMapTransformA: "uBaseMapTransformA",
			baseMapTransformB: "uBaseMapTransformB",
			metallicRoughnessMap: "uMetallicRoughnessMap",
			hasMetallicRoughnessMap: "uHasMetallicRoughnessMap",
			metallicRoughnessMapUV: "uMetallicRoughnessMapUV",
			metallicRoughnessMapTransformA: "uMetallicRoughnessMapTransformA",
			metallicRoughnessMapTransformB: "uMetallicRoughnessMapTransformB",
			normalMap: "uNormalMap",
			hasNormalMap: "uHasNormalMap",
			normalMapUV: "uNormalMapUV",
			normalMapTransformA: "uNormalMapTransformA",
			normalMapTransformB: "uNormalMapTransformB",
			normalScale: "uNormalScale",
			emissiveMap: "uEmissiveMap",
			hasEmissiveMap: "uHasEmissiveMap",
			emissiveMapIsLinear: "uEmissiveMapIsLinear",
			emissiveMapUV: "uEmissiveMapUV",
			emissiveMapTransformA: "uEmissiveMapTransformA",
			emissiveMapTransformB: "uEmissiveMapTransformB",
			occlusionMap: "uOcclusionMap",
			hasOcclusionMap: "uHasOcclusionMap",
			occlusionMapUV: "uOcclusionMapUV",
			occlusionMapTransformA: "uOcclusionMapTransformA",
			occlusionMapTransformB: "uOcclusionMapTransformB",
			occlusionStrength: "uOcclusionStrength",
			iridescenceMap: "uIridescenceMap",
			hasIridescenceMap: "uHasIridescenceMap",
			iridescenceMapUV: "uIridescenceMapUV",
			iridescenceMapTransformA: "uIridescenceMapTransformA",
			iridescenceMapTransformB: "uIridescenceMapTransformB",
			iridescenceThicknessMap: "uIridescenceThicknessMap",
			hasIridescenceThicknessMap: "uHasIridescenceThicknessMap",
			iridescenceThicknessMapUV: "uIridescenceThicknessMapUV",
			iridescenceThicknessMapTransformA: "uIridescenceThicknessMapTransformA",
			iridescenceThicknessMapTransformB: "uIridescenceThicknessMapTransformB",
			hasAnisotropyMap: "uHasAnisotropyMap",
			anisotropyMapUV: "uAnisotropyMapUV",
			anisotropyMapTransformA: "uAnisotropyMapTransformA",
			anisotropyMapTransformB: "uAnisotropyMapTransformB",
			doubleSided: null,
			customSamplers: {},
		},
	};
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: 4,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return textureTable.get(texture) ?? { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const packet = {
		id: "packet-pbr-textures",
		meshInstance: { id: "mesh-0", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	const unitFor = (name) =>
		gl.calls.uniform1i.find((entry) => entry.location === name)?.value;
	assert.equal(unitFor("uBaseMap"), 0);
	assert.equal(unitFor("uNormalMap"), 1);
	assert.equal(unitFor("uMetallicRoughnessMap"), 2);
	assert.equal(unitFor("uEmissiveMap"), 3);
	assert.equal(unitFor("uOcclusionMap"), 4);
	assert.equal(unitFor("uIridescenceMap"), 5);
	assert.equal(unitFor("uIridescenceThicknessMap"), 6);
	assert.equal(unitFor("uBaseMapUV"), 2);
	assert.equal(unitFor("uNormalMapUV"), 3);
	assert.equal(unitFor("uMetallicRoughnessMapUV"), 2);
	assert.equal(unitFor("uEmissiveMapUV"), 3);
	assert.equal(unitFor("uOcclusionMapUV"), 2);
	assert.equal(unitFor("uIridescenceMapUV"), 3);
	assert.equal(unitFor("uIridescenceThicknessMapUV"), 2);
	assert.equal(unitFor("uAnisotropyMapUV"), 3);
	assert.equal(unitFor("uHasNormalMap"), 1);
	assert.equal(unitFor("uHasMetallicRoughnessMap"), 1);
	assert.equal(unitFor("uHasEmissiveMap"), 1);
	assert.equal(unitFor("uHasOcclusionMap"), 1);
	assert.equal(unitFor("uHasIridescenceMap"), 1);
	assert.equal(unitFor("uHasIridescenceThicknessMap"), 1);
	assert.equal(unitFor("uHasAnisotropyMap"), 1);
	assert.equal(unitFor("uBaseMapIsLinear"), 0);
	assert.equal(unitFor("uEmissiveMapIsLinear"), 0);
	assert.ok(
		gl.calls.uniform1f.some(
			(entry) =>
				entry.location === "uNormalScale" &&
				Math.abs(entry.value - 0.35) < 1e-6
		)
	);
	assert.ok(
		gl.calls.uniform1f.some(
			(entry) =>
				entry.location === "uOcclusionStrength" &&
				Math.abs(entry.value - 0.4) < 1e-6
		)
	);
	assert.ok(
		gl.calls.uniform4fv.some(
			(entry) =>
				entry.location === "uAnisotropy" &&
				Math.abs(entry.values[0] - 0.6) < 1e-6
		)
	);

	const transformAFor = (name) =>
		gl.calls.uniform4f.find((entry) => entry.location === name);
	const transformBFor = (name) =>
		gl.calls.uniform2f.find((entry) => entry.location === name);
	const assertUVTransform = (nameA, nameB, map) => {
		const transformA = transformAFor(nameA);
		const transformB = transformBFor(nameB);
		assert.ok(transformA);
		assert.ok(transformB);
		assert.ok(Math.abs(transformA.x - map.repeat.x) < 1e-6);
		assert.ok(Math.abs(transformA.y - map.repeat.y) < 1e-6);
		assert.ok(Math.abs(transformA.z - map.offset.x) < 1e-6);
		assert.ok(Math.abs(transformA.w - map.offset.y) < 1e-6);
		assert.ok(Math.abs(transformB.x - Math.cos(map.rotation)) < 1e-6);
		assert.ok(Math.abs(transformB.y - Math.sin(map.rotation)) < 1e-6);
	};
	assertUVTransform("uBaseMapTransformA", "uBaseMapTransformB", baseMap);
	assertUVTransform(
		"uMetallicRoughnessMapTransformA",
		"uMetallicRoughnessMapTransformB",
		metallicRoughnessMap
	);
	assertUVTransform("uNormalMapTransformA", "uNormalMapTransformB", normalMap);
	assertUVTransform("uEmissiveMapTransformA", "uEmissiveMapTransformB", emissiveMap);
	assertUVTransform(
		"uOcclusionMapTransformA",
		"uOcclusionMapTransformB",
		occlusionMap
	);
	assertUVTransform(
		"uIridescenceMapTransformA",
		"uIridescenceMapTransformB",
		iridescenceMap
	);
	assertUVTransform(
		"uIridescenceThicknessMapTransformA",
		"uIridescenceThicknessMapTransformB",
		iridescenceThicknessMap
	);
	assertUVTransform(
		"uAnisotropyMapTransformA",
		"uAnisotropyMapTransformB",
		anisotropyMap
	);
}

function testDrawWebGLPacketBindsAnisotropyMapWhenSharedSlotIsFree() {
	const gl = createScenePassCaptureGL();
	const material = new PBRMaterial();
	const anisotropyMap = { id: "anisotropy-map", linear: true };
	material.anisotropyStrength = 0.6;
	material.anisotropyMap = anisotropyMap;
	material.anisotropyMapUV = 3;
	const sceneProgram = {
		program: {},
		samplerLayout: {
			units: { uIridescenceThicknessMap: 0, uAnisotropyMap: 1 },
			activeSamplerNames: ["uIridescenceThicknessMap", "uAnisotropyMap"],
			required: 2,
			available: 16,
		},
		uniforms: {
			model: null,
			normalMatrix: null,
			prevModel: null,
			shadingModel: null,
			baseColor: null,
			emissive: null,
			pbr: null,
			transmissionVolume: null,
			iridescence: null,
			attenuationColor: null,
			anisotropy: null,
			phong: null,
			alpha: null,
			iridescenceThicknessMap: "uIridescenceThicknessMap",
			hasIridescenceThicknessMap: "uHasIridescenceThicknessMap",
			hasAnisotropyMap: "uHasAnisotropyMap",
			anisotropyMapUV: "uAnisotropyMapUV",
			anisotropyMapTransformA: null,
			anisotropyMapTransformB: null,
			doubleSided: null,
			customSamplers: {},
		},
	};
	const textureTable = new Map([
		[anisotropyMap, { texture: { id: "anisotropy" }, isLinear: true }],
	]);
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: 4,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return textureTable.get(texture) ?? { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const packet = {
		id: "packet-pbr-anisotropy",
		meshInstance: { id: "mesh-0", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	const anisotropyTextureUnitIndex = gl.calls.activeTextures.findLastIndex(
		(unit) => unit === gl.TEXTURE0 + 1
	);
	assert.notEqual(anisotropyTextureUnitIndex, -1);
	assert.equal(
		gl.calls.boundTextures[anisotropyTextureUnitIndex].texture.id,
		"anisotropy"
	);
	const unitFor = (name) =>
		gl.calls.uniform1i.find((entry) => entry.location === name)?.value;
	assert.equal(unitFor("uIridescenceThicknessMap"), 0);
	assert.equal(unitFor("uHasIridescenceThicknessMap"), 0);
	assert.equal(unitFor("uHasAnisotropyMap"), 1);
	assert.equal(unitFor("uAnisotropyMapUV"), 3);
}

function testDrawWebGLPacketPreservesInactiveGlobalSamplerUnits() {
	const gl = createScenePassCaptureGL();
	const shadowAtlas = { id: "shadow-atlas" };
	gl.activeTexture(gl.TEXTURE0);
	gl.bindTexture(gl.TEXTURE_2D, shadowAtlas);
	const sceneProgram = {
		program: {},
		samplerLayout: {
			units: {
				uShadowAtlas: 0,
				uParticleShadowVolumeAtlas: 1,
			},
			activeSamplerNames: [
				"uShadowAtlas",
				"uParticleShadowVolumeAtlas",
			],
			required: 2,
			available: 16,
		},
		uniforms: {
			pbrExtensionUniforms: {},
			customSamplers: {},
		},
	};
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture() {
				return { texture: { id: "material-fallback" }, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const packet = {
		id: "packet-shadow-only-samplers",
		meshInstance: { id: "mesh-shadow-only", skeleton: null },
		primitive: { receiveShadows: true },
		material: new Material(),
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});

	const lastUnitZeroBinding = gl.calls.activeTextures.findLastIndex(
		(unit) => unit === gl.TEXTURE0,
	);
	assert.equal(lastUnitZeroBinding, 0);
	assert.equal(
		gl.calls.boundTextures[lastUnitZeroBinding].texture,
		shadowAtlas,
	);
}

await runWebGLBackendFile([
	testGeometryRegistryRejectsOutOfRangeIndices,
	testGeometryRegistryRetriesAfterUploadAllocationFailure,
	testGeometryRegistryUploadsUV1Attribute,
	testGeometryRegistryUploadsSkinAndMorphResources,
	testDrawWebGLPacketBindsPBRTexturesAndUVSets,
	testDrawWebGLPacketBindsAnisotropyMapWhenSharedSlotIsFree,
	testDrawWebGLPacketPreservesInactiveGlobalSamplerUnits,
], "WebGL geometry and draw tests");
