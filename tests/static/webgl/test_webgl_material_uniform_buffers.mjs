import assert from "node:assert/strict";

import { BasicMaterial } from "../../../src/materials/BasicMaterial.ts";
import { Material, ShadingModel } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";
import { UnlitMaterial } from "../../../src/materials/UnlitMaterial.ts";
import {
	packWebGLMaterialCommonState,
	packWebGLPBRMaterialState,
	WebGLMaterialBufferCache,
} from "../../../src/backends/webgl/WebGLMaterialBufferCache.ts";
import { WebGLMaterialSnapshotCache } from "../../../src/backends/webgl/WebGLMaterialSnapshotCache.ts";
import {
	resolveWebGLMaterialState,
} from "../../../src/backends/webgl/WebGLMaterialState.ts";
import {
	configureWebGLSceneMaterialBlocks,
	validateWebGLMaterialUniformBufferCapabilities,
} from "../../../src/backends/webgl/WebGLMaterialUniformBlocks.ts";
import {
	createWebGLSceneDrawState,
	drawWebGLPacket,
} from "../../../src/backends/webgl/WebGLScenePass.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";
import {
	createTestBuiltinSceneVariant,
	prepareTestBuiltinSceneVariant,
} from "../../helpers/webgl-backend.mjs";

function testMaterialFamiliesAreSeparated() {
	assert.equal(resolveWebGLMaterialState(new PBRMaterial()).shadingFamily, "pbr");
	assert.equal(resolveWebGLMaterialState(new PhongMaterial()).shadingFamily, "phong");
	assert.equal(resolveWebGLMaterialState(new BasicMaterial()).shadingFamily, "flat");
	const gouraud = new PhongMaterial();
	gouraud.shading = ShadingModel.Gouraud;
	assert.equal(resolveWebGLMaterialState(gouraud).shadingFamily, "phong");
	const unlit = resolveWebGLMaterialState(new UnlitMaterial());
	assert.equal(unlit.shadingFamily, "unlit");
	assert.equal(unlit.lighting, null);
}

function testStd140PackingUsesOnlyActiveTransformSlots() {
	const normalMap = {
		repeat: { x: 2, y: 3 },
		offset: { x: 0.25, y: 0.5 },
		rotation: Math.PI / 2,
	};
	const material = new PBRMaterial({ map: {}, normalMap });
	const state = resolveWebGLMaterialState(material);
	assert.equal(state.shadingFamily, "pbr");
	const variant = createTestBuiltinSceneVariant({
		material: { model: "pbr", baseMap: true, normalMap: true },
	}).material;
	const common = packWebGLMaterialCommonState(state.common, variant);
	const pbr = packWebGLPBRMaterialState(state.lighting, variant);
	assert.equal(common.byteLength, 6 * 16);
	assert.equal(pbr.byteLength, 11 * 16);
	assert.deepEqual(Array.from(pbr.slice(-8, -4)), [2, 3, 0.25, 0.5]);
	assert.ok(Math.abs(pbr.at(-4)) < 1e-6);
	assert.ok(Math.abs(pbr.at(-3) - 1) < 1e-6);
	assert.equal(pbr.at(-2), 0);

	const withoutNormal = {
		...variant,
		normalMap: false,
	};
	assert.equal(
		packWebGLPBRMaterialState(state.lighting, withoutNormal).byteLength,
		9 * 16,
	);
}

function createBufferCaptureGL(failAt = -1) {
	let nextBuffer = 0;
	const calls = { data: [], subData: [], deleted: [], blockBindings: [] };
	return {
		UNIFORM_BUFFER: 1,
		DYNAMIC_DRAW: 2,
		INVALID_INDEX: 0xffffffff,
		calls,
		createBuffer() {
			nextBuffer++;
			return nextBuffer === failAt ? null : { id: nextBuffer };
		},
		bindBuffer() {},
		bufferData(_target, data) {
			calls.data.push(data.byteLength);
		},
		bufferSubData(_target, _offset, data) {
			calls.subData.push(data.byteLength);
		},
		deleteBuffer(buffer) {
			calls.deleted.push(buffer?.id);
		},
		getUniformBlockIndex(_program, name) {
			return name === "IgnisMaterialCommon" ? 0
				: name === "IgnisPBRMaterial" ? 1
				: 0xffffffff;
		},
		uniformBlockBinding(_program, index, binding) {
			calls.blockBindings.push([index, binding]);
		},
	};
}

function testRevisionCacheAndUniformBuffersAreReused() {
	const gl = createBufferCaptureGL();
	const snapshots = new WebGLMaterialSnapshotCache();
	const buffers = new WebGLMaterialBufferCache(gl);
	const material = new PBRMaterial({ roughness: 0.4 });
	const variant = createTestBuiltinSceneVariant({
		material: { model: "pbr" },
	}).material;
	snapshots.beginFrame();
	const first = snapshots.resolve(material);
	assert.strictEqual(first, snapshots.resolve(material));
	const firstBuffers = buffers.resolve(material, first, variant);
	assert.strictEqual(firstBuffers, buffers.resolve(material, first, variant));
	assert.equal(gl.calls.data.length, 2);
	assert.equal(gl.calls.subData.length, 0);

	material.roughness = 0.8;
	snapshots.beginFrame();
	const changed = snapshots.resolve(material);
	assert.notStrictEqual(changed, first);
	buffers.resolve(material, changed, variant);
	assert.equal(gl.calls.data.length, 2);
	assert.equal(gl.calls.subData.length, 2);
	buffers.destroy();
	assert.equal(gl.calls.deleted.length, 2);
}

function testAllocationFailureReleasesPartialResources() {
	const gl = createBufferCaptureGL(2);
	const snapshots = new WebGLMaterialSnapshotCache();
	const buffers = new WebGLMaterialBufferCache(gl);
	const material = new PBRMaterial();
	const variant = createTestBuiltinSceneVariant({ material: { model: "pbr" } }).material;
	assert.throws(
		() => buffers.resolve(material, snapshots.resolve(material), variant),
		(error) => error?.code === "material-uniform-buffer-unavailable",
	);
	assert.deepEqual(gl.calls.deleted, [1]);
}

function testBufferCacheEvictsOldestMaterial() {
	const gl = createBufferCaptureGL();
	const snapshots = new WebGLMaterialSnapshotCache();
	const buffers = new WebGLMaterialBufferCache(gl, 1);
	const variant = createTestBuiltinSceneVariant({ material: { model: "pbr" } }).material;
	const first = new PBRMaterial();
	const second = new PBRMaterial();
	buffers.resolve(first, snapshots.resolve(first), variant);
	buffers.resolve(second, snapshots.resolve(second), variant);
	assert.deepEqual(gl.calls.deleted, [1, 2]);
	buffers.destroy();
	assert.deepEqual(gl.calls.deleted, [1, 2, 3, 4]);
}

function testProgramBlocksUseStableBindingPoints() {
	const gl = createBufferCaptureGL();
	const variant = createTestBuiltinSceneVariant({ material: { model: "pbr" } }).material;
	const binding = configureWebGLSceneMaterialBlocks(gl, {}, variant);
	assert.equal(binding.mode, "ubo");
	assert.deepEqual(gl.calls.blockBindings, [[0, 0], [1, 1]]);
	assert.throws(
		() => configureWebGLSceneMaterialBlocks(
			{ ...gl, getUniformBlockIndex: () => gl.INVALID_INDEX },
			{},
			variant,
		),
		(error) => error?.code === "material-uniform-buffer-unavailable",
	);
}

function testUniformBufferCapabilityFailureIsExplicit() {
	const gl = {
		MAX_UNIFORM_BUFFER_BINDINGS: 1,
		MAX_FRAGMENT_UNIFORM_BLOCKS: 2,
		MAX_UNIFORM_BLOCK_SIZE: 3,
		getParameter(parameter) {
			if (parameter === 1) return 1;
			if (parameter === 2) return 12;
			return 16_384;
		},
	};
	assert.throws(
		() => validateWebGLMaterialUniformBufferCapabilities(gl),
		(error) => error?.code === "material-uniform-buffer-unavailable",
	);
}

function testDrawPathBindsCachedUniformBuffersWithoutReuploading() {
	const gl = createBufferCaptureGL();
	gl.TEXTURE0 = 10;
	gl.TEXTURE_2D = 11;
	gl.TRIANGLES = 4;
	gl.CULL_FACE = 12;
	gl.CCW = 13;
	gl.BACK = 14;
	gl.LESS = 15;
	gl.LEQUAL = 16;
	gl.bindBufferBaseCalls = [];
	gl.bindBufferBase = (_target, binding, buffer) => {
		gl.bindBufferBaseCalls.push([binding, buffer.id]);
	};
	gl.activeTexture = () => {};
	gl.bindTexture = () => {};
	gl.bindVertexArray = () => {};
	gl.uniformMatrix4fv = () => {};
	gl.uniformMatrix3fv = () => {};
	gl.uniform1i = () => {};
	gl.uniform1f = () => {};
	gl.uniform2f = () => {};
	gl.enable = () => {};
	gl.disable = () => {};
	gl.frontFace = () => {};
	gl.cullFace = () => {};
	gl.depthFunc = () => {};
	gl.depthMask = () => {};
	gl.drawElementsCalls = 0;
	gl.drawElements = () => gl.drawElementsCalls++;

	const material = new PBRMaterial();
	const materialVariant = createTestBuiltinSceneVariant({
		material: { model: "pbr" },
	}).material;
	const snapshots = new WebGLMaterialSnapshotCache();
	const buffers = new WebGLMaterialBufferCache(gl);
	const deps = {
		gl,
		targets: {
			_materialGBufferEnabled: false,
			_transmissionBackgroundTexture: null,
			_transmissionDepthTexture: null,
		},
		drawState: createWebGLSceneDrawState(),
		geometry: {
			getGeometry: () => ({ vao: {}, topology: 4, indexCount: 3, indexType: 5123 }),
		},
		textures: {
			getBaseColorTexture: () => ({ texture: null, isLinear: true }),
		},
		materialSnapshots: snapshots,
		materialBuffers: buffers,
		animationPayloads: null,
		modelMatrixCache: new Map(),
		modelMatrixKeysThisFrame: new Set(),
		getWidth: () => 64,
		getHeight: () => 64,
		bindAnimationPayload: () => true,
		getShadowSamplingState: () => ({ enabled: false, transmittanceAvailable: false }),
	};
	const sceneProgram = {
		program: {},
		materialBinding: { mode: "ubo", family: "pbr", materialVariant },
		samplerLayout: { units: {}, activeSamplerNames: [], required: 0, available: 16 },
		uniforms: { pbrExtensionUniforms: {} },
		targetMode: "single",
		colorOutputCount: 1,
	};
	const packet = createTestDrawPacket({ material });
	const context = { features: { enableShadows: false } };
	snapshots.beginFrame();
	drawWebGLPacket(deps, sceneProgram, packet, false, context);
	drawWebGLPacket(deps, sceneProgram, packet, false, context);
	assert.equal(gl.calls.data.length, 2);
	assert.equal(gl.calls.subData.length, 0);
	assert.deepEqual(gl.bindBufferBaseCalls, [[0, 1], [1, 2]]);
	assert.equal(gl.drawElementsCalls, 2);
	buffers.destroy();
}

async function testExactAndCompatibilityShaderABIsRemainDistinct() {
	const exact = createTestBuiltinSceneVariant({ material: { model: "pbr" } });
	await prepareTestBuiltinSceneVariant(exact);
	const exactSource = ShaderSource.get("webgl.scene", {
		specialization: exact,
	}).stages.fragment.code;
	assert.ok(exactSource.includes("uniform IgnisMaterialCommon"));
	assert.ok(exactSource.includes("uniform IgnisPBRMaterial"));

	const full = createTestBuiltinSceneVariant({ material: { model: "full" } });
	await prepareTestBuiltinSceneVariant(full);
	const fullSource = ShaderSource.get("webgl.scene", {
		specialization: full,
	}).stages.fragment.code;
	assert.ok(!fullSource.includes("uniform IgnisMaterialCommon"));
	assert.ok(fullSource.includes("uniform vec4 uBaseColor;"));
}

testMaterialFamiliesAreSeparated();
testStd140PackingUsesOnlyActiveTransformSlots();
testRevisionCacheAndUniformBuffersAreReused();
testAllocationFailureReleasesPartialResources();
testBufferCacheEvictsOldestMaterial();
testProgramBlocksUseStableBindingPoints();
testUniformBufferCapabilityFailureIsExplicit();
testDrawPathBindsCachedUniformBuffersWithoutReuploading();
await testExactAndCompatibilityShaderABIsRemainDistinct();
console.log("WebGL material uniform-buffer tests passed");
