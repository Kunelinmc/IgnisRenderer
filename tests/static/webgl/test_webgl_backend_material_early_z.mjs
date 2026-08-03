import assert from "node:assert/strict";import { Material } from "../../../src/materials/Material.ts";import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";import { Matrix4 } from "../../../src/maths/Matrix4.ts";import { bindWebGLShaderMaterialUniforms, bindWebGLShaderMaterialTextures, drawWebGLPacket, renderWebGLEarlyZPrepass, renderWebGLPackets } from "../../../src/backends/webgl/WebGLScenePass.ts";import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";import { createScenePassCaptureGL, createScenePassContext, createEarlyZScenePassHost, createEarlyZPacket, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

function testDrawWebGLPacketAppliesMaterialDepthWriteState() {
	const gl = createScenePassCaptureGL();
	const sceneProgram = {
		program: {},
		uniforms: {},
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
			getBaseColorTexture() {
				return { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const material = new Material({
		depthWrite: false,
	});
	const packet = {
		id: "packet-depth-read",
		meshInstance: { id: "mesh-depth-read", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	assert.deepEqual(gl.calls.depthMask, [false]);
	assert.deepEqual(gl.calls.depthFunc, [gl.LESS]);

	material.depthWrite = true;
	drawWebGLPacket(host, sceneProgram, packet, false, {});
	assert.deepEqual(gl.calls.depthMask, [false, true]);
	assert.deepEqual(gl.calls.depthFunc, [gl.LESS, gl.LESS]);
}

function testEarlyZPrepassUsesDepthOnlyStateAndDrivesColorLEQUAL() {
	const gl = createScenePassCaptureGL();
	const material = new Material();
	const packet = createEarlyZPacket("early-z", material);
	const context = createScenePassContext();
	const host = createEarlyZScenePassHost(gl);

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [packet]);

	assert.equal(prepassedIds.has(packet.id), true);
	assert.deepEqual(gl.calls.drawBuffers[0], [gl.NONE]);
	assert.deepEqual(gl.calls.colorMask[0], [false, false, false, false]);
	assert.deepEqual(
		gl.calls.colorMask[gl.calls.colorMask.length - 1],
		[true, true, true, true]
	);
	assert.deepEqual(
		gl.calls.drawBuffers[gl.calls.drawBuffers.length - 1],
		[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]
	);
	assert.ok(gl.calls.disable.includes(gl.BLEND));
	assert.ok(gl.calls.depthMask.includes(true));

	renderWebGLPackets(host, context, [packet], false, {
		earlyZPacketIds: prepassedIds,
	});

	assert.ok(gl.calls.depthFunc.includes(gl.LEQUAL));
	assert.equal(gl.calls.depthMask.includes(false), true);
}

function testEarlyZPrepassSkipsDepthWriteDisabledPackets() {
	const gl = createScenePassCaptureGL();
	const material = new Material({
		depthWrite: false,
	});
	const packet = createEarlyZPacket("depth-read", material);
	const context = createScenePassContext();
	const host = createEarlyZScenePassHost(gl);

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [packet]);
	assert.equal(prepassedIds.size, 0);
	assert.equal(gl.calls.drawElements.length, 0);

	renderWebGLPackets(host, context, [packet], false, {
		earlyZPacketIds: prepassedIds,
	});
	assert.equal(gl.calls.depthFunc.includes(gl.LEQUAL), false);
	assert.equal(gl.calls.depthFunc.includes(gl.LESS), true);
	assert.equal(gl.calls.depthMask.includes(false), true);
}

function testBuiltInMaskDepthPrepassShaderContract() {
	const fragment = ShaderSource.get("webgl.part.sceneDepthPrepassFragment.raw");
	assert.ok(fragment.includes("uBaseColor.a"));
	assert.ok(fragment.includes("texture(uBaseMap"));
	assert.ok(fragment.includes("uAlpha.x"));
	assert.ok(fragment.includes("discard"));
}

function testEarlyZPrepassUsesDirtyRectPacketSelection() {
	const gl = createScenePassCaptureGL();
	const packetA = createEarlyZPacket("a");
	const packetB = createEarlyZPacket("b");
	const resolvedRects = [];
	const context = createScenePassContext({
		incremental: {
			enabled: true,
			forceFullFrame: false,
			dirtyRects: [
				{ x: 0, y: 0, width: 16, height: 16 },
				{ x: 32, y: 32, width: 16, height: 16 },
			],
		},
	});
	const host = createEarlyZScenePassHost(gl, {
		resolvePacketsForRect(_context, _packets, rect) {
			resolvedRects.push(rect);
			return rect.x === 0 ? [packetA] : [];
		},
	});

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [
		packetA,
		packetB,
	]);

	assert.deepEqual(resolvedRects, context.incremental.dirtyRects);
	assert.equal(prepassedIds.has(packetA.id), true);
	assert.equal(prepassedIds.has(packetB.id), false);
	assert.equal(gl.calls.scissor.length, 1);
}

function testShaderMaterialCustomUniformBinding() {
	const gl = createScenePassCaptureGL();
	const material = new ShaderMaterial({
		uniformBindings: [
			{ name: "time", type: "f32", value: 1.5, webglUniform: "uTime" },
			{ name: "mode", type: "i32", value: 2, webglUniform: "uMode" },
			{ name: "flags", type: "u32", value: 3, webglUniform: "uFlags" },
			{ name: "uvScale", type: "vec2f", value: [4, 5], webglUniform: "uUVScale" },
			{ name: "normal", type: "vec3f", value: [6, 7, 8], webglUniform: "uNormal" },
			{ name: "tint", type: "vec4f", value: [1, 0.5, 0.25, 1], webglUniform: "uTint" },
			{ name: "offset", type: "vec2i", value: [9, 10], webglUniform: "uOffset" },
			{ name: "indices", type: "vec3i", value: [11, 12, 13], webglUniform: "uIndices" },
			{ name: "mask", type: "vec4i", value: [14, 15, 16, 17], webglUniform: "uMask" },
			{ name: "uoffset", type: "vec2u", value: [18, 19], webglUniform: "uUOffset" },
			{ name: "uindices", type: "vec3u", value: [20, 21, 22], webglUniform: "uUIndices" },
			{ name: "umask", type: "vec4u", value: [23, 24, 25, 26], webglUniform: "uUMask" },
			{
				name: "matrix",
				type: "mat4x4f",
				value: [
					[1, 2, 3, 4],
					[5, 6, 7, 8],
					[9, 10, 11, 12],
					[13, 14, 15, 16],
				],
				webglUniform: "uMatrix",
			},
			{ name: "unused", type: "f32", value: 99, webglUniform: "uUnused" },
		],
	});
	const customUniforms = {};
	for (const binding of material.getUniformBindings()) {
		customUniforms[binding.webglUniform] =
			binding.webglUniform === "uUnused" ? null : binding.webglUniform;
	}
	const sceneProgram = { uniforms: { customUniforms } };
	const host = { _gl: gl };

	bindWebGLShaderMaterialUniforms(host, sceneProgram, material);

	assert.deepEqual(gl.calls.uniform1f, [{ location: "uTime", value: 1.5 }]);
	assert.deepEqual(gl.calls.uniform1i, [{ location: "uMode", value: 2 }]);
	assert.deepEqual(gl.calls.uniform1ui, [{ location: "uFlags", value: 3 }]);
	assert.deepEqual(gl.calls.uniform2fv[0], {
		location: "uUVScale",
		values: [4, 5],
	});
	assert.deepEqual(gl.calls.uniform3fv[0], {
		location: "uNormal",
		values: [6, 7, 8],
	});
	assert.deepEqual(gl.calls.uniform4fv[0], {
		location: "uTint",
		values: [1, 0.5, 0.25, 1],
	});
	assert.deepEqual(gl.calls.uniform2iv[0].values, [9, 10]);
	assert.deepEqual(gl.calls.uniform3iv[0].values, [11, 12, 13]);
	assert.deepEqual(gl.calls.uniform4iv[0].values, [14, 15, 16, 17]);
	assert.deepEqual(gl.calls.uniform2uiv[0].values, [18, 19]);
	assert.deepEqual(gl.calls.uniform3uiv[0].values, [20, 21, 22]);
	assert.deepEqual(gl.calls.uniform4uiv[0].values, [23, 24, 25, 26]);
	assert.deepEqual(gl.calls.uniformMatrix4fv[0], {
		location: "uMatrix",
		transpose: false,
		values: [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16],
	});
	assert.equal(gl.calls.uniform1f.length, 1);
}

function testShaderMaterialCustomTextureBinding() {
	const gl = createScenePassCaptureGL();
	const fakeTexture = { id: "test-tex", colorSpace: "Linear" };
	const material = new ShaderMaterial({
		textureBindings: [
			{ name: "customTex", texture: fakeTexture, webglUniform: "uCustomTex" }
		]
	});
	const customSamplers = {
		uCustomTex: "uCustomTex"
	};
	const sceneProgram = { uniforms: { customSamplers } };

	// Case 1: _maxTextureImageUnits is 32 (starts at WEBGL_TEXTURE_UNIT_CUSTOM_START = 17)
	{
		gl.calls.activeTextures = [];
		gl.calls.boundTextures = [];
		gl.calls.uniform1i = [];
		const host = {
			_gl: gl,
			_maxTextureImageUnits: 32,
			_textures: {
				getBaseColorTexture(texture) {
					return { texture };
				}
			}
		};
		bindWebGLShaderMaterialTextures(host, sceneProgram, material);
		assert.equal(gl.calls.activeTextures.length, 2);
		assert.equal(gl.calls.activeTextures[0], gl.TEXTURE0 + 17);
		assert.equal(gl.calls.activeTextures[1], gl.TEXTURE0 + 0);
		assert.deepEqual(gl.calls.boundTextures, [{ target: gl.TEXTURE_2D, texture: fakeTexture }]);
		assert.deepEqual(gl.calls.uniform1i, [{ location: "uCustomTex", value: 17 }]);
	}

	// Case 2: _maxTextureImageUnits is 16 (falls back to start at 8)
	{
		gl.calls.activeTextures = [];
		gl.calls.boundTextures = [];
		gl.calls.uniform1i = [];
		const host = {
			_gl: gl,
			_maxTextureImageUnits: 16,
			_textures: {
				getBaseColorTexture(texture) {
					return { texture };
				}
			}
		};
		bindWebGLShaderMaterialTextures(host, sceneProgram, material);
		assert.equal(gl.calls.activeTextures.length, 2);
		assert.equal(gl.calls.activeTextures[0], gl.TEXTURE0 + 8);
		assert.equal(gl.calls.activeTextures[1], gl.TEXTURE0 + 0);
		assert.deepEqual(gl.calls.boundTextures, [{ target: gl.TEXTURE_2D, texture: fakeTexture }]);
		assert.deepEqual(gl.calls.uniform1i, [{ location: "uCustomTex", value: 8 }]);
	}
}

await runWebGLBackendFile([
	testDrawWebGLPacketAppliesMaterialDepthWriteState,
	testEarlyZPrepassUsesDepthOnlyStateAndDrivesColorLEQUAL,
	testEarlyZPrepassSkipsDepthWriteDisabledPackets,
	testBuiltInMaskDepthPrepassShaderContract,
	testEarlyZPrepassUsesDirtyRectPacketSelection,
	testShaderMaterialCustomUniformBinding,
	testShaderMaterialCustomTextureBinding,
], "WebGL material and Early-Z tests");
