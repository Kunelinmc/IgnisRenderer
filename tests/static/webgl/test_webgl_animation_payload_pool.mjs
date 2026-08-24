import assert from "node:assert/strict";

import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGLAnimationPayloadPool } from "../../../src/backends/webgl/WebGLAnimationPayloadPool.ts";
import { createWebGLVertexTextureUnitLayout } from "../../../src/backends/webgl/WebGLVertexTextureUnits.ts";
import {
	ANIMATION_JOINT_MATRICES_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
} from "../../../src/simulation/animation/types.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createGL(limits = {}) {
	let nextTexture = 1;
	const uploads = [];
	const deleted = [];
	const maxFragmentUnits = limits.maxFragmentUnits ?? 8;
	const maxVertexUnits = limits.maxVertexUnits ?? 8;
	const maxCombinedUnits = limits.maxCombinedUnits ?? 24;
	return {
		MAX_TEXTURE_IMAGE_UNITS: 1,
		MAX_VERTEX_TEXTURE_IMAGE_UNITS: 2,
		MAX_COMBINED_TEXTURE_IMAGE_UNITS: 3,
		TEXTURE_2D: 4,
		TEXTURE0: 100,
		TEXTURE_MIN_FILTER: 5,
		TEXTURE_MAG_FILTER: 6,
		TEXTURE_WRAP_S: 7,
		TEXTURE_WRAP_T: 8,
		NEAREST: 9,
		CLAMP_TO_EDGE: 10,
		RGBA32F: 11,
		RGBA: 12,
		FLOAT: 13,
		uploads,
		deleted,
		getParameter(parameter) {
			if (parameter === this.MAX_TEXTURE_IMAGE_UNITS) return maxFragmentUnits;
			if (parameter === this.MAX_VERTEX_TEXTURE_IMAGE_UNITS) return maxVertexUnits;
			if (parameter === this.MAX_COMBINED_TEXTURE_IMAGE_UNITS) return maxCombinedUnits;
			return 0;
		},
		createTexture() {
			return { id: nextTexture++ };
		},
		deleteTexture(texture) {
			deleted.push(texture);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D() {},
		texSubImage2D(_target, _level, _x, _y, width, height, _format, _type, data) {
			uploads.push({ width, height, data: new Float32Array(data) });
		},
		activeTexture() {},
		uniform1i() {},
		uniform4i() {},
		vertexAttrib4f() {},
	};
}

function matrixPayload(translationX) {
	const matrix = Matrix4.identity();
	matrix.elements[0][3] = translationX;
	return matrix.toFloat32Array ? matrix.toFloat32Array() : new Float32Array([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		translationX, 0, 0, 1,
	]);
}

function createPacket() {
	const primitive = { id: "primitive", geometry: {}, geometryVersion: 0 };
	return createTestDrawPacket({
		id: "instance:primitive",
		primitive,
		meshInstance: {
			id: "instance",
		},
		deformationRevision: 1,
		deformationMode: "skin-morph",
		jointPayloadKey: "instance",
		morphPayloadKey: "instance:primitive",
	});
}

function createGeometry() {
	return {
		vertexCount: 3,
		skinProfile: "skin4",
		morphTargetCount: 1,
		morphSemanticMask: 1,
		morphPositionTexture: { id: "morph-position" },
		morphNormalTexture: null,
		morphTextureWidth: 4,
	};
}

function createContext(joints, weights) {
	return {
		transient: new Map([
			[ANIMATION_JOINT_MATRICES_KEY, new Map([
				["instance", { skeleton: null, matrices: joints }],
			])],
			[ANIMATION_MORPH_WEIGHTS_KEY, new Map([
				["instance:primitive", {
					packetId: "instance:primitive",
					weights,
					targetCount: weights.length,
				}],
			])],
		]),
	};
}

function run() {
	const staticGL = createGL({ maxVertexUnits: 0, maxCombinedUnits: 8 });
	const staticPool = new WebGLAnimationPayloadPool(
		staticGL,
		createWebGLVertexTextureUnitLayout(staticGL),
		64,
	);
	staticPool.beginFrame({ transient: new Map() });
	assert.equal(
		staticPool.bind(
			{
				animationPayload: null,
				morphPositionDeltas: null,
				morphNormalDeltas: null,
				animationCounts: null,
				animationOffsets: null,
				animationTextureWidths: null,
			},
			createPacket(),
			createGeometry(),
		),
		true,
		"static shadow shaders do not require vertex texture units",
	);
	assert.equal(staticPool.getDebugStats().textureCount, 0);
	staticPool.destroy();

	const gl = createGL();
	const pool = new WebGLAnimationPayloadPool(
		gl,
		createWebGLVertexTextureUnitLayout(gl),
		64,
	);
	const packet = createPacket();
	const geometry = createGeometry();
	const uniforms = {
		animationPayload: {},
		morphPositionDeltas: {},
		morphNormalDeltas: {},
		animationCounts: {},
		animationOffsets: {},
		animationTextureWidths: {},
	};

	const firstJoints = matrixPayload(0);
	pool.beginFrame(createContext(firstJoints, new Float32Array([0.25])));
	assert.equal(pool.bind(uniforms, packet, geometry), true);
	assert.equal(pool.bind(uniforms, packet, geometry), true);
	assert.equal(gl.uploads.length, 1, "one packet uploads once per logical frame");
	assert.deepEqual(
		Array.from(gl.uploads[0].data.slice(0, 16)),
		Array.from(gl.uploads[0].data.slice(16, 32)),
		"first previous joint palette equals current",
	);

	const secondJoints = matrixPayload(2);
	packet.submission.deformation.revision = 2;
	pool.beginFrame(createContext(secondJoints, new Float32Array([0.75])));
	pool.bind(uniforms, packet, geometry);
	assert.equal(gl.uploads.length, 2);
	assert.deepEqual(
		Array.from(gl.uploads[1].data.slice(16, 32)),
		Array.from(firstJoints),
		"changed pose retains the former current palette as previous",
	);

	pool.beginFrame(createContext(secondJoints, new Float32Array([0.75])));
	pool.bind(uniforms, packet, geometry);
	assert.equal(gl.uploads.length, 3, "first unchanged frame settles history");
	assert.deepEqual(
		Array.from(gl.uploads[2].data.slice(0, 16)),
		Array.from(gl.uploads[2].data.slice(16, 32)),
	);

	pool.beginFrame(createContext(secondJoints, new Float32Array([0.75])));
	pool.bind(uniforms, packet, geometry);
	assert.equal(gl.uploads.length, 3, "settled unchanged payload skips upload");

	for (let frame = 0; frame < 61; frame++) {
		pool.beginFrame({ transient: new Map() });
	}
	assert.equal(pool.getDebugStats().entryCount, 0);
	assert.equal(pool.getDebugStats().graceReleases, 1);
	pool.destroy();

	const missingGL = createGL();
	const missingPool = new WebGLAnimationPayloadPool(
		missingGL,
		createWebGLVertexTextureUnitLayout(missingGL),
		64,
	);
	missingPool.beginFrame({ transient: new Map() });
	const missingPacket = createPacket();
	assert.equal(missingPool.bind(uniforms, missingPacket, geometry), false);
	assert.equal(missingPool.bind(uniforms, missingPacket, geometry), false);
	assert.equal(missingGL.uploads.length, 0);
	missingPool.destroy();
	console.log("WebGL animation payload pool tests passed");
}

run();
