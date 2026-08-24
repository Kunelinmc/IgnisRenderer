import assert from "node:assert/strict";

import {
	WEBGL_STATIC_DEFORMATION_PROFILE,
	resolveWebGLGeometryDeformationProfile,
	resolveWebGLPacketDeformationProfile,
} from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createPrimitive(geometry, geometryVersion = 0) {
	return { geometry, geometryVersion };
}

function createPacket(primitive) {
	return createTestDrawPacket({ primitive });
}

function morphTarget(positions, normals) {
	return { positions: positions ?? null, normals: normals ?? null };
}

function testSkinProfileDerivation() {
	const staticProfile = resolveWebGLPacketDeformationProfile(
		createPacket(createPrimitive({
			positions: new Float32Array(3),
			indices: new Uint32Array(),
		})),
	);
	assert.equal(staticProfile.skinProfile, "static");
	assert.equal(staticProfile.morphSemanticMask, 0);

	const skin4 = resolveWebGLPacketDeformationProfile(
		createPacket(createPrimitive({
			joints0: new Uint16Array(4),
			weights0: new Float32Array(4),
			indices: new Uint32Array(),
		})),
	);
	assert.equal(skin4.skinProfile, "skin4");

	const skin8 = resolveWebGLPacketDeformationProfile(
		createPacket(createPrimitive({
			joints0: new Uint16Array(4),
			weights0: new Float32Array(4),
			joints1: new Uint16Array(4),
			indices: new Uint32Array(),
		})),
	);
	assert.equal(skin8.skinProfile, "skin8");
}

function testMorphSemanticMaskBitsAndCap() {
	const profile = resolveWebGLGeometryDeformationProfile({
		morphTargets: [
			morphTarget(new Float32Array(3)),
			morphTarget(null, new Float32Array(3)),
			morphTarget(new Float32Array(3), new Float32Array(3)),
		],
	});
	assert.equal(profile.morphSemanticMask, 1 | 2);

	const capped = resolveWebGLGeometryDeformationProfile({
		morphTargets: Array.from({ length: 9 }, (_, index) =>
			morphTarget(index < 8 ? null : new Float32Array(3))
		),
	});
	assert.equal(capped.morphSemanticMask, 0);
}

function testMissingGeometryFallsBackToStatic() {
	assert.deepEqual(
		resolveWebGLPacketDeformationProfile(createTestDrawPacket()),
		WEBGL_STATIC_DEFORMATION_PROFILE,
	);
	assert.deepEqual(
		resolveWebGLPacketDeformationProfile(createTestDrawPacket()),
		WEBGL_STATIC_DEFORMATION_PROFILE,
	);
}

function testProfileIsMemoizedPerPrimitiveUntilVersionChanges() {
	const primitive = createPrimitive({
		morphTargets: [morphTarget(new Float32Array(3))],
		indices: new Uint32Array(),
	});
	const packetA = createPacket(primitive);
	const packetB = createPacket(primitive);

	const first = resolveWebGLPacketDeformationProfile(packetA);
	assert.equal(first.skinProfile, "static");
	assert.equal(first.morphSemanticMask, 1);

	assert.equal(resolveWebGLPacketDeformationProfile(packetB), first);

	primitive.geometryVersion = 7;
	primitive.geometry.joints0 = new Uint16Array(4);
	primitive.geometry.weights0 = new Float32Array(4);

	const updated = resolveWebGLPacketDeformationProfile(createPacket(primitive));
	assert.notEqual(updated, first);
	assert.equal(updated.skinProfile, "skin4");
	assert.equal(updated.morphSemanticMask, 1);
}

function run() {
	testSkinProfileDerivation();
	testMorphSemanticMaskBitsAndCap();
	testMissingGeometryFallsBackToStatic();
	testProfileIsMemoizedPerPrimitiveUntilVersionChanges();
	console.log("WebGL deformation profile tests passed");
}

run();
