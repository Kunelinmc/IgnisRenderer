import assert from "node:assert/strict";

import {
	WEBGPU_PLANAR_REFLECTION_SIDE_EPSILON,
	filterPlanarReflectionCapturePackets,
} from "../../../src/renderers/webgpu/WebGPUPlanarReflectionPass.ts";
import { Plane } from "../../../src/maths/Plane.ts";

function createPacket(id, centerY, radius, mirrorPlane = null) {
	return {
		id,
		material: { mirrorPlane },
		worldBounds: {
			center: { x: 0, y: centerY, z: 0 },
			radius,
		},
	};
}

function testCapturePacketFilterRejectsBackSideSphereOverlap() {
	const mirrorPlane = new Plane({ x: 0, y: 1, z: 0 }, -0.03);
	const planeKey = "0.000000,1.000000,0.000000,-0.030000";
	const samePlane = {
		normal: { x: 0, y: 1, z: 0 },
		constant: -0.03,
	};
	const packets = [
		createPacket("same-plane", 0.03, 8, samePlane),
		createPacket("floor-behind-plane", 0, 20),
		createPacket("near-plane-tolerance", 0.03 - WEBGPU_PLANAR_REFLECTION_SIDE_EPSILON * 0.5, 1),
		createPacket("above-plane", 1, 1),
	];

	const result = filterPlanarReflectionCapturePackets(
		packets,
		mirrorPlane,
		planeKey,
		true
	);
	assert.deepEqual(
		result.map((packet) => packet.id),
		["near-plane-tolerance", "above-plane"]
	);
}

function testCapturePacketFilterHandlesCameraBelowPlane() {
	const mirrorPlane = new Plane({ x: 0, y: 1, z: 0 }, -0.03);
	const planeKey = "0.000000,1.000000,0.000000,-0.030000";
	const packets = [
		createPacket("below-plane", -1, 1),
		createPacket("above-plane", 1, 20),
	];

	const result = filterPlanarReflectionCapturePackets(
		packets,
		mirrorPlane,
		planeKey,
		false
	);
	assert.deepEqual(result.map((packet) => packet.id), ["below-plane"]);
}

testCapturePacketFilterRejectsBackSideSphereOverlap();
testCapturePacketFilterHandlesCameraBelowPlane();

console.log("WebGPU planar reflection pass tests passed");
