import assert from "node:assert/strict";
import { CubeFaceCamera } from "../../../src/cameras/CubeFaceCamera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

const directions = [
	{ x: 1, y: 0, z: 0 },
	{ x: -1, y: 0, z: 0 },
	{ x: 0, y: 1, z: 0 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: 0, z: 1 },
	{ x: 0, y: 0, z: -1 },
];

for (let faceIndex = 0; faceIndex < directions.length; faceIndex++) {
	const position = { x: 3, y: 4, z: 5 };
	const direction = directions[faceIndex];
	const camera = new CubeFaceCamera(position, 50, faceIndex);
	const cameraSpace = Matrix4.transformPoint(camera.viewMatrix, {
		x: position.x + direction.x,
		y: position.y + direction.y,
		z: position.z + direction.z,
	});
	assert.equal(camera.faceIndex, faceIndex);
	assert.equal(camera.fov, 90);
	assert.equal(camera.aspectRatio, 1);
	assert.ok(Math.abs(cameraSpace.x) < 1e-6);
	assert.ok(Math.abs(cameraSpace.y) < 1e-6);
	assert.ok(cameraSpace.z < -0.99);
}

assert.equal(new CubeFaceCamera({ x: 0, y: 0, z: 0 }, 0, 99).faceIndex, 5);
assert.equal(new CubeFaceCamera({ x: 0, y: 0, z: 0 }, 10, Number.NaN).faceIndex, 0);
console.log("Cube face camera tests passed");
