import assert from "node:assert/strict";
import { ModelFactory } from "../src/models/ModelFactory";
import * as Vector3 from "../src/maths/Vector3";

function assertWindingMatchesFaceNormal(name, model) {
	let reversed = 0;

	for (const face of model.faces) {
		const geometricNormal = Vector3.calculateNormal(face.vertices);
		const faceNormal = face.normal || geometricNormal;
		const alignment = Vector3.dot(geometricNormal, faceNormal);
		if (alignment < 0) reversed++;
	}

	assert.equal(
		reversed,
		0,
		`${name} has ${reversed} reversed face(s): winding does not match face normal`
	);
}

function run() {
	assertWindingMatchesFaceNormal(
		"Box",
		ModelFactory.createBox({ x: 0, y: 0, z: 0 }, 2, 2, 2)
	);
	assertWindingMatchesFaceNormal(
		"Sphere",
		ModelFactory.createSphere({ x: 0, y: 0, z: 0 }, 1, 24, 12)
	);
	assertWindingMatchesFaceNormal(
		"Cylinder",
		ModelFactory.createCylinder({ x: 0, y: 0, z: 0 }, 1, 2, 24)
	);
	console.log("ModelFactory winding tests passed.");
}

run();
