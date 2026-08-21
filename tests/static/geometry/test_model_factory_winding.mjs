import assert from "node:assert/strict";
import { GeometryBuilder } from "../../../src/meshes/GeometryBuilder.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { Vector3 } from "../../../src/maths/Vector3.ts";

function assertWindingMatchesPrimitiveNormals(name, mesh) {
	let reversed = 0;

	for (const primitive of mesh.primitives) {
		const triangleCount = (primitive.geometry.indices.length / 3) | 0;
		for (
			let triangleIndex = 0;
			triangleIndex < triangleCount;
			triangleIndex++
		) {
			const vertices = GeometryBuilder.createVerticesForTriangle(
				primitive,
				triangleIndex
			);
			const geometricNormal = Vector3.calculateNormal(vertices);
			const averagedNormal = averageVertexNormal(vertices);
			const alignment = Vector3.dot(geometricNormal, averagedNormal);
			if (alignment < 0) reversed++;
		}
	}

	assert.equal(
		reversed,
		0,
		`${name} has ${reversed} reversed triangle(s): winding does not match primitive normals`
	);
}

function averageVertexNormal(vertices) {
	let x = 0;
	let y = 0;
	let z = 0;

	for (const vertex of vertices) {
		x += vertex.normal?.x ?? 0;
		y += vertex.normal?.y ?? 0;
		z += vertex.normal?.z ?? 0;
	}

	const length = Math.hypot(x, y, z) || 1;
	return { x: x / length, y: y / length, z: z / length };
}

function createBoxMesh() {
	return MeshFactory.createBox({ x: 0, y: 0, z: 0 }, 2, 2, 2).mesh;
}

function createPlaneMesh() {
	return MeshFactory.createPlane({ x: 0, y: 0, z: 0 }, 2, 2).mesh;
}

function assertPlaneFacesUpward(mesh) {
	for (const primitive of mesh.primitives) {
		const triangleCount = (primitive.geometry.indices.length / 3) | 0;
		for (
			let triangleIndex = 0;
			triangleIndex < triangleCount;
			triangleIndex++
		) {
			const vertices = GeometryBuilder.createVerticesForTriangle(
				primitive,
				triangleIndex
			);
			const geometricNormal = Vector3.calculateNormal(vertices);
			assert.ok(
				geometricNormal.y > 0,
				`PlaneMeshAsset triangle ${triangleIndex} does not face positive Y`
			);
		}
	}
}

function run() {
	assertWindingMatchesPrimitiveNormals("BoxMeshAsset", createBoxMesh());
	const planeMesh = createPlaneMesh();
	assertWindingMatchesPrimitiveNormals("PlaneMeshAsset", planeMesh);
	assertPlaneFacesUpward(planeMesh);
	console.log("Mesh winding tests passed.");
}

run();
