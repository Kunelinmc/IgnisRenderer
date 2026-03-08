import assert from 'node:assert/strict'
import { GeometryBuilder } from '../src/models/GeometryBuilder.ts'
import { Vector3 } from '../src/maths/Vector3.ts'
import { MeshAsset } from '../src/meshes/MeshAsset.ts'

function assertWindingMatchesPrimitiveNormals(name, mesh) {
	let reversed = 0

	for (const primitive of mesh.primitives) {
		const triangleCount = (primitive.geometry.indices.length / 3) | 0
		for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
			const vertices = GeometryBuilder.createVerticesForTriangle(
				primitive,
				triangleIndex
			)
			const geometricNormal = Vector3.calculateNormal(vertices)
			const averagedNormal = averageVertexNormal(vertices)
			const alignment = Vector3.dot(geometricNormal, averagedNormal)
			if (alignment < 0) reversed++
		}
	}

	assert.equal(
		reversed,
		0,
		`${name} has ${reversed} reversed triangle(s): winding does not match primitive normals`
	)
}

function averageVertexNormal(vertices) {
	let x = 0
	let y = 0
	let z = 0

	for (const vertex of vertices) {
		x += vertex.normal?.x ?? 0
		y += vertex.normal?.y ?? 0
		z += vertex.normal?.z ?? 0
	}

	const length = Math.hypot(x, y, z) || 1
	return { x: x / length, y: y / length, z: z / length }
}

function createBoxMesh() {
	const w2 = 1
	const h2 = 1
	const d2 = 1

	const vertices = [
		{ x: -w2, y: -h2, z: -d2 },
		{ x: w2, y: -h2, z: -d2 },
		{ x: w2, y: -h2, z: d2 },
		{ x: -w2, y: -h2, z: d2 },
		{ x: -w2, y: h2, z: -d2 },
		{ x: w2, y: h2, z: -d2 },
		{ x: w2, y: h2, z: d2 },
		{ x: -w2, y: h2, z: d2 },
	]

	const faceSpecs = [
		{ indices: [0, 1, 2, 3], normal: { x: 0, y: -1, z: 0 } },
		{ indices: [4, 7, 6, 5], normal: { x: 0, y: 1, z: 0 } },
		{ indices: [0, 4, 5, 1], normal: { x: 0, y: 0, z: -1 } },
		{ indices: [3, 2, 6, 7], normal: { x: 0, y: 0, z: 1 } },
		{ indices: [0, 3, 7, 4], normal: { x: -1, y: 0, z: 0 } },
		{ indices: [1, 5, 6, 2], normal: { x: 1, y: 0, z: 0 } },
	]

	return MeshAsset.fromFaces(
		faceSpecs.map((face) => ({
			vertices: face.indices.map((index) => ({
				...vertices[index],
				normal: { ...face.normal },
			})),
			normal: face.normal,
		}))
	)
}

function run() {
	assertWindingMatchesPrimitiveNormals('BoxMeshAsset', createBoxMesh())
	console.log('Mesh winding tests passed.')
}

run()
