import assert from 'node:assert/strict'
import { ModelFactory } from '../src/models/ModelFactory.ts'
import { GeometryBuilder } from '../src/core/geometry/GeometryBuilder.ts'
import { Vector3 } from '../src/maths/Vector3.ts'

function assertWindingMatchesPrimitiveNormals(name, model) {
	let reversed = 0

	for (const primitive of model.primitives) {
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

function run() {
	assertWindingMatchesPrimitiveNormals(
		'Box',
		ModelFactory.createBox({ x: 0, y: 0, z: 0 }, 2, 2, 2)
	)
	assertWindingMatchesPrimitiveNormals(
		'Sphere',
		ModelFactory.createSphere({ x: 0, y: 0, z: 0 }, 1, 24, 12)
	)
	assertWindingMatchesPrimitiveNormals(
		'Cylinder',
		ModelFactory.createCylinder({ x: 0, y: 0, z: 0 }, 1, 2, 24)
	)
	assertWindingMatchesPrimitiveNormals(
		'Torus',
		ModelFactory.createTorus({ x: 0, y: 0, z: 0 }, 2, 0.5, 16, 32)
	)
	assertWindingMatchesPrimitiveNormals(
		'Tube',
		ModelFactory.createTube({ x: 0, y: 0, z: 0 }, 1, 2, 4, 16)
	)
	assertWindingMatchesPrimitiveNormals(
		'Cone',
		ModelFactory.createCone({ x: 0, y: 0, z: 0 }, 1, 2, 16)
	)
	console.log('ModelFactory winding tests passed.')
}

run()
