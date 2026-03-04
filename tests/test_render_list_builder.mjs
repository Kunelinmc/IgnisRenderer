import assert from 'node:assert/strict'
import { Camera } from '../src/cameras/Camera.ts'
import { Scene } from '../src/core/Scene.ts'
import { PreparedSceneBuilder } from '../src/core/pipeline/PreparedSceneBuilder.ts'
import { Material } from '../src/materials/Material.ts'
import { SimpleModel } from '../src/models/SimpleModel.ts'

function createTriangleModel(material) {
	return SimpleModel.fromFaces([
		{
			material,
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	])
}

function run() {
	const camera = new Camera()
	camera.position.set(0, 0, 5)
	camera.updateMatrices()

	const opaqueMaterial = new Material({
		name: 'Opaque',
		alphaMode: 'OPAQUE',
	})
	const transparentMaterial = new Material({
		name: 'Transparent',
		alphaMode: 'BLEND',
	})
	const reflectiveMaterial = new Material({
		name: 'Reflective',
		alphaMode: 'OPAQUE',
		reflectivity: 0.7,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	})

	const sharedOpaquePrimitive = createTriangleModel(opaqueMaterial).primitives[0]
	const sharedTransparentPrimitive =
		createTriangleModel(transparentMaterial).primitives[0]
	const reflectiveModel = createTriangleModel(reflectiveMaterial)

	const nearOpaque = new SimpleModel([sharedOpaquePrimitive])
	nearOpaque.transform.position.z = 0
	const farOpaque = new SimpleModel([sharedOpaquePrimitive])
	farOpaque.transform.position.z = -4

	const nearTransparent = new SimpleModel([sharedTransparentPrimitive])
	nearTransparent.transform.position.z = -1
	const farTransparent = new SimpleModel([sharedTransparentPrimitive])
	farTransparent.transform.position.z = -6

	reflectiveModel.transform.position.x = 3

	const scene = new Scene()
	scene.addModel(farOpaque)
	scene.addModel(nearOpaque)
	scene.addModel(nearTransparent)
	scene.addModel(farTransparent)
	scene.addModel(reflectiveModel)

	const frame = PreparedSceneBuilder.build({
		scene,
		camera,
		shadowMaps: new Map(),
	})

	assert.equal(frame.opaquePackets.length, 3)
	assert.equal(frame.transparentPackets.length, 2)
	assert.equal(frame.reflectivePackets.length, 1)
	assert.equal(frame.shadowCasterPackets.length, 3)
	assert.equal(frame.shadowTransmitterPackets.length, 2)

	assert.equal(frame.opaquePackets[0].model.id, nearOpaque.id)
	assert.equal(frame.opaquePackets[1].model.id, farOpaque.id)
	assert.equal(frame.transparentPackets[0].model.id, farTransparent.id)
	assert.equal(frame.transparentPackets[1].model.id, nearTransparent.id)
	assert.equal(frame.reflectivePackets[0].model.id, reflectiveModel.id)

	console.log('Render list builder tests passed')
}

run()
