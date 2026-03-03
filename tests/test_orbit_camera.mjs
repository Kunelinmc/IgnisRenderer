import assert from 'node:assert/strict'
import { OrbitCamera } from '../src/cameras/OrbitCamera.ts'

function testOrbitCameraAutoAdjustsClipPlanes() {
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500)

	assert.equal(camera.distance, 500)
	assert.ok(camera.near >= camera.clipNearMin)
	assert.ok(camera.far >= camera.distance * camera.clipFarFactor)
	assert.ok(camera.near < camera.far)

	camera.zoom(4000)

	assert.equal(camera.distance, camera.maxDistance)
	assert.ok(camera.near >= camera.distance * camera.clipNearFactor)
	assert.ok(camera.far >= camera.distance * camera.clipFarFactor)
	assert.ok(camera.near < camera.far)
}

function testOrbitCameraCanDisableAutoClipAdjustment() {
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500)

	camera.autoAdjustClipPlanes = false
	camera.near = 0.25
	camera.far = 777

	camera.zoom(250)

	assert.equal(camera.near, 0.25)
	assert.equal(camera.far, 777)
}

function run() {
	testOrbitCameraAutoAdjustsClipPlanes()
	testOrbitCameraCanDisableAutoClipAdjustment()
	console.log('Orbit camera clip plane tests passed')
}

run()
