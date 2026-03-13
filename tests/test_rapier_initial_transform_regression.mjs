import assert from 'node:assert/strict'
import { Scene } from '../src/core/Scene.ts'
import { PBRMaterial } from '../src/materials/PBRMaterial.ts'
import { MeshFactory } from '../src/meshes/MeshFactory.ts'
import { PhysicsSystem } from '../src/physics/PhysicsSystem.ts'
import { RapierPhysicsAdapter } from '../src/physics/adapters/RapierPhysicsAdapter.ts'

async function loadRapier() {
	try {
		return await import('@dimforge/rapier3d-compat')
	} catch {
		return null
	}
}

async function run() {
	const rapier = await loadRapier()
	if (!rapier) {
		console.log(
			'Rapier initial transform regression skipped (@dimforge/rapier3d-compat not installed)'
		)
		return
	}

	const physics = new PhysicsSystem({
		adapter: new RapierPhysicsAdapter({
			moduleLoader: async () => rapier,
			strict: true,
		}),
	})

	await physics.init()
	physics.createWorld({
		worldId: 'main',
		gravity: { x: 0, y: -2, z: 0 },
		mode: 'fixed',
		fixedDeltaSeconds: 1 / 60,
		maxSubsteps: 4,
		maxDeltaSeconds: 1 / 10,
	})

	const scene = new Scene()
	const cube = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		16,
		16,
		16,
		new PBRMaterial()
	)
	cube.position.set(12, 150, -34)
	cube.setRotationFromEuler(0.25, 0.5, -0.1)
	scene.add(cube)

	const body = physics.attachBody(cube, {
		worldId: 'main',
		body: { type: 'dynamic' },
		authority: 'physics',
	})

	physics.addCollider(body, {
		mode: 'explicit',
		shape: {
			kind: 'box',
			halfExtents: { x: 8, y: 8, z: 8 },
		},
	})

	physics.step(1 / 60)

	assert.ok(Number.isFinite(cube.position.x))
	assert.ok(Number.isFinite(cube.position.y))
	assert.ok(Number.isFinite(cube.position.z))
	assert.ok(Number.isFinite(cube.quaternion.x))
	assert.ok(Number.isFinite(cube.quaternion.y))
	assert.ok(Number.isFinite(cube.quaternion.z))
	assert.ok(Number.isFinite(cube.quaternion.w))
	assert.ok(Math.abs(cube.position.x - 12) < 1e-3)
	assert.ok(cube.position.y > 149.9 && cube.position.y < 150)
	assert.ok(Math.abs(cube.position.z + 34) < 1e-3)

	console.log('Rapier initial transform regression passed')
}

await run()
