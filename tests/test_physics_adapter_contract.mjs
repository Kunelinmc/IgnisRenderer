import assert from 'node:assert/strict'
import { Node } from '../src/core/Node.ts'
import { PhysicsSystem } from '../src/physics/PhysicsSystem.ts'
import { RapierPhysicsAdapter } from '../src/physics/adapters/RapierPhysicsAdapter.ts'
import { AmmoPhysicsAdapter } from '../src/physics/adapters/AmmoPhysicsAdapter.ts'

async function runContract(adapter, label) {
	const physics = new PhysicsSystem({ adapter })
	await physics.init()
	physics.createWorld({
		worldId: 'main',
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: 'fixed',
		fixedDeltaMs: 16,
	})

	const nodeA = new Node({ position: { x: 0, y: 1, z: 0 } })
	const nodeB = new Node({ position: { x: 0, y: 0, z: 0 } })
	const nodeC = new Node({ position: { x: 0, y: 1, z: 0 } })

	const bodyA = physics.attachBody(nodeA, {
		worldId: 'main',
		body: { type: 'dynamic' },
		authority: 'physics',
	})
	const bodyB = physics.attachBody(nodeB, {
		worldId: 'main',
		body: { type: 'fixed' },
		authority: 'physics',
	})
	const bodyC = physics.attachBody(nodeC, {
		worldId: 'main',
		body: { type: 'kinematic' },
		authority: 'animation',
	})

	physics.addCollider(bodyA, {
		mode: 'explicit',
		shape: { kind: 'sphere', radius: 1 },
	})
	physics.addCollider(bodyB, {
		mode: 'auto-fit',
		shapePreference: 'box',
	})
	const queryHit = physics.raycast({
		worldId: 'main',
		origin: { x: 0, y: 1, z: -5 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
	})
	assert.ok(queryHit)
	assert.equal(queryHit?.bodyId, bodyA.id)
	const overlap = physics.overlapSphere({
		worldId: 'main',
		center: { x: 0, y: 1, z: 0 },
		radius: 0.25,
	})
	assert.ok(overlap.some((item) => item.bodyId === bodyA.id))

	const joint = physics.createJoint({
		worldId: 'main',
		type: 'fixed',
		bodyA,
		bodyB,
	})
	assert.ok(joint.id)

	const controller = physics.createCharacterController({
		worldId: 'main',
		body: bodyC,
		radius: 0.3,
		height: 1.7,
		stepHeight: 0.3,
		maxSlope: 50,
		jumpSpeed: 3,
	})
	controller.moveAndSlide({ x: 1, y: 0, z: 0 }, 16)
	controller.jump(2)

	const report = physics.step(16)
	assert.equal(report.worldReports.length, 1)
	assert.equal(report.worldReports[0].worldId, 'main')
	assert.ok(report.worldReports[0].substeps >= 1)
	assert.ok(typeof controller.isGrounded() === 'boolean')
	assert.ok(Array.isArray(physics.drainEvents('main')))

	physics.destroyWorld('main')
	console.log(`Physics adapter contract passed: ${label}`)
}

async function run() {
	await runContract(
		new RapierPhysicsAdapter({
			moduleLoader: async () => ({ fake: true }),
			strict: true,
		}),
		'rapier'
	)
	await runContract(
		new AmmoPhysicsAdapter({
			moduleLoader: async () => ({ fake: true }),
			strict: true,
		}),
		'ammo'
	)
}

await run()
