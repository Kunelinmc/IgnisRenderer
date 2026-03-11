import assert from 'node:assert/strict'
import { Node } from '../src/core/Node.ts'
import { PhysicsSystem } from '../src/physics/PhysicsSystem.ts'
import { RapierPhysicsAdapter } from '../src/physics/adapters/RapierPhysicsAdapter.ts'
import { AmmoPhysicsAdapter } from '../src/physics/adapters/AmmoPhysicsAdapter.ts'

function createFakeRapierModule() {
	const stats = {
		stepCalls: 0,
	}

	class FakeRigidBodyDesc {
		constructor(type) {
			this.type = type
			this.translation = { x: 0, y: 0, z: 0 }
			this.rotation = { x: 0, y: 0, z: 0, w: 1 }
			this.linvel = { x: 0, y: 0, z: 0 }
			this.angvel = { x: 0, y: 0, z: 0 }
			this.ccd = false
		}
		static dynamic() {
			return new FakeRigidBodyDesc('dynamic')
		}
		static fixed() {
			return new FakeRigidBodyDesc('fixed')
		}
		static kinematicPositionBased() {
			return new FakeRigidBodyDesc('kinematic')
		}
		setTranslation(x, y, z) {
			if (typeof x === 'object') {
				this.translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this.translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
			return this
		}
		setRotation(q) {
			if (q && typeof q === 'object') {
				this.rotation = {
					x: q.x ?? 0,
					y: q.y ?? 0,
					z: q.z ?? 0,
					w: q.w ?? 1,
				}
			}
			return this
		}
		setLinvel(x, y, z) {
			if (typeof x === 'object') {
				this.linvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this.linvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
			return this
		}
		setAngvel(x, y, z) {
			if (typeof x === 'object') {
				this.angvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this.angvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
			return this
		}
		setCcdEnabled(value) {
			this.ccd = !!value
			return this
		}
		setLinearDamping() {
			return this
		}
		setAngularDamping() {
			return this
		}
		setCanSleep() {
			return this
		}
		setEnabledTranslations() {
			return this
		}
		setEnabledRotations() {
			return this
		}
		restrictTranslations() {
			return this
		}
		restrictRotations() {
			return this
		}
	}

	class FakeColliderDesc {
		constructor(kind) {
			this.kind = kind
			this.sensor = false
			this.translation = { x: 0, y: 0, z: 0 }
		}
		static cuboid(x, y, z) {
			return new FakeColliderDesc({ kind: 'box', x, y, z })
		}
		static ball(radius) {
			return new FakeColliderDesc({ kind: 'sphere', radius })
		}
		static capsule(halfHeight, radius) {
			return new FakeColliderDesc({ kind: 'capsule', halfHeight, radius })
		}
		static cylinder(halfHeight, radius) {
			return new FakeColliderDesc({ kind: 'cylinder', halfHeight, radius })
		}
		static trimesh(vertices, indices) {
			return new FakeColliderDesc({ kind: 'trimesh', vertices, indices })
		}
		setSensor(value) {
			this.sensor = !!value
			return this
		}
		setTranslation(x, y, z) {
			if (typeof x === 'object') {
				this.translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this.translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
			return this
		}
		setFriction() {
			return this
		}
		setRestitution() {
			return this
		}
		setDensity() {
			return this
		}
	}

	class FakeRigidBody {
		constructor(desc) {
			this._type = desc.type
			this._translation = { ...desc.translation }
			this._rotation = { ...desc.rotation }
			this._linvel = { ...desc.linvel }
			this._angvel = { ...desc.angvel }
			this._ccd = !!desc.ccd
		}
		setTranslation(x, y, z) {
			if (typeof x === 'object') {
				this._translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this._translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
		}
		setNextKinematicTranslation(x, y, z) {
			this.setTranslation(x, y, z)
		}
		setRotation(q) {
			if (q && typeof q === 'object') {
				this._rotation = {
					x: q.x ?? 0,
					y: q.y ?? 0,
					z: q.z ?? 0,
					w: q.w ?? 1,
				}
			}
		}
		setNextKinematicRotation(q) {
			this.setRotation(q)
		}
		setLinvel(x, y, z) {
			if (typeof x === 'object') {
				this._linvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this._linvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
		}
		setAngvel(x, y, z) {
			if (typeof x === 'object') {
				this._angvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 }
			} else {
				this._angvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 }
			}
		}
		translation() {
			return { ...this._translation }
		}
		rotation() {
			return { ...this._rotation }
		}
		linvel() {
			return { ...this._linvel }
		}
		isSleeping() {
			return false
		}
		isCcdEnabled() {
			return this._ccd
		}
	}

	class FakeWorld {
		constructor() {
			this._bodies = new Set()
			this._colliders = new Set()
			this._joints = new Set()
		}
		createRigidBody(desc) {
			const body = new FakeRigidBody(desc)
			this._bodies.add(body)
			return body
		}
		removeRigidBody(body) {
			this._bodies.delete(body)
		}
		createCollider(desc, body) {
			const collider = { desc, body }
			this._colliders.add(collider)
			return collider
		}
		removeCollider(collider) {
			this._colliders.delete(collider)
		}
		createImpulseJoint(data, bodyA, bodyB, collisionEnabled) {
			const joint = { data, bodyA, bodyB, collisionEnabled }
			this._joints.add(joint)
			return joint
		}
		removeImpulseJoint(joint) {
			this._joints.delete(joint)
		}
		step() {
			stats.stepCalls++
		}
		free() {}
	}

	const module = {
		init: async () => {},
		World: FakeWorld,
		Vector3: class FakeVector3 {
			constructor(x, y, z) {
				this.x = x
				this.y = y
				this.z = z
			}
		},
		Quaternion: class FakeQuaternion {
			constructor(x, y, z, w) {
				this.x = x
				this.y = y
				this.z = z
				this.w = w
			}
		},
		RigidBodyDesc: FakeRigidBodyDesc,
		ColliderDesc: FakeColliderDesc,
		JointData: {
			fixed: () => ({ kind: 'fixed' }),
			revolute: () => ({ kind: 'revolute' }),
			spring: () => ({ kind: 'spring' }),
		},
	}
	return { module, stats }
}

async function runContract(adapter, label, opts = {}) {
	const physics = new PhysicsSystem({ adapter })
	await physics.init()
	physics.createWorld({
		worldId: 'main',
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: 'fixed',
		fixedDeltaSeconds: 0.016,
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
	controller.moveAndSlide({ x: 1, y: 0, z: 0 }, 0.016)
	controller.jump(2)

	const report = physics.step(0.016)
	assert.equal(report.worldReports.length, 1)
	assert.equal(report.worldReports[0].worldId, 'main')
	assert.ok(report.worldReports[0].substeps >= 1)
	assert.ok(typeof controller.isGrounded() === 'boolean')
	assert.ok(Array.isArray(physics.drainEvents('main')))

	physics.destroyWorld('main')

	if (typeof opts.afterRun === 'function') {
		opts.afterRun()
	}

	console.log(`Physics adapter contract passed: ${label}`)
}

async function run() {
	const fakeRapier = createFakeRapierModule()
	await runContract(
		new RapierPhysicsAdapter({
			moduleLoader: async () => fakeRapier.module,
			strict: true,
		}),
		'rapier',
		{
			afterRun: () => {
				assert.ok(
					fakeRapier.stats.stepCalls > 0,
					'Expected Rapier world.step() to be used at least once'
				)
			},
		}
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
