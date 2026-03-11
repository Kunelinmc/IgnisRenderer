import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { PhysicsSystem } from "../src/physics/PhysicsSystem.ts";
import { RapierPhysicsAdapter } from "../src/physics/adapters/RapierPhysicsAdapter.ts";
import { AmmoPhysicsAdapter } from "../src/physics/adapters/AmmoPhysicsAdapter.ts";

function createFakeRapierModule() {
	const stats = {
		stepCalls: 0,
	};

	class FakeRigidBodyDesc {
		constructor(type) {
			this.type = type;
			this.translation = { x: 0, y: 0, z: 0 };
			this.rotation = { x: 0, y: 0, z: 0, w: 1 };
			this.linvel = { x: 0, y: 0, z: 0 };
			this.angvel = { x: 0, y: 0, z: 0 };
			this.ccd = false;
		}
		static dynamic() {
			return new FakeRigidBodyDesc("dynamic");
		}
		static fixed() {
			return new FakeRigidBodyDesc("fixed");
		}
		static kinematicPositionBased() {
			return new FakeRigidBodyDesc("kinematic");
		}
		setTranslation(x, y, z) {
			if (typeof x === "object") {
				this.translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this.translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
			return this;
		}
		setRotation(q) {
			if (q && typeof q === "object") {
				this.rotation = {
					x: q.x ?? 0,
					y: q.y ?? 0,
					z: q.z ?? 0,
					w: q.w ?? 1,
				};
			}
			return this;
		}
		setLinvel(x, y, z) {
			if (typeof x === "object") {
				this.linvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this.linvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
			return this;
		}
		setAngvel(x, y, z) {
			if (typeof x === "object") {
				this.angvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this.angvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
			return this;
		}
		setCcdEnabled(value) {
			this.ccd = !!value;
			return this;
		}
		setLinearDamping() {
			return this;
		}
		setAngularDamping() {
			return this;
		}
		setCanSleep() {
			return this;
		}
		setEnabledTranslations() {
			return this;
		}
		setEnabledRotations() {
			return this;
		}
		restrictTranslations() {
			return this;
		}
		restrictRotations() {
			return this;
		}
	}

	class FakeColliderDesc {
		constructor(kind) {
			this.kind = kind;
			this.sensor = false;
			this.translation = { x: 0, y: 0, z: 0 };
		}
		static cuboid(x, y, z) {
			return new FakeColliderDesc({ kind: "box", x, y, z });
		}
		static ball(radius) {
			return new FakeColliderDesc({ kind: "sphere", radius });
		}
		static capsule(halfHeight, radius) {
			return new FakeColliderDesc({ kind: "capsule", halfHeight, radius });
		}
		static cylinder(halfHeight, radius) {
			return new FakeColliderDesc({ kind: "cylinder", halfHeight, radius });
		}
		static trimesh(vertices, indices) {
			return new FakeColliderDesc({ kind: "trimesh", vertices, indices });
		}
		setSensor(value) {
			this.sensor = !!value;
			return this;
		}
		setTranslation(x, y, z) {
			if (typeof x === "object") {
				this.translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this.translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
			return this;
		}
		setFriction() {
			return this;
		}
		setRestitution() {
			return this;
		}
		setDensity() {
			return this;
		}
	}

	class FakeRigidBody {
		constructor(desc) {
			this._type = desc.type;
			this._translation = { ...desc.translation };
			this._rotation = { ...desc.rotation };
			this._linvel = { ...desc.linvel };
			this._angvel = { ...desc.angvel };
			this._ccd = !!desc.ccd;
		}
		setTranslation(x, y, z) {
			if (typeof x === "object") {
				this._translation = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this._translation = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
		}
		setNextKinematicTranslation(x, y, z) {
			this.setTranslation(x, y, z);
		}
		setRotation(q) {
			if (q && typeof q === "object") {
				this._rotation = {
					x: q.x ?? 0,
					y: q.y ?? 0,
					z: q.z ?? 0,
					w: q.w ?? 1,
				};
			}
		}
		setNextKinematicRotation(q) {
			this.setRotation(q);
		}
		setLinvel(x, y, z) {
			if (typeof x === "object") {
				this._linvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this._linvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
		}
		setAngvel(x, y, z) {
			if (typeof x === "object") {
				this._angvel = { x: x.x ?? 0, y: x.y ?? 0, z: x.z ?? 0 };
			} else {
				this._angvel = { x: x ?? 0, y: y ?? 0, z: z ?? 0 };
			}
		}
		translation() {
			return { ...this._translation };
		}
		rotation() {
			return { ...this._rotation };
		}
		linvel() {
			return { ...this._linvel };
		}
		isSleeping() {
			return false;
		}
		isCcdEnabled() {
			return this._ccd;
		}
	}

	class FakeWorld {
		constructor() {
			this._bodies = new Set();
			this._colliders = new Set();
			this._joints = new Set();
		}
		createRigidBody(desc) {
			const body = new FakeRigidBody(desc);
			this._bodies.add(body);
			return body;
		}
		removeRigidBody(body) {
			this._bodies.delete(body);
		}
		createCollider(desc, body) {
			const collider = { desc, body };
			this._colliders.add(collider);
			return collider;
		}
		removeCollider(collider) {
			this._colliders.delete(collider);
		}
		createImpulseJoint(data, bodyA, bodyB, collisionEnabled) {
			const joint = { data, bodyA, bodyB, collisionEnabled };
			this._joints.add(joint);
			return joint;
		}
		removeImpulseJoint(joint) {
			this._joints.delete(joint);
		}
		step() {
			stats.stepCalls++;
		}
		free() {}
	}

	const module = {
		init: async () => {},
		World: FakeWorld,
		Vector3: class FakeVector3 {
			constructor(x, y, z) {
				this.x = x;
				this.y = y;
				this.z = z;
			}
		},
		Quaternion: class FakeQuaternion {
			constructor(x, y, z, w) {
				this.x = x;
				this.y = y;
				this.z = z;
				this.w = w;
			}
		},
		RigidBodyDesc: FakeRigidBodyDesc,
		ColliderDesc: FakeColliderDesc,
		JointData: {
			fixed: () => ({ kind: "fixed" }),
			revolute: () => ({ kind: "revolute" }),
			spring: () => ({ kind: "spring" }),
		},
	};
	return { module, stats };
}

function createFakeAmmoModule() {
	const stats = {
		stepCalls: 0,
	};

	function readNumberLike(value, key, fallback = 0) {
		if (!value || typeof value !== "object") return fallback;
		const member = value[key];
		if (typeof member === "number" && Number.isFinite(member)) {
			return member;
		}
		if (typeof member === "function") {
			try {
				const result = member.call(value);
				if (typeof result === "number" && Number.isFinite(result)) {
					return result;
				}
			} catch {}
		}
		return fallback;
	}

	class FakeBtVector3 {
		constructor(x = 0, y = 0, z = 0) {
			this._x = x;
			this._y = y;
			this._z = z;
		}
		x() {
			return this._x;
		}
		y() {
			return this._y;
		}
		z() {
			return this._z;
		}
		setValue(x, y, z) {
			this._x = x ?? 0;
			this._y = y ?? 0;
			this._z = z ?? 0;
		}
		clone() {
			return new FakeBtVector3(this._x, this._y, this._z);
		}
	}

	class FakeBtQuaternion {
		constructor(x = 0, y = 0, z = 0, w = 1) {
			this._x = x;
			this._y = y;
			this._z = z;
			this._w = w;
		}
		x() {
			return this._x;
		}
		y() {
			return this._y;
		}
		z() {
			return this._z;
		}
		w() {
			return this._w;
		}
		clone() {
			return new FakeBtQuaternion(this._x, this._y, this._z, this._w);
		}
	}

	class FakeBtTransform {
		constructor() {
			this.setIdentity();
		}
		setIdentity() {
			this._origin = new FakeBtVector3(0, 0, 0);
			this._rotation = new FakeBtQuaternion(0, 0, 0, 1);
		}
		setOrigin(value) {
			this._origin = new FakeBtVector3(
				readNumberLike(value, "x"),
				readNumberLike(value, "y"),
				readNumberLike(value, "z")
			);
		}
		setRotation(value) {
			this._rotation = new FakeBtQuaternion(
				readNumberLike(value, "x"),
				readNumberLike(value, "y"),
				readNumberLike(value, "z"),
				readNumberLike(value, "w", 1)
			);
		}
		getOrigin() {
			return this._origin;
		}
		getRotation() {
			return this._rotation;
		}
		clone() {
			const next = new FakeBtTransform();
			next.setOrigin(this._origin);
			next.setRotation(this._rotation);
			return next;
		}
	}

	class FakeBtDefaultMotionState {
		constructor(transform) {
			this._transform =
				transform instanceof FakeBtTransform ?
					transform.clone()
				:	new FakeBtTransform();
		}
		getWorldTransform(out) {
			if (out && typeof out.setOrigin === "function") {
				out.setOrigin(this._transform.getOrigin());
				out.setRotation(this._transform.getRotation());
			}
		}
		setWorldTransform(transform) {
			if (transform instanceof FakeBtTransform) {
				this._transform = transform.clone();
			}
		}
	}

	class FakeShape {
		calculateLocalInertia(_mass, out) {
			if (out && typeof out.setValue === "function") {
				out.setValue(0, 0, 0);
			}
		}
	}

	class FakeBtSphereShape extends FakeShape {
		constructor(radius) {
			super();
			this.radius = radius;
		}
	}

	class FakeBtBoxShape extends FakeShape {
		constructor(halfExtents) {
			super();
			this.halfExtents = halfExtents;
		}
	}

	class FakeBtRigidBodyConstructionInfo {
		constructor(mass, motionState, shape, localInertia) {
			this.mass = mass;
			this.motionState = motionState;
			this.shape = shape;
			this.localInertia = localInertia;
		}
	}

	class FakeBtRigidBody {
		constructor(info) {
			this._info = info;
			this._mass = Number.isFinite(info?.mass) ? info.mass : 0;
			this._motionState = info?.motionState;
			this._transform = new FakeBtTransform();
			if (this._motionState?.getWorldTransform) {
				this._motionState.getWorldTransform(this._transform);
			}
			this._linearVelocity = new FakeBtVector3(0, 0, 0);
			this._flags = 0;
			this._active = true;
			this._sleeping = false;
		}
		setWorldTransform(transform) {
			if (transform instanceof FakeBtTransform) {
				this._transform = transform.clone();
				this._motionState?.setWorldTransform?.(this._transform);
			}
		}
		getWorldTransform() {
			return this._transform;
		}
		setLinearVelocity(value) {
			this._linearVelocity = new FakeBtVector3(
				readNumberLike(value, "x"),
				readNumberLike(value, "y"),
				readNumberLike(value, "z")
			);
		}
		getLinearVelocity() {
			return this._linearVelocity;
		}
		setAngularVelocity() {}
		setCollisionFlags(flags) {
			this._flags = flags ?? 0;
		}
		getCollisionFlags() {
			return this._flags;
		}
		setActivationState() {}
		setCcdMotionThreshold() {}
		setCcdSweptSphereRadius() {}
		activate() {
			this._active = true;
			this._sleeping = false;
		}
		isActive() {
			return this._active;
		}
		isSleeping() {
			return this._sleeping;
		}
	}

	class FakeBtDiscreteDynamicsWorld {
		constructor() {
			this._gravity = new FakeBtVector3(0, -9.8, 0);
			this._bodies = new Set();
		}
		setGravity(value) {
			this._gravity = new FakeBtVector3(
				readNumberLike(value, "x"),
				readNumberLike(value, "y"),
				readNumberLike(value, "z")
			);
		}
		addRigidBody(body) {
			this._bodies.add(body);
		}
		removeRigidBody(body) {
			this._bodies.delete(body);
		}
		stepSimulation(deltaSeconds) {
			stats.stepCalls++;
			const dt = Math.max(0, Number(deltaSeconds) || 0);
			for (const body of this._bodies) {
				if (!body || body._mass <= 0) continue;
				const velocity = body.getLinearVelocity();
				velocity.setValue(
					velocity.x() + this._gravity.x() * dt,
					velocity.y() + this._gravity.y() * dt,
					velocity.z() + this._gravity.z() * dt
				);
				const transform = body.getWorldTransform();
				const origin = transform.getOrigin();
				origin.setValue(
					origin.x() + velocity.x() * dt,
					origin.y() + velocity.y() * dt,
					origin.z() + velocity.z() * dt
				);
				body.setWorldTransform(transform);
			}
		}
	}

	const module = {
		btDefaultCollisionConfiguration: class {},
		btCollisionDispatcher: class {
			constructor() {}
		},
		btDbvtBroadphase: class {},
		btSequentialImpulseConstraintSolver: class {},
		btDiscreteDynamicsWorld: FakeBtDiscreteDynamicsWorld,
		btVector3: FakeBtVector3,
		btQuaternion: FakeBtQuaternion,
		btTransform: FakeBtTransform,
		btDefaultMotionState: FakeBtDefaultMotionState,
		btRigidBodyConstructionInfo: FakeBtRigidBodyConstructionInfo,
		btRigidBody: FakeBtRigidBody,
		btSphereShape: FakeBtSphereShape,
		btBoxShape: FakeBtBoxShape,
		destroy: () => {},
	};

	return { module, stats };
}

async function runContract(adapter, label, opts = {}) {
	const physics = new PhysicsSystem({ adapter });
	await physics.init();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: "fixed",
		fixedDeltaSeconds: 0.016,
	});

	const nodeA = new Node({ position: { x: 0, y: 1, z: 0 } });
	const nodeB = new Node({ position: { x: 0, y: 0, z: 0 } });
	const nodeC = new Node({ position: { x: 0, y: 1, z: 0 } });

	const bodyA = physics.attachBody(nodeA, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	const bodyB = physics.attachBody(nodeB, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const bodyC = physics.attachBody(nodeC, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});

	physics.addCollider(bodyA, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});
	physics.addCollider(bodyB, {
		mode: "auto-fit",
		shapePreference: "box",
	});
	const queryHit = physics.raycast({
		worldId: "main",
		origin: { x: 0, y: 1, z: -5 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
	});
	assert.ok(queryHit);
	assert.equal(queryHit?.bodyId, bodyA.id);
	const overlap = physics.overlapSphere({
		worldId: "main",
		center: { x: 0, y: 1, z: 0 },
		radius: 0.25,
	});
	assert.ok(overlap.some((item) => item.bodyId === bodyA.id));

	const joint = physics.createJoint({
		worldId: "main",
		type: "fixed",
		bodyA,
		bodyB,
	});
	assert.ok(joint.id);

	const controller = physics.createCharacterController({
		worldId: "main",
		body: bodyC,
		radius: 0.3,
		height: 1.7,
		stepHeight: 0.3,
		maxSlope: 50,
		jumpSpeed: 3,
	});
	controller.moveAndSlide({ x: 1, y: 0, z: 0 }, 0.016);
	controller.jump(2);

	const report = physics.step(0.016);
	assert.equal(report.worldReports.length, 1);
	assert.equal(report.worldReports[0].worldId, "main");
	assert.ok(report.worldReports[0].substeps >= 1);
	assert.ok(typeof controller.isGrounded() === "boolean");
	assert.ok(Array.isArray(physics.drainEvents("main")));

	physics.destroyWorld("main");

	if (typeof opts.afterRun === "function") {
		opts.afterRun();
	}

	console.log(`Physics adapter contract passed: ${label}`);
}

async function run() {
	const fakeRapier = createFakeRapierModule();
	const fakeAmmo = createFakeAmmoModule();
	await runContract(
		new RapierPhysicsAdapter({
			moduleLoader: async () => fakeRapier.module,
			strict: true,
		}),
		"rapier",
		{
			afterRun: () => {
				assert.ok(
					fakeRapier.stats.stepCalls > 0,
					"Expected Rapier world.step() to be used at least once"
				);
			},
		}
	);
	await runContract(
		new AmmoPhysicsAdapter({
			moduleLoader: async () => fakeAmmo.module,
			strict: true,
		}),
		"ammo",
		{
			afterRun: () => {
				assert.ok(
					fakeAmmo.stats.stepCalls > 0,
					"Expected Ammo world.stepSimulation() to be used at least once"
				);
			},
		}
	);
}

await run();
