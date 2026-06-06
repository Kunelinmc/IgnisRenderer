import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../../../src/physics/adapters/SimplePhysicsAdapter.ts";

class TrackingPhysicsAdapter extends SimplePhysicsAdapter {
	constructor() {
		super("tracking");
		this.stepWorldCalls = 0;
		this.setBodyTransformCalls = 0;
		this.setBodyLinearVelocityCalls = 0;
		this.setAngularVelocityCalls = 0;
		this.applyForceCalls = 0;
		this.applyTorqueCalls = 0;
		this.applyImpulseCalls = 0;
		this.lastSetBodyLinearVelocityCall = null;
		this.lastSetAngularVelocityCall = null;
		this.lastApplyForceCall = null;
		this.lastApplyTorqueCall = null;
		this.lastApplyImpulseCall = null;
	}

	stepWorld(worldId, deltaSeconds) {
		this.stepWorldCalls++;
		return super.stepWorld(worldId, deltaSeconds);
	}

	setBodyTransform(worldId, bodyId, transform) {
		this.setBodyTransformCalls++;
		super.setBodyTransform(worldId, bodyId, transform);
	}

	setBodyLinearVelocity(worldId, bodyId, velocity) {
		this.setBodyLinearVelocityCalls++;
		this.lastSetBodyLinearVelocityCall = {
			worldId,
			bodyId,
			velocity: { ...velocity },
		};
		super.setBodyLinearVelocity(worldId, bodyId, velocity);
	}

	setAngularVelocity(worldId, bodyId, velocity) {
		this.setAngularVelocityCalls++;
		this.lastSetAngularVelocityCall = {
			worldId,
			bodyId,
			velocity: { ...velocity },
		};
		super.setAngularVelocity(worldId, bodyId, velocity);
	}

	applyForce(worldId, bodyId, force) {
		this.applyForceCalls++;
		this.lastApplyForceCall = {
			worldId,
			bodyId,
			force: { ...force },
		};
		super.applyForce(worldId, bodyId, force);
	}

	applyTorque(worldId, bodyId, torque) {
		this.applyTorqueCalls++;
		this.lastApplyTorqueCall = {
			worldId,
			bodyId,
			torque: { ...torque },
		};
		super.applyTorque(worldId, bodyId, torque);
	}

	applyImpulse(worldId, bodyId, impulse) {
		this.applyImpulseCalls++;
		this.lastApplyImpulseCall = {
			worldId,
			bodyId,
			impulse: { ...impulse },
		};
		super.applyImpulse(worldId, bodyId, impulse);
	}
}

function testAnimationAuthorityDirtySync() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		0,
		"Expected no animation transform upload when node is unchanged"
	);

	node.position.x = 2;
	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		1,
		"Expected one transform upload when animation body moved"
	);

	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		1,
		"Expected unchanged animation body to skip transform upload"
	);
}

function testSleepingIslandSkipsWorldStep() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	physics.step(0.016);
	const firstStepCalls = adapter.stepWorldCalls;
	assert.ok(firstStepCalls > 0, "Expected first frame to step physics world");

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls,
		"Expected sleeping island to skip redundant world step"
	);
}

function createSleepingWorldFixture() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	physics.step(0.016);
	const firstStepCalls = adapter.stepWorldCalls;
	assert.ok(firstStepCalls > 0, "Expected fixture warmup step to run");

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls,
		"Expected fixture to be in a skipped-step state"
	);

	return { adapter, physics, body, firstStepCalls };
}

function testSetLinearVelocityMarksWorldDirty() {
	const { adapter, physics, body, firstStepCalls } = createSleepingWorldFixture();
	const velocity = { x: 1, y: 2, z: 3 };

	physics.setLinearVelocity(body, velocity);

	assert.equal(
		adapter.setBodyLinearVelocityCalls,
		1,
		"Expected setLinearVelocity() to forward to adapter"
	);
	assert.deepEqual(adapter.lastSetBodyLinearVelocityCall, {
		worldId: "main",
		bodyId: body.id,
		velocity,
	});

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls + 1,
		"Expected setLinearVelocity() to force next frame simulation"
	);
}

function testSetAngularVelocityMarksWorldDirty() {
	const { adapter, physics, body, firstStepCalls } = createSleepingWorldFixture();
	const velocity = { x: 0, y: 1, z: 2 };

	physics.setAngularVelocity(body.id, velocity);

	assert.equal(
		adapter.setAngularVelocityCalls,
		1,
		"Expected setAngularVelocity() to forward to adapter"
	);
	assert.deepEqual(adapter.lastSetAngularVelocityCall, {
		worldId: "main",
		bodyId: body.id,
		velocity,
	});

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls + 1,
		"Expected setAngularVelocity() to force next frame simulation"
	);
}

function testApplyForceMarksWorldDirty() {
	const { adapter, physics, body, firstStepCalls } = createSleepingWorldFixture();
	const force = { x: 3, y: 0, z: 0 };

	physics.applyForce(body, force);

	assert.equal(
		adapter.applyForceCalls,
		1,
		"Expected applyForce() to forward to adapter"
	);
	assert.deepEqual(adapter.lastApplyForceCall, {
		worldId: "main",
		bodyId: body.id,
		force,
	});

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls + 1,
		"Expected applyForce() to force next frame simulation"
	);
}

function testApplyTorqueMarksWorldDirty() {
	const { adapter, physics, body, firstStepCalls } = createSleepingWorldFixture();
	const torque = { x: 0, y: 2, z: 0 };

	physics.applyTorque(body, torque);

	assert.equal(
		adapter.applyTorqueCalls,
		1,
		"Expected applyTorque() to forward to adapter"
	);
	assert.deepEqual(adapter.lastApplyTorqueCall, {
		worldId: "main",
		bodyId: body.id,
		torque,
	});

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls + 1,
		"Expected applyTorque() to force next frame simulation"
	);
}

function testApplyImpulseMarksWorldDirty() {
	const { adapter, physics, body, firstStepCalls } = createSleepingWorldFixture();
	const impulse = { x: 0, y: 1, z: 0 };

	physics.applyImpulse(body, impulse);

	assert.equal(
		adapter.applyImpulseCalls,
		1,
		"Expected applyImpulse() to forward to adapter"
	);
	assert.deepEqual(adapter.lastApplyImpulseCall, {
		worldId: "main",
		bodyId: body.id,
		impulse,
	});

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls + 1,
		"Expected applyImpulse() to force next frame simulation"
	);
}

function run() {
	testAnimationAuthorityDirtySync();
	testSleepingIslandSkipsWorldStep();
	testSetLinearVelocityMarksWorldDirty();
	testSetAngularVelocityMarksWorldDirty();
	testApplyForceMarksWorldDirty();
	testApplyTorqueMarksWorldDirty();
	testApplyImpulseMarksWorldDirty();
	console.log("Physics system optimization tests passed");
}

run();
