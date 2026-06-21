import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../../../src/physics/adapters/SimplePhysicsAdapter.ts";

class TrackingPhysicsAdapter extends SimplePhysicsAdapter {
	constructor() {
		super("tracking-runtime-gap");
		this.destroyJointCalls = [];
		this.destroyControllerCalls = [];
		this.destroyColliderCalls = [];
		this.sensorCalls = [];
		this.maskCalls = [];
		this.materialCalls = [];
	}

	destroyJoint(worldId, jointId) {
		this.destroyJointCalls.push({ worldId, jointId });
		super.destroyJoint(worldId, jointId);
	}

	destroyCharacterController(worldId, controllerId) {
		this.destroyControllerCalls.push({ worldId, controllerId });
		super.destroyCharacterController(worldId, controllerId);
	}

	destroyCollider(worldId, colliderId) {
		this.destroyColliderCalls.push({ worldId, colliderId });
		super.destroyCollider(worldId, colliderId);
	}

	setColliderSensor(worldId, colliderId, isSensor) {
		this.sensorCalls.push({ worldId, colliderId, isSensor });
		super.setColliderSensor(worldId, colliderId, isSensor);
	}

	setCollisionMask(worldId, colliderId, mask) {
		this.maskCalls.push({ worldId, colliderId, mask });
		super.setCollisionMask(worldId, colliderId, mask);
	}

	setColliderMaterial(worldId, colliderId, material) {
		this.materialCalls.push({ worldId, colliderId, material: { ...material } });
		super.setColliderMaterial(worldId, colliderId, material);
	}
}

function assertAlmostEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${actual} to be close to ${expected}`,
	);
}

function createPhysics() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});
	return { adapter, physics };
}

function addSphere(physics, body, radius = 1) {
	return physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius },
	});
}

function testReadBackApi() {
	const { physics } = createPhysics();
	const node = new Node({ position: { x: 1, y: 0, z: 0 } });
	const body = physics.attachBody(node, {
		worldId: "main",
		body: {
			type: "dynamic",
			linearDamping: 0,
			angularDamping: 0,
			linearVelocity: { x: 2, y: 0, z: 0 },
			angularVelocity: { x: 0, y: 3, z: 0 },
		},
		authority: "physics",
	});
	addSphere(physics, body);

	assert.deepEqual(physics.getLinearVelocity(body), { x: 2, y: 0, z: 0 });
	assert.deepEqual(physics.getAngularVelocity(body.id), { x: 0, y: 3, z: 0 });
	assert.equal(physics.isSleeping(body), false);
	assert.deepEqual(physics.getBodyStats("main"), {
		bodyCount: 1,
		activeBodies: 1,
		sleepingBodies: 0,
		ccdBodies: 0,
	});

	physics.setLinearVelocity(body, { x: 4, y: 0, z: 0 });
	physics.setAngularVelocity(body, { x: 0, y: 5, z: 0 });
	assert.deepEqual(physics.getLinearVelocity(body), { x: 4, y: 0, z: 0 });
	assert.deepEqual(physics.getAngularVelocity(body), { x: 0, y: 5, z: 0 });

	physics.step(0.05);
	const transform = physics.getBodyTransform(body);
	assert.ok(transform);
	assertAlmostEqual(transform.position.x, 1.2);
	assert.deepEqual(physics.getLinearVelocity(body), { x: 4, y: 0, z: 0 });

	physics.setLinearVelocity(body, { x: 0, y: 0, z: 0 });
	physics.setAngularVelocity(body, { x: 0, y: 0, z: 0 });
	physics.step(0.016);
	assert.equal(physics.isSleeping(body), true);
	assert.equal(physics.getBodyStats("main").sleepingBodies, 1);
}

function testDestroyJointAndControllerApi() {
	const { adapter, physics } = createPhysics();
	const a = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "dynamic" },
	});
	const b = physics.attachBody(new Node({ position: { x: 1, y: 0, z: 0 } }), {
		worldId: "main",
		body: { type: "fixed" },
	});
	const joint = physics.createJoint({
		worldId: "main",
		type: "fixed",
		bodyA: a,
		bodyB: b,
	});
	physics.destroyJoint(joint);
	physics.destroyJoint(joint);
	assert.equal(adapter.destroyJointCalls.length, 1);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);

	const controller = physics.createCharacterController({
		worldId: "main",
		body: b,
		radius: 0.3,
		height: 1.7,
	});
	physics.destroyCharacterController(controller);
	physics.destroyCharacterController(controller);
	assert.equal(adapter.destroyControllerCalls.length, 1);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);
}

function testColliderRuntimeApi() {
	const { adapter, physics } = createPhysics();
	const left = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "dynamic" },
	});
	const right = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
	});
	const leftCollider = addSphere(physics, left);
	const removableCollider = addSphere(physics, left, 0.25);
	const rightCollider = addSphere(physics, right);

	physics.removeCollider(removableCollider);
	physics.removeCollider(removableCollider);
	assert.equal(adapter.destroyColliderCalls.length, 1);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);
	assert.equal(
		physics.raycast({
			worldId: "main",
			origin: { x: 0, y: 0, z: -3 },
			direction: { x: 0, y: 0, z: 1 },
			maxDistance: 10,
			filter: { includeColliderIds: [removableCollider.id] },
		}),
		null,
	);

	let report = physics.step(0.016);
	assert.ok(report.events.some((event) => event.type === "collisionBegin"));

	physics.setColliderSensor(rightCollider, true);
	report = physics.step(0.016);
	assert.equal(adapter.sensorCalls.length, 1);
	assert.equal(adapter.sensorCalls[0].isSensor, true);
	assert.ok(
		report.events.some(
			(event) => event.type === "triggerBegin" || event.type === "triggerStay",
		),
	);

	physics.setCollisionMask(leftCollider, 0);
	report = physics.step(0.016);
	assert.equal(adapter.maskCalls.length, 1);
	assert.equal(adapter.maskCalls[0].mask, 0);
	assert.ok(report.events.some((event) => event.type === "triggerEnd"));

	physics.setColliderFriction(rightCollider, 0.4);
	physics.setColliderRestitution(rightCollider, 0.8);
	assert.deepEqual(adapter.materialCalls, [
		{
			worldId: "main",
			colliderId: rightCollider.id,
			material: { friction: 0.4 },
		},
		{
			worldId: "main",
			colliderId: rightCollider.id,
			material: { restitution: 0.8 },
		},
	]);
}

function run() {
	testReadBackApi();
	testDestroyJointAndControllerApi();
	testColliderRuntimeApi();
	console.log("Physics runtime gap API tests passed");
}

run();
