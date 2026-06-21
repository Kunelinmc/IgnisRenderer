import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { PhysicsBodyNode } from "../../../src/physics/PhysicsBodyNode.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../../../src/physics/adapters/SimplePhysicsAdapter.ts";

class TrackingPhysicsAdapter extends SimplePhysicsAdapter {
	constructor() {
		super("tracking-gap-extensions");
		this.colliderFilterCalls = [];
		this.bodyTypeCalls = [];
		this.bodyMassCalls = [];
		this.bodyGravityScaleCalls = [];
		this.bodyLinearDampingCalls = [];
		this.bodyAngularDampingCalls = [];
		this.wakeUpCalls = [];
	}

	setColliderCollisionFilter(worldId, colliderId, filter) {
		this.colliderFilterCalls.push({ worldId, colliderId, filter });
		super.setColliderCollisionFilter(worldId, colliderId, filter);
	}

	setBodyType(worldId, bodyId, type) {
		this.bodyTypeCalls.push({ worldId, bodyId, type });
		super.setBodyType(worldId, bodyId, type);
	}

	setBodyMass(worldId, bodyId, mass) {
		this.bodyMassCalls.push({ worldId, bodyId, mass });
		super.setBodyMass(worldId, bodyId, mass);
	}

	setBodyGravityScale(worldId, bodyId, scale) {
		this.bodyGravityScaleCalls.push({ worldId, bodyId, scale });
		super.setBodyGravityScale(worldId, bodyId, scale);
	}

	setBodyLinearDamping(worldId, bodyId, value) {
		this.bodyLinearDampingCalls.push({ worldId, bodyId, value });
		super.setBodyLinearDamping(worldId, bodyId, value);
	}

	setBodyAngularDamping(worldId, bodyId, value) {
		this.bodyAngularDampingCalls.push({ worldId, bodyId, value });
		super.setBodyAngularDamping(worldId, bodyId, value);
	}

	wakeUpBody(worldId, bodyId) {
		this.wakeUpCalls.push({ worldId, bodyId });
		super.wakeUpBody(worldId, bodyId);
	}
}

function createPhysics(gravity = { x: 0, y: 0, z: 0 }) {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity,
		mode: "variable",
	});
	return { adapter, physics };
}

function addSphere(physics, body, collision) {
	return physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
		collision,
	});
}

function testNamedCollisionLayers() {
	const { adapter, physics } = createPhysics();
	physics.defineCollisionLayer("player", 1);
	physics.defineCollisionLayer("enemy", 2);

	assert.throws(
		() => physics.defineCollisionLayer("player", 3),
		/already exists/,
	);
	assert.throws(
		() => physics.defineCollisionLayer("invalid", 16),
		/between 0 and 15/,
	);

	const player = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "dynamic" },
	});
	const enemy = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
	});

	const playerCollider = addSphere(physics, player, {
		groups: ["player"],
		collidesWith: ["enemy"],
	});
	addSphere(physics, enemy, {
		groups: ["enemy"],
		collidesWith: ["player"],
	});

	assert.equal(adapter.colliderFilterCalls[0].filter, 0x00020004);
	assert.equal(adapter.colliderFilterCalls[1].filter, 0x00040002);

	let report = physics.step(0.016);
	assert.ok(report.events.some((event) => event.type === "collisionBegin"));

	physics.setColliderCollisionFilter(playerCollider, {
		groups: ["player"],
		collidesWith: [],
	});
	assert.equal(adapter.colliderFilterCalls.at(-1).filter, 0x00020000);
	report = physics.step(0.016);
	assert.ok(report.events.some((event) => event.type === "collisionEnd"));

	assert.throws(
		() =>
			physics.setColliderCollisionFilter(playerCollider, {
				groups: ["missing"],
				collidesWith: "all",
			}),
		/not defined/,
	);
}

function testRuntimeBodySettersPreserveHandles() {
	const { adapter, physics } = createPhysics({ x: 0, y: -10, z: 0 });
	const dynamic = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "dynamic", linearDamping: 0, angularDamping: 0 },
	});
	const fixed = physics.attachBody(new Node({ position: { x: 3, y: 0, z: 0 } }), {
		worldId: "main",
		body: { type: "fixed" },
	});
	const collider = addSphere(physics, dynamic);
	const joint = physics.createJoint({
		worldId: "main",
		type: "fixed",
		bodyA: dynamic,
		bodyB: fixed,
	});

	physics.setBodyGravityScale(dynamic, 0);
	physics.step(1);
	assert.equal(physics.getLinearVelocity(dynamic).y, 0);
	assert.equal(adapter.bodyGravityScaleCalls.length, 1);

	physics.setBodyMass(dynamic, 4);
	physics.setBodyLinearDamping(dynamic, 0.25);
	physics.setBodyAngularDamping(dynamic, 0.5);
	assert.equal(adapter.bodyMassCalls.at(-1).mass, 4);
	assert.equal(adapter.bodyLinearDampingCalls.at(-1).value, 0.25);
	assert.equal(adapter.bodyAngularDampingCalls.at(-1).value, 0.5);

	physics.setLinearVelocity(dynamic, { x: 0, y: 0, z: 0 });
	physics.setAngularVelocity(dynamic, { x: 0, y: 0, z: 0 });
	physics.step(0.016);
	assert.equal(physics.isSleeping(dynamic), true);
	physics.wakeUpBody(dynamic);
	assert.equal(adapter.wakeUpCalls.length, 1);
	assert.equal(physics.isSleeping(dynamic), false);

	physics.setBodyType(dynamic, "fixed");
	assert.equal(adapter.bodyTypeCalls.at(-1).type, "fixed");
	assert.equal(physics.getBodyStats("main").bodyCount, 2);
	physics.setColliderSensor(collider, true);
	physics.destroyJoint(joint);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);
}

function testSceneLifecycleBinding() {
	const { physics } = createPhysics();
	const scene = new Scene();
	const dispose = physics.bindSceneLifecycle(scene, {
		attachExisting: false,
		detachRemoved: true,
	});

	const node = new PhysicsBodyNode({
		bodyBinding: {
			worldId: "main",
			body: { type: "dynamic" },
			colliders: [
				{
					mode: "explicit",
					shape: { kind: "sphere", radius: 1 },
				},
			],
		},
	});
	scene.add(node);
	assert.equal(physics.getBodyStats("main").bodyCount, 1);

	const clone = node.clone();
	scene.add(clone);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);

	const parent = scene.add(new Node());
	parent.addChild(clone);
	assert.equal(physics.getBodyStats("main").bodyCount, 2);

	scene.remove(node);
	assert.equal(physics.getBodyStats("main").bodyCount, 1);

	dispose();
	const unbound = node.clone();
	scene.add(unbound);
	assert.equal(physics.getBodyStats("main").bodyCount, 1);
}

function run() {
	testNamedCollisionLayers();
	testRuntimeBodySettersPreserveHandles();
	testSceneLifecycleBinding();
	console.log("Physics GAP extension tests passed");
}

run();
