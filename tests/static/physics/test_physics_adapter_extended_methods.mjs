import assert from "node:assert/strict";
import { SimplePhysicsAdapter } from "../../../src/physics/adapters/SimplePhysicsAdapter.ts";

function assertAlmostEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${actual} to be close to ${expected}`
	);
}

function findBodyState(stepResult, bodyId) {
	const state = stepResult.bodyStates.find((item) => item.bodyId === bodyId);
	assert.ok(state, `Missing body state for "${bodyId}"`);
	return state;
}

function createTransform(x = 0, y = 0, z = 0) {
	return {
		position: { x, y, z },
		rotation: [0, 0, 0, 1],
	};
}

function testForcesAndVelocities() {
	const adapter = new SimplePhysicsAdapter("extended-forces");
	adapter.initSync();
	adapter.createWorld({
		worldId: "forces",
		gravity: { x: 0, y: 0, z: 0 },
	});
	adapter.createBody(
		"forces",
		"body",
		{
			type: "dynamic",
			mass: 2,
		},
		createTransform()
	);

	adapter.applyImpulse("forces", "body", { x: 2, y: 0, z: 0 });
	let step = adapter.stepWorld("forces", 1);
	let state = findBodyState(step, "body");
	assertAlmostEqual(state.transform.position.x, 1);

	adapter.applyForce("forces", "body", { x: 2, y: 0, z: 0 });
	step = adapter.stepWorld("forces", 1);
	state = findBodyState(step, "body");
	assertAlmostEqual(state.transform.position.x, 3);

	adapter.setAngularVelocity("forces", "body", { x: 0, y: 0, z: 0 });
	adapter.applyTorque("forces", "body", { x: 0, y: 1, z: 0 });
	step = adapter.stepWorld("forces", 1);
	state = findBodyState(step, "body");
	assert.equal(state.sleeping, false);

	adapter.destroyWorld("forces");
}

function testColliderSensorAndMask() {
	const adapter = new SimplePhysicsAdapter("extended-collider");
	adapter.initSync();
	adapter.createWorld({
		worldId: "mask",
		gravity: { x: 0, y: 0, z: 0 },
	});
	adapter.createBody("mask", "a", { type: "dynamic" }, createTransform(0, 0, 0));
	adapter.createBody("mask", "b", { type: "fixed" }, createTransform(0, 0, 0));
	adapter.addCollider(
		"mask",
		"a",
		"collider-a",
		{ mode: "explicit", shape: { kind: "sphere", radius: 1 } },
		{ kind: "sphere", radius: 1 }
	);
	adapter.addCollider(
		"mask",
		"b",
		"collider-b",
		{ mode: "explicit", shape: { kind: "sphere", radius: 1 } },
		{ kind: "sphere", radius: 1 }
	);

	let step = adapter.stepWorld("mask", 0.016);
	assert.ok(step.events.some((event) => event.type === "collisionBegin"));

	adapter.setColliderSensor("mask", "collider-b", true);
	step = adapter.stepWorld("mask", 0.016);
	assert.ok(
		step.events.some((event) => event.type === "triggerBegin" || event.type === "triggerStay")
	);

	adapter.setCollisionMask("mask", "collider-a", 0);
	step = adapter.stepWorld("mask", 0.016);
	assert.ok(step.events.some((event) => event.type === "triggerEnd"));

	adapter.destroyWorld("mask");
}

function testRaycastAll() {
	const adapter = new SimplePhysicsAdapter("extended-ray");
	adapter.initSync();
	adapter.createWorld({
		worldId: "ray",
		gravity: { x: 0, y: 0, z: 0 },
	});

	for (const [index, z] of [3, 6, 9].entries()) {
		const bodyId = `body-${index}`;
		const colliderId = `collider-${index}`;
		adapter.createBody("ray", bodyId, { type: "fixed" }, createTransform(0, 0, z));
		adapter.addCollider(
			"ray",
			bodyId,
			colliderId,
			{ mode: "explicit", shape: { kind: "sphere", radius: 1 } },
			{ kind: "sphere", radius: 1 }
		);
	}

	const hits = adapter.raycastAll("ray", {
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
	});
	assert.equal(hits.length, 3);
	assert.ok(hits[0].distance < hits[1].distance);
	assert.ok(hits[1].distance < hits[2].distance);

	const limitedHits = adapter.raycastAll("ray", {
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		maxHits: 2,
	});
	assert.equal(limitedHits.length, 2);

	adapter.destroyWorld("ray");
}

function testBodyOptionsAndControllerGravityScale() {
	const adapter = new SimplePhysicsAdapter("extended-options");
	adapter.initSync();
	adapter.createWorld({
		worldId: "options",
		gravity: { x: 0, y: 0, z: 0 },
		allowSleep: false,
	});
	adapter.createBody(
		"options",
		"locked",
		{
			type: "dynamic",
			linearVelocity: { x: 1, y: 2, z: 0 },
			linearDamping: 0.5,
			lockTranslations: [false, true, false],
		},
		createTransform()
	);
	adapter.createBody(
		"options",
		"awake",
		{
			type: "dynamic",
		},
		createTransform()
	);
	adapter.createBody(
		"options",
		"sleepy",
		{
			type: "dynamic",
			canSleep: true,
		},
		createTransform()
	);

	let step = adapter.stepWorld("options", 1);
	const lockedState = findBodyState(step, "locked");
	assertAlmostEqual(lockedState.transform.position.x, 0.5);
	assertAlmostEqual(lockedState.transform.position.y, 0);
	assert.equal(findBodyState(step, "awake").sleeping, false);
	assert.equal(findBodyState(step, "sleepy").sleeping, true);
	adapter.destroyWorld("options");

	adapter.createWorld({
		worldId: "kcc",
		gravity: { x: 0, y: -10, z: 0 },
	});
	adapter.createBody("kcc", "player", { type: "kinematic" }, createTransform(0, 1, 0));
	adapter.createCharacterController("kcc", "controller", {
		worldId: "kcc",
		body: "player",
		radius: 0.3,
		height: 1.7,
		gravityScale: 0.5,
	});
	adapter.moveCharacterController("kcc", "controller", { x: 0, y: 0, z: 0 }, 1);
	step = adapter.stepWorld("kcc", 1);
	assertAlmostEqual(findBodyState(step, "player").transform.position.y, -4);
	adapter.destroyWorld("kcc");
}

function run() {
	testForcesAndVelocities();
	testColliderSensorAndMask();
	testRaycastAll();
	testBodyOptionsAndControllerGravityScale();
	console.log("Physics adapter extended method tests passed");
}

run();
