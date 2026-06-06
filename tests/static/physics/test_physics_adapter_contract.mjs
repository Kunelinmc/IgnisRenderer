import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";
import { RapierPhysicsAdapter } from "../../../src/physics/adapters/RapierPhysicsAdapter.ts";
import { AmmoPhysicsAdapter } from "../../../src/physics/adapters/AmmoPhysicsAdapter.ts";

import { createFakeRapierModule, createFakeAmmoModule } from "../../helpers/fakes.mjs";



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

async function runRapierCharacterControllerContract() {
	const fakeRapier = createFakeRapierModule();
	const adapter = new RapierPhysicsAdapter({
		moduleLoader: async () => fakeRapier.module,
		strict: true,
	});
	await adapter.init();

	adapter.createWorld({
		worldId: "kcc",
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: "fixed",
		fixedDeltaSeconds: 0.016,
	});
	adapter.createBody(
		"kcc",
		"player",
		{ type: "kinematic" },
		{
			position: { x: 0, y: 1, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);
	adapter.createCharacterController("kcc", "cc", {
		worldId: "kcc",
		body: "player",
		radius: 0.3,
		height: 1.7,
		stepHeight: 0.3,
		maxSlope: 50,
		jumpSpeed: 3,
	});
	adapter.moveCharacterController("kcc", "cc", { x: 1, y: 0, z: 0 }, 1);
	const stepResult = adapter.stepWorld("kcc", 1);
	const playerState = stepResult.bodyStates.find(
		(state) => state.bodyId === "player"
	);

	assert.ok(playerState, "Expected step state for kinematic player body");
	assert.equal(
		playerState?.transform.position.x,
		0.5,
		"Expected Rapier KCC computed movement to correct horizontal movement"
	);
	assert.equal(
		fakeRapier.stats.characterControllerCreates,
		1,
		"Expected Rapier world.createCharacterController() to be used"
	);
	assert.ok(
		fakeRapier.stats.characterComputeCalls > 0,
		"Expected Rapier KCC computeColliderMovement() to be used"
	);
	assert.equal(
		adapter.isCharacterControllerGrounded("kcc", "cc"),
		true,
		"Expected grounded state to come from Rapier KCC"
	);

	adapter.destroyWorld("kcc");
	console.log("Physics adapter contract passed: rapier-kcc");
}

async function runRapierBodyDescriptorContract() {
	const fakeRapier = createFakeRapierModule();
	const adapter = new RapierPhysicsAdapter({
		moduleLoader: async () => fakeRapier.module,
		strict: true,
	});
	await adapter.init();

	adapter.createWorld({
		worldId: "opts",
		gravity: { x: 0, y: -9.8, z: 0 },
		allowSleep: false,
	});
	adapter.createBody(
		"opts",
		"player-a",
		{ type: "dynamic", mass: 3 },
		{
			position: { x: 0, y: 1, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);
	adapter.createBody(
		"opts",
		"player-b",
		{ type: "dynamic", canSleep: true },
		{
			position: { x: 0, y: 2, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);

	const lastCanSleepCall =
		fakeRapier.stats.descriptorCanSleepCalls[
			fakeRapier.stats.descriptorCanSleepCalls.length - 1
		];
	assert.equal(lastCanSleepCall, true);
	assert.ok(
		fakeRapier.stats.descriptorCanSleepCalls.includes(false),
		"Expected allowSleep=false world default to call setCanSleep(false)"
	);
	assert.ok(
		fakeRapier.stats.descriptorAdditionalMassCalls.includes(3) ||
			fakeRapier.stats.bodyAdditionalMassCalls.includes(3),
		"Expected dynamic body mass to be forwarded to Rapier"
	);

	adapter.destroyWorld("opts");
	console.log("Physics adapter contract passed: rapier-body-descriptor");
}

async function runRapierCollisionFilterContract() {
	const fakeRapier = createFakeRapierModule();
	const adapter = new RapierPhysicsAdapter({
		moduleLoader: async () => fakeRapier.module,
		strict: true,
	});
	await adapter.init();

	adapter.createWorld({
		worldId: "filter",
		gravity: { x: 0, y: -9.8, z: 0 },
	});
	adapter.createBody(
		"filter",
		"body",
		{ type: "dynamic" },
		{
			position: { x: 0, y: 1, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);
	adapter.addCollider(
		"filter",
		"body",
		"collider",
		{ mode: "explicit", shape: { kind: "sphere", radius: 1 } },
		{ kind: "sphere", radius: 1 }
	);
	adapter.setCollisionMask("filter", "collider", 0x00020001);

	assert.ok(
		fakeRapier.stats.collisionGroupUpdates.includes(0x00020001),
		"Expected Rapier setCollisionGroups() to receive encoded group/filter mask"
	);
	adapter.destroyWorld("filter");
	console.log("Physics adapter contract passed: rapier-collision-filter");
}

async function runAmmoCollisionFilterContract() {
	const fakeAmmo = createFakeAmmoModule();
	const adapter = new AmmoPhysicsAdapter({
		moduleLoader: async () => fakeAmmo.module,
		strict: true,
	});
	await adapter.init();

	adapter.createWorld({
		worldId: "filter",
		gravity: { x: 0, y: -9.8, z: 0 },
	});
	adapter.createBody(
		"filter",
		"body",
		{ type: "dynamic" },
		{
			position: { x: 0, y: 1, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);
	adapter.addCollider(
		"filter",
		"body",
		"collider",
		{ mode: "explicit", shape: { kind: "sphere", radius: 1 } },
		{ kind: "sphere", radius: 1 }
	);
	adapter.setCollisionMask("filter", "collider", 0x00040002);

	assert.ok(
		fakeAmmo.stats.addRigidBodyCalls.some(
			(call) => call.length >= 3 && call[1] === 0x0004 && call[2] === 0x0002
		),
		"Expected Ammo addRigidBody(body, group, mask) to be called with updated filter"
	);
	adapter.destroyWorld("filter");
	console.log("Physics adapter contract passed: ammo-collision-filter");
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
	await runRapierCharacterControllerContract();
	await runRapierBodyDescriptorContract();
	await runRapierCollisionFilterContract();
	await runAmmoCollisionFilterContract();
}

await run();
