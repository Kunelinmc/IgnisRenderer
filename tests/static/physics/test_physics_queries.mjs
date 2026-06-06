import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";

function assertAlmostEqual(actual, expected, epsilon = 1e-6) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${actual} to be close to ${expected}`
	);
}

function run() {
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const sphereBody = physics.attachBody(
		new Node({ position: { x: 0, y: 0, z: 5 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);
	const boxBody = physics.attachBody(
		new Node({ position: { x: 0, y: 0, z: 8 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);
	const triggerBody = physics.attachBody(
		new Node({ position: { x: 0, y: 0, z: 2 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);
	const rotatedOverlapBody = physics.attachBody(
		new Node({ position: { x: 0.8, y: 0.8, z: 0 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);
	const rotatedCastBody = physics.attachBody(
		new Node({ position: { x: 0, y: 0.9, z: 6 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);

	const sphereCollider = physics.addCollider(sphereBody, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});
	const boxCollider = physics.addCollider(boxBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 1, y: 1, z: 1 } },
	});
	physics.addCollider(triggerBody, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 0.25 },
		isTrigger: true,
	});
	const rotatedOverlapCollider = physics.addCollider(rotatedOverlapBody, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 0.1 },
	});
	const rotatedCastCollider = physics.addCollider(rotatedCastBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 0.2, y: 0.2, z: 0.2 } },
	});

	const rayHit = physics.raycast({
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeTriggers: false },
	});
	assert.ok(rayHit);
	assert.equal(rayHit?.bodyId, sphereBody.id);
	assert.equal(rayHit?.colliderId, sphereCollider.id);
	assertAlmostEqual(rayHit.distance, 4);

	const sphereCastHit = physics.sphereCast({
		center: { x: 0, y: 0, z: 0 },
		radius: 0.5,
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeTriggers: false },
	});
	assert.ok(sphereCastHit);
	assert.equal(sphereCastHit?.bodyId, sphereBody.id);
	assertAlmostEqual(sphereCastHit.distance, 3.5);

	const boxCastHit = physics.boxCast({
		center: { x: 0, y: 0, z: 0 },
		halfExtents: { x: 0.5, y: 0.5, z: 0.5 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeBodyIds: [boxBody.id] },
	});
	assert.ok(boxCastHit);
	assert.equal(boxCastHit?.bodyId, boxBody.id);
	assertAlmostEqual(boxCastHit.distance, 6.5);

	const overlapSphereHits = physics.overlapSphere({
		center: { x: 0, y: 0, z: 5 },
		radius: 0.25,
		filter: { includeTriggers: false },
	});
	assert.equal(overlapSphereHits.length, 1);
	assert.equal(overlapSphereHits[0].bodyId, sphereBody.id);

	const overlapBoxHits = physics.overlapBox({
		center: { x: 0, y: 0, z: 8 },
		halfExtents: { x: 0.5, y: 0.5, z: 0.5 },
		filter: { includeBodyIds: [boxBody.id] },
	});
	assert.equal(overlapBoxHits.length, 1);
	assert.equal(overlapBoxHits[0].bodyId, boxBody.id);
	assert.equal(overlapBoxHits[0].colliderId, boxCollider.id);

	const overlapWithoutRotation = physics.overlapBox({
		center: { x: 0, y: 0, z: 0 },
		halfExtents: { x: 1, y: 0.2, z: 0.2 },
		filter: { includeBodyIds: [rotatedOverlapBody.id] },
	});
	assert.equal(overlapWithoutRotation.length, 0);

	const halfAngle = Math.PI / 8;
	const boxRotationZ45 = [0, 0, Math.sin(halfAngle), Math.cos(halfAngle)];
	const overlapWithRotation = physics.overlapBox({
		center: { x: 0, y: 0, z: 0 },
		halfExtents: { x: 1, y: 0.2, z: 0.2 },
		rotation: boxRotationZ45,
		filter: { includeBodyIds: [rotatedOverlapBody.id] },
	});
	assert.equal(overlapWithRotation.length, 1);
	assert.equal(overlapWithRotation[0].bodyId, rotatedOverlapBody.id);
	assert.equal(overlapWithRotation[0].colliderId, rotatedOverlapCollider.id);

	const boxCastWithoutRotation = physics.boxCast({
		center: { x: 0, y: 0, z: 0 },
		halfExtents: { x: 1, y: 0.2, z: 0.2 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeBodyIds: [rotatedCastBody.id] },
	});
	assert.equal(boxCastWithoutRotation, null);

	const boxCastWithRotation = physics.boxCast({
		center: { x: 0, y: 0, z: 0 },
		halfExtents: { x: 1, y: 0.2, z: 0.2 },
		rotation: boxRotationZ45,
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeBodyIds: [rotatedCastBody.id] },
	});
	assert.ok(boxCastWithRotation);
	assert.equal(boxCastWithRotation?.bodyId, rotatedCastBody.id);
	assert.equal(boxCastWithRotation?.colliderId, rotatedCastCollider.id);

	const inferredWorldHit = physics.raycast({
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeBodyIds: [sphereBody.id], includeTriggers: false },
	});
	assert.ok(inferredWorldHit);

	physics.createWorld({ worldId: "fx" });
	assert.throws(
		() =>
			physics.raycast({
				origin: { x: 0, y: 0, z: 0 },
				direction: { x: 1, y: 0, z: 0 },
				maxDistance: 10,
			}),
		/worldId is required/
	);

	physics.destroyWorld("fx");
	physics.destroyWorld("main");
	console.log("Physics query tests passed");
}

run();
