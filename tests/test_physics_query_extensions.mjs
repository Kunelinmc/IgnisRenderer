import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Node } from "../src/core/Node.ts";
import { PhysicsSystem } from "../src/physics/PhysicsSystem.ts";

function run() {
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const nodeA = new Node({ position: { x: 0, y: 0, z: 5 } });
	const nodeB = new Node({ position: { x: 0, y: 0, z: 8 } });
	const bodyA = physics.attachBody(nodeA, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const bodyB = physics.attachBody(nodeB, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(bodyA, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});
	physics.addCollider(bodyB, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	const hits = physics.raycastAll({
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeTriggers: false },
	});
	assert.equal(hits.length, 2);
	assert.ok(hits[0].distance <= hits[1].distance);
	assert.equal(physics.resolveHitNode(hits[0]), nodeA);
	assert.equal(physics.resolveHitNode(hits[1]), nodeB);

	const scene = new Scene();
	const entityNode = new Node({ position: { x: 2, y: 0, z: 5 } });
	scene.add(entityNode);
	scene.updateWorldMatrices();
	const entityId = entityNode.entityId;
	assert.ok(typeof entityId === "number");
	physics.setEntityNodeResolver((id) => scene.ecs.getNodeByEntity(id));
	const entityBody = physics.attachBody(entityId, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(entityBody, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	const entityHit = physics.raycast({
		origin: { x: 2, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
		filter: { includeTriggers: false },
	});
	assert.ok(entityHit);
	assert.equal(physics.resolveHitEntityId(entityHit), entityId);

	console.log("Physics query extension tests passed");
}

run();
