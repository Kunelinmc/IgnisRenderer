import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";

function run() {
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const nodeA = new Node({ name: "a" });
	const nodeB = new Node({ name: "b", position: { x: 0, y: 0, z: 0.5 } });

	const bodyA = physics.attachBody(nodeA, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	const bodyB = physics.attachBody(nodeB, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});

	physics.addCollider(bodyA, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
		isTrigger: true,
	});
	physics.addCollider(bodyB, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
		isTrigger: true,
	});

	const callbackEvents = [];
	physics.on("triggerBegin", (event) => callbackEvents.push(event.type));
	physics.on("triggerStay", (event) => callbackEvents.push(event.type));
	physics.on("triggerEnd", (event) => callbackEvents.push(event.type));

	physics.step(0.016);
	physics.step(0.016);
	nodeB.position.z = 10;
	physics.step(0.016);

	const queued = physics.drainEvents("main");
	assert.ok(queued.length >= 3);
	assert.ok(callbackEvents.includes("triggerBegin"));
	assert.ok(callbackEvents.includes("triggerStay"));
	assert.ok(callbackEvents.includes("triggerEnd"));
	assert.equal(physics.drainEvents("main").length, 0);

	console.log("Physics event tests passed");
}

run();
