import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";

function run() {
	const node = new Node({ name: "physics-node" });
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({ worldId: "main" });

	const handle = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	assert.equal(handle.node, node);

	physics.detachBody(handle);
	console.log("Physics node target tests passed");
}

run();
