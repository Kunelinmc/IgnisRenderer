import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsBodyNode } from "../../../src/physics/PhysicsBodyNode.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";

function run() {
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({ worldId: "main" });
	physics.createWorld({ worldId: "fx" });

	const node = new Node({ name: "bodyA" });
	const handle = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	assert.equal(handle.worldId, "main");

	assert.throws(
		() =>
			physics.attachBody(node, {
				worldId: "main",
				body: { type: "dynamic" },
			}),
		/already bound/
	);

	const bodyNode = new PhysicsBodyNode({
		name: "bodyNode",
		bodyBinding: {
			worldId: "fx",
			body: { type: "kinematic" },
			authority: "animation",
		},
	});
	const bodyNodeHandle = physics.attachBody(bodyNode);
	assert.equal(bodyNodeHandle.worldId, "fx");

	assert.throws(
		() =>
			physics.attachBody(new Node(), {
				worldId: "main",
				body: { type: "dynamic" },
				authority: "animation",
			}),
		/does not allow dynamic rigid bodies/
	);

	physics.detachBody(handle);
	physics.destroyWorld("main");
	physics.destroyWorld("fx");

	console.log("Physics system binding tests passed");
}

run();
