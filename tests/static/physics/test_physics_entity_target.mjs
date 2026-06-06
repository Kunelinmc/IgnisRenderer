import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";

function run() {
	const scene = new Scene();
	const node = new Node({ name: "physics-node" });
	scene.add(node);
	scene.updateWorldMatrices();

	const entityId = node.entityId;
	assert.ok(entityId !== null);

	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({ worldId: "main" });
	physics.setEntityNodeResolver((id) => scene.ecs.getNodeByEntity(id));

	const handle = physics.attachBody(entityId, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	assert.equal(handle.entityId, entityId);
	assert.equal(handle.node, node);

	physics.detachBody(handle);
	console.log("Physics entity target tests passed");
}

run();
