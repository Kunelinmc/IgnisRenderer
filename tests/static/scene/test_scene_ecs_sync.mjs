import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { Node } from "../../../src/core/Node.ts";

function run() {
	const scene = new Scene();
	const parent = new Node({ name: "parent" });
	const child = new Node({ name: "child" });
	parent.addChild(child);
	scene.add(parent);
	scene.updateWorldMatrices();

	assert.ok(parent.entityId !== null);
	assert.ok(child.entityId !== null);

	const childEntity = child.entityId;
	const local = scene.ecs.getComponent(childEntity, "LocalTransform");
	assert.ok(local);
	assert.equal(local.positionX, 0);

	local.positionX = 12;
	scene.ecs.setComponent(childEntity, "LocalTransform", local);
	scene.syncECSToNode();
	assert.equal(child.position.x, 12);

	scene.remove(parent);
	assert.equal(scene.contains(parent), false);
	assert.equal(parent.entityId, null);

	console.log("Scene ECS sync tests passed");
}

run();
