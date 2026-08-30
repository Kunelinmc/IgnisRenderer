import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { ECSWorld } from "../../../src/ecs/ECSWorld.ts";

function run() {
	const scene = new Scene();
	const parent = new Node({ name: "parent" });
	const child = new Node({ name: "child" });
	parent.addChild(child);
	scene.add(parent);
	scene.updateWorldMatrices();

	assert.equal("ecs" in scene, false);
	assert.equal("entityId" in child, false);

	const world = new ECSWorld(scene);
	const secondWorld = new ECSWorld(scene);
	const parentEntity = world.getEntityByNode(parent);
	const childEntity = world.getEntityByNode(child);
	assert.ok(parentEntity !== null);
	assert.ok(childEntity !== null);
	assert.ok(secondWorld.getEntityByNode(child) !== null);
	assert.equal(world.getNodeByEntity(childEntity), child);

	const local = world.getComponent(childEntity, "LocalTransform");
	const worldTransform = world.getComponent(childEntity, "WorldTransform");
	assert.ok(local);
	assert.ok(worldTransform);
	local.positionX = 12;
	world.setComponent(childEntity, "LocalTransform", local);
	assert.equal(child.position.x, 12);

	scene.updateWorldMatrices();
	assert.equal(world.getComponent(childEntity, "LocalTransform"), local);
	assert.equal(world.getComponent(childEntity, "WorldTransform"), worldTransform);
	const secondChildEntity = secondWorld.getEntityByNode(child);
	assert.equal(
		secondWorld.getComponent(secondChildEntity, "LocalTransform").positionX,
		12,
	);
	const versionAfterChange = world.version;
	scene.updateWorldMatrices();
	assert.equal(world.version, versionAfterChange);

	child.visible = false;
	child.name = "renamed";
	assert.equal(world.getComponent(childEntity, "Visibility").visible, false);
	assert.equal(world.getComponent(childEntity, "Name").value, "renamed");
	assert.match(world.getComponent(childEntity, "PathBinding").path, /renamed/);

	scene.add(child);
	assert.equal(world.getEntityByNode(child), childEntity);
	assert.equal(
		world.getComponent(childEntity, "Hierarchy").parent,
		world.getEntityByNode(scene.root),
	);
	assert.deepEqual(world.getComponent(parentEntity, "Hierarchy").children, []);

	assert.throws(
		() => world.setComponent(childEntity, "WorldTransform", worldTransform),
		/read-only/,
	);
	assert.throws(() => world.removeComponent(childEntity, "Name"), /required/);
	assert.throws(() => world.destroyEntity(childEntity), /scene-backed/);

	world.destroy();
	scene.remove(child);
	assert.equal(world.getEntityByNode(child), null);
	assert.equal(secondWorld.getEntityByNode(child), null);
	secondWorld.destroy();

	console.log("Scene ECS projection tests passed");
}

run();
