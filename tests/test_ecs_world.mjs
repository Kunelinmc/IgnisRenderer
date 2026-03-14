import assert from "node:assert/strict";
import { ECSWorld } from "../src/ecs/ECSWorld.ts";
import { Node } from "../src/core/Node.ts";

function run() {
	const world = new ECSWorld();
	const entity = world.createEntity("entity:root");
	world.setComponent(entity, "Name", { value: "root" });
	world.setComponent(entity, "Visibility", { visible: true });

	const query = world.query(["Name", "Visibility"]);
	assert.deepEqual(query, [entity]);
	assert.equal(world.getExternalId(entity), "entity:root");
	assert.equal(world.getEntityByExternalId("entity:root"), entity);

	const node = new Node({ name: "nodeA" });
	const nodeEntity = world.registerNode(node, null);
	assert.equal(world.getEntityByNode(node), nodeEntity);
	assert.equal(world.getNodeByEntity(nodeEntity), node);

	world.unregisterNode(node);
	assert.equal(world.getEntityByNode(node), null);

	console.log("ECS world tests passed");
}

run();
