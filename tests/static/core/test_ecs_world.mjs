import assert from "node:assert/strict";
import { ECSWorld } from "../../../src/ecs/ECSWorld.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Node } from "../../../src/core/Node.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { ParticleSystem } from "../../../src/particles/ParticleSystem.ts";

function run() {
	const scene = new Scene();
	const world = new ECSWorld(scene);
	const entity = world.createEntity("entity:root");
	world.setComponent(entity, "Name", { value: "root" });
	world.setComponent(entity, "Visibility", { visible: true });

	const query = world.query(["Name", "Visibility"]);
	assert.ok(query.includes(entity));
	assert.ok(query.includes(world.getEntityByNode(scene.root)));
	assert.equal(world.getExternalId(entity), "entity:root");
	assert.equal(world.getEntityByExternalId("entity:root"), entity);

	const node = new Node({ name: "nodeA" });
	scene.add(node);
	const nodeEntity = world.getEntityByNode(node);
	assert.equal(world.getEntityByNode(node), nodeEntity);
	assert.equal(world.getNodeByEntity(nodeEntity), node);

	scene.remove(node);
	assert.equal(world.getEntityByNode(node), null);

	const meshInstance = new MeshInstance({
		mesh: new MeshAsset(),
	});
	const light = new AmbientLight();
	const camera = new Camera();
	const particleSystem = new ParticleSystem();
	const plainNode = new Node({ name: "plainNode" });

	scene.add(meshInstance);
	scene.add(light);
	scene.add(camera);
	scene.add(particleSystem);
	scene.add(plainNode);

	assert.deepEqual(world.findMeshInstances(), [meshInstance]);
	assert.deepEqual(world.findLights(), [light]);
	assert.deepEqual(world.findCameras(), [camera]);
	assert.deepEqual(world.findParticleSystems(), [particleSystem]);

	world.destroy();
	assert.equal(world.getEntityByNode(meshInstance), null);

}

function testExternalIdTransfersKeepBothIndexesConsistent() {
	const world = new ECSWorld(new Scene());
	const first = world.createEntity("shared");
	const second = world.createEntity("shared");
	assert.equal(world.getExternalId(first), null);
	assert.equal(world.getEntityByExternalId("shared"), second);
	world.destroyEntity(first);
	assert.equal(world.getEntityByExternalId("shared"), second);

	const third = world.createEntity("third");
	world.setExternalId(third, "shared");
	assert.equal(world.getExternalId(second), null);
	assert.equal(world.getEntityByExternalId("third"), null);
	world.setExternalId(second, "second");
	assert.equal(world.getEntityByExternalId("shared"), third);
	world.destroyEntity(third);
	assert.equal(world.getEntityByExternalId("shared"), null);
	assert.equal(world.getEntityByExternalId("second"), second);
	world.destroy();
}

function testEmptyExternalIdsAreCreatedTransferredAndRemoved() {
	const world = new ECSWorld(new Scene());
	const first = world.createEntity("");
	assert.equal(world.getExternalId(first), "");
	assert.equal(world.getEntityByExternalId(""), first);
	const second = world.createEntity();
	world.setExternalId(second, "");
	assert.equal(world.getExternalId(first), null);
	world.setExternalId(first, "first");
	assert.equal(world.getEntityByExternalId(""), second);
	world.setExternalId(second, "second");
	assert.equal(world.getEntityByExternalId(""), null);
	world.setExternalId(second, "");
	world.destroyEntity(second);
	assert.equal(world.getEntityByExternalId(""), null);
	world.destroy();
}

function testProjectionTransfersConflictingExternalId() {
	const scene = new Scene();
	const world = new ECSWorld(scene);
	const node = new Node();
	const standalone = world.createEntity(node.id);
	scene.add(node);
	const projected = world.getEntityByNode(node);
	assert.ok(projected !== null);
	assert.equal(world.getExternalId(standalone), null);
	world.destroyEntity(standalone);
	assert.equal(world.getEntityByExternalId(node.id), projected);
	scene.remove(node);
	assert.equal(world.getEntityByExternalId(node.id), null);
	world.destroy();
}

for (const check of [
	run,
	testExternalIdTransfersKeepBothIndexesConsistent,
	testEmptyExternalIdsAreCreatedTransferredAndRemoved,
	testProjectionTransfersConflictingExternalId,
]) {
	try {
		check();
	} catch (error) {
		console.error(check.name, error);
		process.exitCode = 1;
	}
}

if (!process.exitCode) console.log("ECS world tests passed");
