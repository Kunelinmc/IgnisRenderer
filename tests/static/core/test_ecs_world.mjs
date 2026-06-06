import assert from "node:assert/strict";
import { ECSWorld } from "../../../src/ecs/ECSWorld.ts";
import { Node } from "../../../src/core/Node.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { ParticleSystem } from "../../../src/particles/ParticleSystem.ts";

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

	const meshInstance = new MeshInstance({
		mesh: new MeshAsset(),
	});
	const light = new AmbientLight();
	const camera = new Camera();
	const particleSystem = new ParticleSystem();
	const plainNode = new Node({ name: "plainNode" });

	world.registerNode(meshInstance, null);
	world.registerNode(light, null);
	world.registerNode(camera, null);
	world.registerNode(particleSystem, null);
	world.registerNode(plainNode, null);

	assert.deepEqual(world.findMeshInstances(), [meshInstance]);
	assert.deepEqual(world.findLights(), [light]);
	assert.deepEqual(world.findCameras(), [camera]);
	assert.deepEqual(world.findParticleSystems(), [particleSystem]);

	console.log("ECS world tests passed");
}

run();
