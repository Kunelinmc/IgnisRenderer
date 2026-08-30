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

	console.log("ECS world tests passed");
}

run();
