import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { ParticleSystem, ParticleSpaceMode } from "../../../src/particles/index.ts";
import { PhysicsBodyNode } from "../../../src/physics/PhysicsBodyNode.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { OrbitCamera } from "../../../src/cameras/OrbitCamera.ts";

function createTriangleMesh() {
	return MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: -1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
}

function testNodeHierarchyClone() {
	const root = new Node({ name: "root", visible: false });
	root.position.copy({ x: 2, y: 3, z: 4 });
	const child = new Node({ name: "child" });
	root.addChild(child);

	const clonedRoot = root.clone();

	assert.ok(clonedRoot instanceof Node);
	assert.notEqual(clonedRoot, root);
	assert.notEqual(clonedRoot.id, root.id);
	assert.equal(clonedRoot.name, "root");
	assert.equal(clonedRoot.visible, false);
	assert.equal(clonedRoot.children.length, 1);
	assert.notEqual(clonedRoot.children[0], child);
	assert.equal(clonedRoot.children[0].parent, clonedRoot);
	assert.equal(clonedRoot.children[0].name, "child");
}

function testMeshInstanceClone() {
	const mesh = createTriangleMesh();
	const meshInstance = new MeshInstance({
		mesh,
		morphWeights: [new Float32Array([0.25, 0.75])],
	});
	meshInstance.position.copy({ x: 9, y: 8, z: 7 });

	const cloned = meshInstance.clone();

	assert.ok(cloned instanceof MeshInstance);
	assert.notEqual(cloned.id, meshInstance.id);
	assert.equal(cloned.mesh, meshInstance.mesh);
	assert.notEqual(cloned.morphWeights[0], meshInstance.morphWeights[0]);
	assert.equal(cloned.morphWeights[0][0], meshInstance.morphWeights[0][0]);

	cloned.morphWeights[0][0] = 0.9;
	assert.equal(meshInstance.morphWeights[0][0], 0.25);
}

function testSpecializedNodeClone() {
	const particle = new ParticleSystem({
		space: ParticleSpaceMode.World,
		maxParticles: 123,
		castShadows: false,
		shadowDensity: 3.5,
		shadowSoftness: 0.25,
		emit: {
			rate: 42,
			direction: { x: 1, y: 0.5, z: -1 },
			bursts: [{ time: 0.25, count: 9 }],
		},
		colliders: [
			{
				type: "sphere",
				center: { x: 1, y: 2, z: 3 },
				radius: 5,
			},
		],
	});

	const physicsNode = new PhysicsBodyNode({
		bodyBinding: {
			worldId: "mainWorld",
			body: {
				type: "dynamic",
				mass: 5,
			},
			colliders: [
				{
					mode: "explicit",
					shape: { kind: "sphere", radius: 2 },
				},
			],
		},
	});

	const light = new DirectionalLight({
		direction: { x: -1, y: -2, z: -3 },
		intensity: 2.5,
	});

	const camera = new OrbitCamera({ x: 5, y: 1, z: 2 }, 320);
	camera.theta = 0.75;
	camera.fov = 65;
	camera.updatePosition();

	const root = new Node({ name: "root-specialized" });
	root.addChild(particle);
	root.addChild(physicsNode);
	root.addChild(light);
	root.addChild(camera);

	const cloned = root.clone();

	const [clonedParticle, clonedPhysicsNode, clonedLight, clonedCamera] =
		cloned.children;

	assert.ok(clonedParticle instanceof ParticleSystem);
	assert.equal(clonedParticle.maxParticles, 123);
	assert.equal(clonedParticle.castShadows, false);
	assert.equal(clonedParticle.shadowDensity, 3.5);
	assert.equal(clonedParticle.shadowSoftness, 0.25);
	assert.notEqual(clonedParticle.emit, particle.emit);
	assert.notEqual(clonedParticle.emit.direction, particle.emit.direction);
	assert.deepEqual(clonedParticle.emit.direction, particle.emit.direction);
	assert.notEqual(clonedParticle.colliders[0], particle.colliders[0]);
	assert.deepEqual(clonedParticle.colliders[0], particle.colliders[0]);

	assert.ok(clonedPhysicsNode instanceof PhysicsBodyNode);
	assert.notEqual(clonedPhysicsNode.bodyBinding, physicsNode.bodyBinding);
	assert.deepEqual(clonedPhysicsNode.bodyBinding, physicsNode.bodyBinding);
	assert.notEqual(
		clonedPhysicsNode.bodyBinding.colliders[0],
		physicsNode.bodyBinding.colliders[0]
	);

	assert.ok(clonedLight instanceof DirectionalLight);
	assert.deepEqual(clonedLight.direction, light.direction);
	assert.equal(clonedLight.intensity, light.intensity);

	assert.ok(clonedCamera instanceof OrbitCamera);
	assert.equal(clonedCamera.theta, camera.theta);
	assert.equal(clonedCamera.distance, camera.distance);
	assert.deepEqual(clonedCamera.target, camera.target);
}

function run() {
	testNodeHierarchyClone();
	testMeshInstanceClone();
	testSpecializedNodeClone();
	console.log("Node clone tests passed");
}

run();
