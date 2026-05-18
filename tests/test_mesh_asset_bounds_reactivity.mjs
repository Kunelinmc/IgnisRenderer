import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";

function createTrianglePrimitive(offsetX = 0, size = 1) {
	return MeshAsset.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: offsetX, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: offsetX + size, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: offsetX, y: size, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]).primitives[0];
}

function testArrayMutationRefreshesMeshBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	assert.equal(mesh.boundingBox.max.x, 1);

	mesh.primitives.push(createTrianglePrimitive(10, 2));

	assert.equal(mesh.boundingBox.min.x, 0);
	assert.equal(mesh.boundingBox.max.x, 12);
	assert.ok(mesh.boundingSphere.radius > 5);
}

function testPrimitiveReplacementRefreshesMeshBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	mesh.primitives = [createTrianglePrimitive(-4, 2)];

	assert.equal(mesh.boundingBox.min.x, -4);
	assert.equal(mesh.boundingBox.max.x, -2);
}

function testConstructorArrayMutationRefreshesMeshBounds() {
	const primitives = [createTrianglePrimitive(0, 1)];
	const mesh = new MeshAsset(primitives);

	primitives.push(createTrianglePrimitive(20, 3));

	assert.equal(mesh.boundingBox.max.x, 23);
}

function testGeometryVersionRefreshesPrimitiveBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const primitive = mesh.primitives[0];

	primitive.geometry.positions[3] = 6;
	primitive.geometryVersion += 1;

	assert.equal(mesh.boundingBox.max.x, 6);
	assert.equal(primitive.boundingBox.max.x, 6);
}

function testMeshInstanceUsesReactiveBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const instance = new MeshInstance({ mesh });
	instance.updateWorldMatrix();

	mesh.primitives.push(createTrianglePrimitive(4, 1));

	assert.equal(instance.getWorldBoundingBox().max.x, 5);
}

function testSceneBoundsCacheUsesReactiveMeshBounds() {
	const scene = new Scene();
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const instance = scene.add(new MeshInstance({ mesh }));
	scene.updateWorldMatrices();

	const initialBounds = scene.getBounds();
	mesh.primitives.push(createTrianglePrimitive(10, 2));
	const updatedBounds = scene.getBounds();

	assert.ok(updatedBounds.radius > initialBounds.radius);
	assert.ok(instance.getWorldBoundingBox().max.x > 10);
}

function testSpatialIndexUsesReactiveMeshBounds() {
	const scene = new Scene();
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const instance = scene.add(new MeshInstance({ mesh }));
	scene.updateWorldMatrices();

	const spatial = scene.rebuildSpatialIndex(scene.getMeshInstances());
	const farBounds = {
		min: { x: 4.5, y: -1, z: -1 },
		max: { x: 5.5, y: 1, z: 1 },
	};
	assert.equal(spatial.queryBounds(farBounds).includes(instance), false);

	mesh.primitives.push(createTrianglePrimitive(4.5, 1));
	scene.rebuildSpatialIndex(scene.getMeshInstances());

	assert.equal(spatial.queryBounds(farBounds).includes(instance), true);
}

function run() {
	testArrayMutationRefreshesMeshBounds();
	testPrimitiveReplacementRefreshesMeshBounds();
	testConstructorArrayMutationRefreshesMeshBounds();
	testGeometryVersionRefreshesPrimitiveBounds();
	testMeshInstanceUsesReactiveBounds();
	testSceneBoundsCacheUsesReactiveMeshBounds();
	testSpatialIndexUsesReactiveMeshBounds();
	console.log("MeshAsset bounds reactivity tests passed");
}

run();
