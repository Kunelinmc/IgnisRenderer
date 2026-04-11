import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { HybridSpatialIndex } from "../src/spatial/HybridSpatialIndex.ts";

function createTriangleMesh(material) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{
					x: -0.5,
					y: -0.5,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0.5,
					y: -0.5,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 0.5,
					z: 0,
					u: 0.5,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

function createInstance(mesh, x, y, z, dynamic = false) {
	const instance = new MeshInstance({
		mesh,
		skeleton: dynamic ? {} : null,
	});
	instance.position.set(x, y, z);
	instance.updateWorldMatrix();
	return instance;
}

function createCamera() {
	const camera = new Camera();
	camera.position.set(0, 0, 5);
	camera.updateMatrices();
	return camera;
}

function testFrustumAndDynamicUpdates() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "HybridFrustum" }));
	const staticMesh = createInstance(mesh, -0.8, 0, -2, false);
	const dynamicMesh = createInstance(mesh, 0.8, 0, -2, true);
	const culledDynamicMesh = createInstance(mesh, 0, 220, -2, true);

	const index = new HybridSpatialIndex([
		staticMesh,
		dynamicMesh,
		culledDynamicMesh,
	]);

	const firstHits = new Set(index.queryFrustum(camera.frustum));
	assert.equal(firstHits.has(staticMesh), true);
	assert.equal(firstHits.has(dynamicMesh), true);
	assert.equal(firstHits.has(culledDynamicMesh), false);

	dynamicMesh.position.y = 280;
	dynamicMesh.updateWorldMatrix();
	index.markDirty(dynamicMesh);

	const secondHits = new Set(index.queryFrustum(camera.frustum));
	assert.equal(secondHits.has(staticMesh), true);
	assert.equal(secondHits.has(dynamicMesh), false);
}

function testRayOrderingAcrossBuckets() {
	const mesh = createTriangleMesh(new Material({ name: "HybridRay" }));
	const nearStatic = createInstance(mesh, 0, 0, -3, false);
	const farDynamic = createInstance(mesh, 0, 0, -7, true);
	const index = new HybridSpatialIndex([nearStatic, farDynamic]);

	const hits = index.queryRayDetailed(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		{ maxDistance: 100 }
	);
	assert.ok(hits.length >= 2, `Expected at least 2 ray hits, got ${hits.length}`);
	assert.equal(hits[0].meshInstance, nearStatic);
	assert.equal(hits[1].meshInstance, farDynamic);
	assert.ok(hits[0].distance < hits[1].distance);
}

function testBucketMigrationViaMarkDirty() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "HybridMigration" }));
	const instance = createInstance(mesh, 0, 0, -2, false);
	const index = new HybridSpatialIndex([instance]);

	assert.equal(index.queryFrustum(camera.frustum).length, 1);

	instance.skeleton = {};
	index.markDirty(instance);
	assert.equal(index.queryFrustum(camera.frustum).length, 1);

	instance.skeleton = null;
	index.markDirty(instance);
	assert.equal(index.queryFrustum(camera.frustum).length, 1);
}

function run() {
	testFrustumAndDynamicUpdates();
	testRayOrderingAcrossBuckets();
	testBucketMigrationViaMarkDirty();
	console.log("Spatial hybrid index tests passed");
}

run();
