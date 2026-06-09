import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { BVH } from "../../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../../src/spatial/HybridSpatialIndex.ts";
import { LooseOctree } from "../../../src/spatial/LooseOctree.ts";

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

function createSparseDynamicGrid(mesh, count) {
	const instances = new Array(count);
	const columns = Math.ceil(Math.sqrt(count));
	for (let i = 0; i < count; i++) {
		instances[i] = createInstance(
			mesh,
			(i % columns) * 2 - columns,
			Math.floor(i / columns) * 0.1,
			-5 - Math.floor(i / columns) * 1.5,
			i % 10 === 0
		);
	}
	return instances;
}

function createAllDynamicGrid(mesh, count) {
	const instances = new Array(count);
	const columns = Math.ceil(Math.sqrt(count));
	for (let i = 0; i < count; i++) {
		instances[i] = createInstance(
			mesh,
			(i % columns) * 2 - columns,
			Math.floor(i / columns) * 0.1,
			-5 - Math.floor(i / columns) * 1.5,
			true
		);
	}
	return instances;
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
	assert.equal(index._dynamicBackend, "bvh");

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

	const nearestOut = [];
	const nearestHits = index.queryRayDetailedInto(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		nearestOut,
		{ maxDistance: 100, maxResults: 1 }
	);
	assert.equal(nearestHits, nearestOut);
	assert.equal(nearestHits.length, 1);
	assert.equal(nearestHits[0].meshInstance, nearStatic);
}

function testAutoSelectsBVHForSparseDynamicMix() {
	const mesh = createTriangleMesh(new Material({ name: "HybridAutoSparse" }));
	const instances = createSparseDynamicGrid(mesh, 1000);
	const index = new HybridSpatialIndex(instances);

	assert.equal(index._dynamicBackend, "bvh");
	assert.ok(index._dynamicIndex instanceof BVH);
	assert.equal(index._dynamicIndex.size, 100);
}

function testAutoSelectsOctreeForLargeCleanDynamicSet() {
	const mesh = createTriangleMesh(new Material({ name: "HybridAutoOctree" }));
	const instances = createAllDynamicGrid(mesh, 9000);
	const index = new HybridSpatialIndex(instances);

	assert.equal(index._dynamicBackend, "octree");
	assert.ok(index._dynamicIndex instanceof LooseOctree);
	assert.equal(index._dynamicIndex.size, 9000);
}

function testExplicitOctreeBackendPreservesDynamicBucket() {
	const mesh = createTriangleMesh(new Material({ name: "HybridOctreeMode" }));
	const instances = createSparseDynamicGrid(mesh, 1000);
	const index = new HybridSpatialIndex(instances, {
		dynamicBackend: "octree",
	});

	assert.equal(index._dynamicBackend, "octree");
	assert.ok(index._dynamicIndex instanceof LooseOctree);
	assert.equal(index._dynamicIndex.size, 100);
}

function testExplicitBVHMatchesOctreeQueryResults() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "HybridBVHMode" }));
	const staticNear = createInstance(mesh, -0.5, 0, -2, false);
	const dynamicNear = createInstance(mesh, 0.5, 0, -2, true);
	const dynamicFar = createInstance(mesh, 8, 0, -8, true);
	const instances = [staticNear, dynamicNear, dynamicFar];
	const bvhIndex = new HybridSpatialIndex(instances, {
		dynamicBackend: "bvh",
	});
	const octreeIndex = new HybridSpatialIndex(instances, {
		dynamicBackend: "octree",
	});

	assert.deepEqual(
		new Set(bvhIndex.queryFrustum(camera.frustum)),
		new Set(octreeIndex.queryFrustum(camera.frustum))
	);
	assert.deepEqual(
		new Set(
			bvhIndex.queryBounds({
				min: { x: -2, y: -2, z: -4 },
				max: { x: 2, y: 2, z: 0 },
			})
		),
		new Set(
			octreeIndex.queryBounds({
				min: { x: -2, y: -2, z: -4 },
				max: { x: 2, y: 2, z: 0 },
			})
		)
	);

	const bvhRayHits = bvhIndex.queryRayDetailed(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		{ maxDistance: 100 }
	);
	const octreeRayHits = octreeIndex.queryRayDetailed(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: -1 },
		{ maxDistance: 100 }
	);
	assert.deepEqual(
		bvhRayHits.map((hit) => hit.meshInstance),
		octreeRayHits.map((hit) => hit.meshInstance)
	);
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
	assert.equal(index._dynamicIndex.size, 1);

	instance.skeleton = null;
	index.markDirty(instance);
	assert.equal(index.queryFrustum(camera.frustum).length, 1);
	assert.equal(index._dynamicIndex.size, 0);
}

function testBoundsQueryAcrossBuckets() {
	const mesh = createTriangleMesh(new Material({ name: "HybridBounds" }));
	const staticNear = createInstance(mesh, -0.4, 0, -2, false);
	const dynamicNear = createInstance(mesh, 0.4, 0, -2, true);
	const dynamicFar = createInstance(mesh, 20, 0, -2, true);
	const index = new HybridSpatialIndex([staticNear, dynamicNear, dynamicFar]);

	const boundsHits = new Set(
		index.queryBounds({
			min: { x: -2, y: -2, z: -4 },
			max: { x: 2, y: 2, z: 0 },
		})
	);
	assert.equal(boundsHits.has(staticNear), true);
	assert.equal(boundsHits.has(dynamicNear), true);
	assert.equal(boundsHits.has(dynamicFar), false);
}

function run() {
	testFrustumAndDynamicUpdates();
	testRayOrderingAcrossBuckets();
	testAutoSelectsBVHForSparseDynamicMix();
	testAutoSelectsOctreeForLargeCleanDynamicSet();
	testExplicitOctreeBackendPreservesDynamicBucket();
	testExplicitBVHMatchesOctreeQueryResults();
	testBucketMigrationViaMarkDirty();
	testBoundsQueryAcrossBuckets();
	console.log("Spatial hybrid index tests passed");
}

run();
