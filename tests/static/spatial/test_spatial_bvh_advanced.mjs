import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { BVH } from "../../../src/spatial/BVH.ts";

function createTriangleMesh(material) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

function createMeshInstance(mesh, x, y, z) {
	const instance = new MeshInstance({ mesh });
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

function testLeafQueryDoesNotRecomputeBounds() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "NoRecompute" }));
	const instance = createMeshInstance(mesh, 0, 0, -2);
	const bvh = new BVH([instance], 8);

	instance.getWorldBoundingBox = () => {
		throw new Error("query should use cached bounds");
	};

	const hits = bvh.queryFrustum(camera.frustum);
	assert.equal(hits.length, 1);
	assert.equal(hits[0], instance);
}

function testVisibilityFiltering() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "Visibility" }));
	const visibleMesh = createMeshInstance(mesh, -1, 0, -2);
	const hiddenMesh = createMeshInstance(mesh, 1, 0, -2);
	hiddenMesh.visible = false;

	const bvh = new BVH([visibleMesh, hiddenMesh], 1);
	const defaultHits = bvh.queryFrustum(camera.frustum);
	const includeInvisibleHits = bvh.queryFrustum(camera.frustum, {
		includeInvisible: true,
	});

	assert.equal(defaultHits.length, 1);
	assert.equal(defaultHits[0], visibleMesh);
	assert.equal(includeInvisibleHits.length, 2);
}

function testDynamicUpdates() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "Dynamic" }));
	const first = createMeshInstance(mesh, -1, 0, -2);
	const second = createMeshInstance(mesh, 1, 0, -2);

	const bvh = new BVH([first], 1);
	assert.equal(bvh.queryFrustum(camera.frustum).length, 1);

	bvh.upsert(second);
	assert.equal(bvh.queryFrustum(camera.frustum).length, 2);

	second.position.set(400, 0, -2);
	second.updateWorldMatrix();
	bvh.markDirty(second);
	const movedHits = bvh.queryFrustum(camera.frustum);
	assert.equal(movedHits.length, 1);
	assert.equal(movedHits[0], first);

	assert.equal(bvh.remove(first), true);
	assert.equal(bvh.queryFrustum(camera.frustum).length, 0);
}

function testEarlyExitMaxResults() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "EarlyExit" }));
	const instances = [];
	for (let index = 0; index < 12; index++) {
		instances.push(createMeshInstance(mesh, index * 0.1, 0, -2));
	}
	const bvh = new BVH(instances, 2);

	assert.equal(
		bvh.queryFrustum(camera.frustum, { maxResults: 0 }).length,
		0
	);
	assert.equal(
		bvh.queryFrustum(camera.frustum, { maxResults: 3 }).length,
		3
	);
}

function testIntoQueriesReuseOutputArrays() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "IntoQueries" }));
	const first = createMeshInstance(mesh, -0.5, 0, -2);
	const second = createMeshInstance(mesh, 0.5, 0, -2);
	const bvh = new BVH([first, second], 1);

	const frustumOut = [second];
	const frustumHits = bvh.queryFrustumInto(camera.frustum, frustumOut, {
		maxResults: 1,
	});
	assert.equal(frustumHits, frustumOut);
	assert.equal(frustumHits.length, 1);

	const boundsOut = [first];
	const boundsHits = bvh.queryBoundsInto(
		{
			min: { x: -2, y: -2, z: -4 },
			max: { x: 2, y: 2, z: 0 },
		},
		boundsOut
	);
	assert.equal(boundsHits, boundsOut);
	assert.equal(boundsHits.length, 2);
	assert.equal(boundsHits.includes(first), true);
	assert.equal(boundsHits.includes(second), true);
}

function testDegenerateSplitDoesNotBreak() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "Degenerate" }));
	const instances = [];

	for (let index = 0; index < 32; index++) {
		instances.push(createMeshInstance(mesh, 0, 0, -2));
	}

	const bvh = new BVH(instances, 1);
	const hits = bvh.queryFrustum(camera.frustum);

	assert.ok(bvh.root);
	assert.equal(hits.length, 32);
}

function testSAHBuildStrategyMatchesMedianResults() {
	const camera = createCamera();
	const mesh = createTriangleMesh(new Material({ name: "SAH" }));
	const instances = [];
	for (let index = 0; index < 24; index++) {
		instances.push(
			createMeshInstance(
				mesh,
				(index % 6) - 3,
				Math.floor(index / 6) * 0.2,
				-2 - index * 0.05
			)
		);
	}

	const medianHits = new Set(new BVH(instances, 2).queryFrustum(camera.frustum));
	const sahHits = new Set(
		new BVH(instances, { leafSize: 2, buildStrategy: "sah" })
			.queryFrustum(camera.frustum)
	);
	assert.deepEqual(sahHits, medianHits);
}

function testDirtyRefitOnlyVisitsAncestorPath() {
	const mesh = createTriangleMesh(new Material({ name: "AncestorRefit" }));
	const instances = [];
	for (let index = 0; index < 1024; index++) {
		instances.push(createMeshInstance(mesh, index * 2, 0, -2));
	}
	const bvh = new BVH(instances, { leafSize: 1, rebuildDirtyRatio: 0.5 });
	let innerRefitCalls = 0;
	const originalRefit = bvh._refitInnerNode.bind(bvh);
	bvh._refitInnerNode = (...args) => {
		innerRefitCalls++;
		return originalRefit(...args);
	};

	instances[500].position.y = 3;
	instances[500].updateWorldMatrix();
	bvh.markDirty(instances[500]);
	bvh.queryBoundsInto(
		{ min: { x: -1, y: -1, z: -3 }, max: { x: 3000, y: 5, z: 1 } },
		[]
	);
	assert.ok(innerRefitCalls > 0);
	assert.ok(innerRefitCalls < 32, `Expected path refit, got ${innerRefitCalls}`);
}

function testSpatialBoundsExcludeDescendantMeshes() {
	const parentMesh = createTriangleMesh(new Material({ name: "ParentOwn" }));
	const childMesh = createTriangleMesh(new Material({ name: "ChildOwn" }));
	const parent = createMeshInstance(parentMesh, 0, 0, 0);
	const child = createMeshInstance(childMesh, 20, 0, 0);
	parent.addChild(child);
	parent.updateWorldMatrix();
	const bvh = new BVH([parent, child], 1);
	const hits = bvh.queryBounds({
		min: { x: 19.5, y: -1, z: -1 },
		max: { x: 21.5, y: 2, z: 1 },
	});
	assert.deepEqual(hits, [child]);
}

function run() {
	testLeafQueryDoesNotRecomputeBounds();
	testVisibilityFiltering();
	testDynamicUpdates();
	testEarlyExitMaxResults();
	testIntoQueriesReuseOutputArrays();
	testDegenerateSplitDoesNotBreak();
	testSAHBuildStrategyMatchesMedianResults();
	testDirtyRefitOnlyVisitsAncestorPath();
	testSpatialBoundsExcludeDescendantMeshes();
	console.log("Spatial BVH advanced tests passed");
}

run();
