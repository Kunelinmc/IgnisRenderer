import assert from "node:assert/strict";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { GeometryBuilder } from "../../../src/meshes/GeometryBuilder.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

function createTrianglePrimitive(offsetX = 0, size = 1) {
	return GeometryBuilder.buildPrimitivesFromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: offsetX, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: offsetX + size, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: offsetX, y: size, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	])[0];
}

function testArrayMutationRefreshesMeshBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	assert.equal(mesh.boundingBox.max.x, 1);

	mesh.addPrimitive(createTrianglePrimitive(10, 2));

	assert.equal(mesh.boundingBox.min.x, 0);
	assert.equal(mesh.boundingBox.max.x, 12);
	assert.ok(mesh.boundingSphere.radius > 5);
}

function testPrimitiveReplacementRefreshesMeshBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	mesh.setPrimitives([createTrianglePrimitive(-4, 2)]);

	assert.equal(mesh.boundingBox.min.x, -4);
	assert.equal(mesh.boundingBox.max.x, -2);
}

function testConstructorCopiesPrimitiveArray() {
	const primitives = [createTrianglePrimitive(0, 1)];
	const mesh = new MeshAsset(primitives);

	primitives.push(createTrianglePrimitive(20, 3));

	assert.equal(mesh.boundingBox.max.x, 1);
	assert.equal(Object.isFrozen(mesh.primitives), true);
	assert.throws(() => mesh.primitives.push(createTrianglePrimitive(2, 1)));
}

function testGeometryVersionRefreshesPrimitiveBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const primitive = mesh.primitives[0];

	primitive.geometry.positions[3] = 6;
	mesh.markPrimitiveGeometryDirty(primitive);

	assert.equal(mesh.boundingBox.max.x, 6);
	assert.equal(primitive.boundingBox.max.x, 6);
}

function testMeshInstanceUsesReactiveBounds() {
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const instance = new MeshInstance({ mesh });
	instance.updateWorldMatrix();

	mesh.addPrimitive(createTrianglePrimitive(4, 1));

	assert.equal(instance.getWorldBoundingBox().max.x, 5);
}

function testSceneBoundsCacheUsesReactiveMeshBounds() {
	const scene = new Scene();
	const mesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const instance = scene.add(new MeshInstance({ mesh }));
	scene.updateWorldMatrices();

	const initialBounds = scene.getBounds();
	mesh.addPrimitive(createTrianglePrimitive(10, 2));
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

	mesh.addPrimitive(createTrianglePrimitive(4.5, 1));
	scene.rebuildSpatialIndex(scene.getMeshInstances());

	assert.equal(spatial.queryBounds(farBounds).includes(instance), true);
}

function testControlledPrimitiveOwnershipAndReplacement() {
	const first = createTrianglePrimitive(0, 1);
	const second = createTrianglePrimitive(5, 2);
	const mesh = new MeshAsset([first]);
	assert.throws(() => mesh.addPrimitive(first), /already owns/);
	assert.throws(() => new MeshAsset([first]), /another MeshAsset/);

	const previous = mesh.replacePrimitive(0, second);
	assert.equal(previous, first);
	assert.equal(mesh.boundingBox.min.x, 5);
	assert.equal(mesh.boundingBox.max.x, 7);
	assert.equal(mesh.removePrimitive(second), true);
	assert.equal(mesh.boundingSphere.radius, 0);
	assert.equal(mesh.boundingBox.max.x, 0);
}

function testGeometryReplacementAdvancesVersion() {
	const primitive = createTrianglePrimitive(0, 1);
	const mesh = new MeshAsset([primitive]);
	const replacement = createTrianglePrimitive(10, 3).geometry;
	const previousVersion = primitive.geometryVersion;
	mesh.setPrimitiveGeometry(primitive, replacement);
	assert.equal(primitive.geometryVersion, previousVersion + 1);
	assert.equal(mesh.boundingBox.min.x, 10);
	assert.equal(mesh.boundingBox.max.x, 13);
}

function testWorldBoundsCacheAndOwnBounds() {
	const parentMesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const childMesh = new MeshAsset([createTrianglePrimitive(0, 1)]);
	const parent = new MeshInstance({ mesh: parentMesh });
	const child = new MeshInstance({ mesh: childMesh });
	child.position.x = 20;
	parent.addChild(child);
	parent.updateWorldMatrix();

	const initialVersion = parent.worldBoundsVersion;
	assert.equal(parent.worldBoundsVersion, initialVersion);
	assert.equal(parent.getOwnWorldBoundingBox().max.x, 1);
	assert.equal(parent.getWorldBoundingBox().max.x, 21);

	parent.position.x = 3;
	parent.updateWorldMatrix();
	assert.ok(parent.worldBoundsVersion > initialVersion);
	assert.equal(parent.getOwnWorldBoundingBox().min.x, 3);
}

function testExactModelSphereAfterIncrementalMutation() {
	const first = createTrianglePrimitive(-10, 2);
	const second = createTrianglePrimitive(8, 2);
	const mesh = new MeshAsset([first, second]);
	const initialCenter = { ...mesh.boundingSphere.center };
	second.geometry.positions[4] = 1.5;
	mesh.markPrimitiveGeometryDirty(second);

	const box = mesh.boundingBox;
	const center = {
		x: (box.min.x + box.max.x) * 0.5,
		y: (box.min.y + box.max.y) * 0.5,
		z: (box.min.z + box.max.z) * 0.5,
	};
	let maxDistanceSq = 0;
	for (const primitive of mesh.primitives) {
		const positions = primitive.geometry.positions;
		for (let i = 0; i < positions.length; i += 3) {
			const dx = positions[i] - center.x;
			const dy = positions[i + 1] - center.y;
			const dz = positions[i + 2] - center.z;
			maxDistanceSq = Math.max(maxDistanceSq, dx * dx + dy * dy + dz * dz);
		}
	}
	assert.deepEqual(mesh.boundingSphere.center, center);
	assert.equal(mesh.boundingSphere.radius, Math.sqrt(maxDistanceSq));
	assert.deepEqual(mesh.boundingSphere.center, initialCenter);
}

function testAffineWorldBoundsMatchCornerReference() {
	const mesh = new MeshAsset([createTrianglePrimitive(-2, 5)]);
	const instance = new MeshInstance({ mesh });
	instance.worldMatrix = new Matrix4([
		[-2, 0.5, 0.25, 7],
		[0.75, 3, -0.5, -4],
		[0.2, 0.4, -1.5, 2],
		[0, 0, 0, 1],
	]);
	const actual = instance.getOwnWorldBoundingBox();
	const local = mesh.boundingBox;
	const expected = {
		min: { x: Infinity, y: Infinity, z: Infinity },
		max: { x: -Infinity, y: -Infinity, z: -Infinity },
	};
	for (let corner = 0; corner < 8; corner++) {
		const transformed = Matrix4.transformPoint(instance.worldMatrix, {
			x: (corner & 1) === 0 ? local.min.x : local.max.x,
			y: (corner & 2) === 0 ? local.min.y : local.max.y,
			z: (corner & 4) === 0 ? local.min.z : local.max.z,
		});
		expected.min.x = Math.min(expected.min.x, transformed.x);
		expected.min.y = Math.min(expected.min.y, transformed.y);
		expected.min.z = Math.min(expected.min.z, transformed.z);
		expected.max.x = Math.max(expected.max.x, transformed.x);
		expected.max.y = Math.max(expected.max.y, transformed.y);
		expected.max.z = Math.max(expected.max.z, transformed.z);
	}
	assert.deepEqual(actual, expected);
}

function run() {
	testArrayMutationRefreshesMeshBounds();
	testPrimitiveReplacementRefreshesMeshBounds();
	testConstructorCopiesPrimitiveArray();
	testGeometryVersionRefreshesPrimitiveBounds();
	testMeshInstanceUsesReactiveBounds();
	testSceneBoundsCacheUsesReactiveMeshBounds();
	testSpatialIndexUsesReactiveMeshBounds();
	testControlledPrimitiveOwnershipAndReplacement();
	testGeometryReplacementAdvancesVersion();
	testWorldBoundsCacheAndOwnBounds();
	testExactModelSphereAfterIncrementalMutation();
	testAffineWorldBoundsMatchCornerReference();
	console.log("MeshAsset bounds reactivity tests passed");
}

run();
