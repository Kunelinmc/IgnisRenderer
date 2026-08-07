import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { BVH } from "../../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../../src/spatial/HybridSpatialIndex.ts";
import { LooseOctree } from "../../../src/spatial/LooseOctree.ts";

function createRng(seed) {
	let state = seed >>> 0;
	return () => {
		state = Math.imul(state ^ (state >>> 15), 1 | state);
		state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
		return ((state ^ (state >>> 14)) >>> 0) / 4294967296;
	};
}

function createMesh() {
	return MeshAsset.fromFaces([
		{
			material: new Material({ name: "SpatialParity" }),
			vertices: [
				{ x: -0.5, y: -0.5, z: -0.5 },
				{ x: 0.5, y: 0.5, z: 0.5 },
				{ x: -0.5, y: 0.5, z: 0.5 },
			],
		},
	]);
}

function intersectsBounds(left, right) {
	return !(
		left.max.x < right.min.x ||
		left.min.x > right.max.x ||
		left.max.y < right.min.y ||
		left.min.y > right.max.y ||
		left.max.z < right.min.z ||
		left.min.z > right.max.z
	);
}

function intersectRay(origin, direction, bounds) {
	let near = 0;
	let far = Infinity;
	for (const axis of ["x", "y", "z"]) {
		if (Math.abs(direction[axis]) < 1e-8) {
			if (origin[axis] < bounds.min[axis] || origin[axis] > bounds.max[axis]) {
				return null;
			}
			continue;
		}
		let first = (bounds.min[axis] - origin[axis]) / direction[axis];
		let second = (bounds.max[axis] - origin[axis]) / direction[axis];
		if (first > second) [first, second] = [second, first];
		near = Math.max(near, first);
		far = Math.min(far, second);
		if (far < near) return null;
	}
	return far >= 0 ? Math.max(0, near) : null;
}

function ids(values) {
	return values.map((value) => value.id).sort();
}

function run() {
	const rng = createRng(0x51a71a1);
	const mesh = createMesh();
	const instances = [];
	for (let index = 0; index < 240; index++) {
		const instance = new MeshInstance({
			mesh,
			skeleton: index % 5 === 0 ? {} : null,
		});
		instance.position.set(
			(rng() - 0.5) * 40,
			(rng() - 0.5) * 20,
			-2 - rng() * 80
		);
		instance.updateWorldMatrix();
		instances.push(instance);
	}
	const indexes = [
		new BVH(instances),
		new LooseOctree(instances),
		new HybridSpatialIndex(instances),
	];

	const camera = new Camera();
	camera.updateWorldMatrix();
	camera.updateMatrices();
	const expectedFrustum = instances.filter((instance) => {
		const bounds = instance.getOwnWorldBoundingBox();
		return camera.frustum.intersectsAABB(bounds.min, bounds.max);
	});
	for (const index of indexes) {
		assert.deepEqual(ids(index.queryFrustum(camera.frustum)), ids(expectedFrustum));
	}

	for (let queryIndex = 0; queryIndex < 20; queryIndex++) {
		const centerX = (rng() - 0.5) * 40;
		const centerZ = -rng() * 80;
		const bounds = {
			min: { x: centerX - 5, y: -5, z: centerZ - 5 },
			max: { x: centerX + 5, y: 5, z: centerZ + 5 },
		};
		const expected = instances.filter((instance) =>
			intersectsBounds(instance.getOwnWorldBoundingBox(), bounds)
		);
		for (const index of indexes) {
			assert.deepEqual(ids(index.queryBounds(bounds)), ids(expected));
		}
	}

	for (let queryIndex = 0; queryIndex < 20; queryIndex++) {
		const origin = { x: (rng() - 0.5) * 30, y: (rng() - 0.5) * 15, z: 4 };
		const direction = { x: 0, y: 0, z: -1 };
		const expected = instances
			.map((meshInstance) => ({
				meshInstance,
				distance: intersectRay(
					origin,
					direction,
					meshInstance.getOwnWorldBoundingBox()
				),
			}))
			.filter((hit) => hit.distance !== null)
			.sort(
				(left, right) =>
					left.distance - right.distance ||
					left.meshInstance.id.localeCompare(right.meshInstance.id)
			)
			.slice(0, 8);
		for (const index of indexes) {
			const actual = index.queryRayDetailed(origin, direction, { maxResults: 8 });
			assert.deepEqual(
				actual.map((hit) => [hit.meshInstance.id, hit.distance]),
				expected.map((hit) => [hit.meshInstance.id, hit.distance])
			);
		}
	}

	console.log("Spatial query parity tests passed");
}

run();
