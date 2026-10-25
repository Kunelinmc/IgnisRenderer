import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { BVH } from "../../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../../src/spatial/HybridSpatialIndex.ts";
import { LooseOctree } from "../../../src/spatial/LooseOctree.ts";

const INDEX_FACTORIES = [
	(instances) => new BVH(instances),
	(instances) => new BVH(instances, { leafSize: 1, buildStrategy: "sah" }),
	(instances) => new LooseOctree(instances),
	(instances) => new HybridSpatialIndex(instances),
	(instances) => new HybridSpatialIndex(instances, { dynamicBackend: "bvh" }),
	(instances) => new HybridSpatialIndex(instances, { dynamicBackend: "octree" }),
];

function testDuplicateBulkMembershipCanBeUpdatedAndRemoved() {
	for (const createIndex of INDEX_FACTORIES) {
		for (const entryPoint of ["constructor", "setMeshInstances", "rebuild"]) {
			const first = MeshFactory.createBox({ x: 0, y: 0, z: -5 }, 2, 2, 2);
			const second = MeshFactory.createBox({ x: 0, y: 0, z: -10 }, 2, 2, 2);
			second.skeleton = {};
			first.updateWorldMatrix();
			second.updateWorldMatrix();
			const source = [first, second, first, second, first];
			const index = createIndex(entryPoint === "constructor" ? source : []);
			if (entryPoint !== "constructor") index[entryPoint](source);
			const bounds = {
				min: { x: -100, y: -100, z: -100 },
				max: { x: 100, y: 100, z: 100 },
			};
			const origin = { x: 0, y: 0, z: 0 };
			const direction = { x: 0, y: 0, z: -1 };
			assert.equal(index.size, 2, `${index.constructor.name}.${entryPoint}`);
			assert.equal(source.length, 5, "Normalization must not mutate the caller's array");
			assert.deepEqual(new Set(index.queryBounds(bounds)), new Set([first, second]));
			assert.equal(index.queryBounds(bounds).length, 2);
			assert.deepEqual(index.queryRay(origin, direction), [first, second]);
			const camera = new Camera();
			camera.updateMatrices();
			assert.equal(index.queryFrustum(camera.frustum).length, 2);

			first.position.x = 20;
			first.updateWorldMatrix();
			index.markDirty(first);
			assert.deepEqual(index.queryRay(origin, direction), [second]);
			assert.deepEqual(index.queryBounds(first.getOwnWorldBoundingBox()), [first]);
			assert.equal(index.remove(first), true);
			assert.equal(index.remove(first), false);
			assert.deepEqual(index.queryBounds(bounds), [second]);
			assert.equal(index.remove(second), true);
			assert.equal(index.size, 0);
			assert.deepEqual(index.queryBounds(bounds), []);
			assert.deepEqual(index.queryRay(origin, direction), []);
			assert.deepEqual(index.queryFrustum(camera.frustum), []);
			index.upsert(first);
			index.upsert(first);
			assert.equal(index.size, 1);
			assert.deepEqual(index.queryBounds(bounds), [first]);
		}
	}
}

function testSceneRemovesDuplicateInitialMembers() {
	for (const mode of ["bvh", "hybrid"]) {
		const scene = new Scene();
		scene.spatialIndexMode = mode;
		const instance = MeshFactory.createBox({ x: 0, y: 0, z: -5 }, 2, 2, 2);
		instance.updateWorldMatrix();
		const index = scene.rebuildSpatialIndex([instance, instance]);
		const bounds = instance.getOwnWorldBoundingBox();
		assert.deepEqual(index.queryBounds(bounds), [instance]);
		scene.rebuildSpatialIndex([]);
		assert.equal(index.size, 0);
		assert.deepEqual(index.queryBounds(bounds), []);
	}
}

function testRaySlabsPreserveSmallNonzeroComponents() {
	const near = MeshFactory.createBox({ x: 0, y: 0, z: -5 }, 2, 2, 2);
	const far = MeshFactory.createBox({ x: 0, y: 0, z: -10 }, 2, 2, 2);
	far.skeleton = {};
	near.updateWorldMatrix();
	far.updateWorldMatrix();
	const cases = [
		{
			origin: { x: 1 + 1e-11, y: 0, z: 0 },
			direction: { x: -5e-12, y: 0, z: -1 },
			distances: [4, 9],
		},
		{
			origin: { x: 1 - 1e-11, y: 0, z: 0 },
			direction: { x: 5e-12, y: 0, z: -1 },
			distances: [],
		},
		{
			origin: { x: 1, y: 0, z: 0 },
			direction: { x: 0, y: -0, z: -1 },
			distances: [4, 9],
		},
		{
			origin: { x: -1, y: 0, z: 0 },
			direction: { x: Number.MIN_VALUE, y: 0, z: -1 },
			distances: [4, 9],
		},
		{
			origin: { x: 1, y: 0, z: 0 },
			direction: { x: -Number.MIN_VALUE, y: 0, z: -1 },
			distances: [4, 9],
		},
		{
			origin: { x: 1, y: 0, z: 0 },
			direction: { x: Number.MIN_VALUE, y: 0, z: -1 },
			distances: [],
		},
		{
			origin: { x: 1, y: 0, z: 0 },
			direction: { x: Number.MIN_VALUE, y: 0, z: -2 },
			distances: [],
		},
		{
			origin: { x: 1, y: 0, z: 0 },
			direction: { x: 1e-300, y: 0, z: -1e300 },
			distances: [],
		},
		{
			origin: { x: 0, y: 0, z: -5 },
			direction: { x: -0, y: 0, z: -1 },
			distances: [0, 4],
		},
	];
	for (const createIndex of INDEX_FACTORIES) {
		const index = createIndex([near, far]);
		for (const { origin, direction, distances } of cases) {
			for (const maxResults of [1, 2, Infinity]) {
				for (const maxDistance of [3, 4, 10]) {
					const hits = index.queryRayDetailed(origin, direction, {
						maxResults,
						maxDistance,
					});
					const expected = distances.filter((value) => value <= maxDistance)
						.slice(0, maxResults);
					assert.deepEqual(hits.map((hit) => hit.distance), expected);
					assert.deepEqual(
						hits.map((hit) => hit.meshInstance),
						[near, far].slice(0, expected.length)
					);
				}
			}
		}
	}
}

function testRayDirectionsNormalizeAcrossScales() {
	const near = MeshFactory.createBox({ x: 5, y: 0, z: -5 }, 2, 2, 2);
	const far = MeshFactory.createBox({ x: 10, y: 0, z: -10 }, 2, 2, 2);
	far.skeleton = {};
	near.updateWorldMatrix();
	far.updateWorldMatrix();
	for (const createIndex of INDEX_FACTORIES) {
		const index = createIndex([near, far]);
		for (const scale of [1, 1e-9, 1e-300, Number.MIN_VALUE, 1e300, 1.7e308]) {
			for (const maxResults of [1, 2, Infinity]) {
				for (const maxDistance of [6, 20]) {
					const hits = index.queryRayDetailed(
						{ x: 0, y: 0, z: 0 },
						{ x: scale, y: 0, z: -scale },
						{ maxResults, maxDistance }
					);
					const expected = [4 * Math.SQRT2, 9 * Math.SQRT2]
						.filter((distance) => distance <= maxDistance).slice(0, maxResults);
					assert.deepEqual(
						hits.map((hit) => hit.meshInstance),
						[near, far].slice(0, expected.length)
					);
					for (let i = 0; i < hits.length; i++) {
						assert.ok(Math.abs(hits[i].distance - expected[i]) < 1e-12);
					}
				}
			}
		}
	}
}

function testSubnormalSlabsKeepWorldSpaceDistances() {
	const cases = [
		{ z: -5, scaleZ: 1, originX: -4 * Number.MIN_VALUE,
			x: 3 * Number.MIN_VALUE, directionZ: -4, distance: 16 / 3 },
		{ z: -2e-16, scaleZ: 1e-16, originX: -Number.MIN_VALUE,
			x: 3, directionZ: -1.7e308, distance: (Number.MIN_VALUE * 1.7e308) / 3 },
		{ z: -2e293, scaleZ: 1e293, originX: -1e-15,
			x: Number.MIN_VALUE, directionZ: -1e-15, distance: 1e-30 / Number.MIN_VALUE },
	];
	for (const sample of cases) {
		const instance = MeshFactory.createBox({ x: Number.MIN_VALUE, y: 0, z: sample.z }, 2, 2, 2);
		instance.skeleton = {};
		instance.scale.set(Number.MIN_VALUE, 1, sample.scaleZ);
		instance.updateWorldMatrix();
		for (const createIndex of INDEX_FACTORIES) {
			const index = createIndex([instance]);
			for (const maxResults of [1, 2, Infinity]) {
				const hits = index.queryRayDetailed(
					{ x: sample.originX, y: 0, z: 0 },
					{ x: sample.x, y: 0, z: sample.directionZ },
					{ maxResults, maxDistance: sample.distance * 1.1 }
				);
				assert.equal(hits.length, 1);
				assert.equal(hits[0].meshInstance, instance);
				assert.ok(Math.abs(hits[0].distance - sample.distance) < sample.distance * 1e-14);
			}
		}
	}
}

function testRayValidationRejectsZeroAndNonFiniteDirections() {
	const instance = MeshFactory.createBox({ x: 0, y: 0, z: -5 }, 2, 2, 2);
	instance.updateWorldMatrix();
	const directions = [
		{ x: 0, y: -0, z: 0 },
		{ x: NaN, y: 0, z: -1 },
		{ x: 0, y: Infinity, z: -1 },
		{ x: 0, y: 0, z: -Infinity },
	];
	for (const createIndex of INDEX_FACTORIES) {
		for (const members of [[], [instance]]) {
			const index = createIndex(members);
			for (const direction of directions) {
				for (const maxResults of [0, 1, Infinity]) {
					const out = [{ meshInstance: instance, distance: 999 }];
					assert.throws(() => index.queryRayDetailedInto(
						{ x: 0, y: 0, z: 0 }, direction, out, { maxResults }
					), /direction must/);
					assert.deepEqual(out, []);
				}
			}
		}
	}
}

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
		if (direction[axis] === 0) {
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

function assertQueryParity(indexes, instances, rng) {
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
		const target = instances[queryIndex * 11].position;
		const origin = queryIndex % 2 === 0 ?
			{ x: target.x, y: target.y, z: 4 }
		: { x: (rng() - 0.5) * 30, y: (rng() - 0.5) * 15, z: 4 };
		const direction = { x: 0, y: 0, z: -1 };
		if (queryIndex % 3 === 0) {
			// Aim through box interiors to avoid ambiguous corner-rounding expectations.
			const dx = target.x - origin.x;
			const dy = target.y - origin.y;
			const dz = target.z - origin.z;
			const length = Math.hypot(dx, dy, dz);
			direction.x = dx / length;
			direction.y = dy / length;
			direction.z = dz / length;
		}
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
			);
		for (const maxResults of [1, 8]) {
			for (const index of indexes) {
				const actual = index.queryRayDetailed(origin, direction, { maxResults });
				assert.deepEqual(
					actual.map((hit) => hit.meshInstance),
					expected.slice(0, maxResults).map((hit) => hit.meshInstance)
				);
				for (let i = 0; i < actual.length; i++) {
					assert.ok(Math.abs(actual[i].distance - expected[i].distance) < 1e-8);
				}
			}
		}
	}
}

function run() {
	testDuplicateBulkMembershipCanBeUpdatedAndRemoved();
	testSceneRemovesDuplicateInitialMembers();
	testRaySlabsPreserveSmallNonzeroComponents();
	testRayDirectionsNormalizeAcrossScales();
	testSubnormalSlabsKeepWorldSpaceDistances();
	testRayValidationRejectsZeroAndNonFiniteDirections();
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
		new LooseOctree(instances, { looseness: 1 }),
		new LooseOctree(instances, { looseness: 2 }),
		new HybridSpatialIndex(instances),
		new HybridSpatialIndex(instances, { dynamicBackend: "octree" }),
	];
	assertQueryParity(indexes, instances, rng);
	// Exercise both in-place loose-node updates and reinsertion after larger moves.
	for (const movement of [0.2, 20]) {
		for (let i = 0; i < instances.length; i += 2) {
			const instance = instances[i];
			instance.position.x += (rng() - 0.5) * movement;
			instance.position.y += (rng() - 0.5) * movement;
			instance.position.z += (rng() - 0.5) * movement;
			instance.updateWorldMatrix();
			for (const index of indexes) index.markDirty(instance);
		}
		assertQueryParity(indexes, instances, rng);
	}

	console.log("Spatial query parity tests passed");
}

run();
