import { performance } from "node:perf_hooks";
import { readFile, writeFile } from "node:fs/promises";

import { Camera } from "../../src/cameras/Camera.ts";
import { Material } from "../../src/materials/Material.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../src/meshes/MeshInstance.ts";
import { BVH } from "../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../src/spatial/HybridSpatialIndex.ts";
import { LooseOctree } from "../../src/spatial/LooseOctree.ts";

const DEFAULT_SEED = 20260609;

function parseArgs(argv) {
	const args = {
		seed: DEFAULT_SEED,
		baseline: null,
		out: null,
		quick: false,
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--quick") {
			args.quick = true;
			continue;
		}
		if (arg === "--seed" && argv[i + 1]) {
			args.seed = Number.parseInt(argv[++i], 10);
			continue;
		}
		if (arg.startsWith("--seed=")) {
			args.seed = Number.parseInt(arg.slice("--seed=".length), 10);
			continue;
		}
		if (arg === "--baseline" && argv[i + 1]) {
			args.baseline = argv[++i];
			continue;
		}
		if (arg.startsWith("--baseline=")) {
			args.baseline = arg.slice("--baseline=".length);
			continue;
		}
		if (arg === "--out" && argv[i + 1]) {
			args.out = argv[++i];
			continue;
		}
		if (arg.startsWith("--out=")) {
			args.out = arg.slice("--out=".length);
		}
	}
	return args;
}

function createRng(seed) {
	let state = seed >>> 0;
	return () => {
		state += 0x6d2b79f5;
		let t = state;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function createTriangleMesh() {
	const material = new Material({ name: "SpatialBench" });
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

function createInstance(mesh, x, y, z, dynamic) {
	const instance = new MeshInstance({
		mesh,
		skeleton: dynamic ? {} : null,
	});
	instance.position.set(x, y, z);
	instance.updateWorldMatrix();
	return instance;
}

function createInstances(count, distribution, rng) {
	const mesh = createTriangleMesh();
	const instances = new Array(count);
	const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
	for (let i = 0; i < count; i++) {
		let x = 0;
		let y = 0;
		let z = 0;
		if (distribution === "uniform") {
			x = (i % columns) * 2 - columns;
			y = Math.floor(i / columns) * 0.2;
			z = -4 - Math.floor(i / columns) * 2;
		} else if (distribution === "dynamic-mix") {
			x = (i % columns) * 2 - columns;
			y = Math.floor(i / columns) * 0.1;
			z = -5 - Math.floor(i / columns) * 1.5;
		} else if (
			distribution === "clustered" ||
			distribution === "dynamic-clustered"
		) {
			const cluster = i % 12;
			const angle = cluster * 0.5235987756;
			const cx = Math.cos(angle) * 80;
			const cz = Math.sin(angle) * 80 - 120;
			x = cx + (rng() - 0.5) * 18;
			y = (rng() - 0.5) * 8;
			z = cz + (rng() - 0.5) * 18;
		} else if (
			distribution === "overlap" ||
			distribution === "dynamic-overlap"
		) {
			x = (rng() - 0.5) * 0.1;
			y = (rng() - 0.5) * 0.1;
			z = -30 + (rng() - 0.5) * 0.1;
		} else {
			x = (i % columns) * 2 - columns;
			y = Math.floor(i / columns) * 0.1;
			z = -5 - Math.floor(i / columns) * 1.5;
		}
		const dynamic =
			distribution === "dynamic-mix" ? i % 10 === 0
			: distribution === "dynamic-clustered" ? i % 2 === 0
			: distribution === "dynamic-overlap" ? i % 2 === 0
			: false;
		instances[i] = createInstance(mesh, x, y, z, dynamic);
	}
	return instances;
}

function createCamera() {
	const camera = new Camera();
	camera.position.set(0, 0, 12);
	camera.updateMatrices();
	return camera;
}

function createIndex(kind, instances) {
	if (kind === "bvh-sah") {
		return new BVH(instances, { buildStrategy: "sah" });
	}
	if (kind === "hybrid") {
		return new HybridSpatialIndex(instances);
	}
	return new BVH(instances, { buildStrategy: "median" });
}

function measureBuild(kind, instances) {
	const start = performance.now();
	const index = createIndex(kind, instances);
	return {
		index,
		buildMs: performance.now() - start,
	};
}

function measureUpdate(index, instances, ratio, iterations, rng, camera) {
	const movedCount = Math.max(1, Math.floor(instances.length * ratio));
	const queryOut = [];
	const samples = [];
	for (let frame = 0; frame < iterations; frame++) {
		const start = performance.now();
		for (let i = 0; i < movedCount; i++) {
			const instance = instances[(i * 997 + frame) % instances.length];
			instance.position.x += (rng() - 0.5) * 0.2;
			instance.position.y += (rng() - 0.5) * 0.1;
			instance.updateWorldMatrix();
			index.markDirty(instance);
		}
		index.queryFrustumInto(camera.frustum, queryOut, { maxResults: Infinity });
		samples.push(performance.now() - start);
	}
	return average(samples);
}

function measureFrustumQueries(index, camera, iterations) {
	const out = [];
	const start = performance.now();
	let hits = 0;
	for (let i = 0; i < iterations; i++) {
		index.queryFrustumInto(camera.frustum, out);
		hits += out.length;
	}
	const elapsedMs = performance.now() - start;
	return {
		qps: iterations / Math.max(1e-6, elapsedMs / 1000),
		avgHits: hits / iterations,
	};
}

function measureBoundsQueries(index, iterations, rng) {
	const out = [];
	const start = performance.now();
	let hits = 0;
	for (let i = 0; i < iterations; i++) {
		const centerX = (rng() - 0.5) * 160;
		const centerZ = -80 + (rng() - 0.5) * 160;
		index.queryBoundsInto(
			{
				min: { x: centerX - 12, y: -8, z: centerZ - 12 },
				max: { x: centerX + 12, y: 8, z: centerZ + 12 },
			},
			out
		);
		hits += out.length;
	}
	const elapsedMs = performance.now() - start;
	return {
		qps: iterations / Math.max(1e-6, elapsedMs / 1000),
		avgHits: hits / iterations,
	};
}

function measureRayQueries(index, iterations, rng, maxResults) {
	const out = [];
	const start = performance.now();
	let hits = 0;
	for (let i = 0; i < iterations; i++) {
		const x = (rng() - 0.5) * 120;
		const y = (rng() - 0.5) * 30;
		index.queryRayDetailedInto(
			{ x, y, z: 20 },
			{ x: 0, y: 0, z: -1 },
			out,
			{ maxDistance: 400, maxResults }
		);
		hits += out.length;
	}
	const elapsedMs = performance.now() - start;
	return {
		qps: iterations / Math.max(1e-6, elapsedMs / 1000),
		avgHits: hits / iterations,
	};
}

function collectScenarioDiagnostics(index, instances, camera, iterations, rng) {
	const bucketCounts = countDynamicBuckets(instances);
	return {
		staticCount: bucketCounts.staticCount,
		dynamicCount: bucketCounts.dynamicCount,
		selectedDynamicBackend: resolveSelectedDynamicBackend(index),
		dynamicOctree: collectSelectedDynamicOctreeStats(index),
		queries: {
			frustum: diagnoseSpatialFrustum(index, camera.frustum),
			bounds: averageDiagnostics(iterations, () => {
				const centerX = (rng() - 0.5) * 160;
				const centerZ = -80 + (rng() - 0.5) * 160;
				return diagnoseSpatialBounds(index, {
					min: { x: centerX - 12, y: -8, z: centerZ - 12 },
					max: { x: centerX + 12, y: 8, z: centerZ + 12 },
				});
			}),
			rayTop1: averageDiagnostics(iterations, () => {
				const x = (rng() - 0.5) * 120;
				const y = (rng() - 0.5) * 30;
				return diagnoseSpatialRay(
					index,
					{ x, y, z: 20 },
					{ x: 0, y: 0, z: -1 },
					{ maxDistance: 400, maxResults: 1 }
				);
			}),
			rayTopN: averageDiagnostics(iterations, () => {
				const x = (rng() - 0.5) * 120;
				const y = (rng() - 0.5) * 30;
				return diagnoseSpatialRay(
					index,
					{ x, y, z: 20 },
					{ x: 0, y: 0, z: -1 },
					{ maxDistance: 400, maxResults: 16 }
				);
			}),
		},
	};
}

function countDynamicBuckets(instances) {
	let dynamicCount = 0;
	for (const instance of instances) {
		if (instance.skeleton) dynamicCount++;
	}
	return {
		staticCount: instances.length - dynamicCount,
		dynamicCount,
	};
}

function resolveSelectedDynamicBackend(index) {
	return index instanceof HybridSpatialIndex ? index._dynamicBackend : null;
}

function collectSelectedDynamicOctreeStats(index) {
	if (
		!(index instanceof HybridSpatialIndex) ||
		index._dynamicBackend !== "octree"
	) {
		return null;
	}
	return summarizeLooseOctree(index._dynamicIndex);
}

function summarizeLooseOctree(index) {
	const root = index?._root ?? null;
	if (!root) {
		return {
			rootResidentCount: 0,
			rootResidentRatio: 0,
			nodeCount: 0,
			maxDepth: 0,
		};
	}
	const totalCount = index.size;
	const stats = {
		rootResidentCount: root.objects.length,
		rootResidentRatio:
			totalCount > 0 ? root.objects.length / totalCount : 0,
		nodeCount: 0,
		maxDepth: 0,
	};
	const stack = [{ node: root, depth: 0 }];
	while (stack.length > 0) {
		const current = stack.pop();
		stats.nodeCount++;
		stats.maxDepth = Math.max(stats.maxDepth, current.depth);
		for (const child of current.node.children ?? []) {
			if (child) stack.push({ node: child, depth: current.depth + 1 });
		}
	}
	return stats;
}

function createDiagnostics() {
	return {
		nodeVisits: 0,
		nodeAabbTests: 0,
		objectAabbTests: 0,
		hits: 0,
		nearestDistance: null,
	};
}

function mergeDiagnostics(left, right) {
	return {
		nodeVisits: left.nodeVisits + right.nodeVisits,
		nodeAabbTests: left.nodeAabbTests + right.nodeAabbTests,
		objectAabbTests: left.objectAabbTests + right.objectAabbTests,
		hits: left.hits + right.hits,
		nearestDistance: chooseNearestDistance(
			left.nearestDistance,
			right.nearestDistance
		),
	};
}

function averageDiagnostics(iterations, sampler) {
	const total = createDiagnostics();
	for (let i = 0; i < iterations; i++) {
		const sample = sampler();
		total.nodeVisits += sample.nodeVisits;
		total.nodeAabbTests += sample.nodeAabbTests;
		total.objectAabbTests += sample.objectAabbTests;
		total.hits += sample.hits;
		total.nearestDistance = chooseNearestDistance(
			total.nearestDistance,
			sample.nearestDistance
		);
	}
	return {
		nodeVisits: total.nodeVisits / iterations,
		nodeAabbTests: total.nodeAabbTests / iterations,
		objectAabbTests: total.objectAabbTests / iterations,
		avgHits: total.hits / iterations,
	};
}

function chooseNearestDistance(left, right) {
	if (left === null) return right;
	if (right === null) return left;
	return Math.min(left, right);
}

function diagnoseSpatialFrustum(index, frustum) {
	if (index instanceof HybridSpatialIndex) {
		return mergeDiagnostics(
			diagnoseSpatialFrustum(index._staticBVH, frustum),
			diagnoseSpatialFrustum(index._dynamicIndex, frustum)
		);
	}
	if (index instanceof BVH) {
		return diagnoseBVHFrustum(index.root, frustum);
	}
	if (index instanceof LooseOctree) {
		return diagnoseLooseOctreeFrustum(index._root, frustum, index._looseness);
	}
	return createDiagnostics();
}

function diagnoseSpatialBounds(index, bounds) {
	if (index instanceof HybridSpatialIndex) {
		return mergeDiagnostics(
			diagnoseSpatialBounds(index._staticBVH, bounds),
			diagnoseSpatialBounds(index._dynamicIndex, bounds)
		);
	}
	if (index instanceof BVH) {
		return diagnoseBVHBounds(index.root, bounds);
	}
	if (index instanceof LooseOctree) {
		return diagnoseLooseOctreeBounds(index._root, bounds, index._looseness);
	}
	return createDiagnostics();
}

function diagnoseSpatialRay(index, origin, direction, options) {
	if (index instanceof HybridSpatialIndex) {
		const staticDiagnostics = diagnoseSpatialRay(
			index._staticBVH,
			origin,
			direction,
			options
		);
		if (options.maxResults === 1) {
			const dynamicDiagnostics = diagnoseSpatialRay(
				index._dynamicIndex,
				origin,
				direction,
				{
					maxDistance:
						staticDiagnostics.nearestDistance ?? options.maxDistance,
					maxResults: 1,
				}
			);
			return mergeDiagnostics(staticDiagnostics, dynamicDiagnostics);
		}
		return mergeDiagnostics(
			staticDiagnostics,
			diagnoseSpatialRay(index._dynamicIndex, origin, direction, options)
		);
	}
	if (index instanceof BVH) {
		return diagnoseBVHRay(index.root, origin, direction, options);
	}
	if (index instanceof LooseOctree) {
		return diagnoseLooseOctreeRay(
			index._root,
			index._looseness,
			origin,
			direction,
			options
		);
	}
	return createDiagnostics();
}

function diagnoseBVHFrustum(root, frustum) {
	const diagnostics = createDiagnostics();
	if (!root) return diagnostics;
	const visit = (node) => {
		diagnostics.nodeVisits++;
		diagnostics.nodeAabbTests++;
		const status = classifyAABBFrustum(
			frustum,
			node.bounds.min.x,
			node.bounds.min.y,
			node.bounds.min.z,
			node.bounds.max.x,
			node.bounds.max.y,
			node.bounds.max.z
		);
		if (status === -1) return;
		if (node.objects && node.objectBounds) {
			if (status === 1) {
				diagnostics.hits += node.objects.length;
				return;
			}
			for (const bounds of node.objectBounds) {
				diagnostics.objectAabbTests++;
				if (
					classifyAABBFrustum(
						frustum,
						bounds.min.x,
						bounds.min.y,
						bounds.min.z,
						bounds.max.x,
						bounds.max.y,
						bounds.max.z
					) !== -1
				) {
					diagnostics.hits++;
				}
			}
			return;
		}
		if (status === 1) {
			appendSubtreeDiagnostics(node, diagnostics);
			return;
		}
		if (node.left) visit(node.left);
		if (node.right) visit(node.right);
	};
	visit(root);
	return diagnostics;
}

function diagnoseBVHBounds(root, bounds) {
	const diagnostics = createDiagnostics();
	if (!root) return diagnostics;
	const stack = [root];
	while (stack.length > 0) {
		const node = stack.pop();
		diagnostics.nodeVisits++;
		diagnostics.nodeAabbTests++;
		if (!intersectsAABB(node.bounds, bounds)) continue;
		if (node.objects && node.objectBounds) {
			for (const objectBounds of node.objectBounds) {
				diagnostics.objectAabbTests++;
				if (intersectsAABB(objectBounds, bounds)) diagnostics.hits++;
			}
			continue;
		}
		if (node.left) stack.push(node.left);
		if (node.right) stack.push(node.right);
	}
	return diagnostics;
}

function diagnoseBVHRay(root, origin, direction, options) {
	const diagnostics = createDiagnostics();
	const normalizedDirection = normalizeRayDirection(direction);
	if (!root) return diagnostics;
	const maxDistance = resolveDiagnosticMaxDistance(options.maxDistance);
	const rootDistance = diagnosticRayAABB(
		origin,
		normalizedDirection,
		maxDistance,
		root.bounds
	);
	diagnostics.nodeVisits++;
	diagnostics.nodeAabbTests++;
	if (rootDistance === null) return diagnostics;
	const stack = [{ node: root, distance: rootDistance }];
	let bestDistance = maxDistance;
	while (stack.length > 0) {
		const current = stack.pop();
		if (!current || current.distance > bestDistance) continue;
		const node = current.node;
		if (node.objects && node.objectBounds) {
			for (const bounds of node.objectBounds) {
				diagnostics.objectAabbTests++;
				const distance = diagnosticRayAABB(
					origin,
					normalizedDirection,
					options.maxResults === 1 ? bestDistance : maxDistance,
					bounds
				);
				if (distance === null) continue;
				diagnostics.hits++;
				diagnostics.nearestDistance = chooseNearestDistance(
					diagnostics.nearestDistance,
					distance
				);
				if (options.maxResults === 1) {
					bestDistance = Math.min(bestDistance, distance);
				}
			}
			continue;
		}
		for (const child of [node.left, node.right]) {
			if (!child) continue;
			diagnostics.nodeVisits++;
			diagnostics.nodeAabbTests++;
			const distance = diagnosticRayAABB(
				origin,
				normalizedDirection,
				options.maxResults === 1 ? bestDistance : maxDistance,
				child.bounds
			);
			if (distance !== null) {
				stack.push({ node: child, distance });
			}
		}
	}
	return diagnostics;
}

function diagnoseLooseOctreeFrustum(root, frustum, looseness) {
	const diagnostics = createDiagnostics();
	if (!root) return diagnostics;
	const visit = (node) => {
		diagnostics.nodeVisits++;
		diagnostics.nodeAabbTests++;
		const bounds = getLooseNodeBounds(node, looseness);
		const status = classifyAABBFrustum(
			frustum,
			bounds.min.x,
			bounds.min.y,
			bounds.min.z,
			bounds.max.x,
			bounds.max.y,
			bounds.max.z
		);
		if (status === -1) return;
		if (status === 1) {
			appendLooseSubtreeDiagnostics(node, diagnostics);
			return;
		}
		for (const objectBounds of node.objectBounds) {
			diagnostics.objectAabbTests++;
			if (
				classifyAABBFrustum(
					frustum,
					objectBounds.min.x,
					objectBounds.min.y,
					objectBounds.min.z,
					objectBounds.max.x,
					objectBounds.max.y,
					objectBounds.max.z
				) !== -1
			) {
				diagnostics.hits++;
			}
		}
		for (const child of node.children ?? []) {
			if (child) visit(child);
		}
	};
	visit(root);
	return diagnostics;
}

function diagnoseLooseOctreeBounds(root, bounds, looseness) {
	const diagnostics = createDiagnostics();
	if (!root) return diagnostics;
	const stack = [root];
	while (stack.length > 0) {
		const node = stack.pop();
		diagnostics.nodeVisits++;
		diagnostics.nodeAabbTests++;
		if (!intersectsAABB(getLooseNodeBounds(node, looseness), bounds)) continue;
		for (const objectBounds of node.objectBounds) {
			diagnostics.objectAabbTests++;
			if (intersectsAABB(objectBounds, bounds)) diagnostics.hits++;
		}
		for (const child of node.children ?? []) {
			if (child) stack.push(child);
		}
	}
	return diagnostics;
}

function diagnoseLooseOctreeRay(root, looseness, origin, direction, options) {
	const diagnostics = createDiagnostics();
	const normalizedDirection = normalizeRayDirection(direction);
	if (!root) return diagnostics;
	const maxDistance = resolveDiagnosticMaxDistance(options.maxDistance);
	const rootDistance = diagnosticRayAABB(
		origin,
		normalizedDirection,
		maxDistance,
		getLooseNodeBounds(root, looseness)
	);
	diagnostics.nodeVisits++;
	diagnostics.nodeAabbTests++;
	if (rootDistance === null) return diagnostics;
	const stack = [{ node: root, distance: rootDistance }];
	const shouldBoundHits = Number.isFinite(options.maxResults);
	const maxResults = resolveDiagnosticMaxResults(options.maxResults);
	let traversalMaxDistance = maxDistance;
	const hits = [];
	while (stack.length > 0) {
		const current = stack.pop();
		if (!current || current.distance > traversalMaxDistance) continue;
		const node = current.node;
		for (const bounds of node.objectBounds) {
			diagnostics.objectAabbTests++;
			const distance = diagnosticRayAABB(
				origin,
				normalizedDirection,
				traversalMaxDistance,
				bounds
			);
			if (distance === null) continue;
			diagnostics.hits++;
			diagnostics.nearestDistance = chooseNearestDistance(
				diagnostics.nearestDistance,
				distance
			);
			hits.push(distance);
			if (shouldBoundHits && hits.length >= maxResults) {
				hits.sort((left, right) => left - right);
				if (hits.length > maxResults) hits.length = maxResults;
				traversalMaxDistance = Math.min(
					traversalMaxDistance,
					hits[hits.length - 1]
				);
			}
		}
		for (const child of node.children ?? []) {
			if (!child) continue;
			diagnostics.nodeVisits++;
			diagnostics.nodeAabbTests++;
			const distance = diagnosticRayAABB(
				origin,
				normalizedDirection,
				traversalMaxDistance,
				getLooseNodeBounds(child, looseness)
			);
			if (distance !== null) {
				stack.push({ node: child, distance });
			}
		}
	}
	return diagnostics;
}

function appendSubtreeDiagnostics(node, diagnostics) {
	if (node.objects) {
		diagnostics.hits += node.objects.length;
		return;
	}
	if (node.left) {
		diagnostics.nodeVisits++;
		appendSubtreeDiagnostics(node.left, diagnostics);
	}
	if (node.right) {
		diagnostics.nodeVisits++;
		appendSubtreeDiagnostics(node.right, diagnostics);
	}
}

function appendLooseSubtreeDiagnostics(node, diagnostics) {
	diagnostics.hits += node.objects.length;
	for (const child of node.children ?? []) {
		if (!child) continue;
		diagnostics.nodeVisits++;
		appendLooseSubtreeDiagnostics(child, diagnostics);
	}
}

function getLooseNodeBounds(node, looseness) {
	const looseHalfSize = node.halfSize * looseness;
	return {
		min: {
			x: node.centerX - looseHalfSize,
			y: node.centerY - looseHalfSize,
			z: node.centerZ - looseHalfSize,
		},
		max: {
			x: node.centerX + looseHalfSize,
			y: node.centerY + looseHalfSize,
			z: node.centerZ + looseHalfSize,
		},
	};
}

function classifyAABBFrustum(frustum, minX, minY, minZ, maxX, maxY, maxZ) {
	let fullyInside = true;
	for (const plane of frustum.planes) {
		const nx = plane.normal.x;
		const ny = plane.normal.y;
		const nz = plane.normal.z;
		const px = nx >= 0 ? maxX : minX;
		const py = ny >= 0 ? maxY : minY;
		const pz = nz >= 0 ? maxZ : minZ;
		if (nx * px + ny * py + nz * pz + plane.constant < 0) {
			return -1;
		}
		const nxPoint = nx >= 0 ? minX : maxX;
		const nyPoint = ny >= 0 ? minY : maxY;
		const nzPoint = nz >= 0 ? minZ : maxZ;
		if (nx * nxPoint + ny * nyPoint + nz * nzPoint + plane.constant < 0) {
			fullyInside = false;
		}
	}
	return fullyInside ? 1 : 0;
}

function intersectsAABB(left, right) {
	return !(
		left.max.x < right.min.x ||
		left.min.x > right.max.x ||
		left.max.y < right.min.y ||
		left.min.y > right.max.y ||
		left.max.z < right.min.z ||
		left.min.z > right.max.z
	);
}

function normalizeRayDirection(direction) {
	const length = Math.hypot(direction.x, direction.y, direction.z);
	if (!(length > 1e-8)) {
		throw new Error("diagnose ray direction must be non-zero");
	}
	return {
		x: direction.x / length,
		y: direction.y / length,
		z: direction.z / length,
	};
}

function diagnosticRayAABB(origin, direction, maxDistance, bounds) {
	let tMin = 0;
	let tMax = maxDistance;
	const axisHit = (originValue, directionValue, minValue, maxValue) => {
		if (Math.abs(directionValue) < 1e-10) {
			return originValue >= minValue && originValue <= maxValue;
		}
		const invDirection = 1 / directionValue;
		let t0 = (minValue - originValue) * invDirection;
		let t1 = (maxValue - originValue) * invDirection;
		if (t0 > t1) {
			const tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = Math.max(tMin, t0);
		tMax = Math.min(tMax, t1);
		return tMax >= tMin;
	};
	if (!axisHit(origin.x, direction.x, bounds.min.x, bounds.max.x)) return null;
	if (!axisHit(origin.y, direction.y, bounds.min.y, bounds.max.y)) return null;
	if (!axisHit(origin.z, direction.z, bounds.min.z, bounds.max.z)) return null;
	if (tMax < 0 || tMin > maxDistance) return null;
	if (tMin >= 0) return tMin;
	if (tMax >= 0) return 0;
	return null;
}

function resolveDiagnosticMaxDistance(value) {
	if (value === undefined) return Infinity;
	if (!Number.isFinite(value)) return Infinity;
	return Math.max(0, value);
}

function resolveDiagnosticMaxResults(value) {
	if (value === undefined) return Infinity;
	if (!Number.isFinite(value)) return Infinity;
	return Math.max(0, Math.floor(value));
}

function average(values) {
	if (values.length === 0) return 0;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getConfig(quick) {
	return quick ?
			{
				counts: [1000, 3000],
				distributions: [
					"uniform",
					"clustered",
					"overlap",
					"dynamic-mix",
					"dynamic-clustered",
					"dynamic-overlap",
				],
				queryIterations: 120,
				diagnosticIterations: 40,
				updateIterations: 12,
			}
		:	{
				counts: [1000, 10000, 50000],
				distributions: [
					"uniform",
					"clustered",
					"overlap",
					"dynamic-mix",
					"dynamic-clustered",
					"dynamic-overlap",
				],
				queryIterations: 1000,
				diagnosticIterations: 200,
				updateIterations: 40,
			};
}

function compareWithBaseline(current, baseline) {
	if (!baseline?.scenarios) return null;
	const baselineByKey = new Map(
		baseline.scenarios.map((scenario) => [scenario.key, scenario])
	);
	const comparisons = [];
	for (const scenario of current.scenarios) {
		const base = baselineByKey.get(scenario.key);
		if (!base) continue;
		comparisons.push({
			key: scenario.key,
			buildSpeedup: base.buildMs / Math.max(1e-6, scenario.buildMs),
			frustumQpsRatio:
				scenario.frustum.qps / Math.max(1e-6, base.frustum.qps),
			boundsQpsRatio:
				scenario.bounds.qps / Math.max(1e-6, base.bounds.qps),
			rayTop1QpsRatio:
				scenario.rayTop1.qps / Math.max(1e-6, base.rayTop1.qps),
			update1PctSpeedup:
				base.update1PctMs / Math.max(1e-6, scenario.update1PctMs),
			update10PctSpeedup:
				base.update10PctMs / Math.max(1e-6, scenario.update10PctMs),
			boundsObjectTestsRatio: ratioOrNull(
				scenario.queryDiagnostics?.bounds?.objectAabbTests,
				base.queryDiagnostics?.bounds?.objectAabbTests
			),
			rayTop1ObjectTestsRatio: ratioOrNull(
				scenario.queryDiagnostics?.rayTop1?.objectAabbTests,
				base.queryDiagnostics?.rayTop1?.objectAabbTests
			),
			rayTopNObjectTestsRatio: ratioOrNull(
				scenario.queryDiagnostics?.rayTopN?.objectAabbTests,
				base.queryDiagnostics?.rayTopN?.objectAabbTests
			),
			frustumNodeVisitsRatio: ratioOrNull(
				scenario.queryDiagnostics?.frustum?.nodeVisits,
				base.queryDiagnostics?.frustum?.nodeVisits
			),
		});
	}
	return comparisons;
}

function ratioOrNull(current, baseline) {
	if (
		typeof current !== "number" ||
		typeof baseline !== "number" ||
		!Number.isFinite(current) ||
		!Number.isFinite(baseline) ||
		Math.abs(baseline) < 1e-6
	) {
		return null;
	}
	return current / baseline;
}

function formatNumber(value) {
	return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	const config = getConfig(args.quick);
	const camera = createCamera();
	const kinds = ["bvh-median", "bvh-sah", "hybrid"];
	const scenarios = [];

	for (const count of config.counts) {
		for (const distribution of config.distributions) {
			for (const kind of kinds) {
				const rng = createRng(
					args.seed ^ count ^ distribution.length ^ kind.length
				);
				const instances = createInstances(count, distribution, rng);
				const { index, buildMs } = measureBuild(kind, instances);
				const update1PctMs = measureUpdate(
					index,
					instances,
					0.01,
					config.updateIterations,
					rng,
					camera
				);
				const update10PctMs = measureUpdate(
					index,
					instances,
					0.1,
					config.updateIterations,
					rng,
					camera
				);
				const frustum = measureFrustumQueries(
					index,
					camera,
					config.queryIterations
				);
				const bounds = measureBoundsQueries(
					index,
					config.queryIterations,
					rng
				);
				const rayTop1 = measureRayQueries(
					index,
					config.queryIterations,
					rng,
					1
				);
				const rayTopN = measureRayQueries(
					index,
					config.queryIterations,
					rng,
					16
				);
				const diagnosticSeed =
					args.seed ^ count ^ distribution.length ^ kind.length ^ 0x9e37;
				const diagnostics = collectScenarioDiagnostics(
					index,
					instances,
					camera,
					config.diagnosticIterations,
					createRng(diagnosticSeed)
				);
				const scenario = {
					key: `${kind}:${distribution}:${count}`,
					kind,
					distribution,
					count,
					staticCount: diagnostics.staticCount,
					dynamicCount: diagnostics.dynamicCount,
					selectedDynamicBackend: diagnostics.selectedDynamicBackend,
					dynamicOctree: diagnostics.dynamicOctree,
					buildMs,
					update1PctMs,
					update10PctMs,
					frustum,
					bounds,
					rayTop1,
					rayTopN,
					queryDiagnostics: diagnostics.queries,
				};
				scenarios.push(scenario);
				console.log(
					[
						"[bench]",
						scenario.key,
						`build=${formatNumber(buildMs)}ms`,
						`upd1=${formatNumber(update1PctMs)}ms`,
						`upd10=${formatNumber(update10PctMs)}ms`,
						`frustum=${formatNumber(frustum.qps)}qps`,
						`bounds=${formatNumber(bounds.qps)}qps`,
						`ray1=${formatNumber(rayTop1.qps)}qps`,
						`rayN=${formatNumber(rayTopN.qps)}qps`,
						`dyn=${scenario.dynamicCount}`,
						`dynBackend=${scenario.selectedDynamicBackend ?? "n/a"}`,
						`ray1ObjTests=${formatNumber(
							scenario.queryDiagnostics.rayTop1.objectAabbTests
						)}`,
					].join(" ")
				);
			}
		}
	}

	const report = {
		seed: args.seed,
		quick: args.quick,
		generatedAt: new Date().toISOString(),
		scenarios,
	};
	if (args.baseline) {
		const baseline = JSON.parse(await readFile(args.baseline, "utf8"));
		report.baselineComparison = compareWithBaseline(report, baseline);
	}
	if (args.out) {
		await writeFile(args.out, `${JSON.stringify(report, null, 2)}\n`, "utf8");
		console.log(`Saved benchmark report: ${args.out}`);
	}
}

await main();
