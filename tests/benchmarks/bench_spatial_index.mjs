import { performance } from "node:perf_hooks";
import { readFile, writeFile } from "node:fs/promises";

import { Camera } from "../../src/cameras/Camera.ts";
import { Material } from "../../src/materials/Material.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../src/meshes/MeshInstance.ts";
import { BVH } from "../../src/spatial/BVH.ts";
import { HybridSpatialIndex } from "../../src/spatial/HybridSpatialIndex.ts";

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
		} else if (distribution === "clustered") {
			const cluster = i % 12;
			const angle = cluster * 0.5235987756;
			const cx = Math.cos(angle) * 80;
			const cz = Math.sin(angle) * 80 - 120;
			x = cx + (rng() - 0.5) * 18;
			y = (rng() - 0.5) * 8;
			z = cz + (rng() - 0.5) * 18;
		} else if (distribution === "overlap") {
			x = (rng() - 0.5) * 0.1;
			y = (rng() - 0.5) * 0.1;
			z = -30 + (rng() - 0.5) * 0.1;
		} else {
			x = (i % columns) * 2 - columns;
			y = Math.floor(i / columns) * 0.1;
			z = -5 - Math.floor(i / columns) * 1.5;
		}
		const dynamic = distribution === "dynamic-mix" && i % 10 === 0;
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

function average(values) {
	if (values.length === 0) return 0;
	return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function getConfig(quick) {
	return quick ?
			{
				counts: [1000, 3000],
				distributions: ["uniform", "clustered", "overlap", "dynamic-mix"],
				queryIterations: 120,
				updateIterations: 12,
			}
		:	{
				counts: [1000, 10000, 50000],
				distributions: ["uniform", "clustered", "overlap", "dynamic-mix"],
				queryIterations: 1000,
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
		});
	}
	return comparisons;
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
				const scenario = {
					key: `${kind}:${distribution}:${count}`,
					kind,
					distribution,
					count,
					buildMs,
					update1PctMs,
					update10PctMs,
					frustum,
					bounds,
					rayTop1,
					rayTopN,
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
