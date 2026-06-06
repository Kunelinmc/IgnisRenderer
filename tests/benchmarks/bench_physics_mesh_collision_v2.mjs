import { performance } from "node:perf_hooks";
import { readFile, writeFile } from "node:fs/promises";
import { Node } from "../../src/core/Node.ts";
import { Scene } from "../../src/core/Scene.ts";
import { Material } from "../../src/materials/Material.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../src/meshes/MeshInstance.ts";
import { PhysicsSystem } from "../../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../../src/physics/adapters/SimplePhysicsAdapter.ts";
import { RapierPhysicsAdapter } from "../../src/physics/adapters/RapierPhysicsAdapter.ts";
import { AmmoPhysicsAdapter } from "../../src/physics/adapters/AmmoPhysicsAdapter.ts";

const DEFAULT_SEED = 20260418;

function parseArgs(argv) {
	const args = {
		seed: DEFAULT_SEED,
		baseline: null,
		out: null,
		quick: false,
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--seed" && argv[i + 1]) {
			args.seed = Number.parseInt(argv[i + 1], 10);
			i++;
			continue;
		}
		if (arg.startsWith("--seed=")) {
			args.seed = Number.parseInt(arg.slice("--seed=".length), 10);
			continue;
		}
		if (arg === "--baseline" && argv[i + 1]) {
			args.baseline = argv[i + 1];
			i++;
			continue;
		}
		if (arg.startsWith("--baseline=")) {
			args.baseline = arg.slice("--baseline=".length);
			continue;
		}
		if (arg === "--out" && argv[i + 1]) {
			args.out = argv[i + 1];
			i++;
			continue;
		}
		if (arg.startsWith("--out=")) {
			args.out = arg.slice("--out=".length);
			continue;
		}
		if (arg === "--quick") {
			args.quick = true;
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

function percentile(values, p) {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const index = Math.max(0, Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length)));
	return sorted[index];
}

function createGridMeshInstance(segments, worldX, worldY, worldZ) {
	const material = new Material({ name: "BenchMesh" });
	const faces = [];
	const step = 1;
	for (let x = 0; x < segments; x++) {
		for (let z = 0; z < segments; z++) {
			const x0 = x * step;
			const z0 = z * step;
			const x1 = (x + 1) * step;
			const z1 = (z + 1) * step;
			faces.push({
				material,
				vertices: [
					{
						x: x0,
						y: 0,
						z: z0,
						u: 0,
						v: 0,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: x1,
						y: 0,
						z: z0,
						u: 1,
						v: 0,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: x1,
						y: 0,
						z: z1,
						u: 1,
						v: 1,
						normal: { x: 0, y: 1, z: 0 },
					},
				],
			});
			faces.push({
				material,
				vertices: [
					{
						x: x0,
						y: 0,
						z: z0,
						u: 0,
						v: 0,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: x1,
						y: 0,
						z: z1,
						u: 1,
						v: 1,
						normal: { x: 0, y: 1, z: 0 },
					},
					{
						x: x0,
						y: 0,
						z: z1,
						u: 0,
						v: 1,
						normal: { x: 0, y: 1, z: 0 },
					},
				],
			});
		}
	}

	const mesh = MeshAsset.fromFaces(faces);
	const meshInstance = new MeshInstance({ mesh });
	meshInstance.position.set(worldX, worldY, worldZ);
	meshInstance.updateWorldMatrix();
	return meshInstance;
}

function addDynamicSphereBodies(physics, count, worldId, rng) {
	for (let i = 0; i < count; i++) {
		const node = new Node({
			position: {
				x: (rng() - 0.5) * 20,
				y: 2 + rng() * 6,
				z: (rng() - 0.5) * 20,
			},
		});
		const body = physics.attachBody(node, {
			worldId,
			body: { type: "dynamic" },
			authority: "physics",
		});
		physics.addCollider(body, {
			mode: "explicit",
			shape: { kind: "sphere", radius: 0.4 + rng() * 0.2 },
		});
	}
}

function measureSteps(physics, iterations, warmup, onBeforeStep = null) {
	for (let i = 0; i < warmup; i++) {
		if (onBeforeStep) onBeforeStep(i);
		physics.step(1 / 60);
	}
	const samples = [];
	for (let i = 0; i < iterations; i++) {
		if (onBeforeStep) onBeforeStep(i);
		const start = performance.now();
		physics.step(1 / 60);
		samples.push(performance.now() - start);
	}
	return {
		p50Ms: percentile(samples, 50),
		p95Ms: percentile(samples, 95),
		samples,
	};
}

function measureQueryThroughput(physics, worldId, iterations, rng) {
	const start = performance.now();
	let queries = 0;
	for (let i = 0; i < iterations; i++) {
		const ox = (rng() - 0.5) * 20;
		const oz = (rng() - 0.5) * 20;
		physics.raycast({
			worldId,
			origin: { x: ox, y: 12, z: oz },
			direction: { x: 0, y: -1, z: 0 },
			maxDistance: 40,
		});
		physics.sphereCast({
			worldId,
			center: { x: ox, y: 12, z: oz },
			radius: 0.5,
			direction: { x: 0, y: -1, z: 0 },
			maxDistance: 40,
		});
		physics.overlapBox({
			worldId,
			center: { x: ox, y: 1, z: oz },
			halfExtents: { x: 1.2, y: 1.2, z: 1.2 },
		});
		queries += 3;
	}
	const elapsedMs = performance.now() - start;
	const throughput = queries / Math.max(1e-6, elapsedMs / 1000);
	return {
		totalQueries: queries,
		elapsedMs,
		throughputQps: throughput,
	};
}

async function runScenarioA(adapterFactory, seed, config) {
	const rng = createRng(seed);
	const scene = new Scene();
	const physics = new PhysicsSystem({ adapter: adapterFactory() });
	await physics.init();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: "variable",
	});
	physics.bindSceneSpatial(scene);

	const staticMesh = createGridMeshInstance(config.sceneASegments, -16, 0, -16);
	scene.add(staticMesh);
	scene.updateWorldMatrices();
	const staticBody = physics.attachBody(staticMesh, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(staticBody, {
		mode: "mesh",
		sourceNode: staticMesh,
		meshPolicy: "fixed",
		narrowphase: "face-bvh",
		backendPreference: "exact",
	});

	addDynamicSphereBodies(physics, config.sceneADynamicBodies, "main", rng);
	const result = measureSteps(
		physics,
		config.sceneAStepIterations,
		config.sceneAWarmup
	);
	physics.destroyWorld("main");
	return result;
}

async function runScenarioB(adapterFactory, seed, config) {
	const rng = createRng(seed ^ 0xabc123);
	const scene = new Scene();
	const physics = new PhysicsSystem({ adapter: adapterFactory() });
	await physics.init();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});
	physics.bindSceneSpatial(scene);

	for (let i = 0; i < config.sceneBMeshCount; i++) {
		const mesh = createGridMeshInstance(
			config.sceneBSegments,
			(i % config.sceneBColumns) * 12,
			0,
			Math.floor(i / config.sceneBColumns) * 12
		);
		scene.add(mesh);
		const body = physics.attachBody(mesh, {
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		});
		physics.addCollider(body, {
			mode: "mesh",
			sourceNode: mesh,
			meshPolicy: "fixed",
			narrowphase: "face-bvh",
			backendPreference: "exact",
		});
	}
	scene.updateWorldMatrices();
	addDynamicSphereBodies(physics, config.sceneBDynamicBodies, "main", rng);
	const result = measureQueryThroughput(
		physics,
		"main",
		config.sceneBQueryIterations,
		rng
	);
	physics.destroyWorld("main");
	return result;
}

async function runScenarioC(adapterFactory, seed, config) {
	const rng = createRng(seed ^ 0xdeadbeef);
	const scene = new Scene();
	const physics = new PhysicsSystem({ adapter: adapterFactory() });
	await physics.init();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: -9.8, z: 0 },
		mode: "variable",
	});
	physics.bindSceneSpatial(scene);

	const dynamicMesh = createGridMeshInstance(config.sceneCSegments, -12, 1.5, -12);
	scene.add(dynamicMesh);
	scene.updateWorldMatrices();
	const dynamicBody = physics.attachBody(dynamicMesh, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});
	physics.addCollider(dynamicBody, {
		mode: "mesh",
		sourceNode: dynamicMesh,
		meshPolicy: "dynamic",
		narrowphase: "face-bvh",
		backendPreference: "exact",
	});
	addDynamicSphereBodies(physics, config.sceneCDynamicBodies, "main", rng);

	const result = measureSteps(
		physics,
		config.sceneCStepIterations,
		config.sceneCWarmup,
		(frame) => {
		dynamicMesh.position.x = -12 + Math.sin(frame * 0.08) * 2;
		dynamicMesh.position.z = -12 + Math.cos(frame * 0.08) * 2;
		dynamicMesh.updateWorldMatrix();
		physics.rebuildColliders(dynamicBody);
		}
	);
	physics.destroyWorld("main");
	return result;
}

function formatNumber(value) {
	return Number.isFinite(value) ? value.toFixed(3) : "n/a";
}

function compareWithBaseline(current, baseline) {
	if (!baseline) return null;
	const out = {};
	for (const [adapterName, adapterResult] of Object.entries(current.adapters)) {
		const baseAdapter = baseline.adapters?.[adapterName];
		if (!baseAdapter) continue;
		out[adapterName] = {
			sceneA_stepP50_speedup:
				baseAdapter.sceneA.p50Ms / Math.max(1e-6, adapterResult.sceneA.p50Ms),
			sceneA_stepP95_speedup:
				baseAdapter.sceneA.p95Ms / Math.max(1e-6, adapterResult.sceneA.p95Ms),
			sceneB_query_speedup:
				adapterResult.sceneB.throughputQps /
				Math.max(1e-6, baseAdapter.sceneB.throughputQps),
			sceneC_stepP50_speedup:
				baseAdapter.sceneC.p50Ms / Math.max(1e-6, adapterResult.sceneC.p50Ms),
			sceneC_stepP95_speedup:
				baseAdapter.sceneC.p95Ms / Math.max(1e-6, adapterResult.sceneC.p95Ms),
		};
	}
	return out;
}

async function run() {
	const args = parseArgs(process.argv.slice(2));
	const config =
		args.quick ?
			{
				sceneASegments: 24,
				sceneADynamicBodies: 80,
				sceneAStepIterations: 60,
				sceneAWarmup: 12,
				sceneBSegments: 18,
				sceneBMeshCount: 4,
				sceneBColumns: 2,
				sceneBDynamicBodies: 24,
				sceneBQueryIterations: 600,
				sceneCSegments: 20,
				sceneCDynamicBodies: 56,
				sceneCStepIterations: 50,
				sceneCWarmup: 10,
			}
		:	{
				sceneASegments: 64,
				sceneADynamicBodies: 220,
				sceneAStepIterations: 180,
				sceneAWarmup: 40,
				sceneBSegments: 40,
				sceneBMeshCount: 8,
				sceneBColumns: 4,
				sceneBDynamicBodies: 80,
				sceneBQueryIterations: 2500,
				sceneCSegments: 48,
				sceneCDynamicBodies: 140,
				sceneCStepIterations: 160,
				sceneCWarmup: 30,
			};
	const baseline =
		args.baseline ? JSON.parse(await readFile(args.baseline, "utf8")) : null;

	const adapters = [
		{ name: "simple", create: () => new SimplePhysicsAdapter("simple-bench") },
		{ name: "rapier", create: () => new RapierPhysicsAdapter({ strict: false }) },
		{ name: "ammo", create: () => new AmmoPhysicsAdapter({ strict: false }) },
	];

	const result = {
		timestampIso: new Date().toISOString(),
		seed: args.seed,
		adapters: {},
	};

	for (const adapter of adapters) {
		console.log(`\n[bench] adapter=${adapter.name} scene=A`);
		const sceneA = await runScenarioA(adapter.create, args.seed, config);
		console.log(`[bench] adapter=${adapter.name} scene=B`);
		const sceneB = await runScenarioB(adapter.create, args.seed, config);
		console.log(`[bench] adapter=${adapter.name} scene=C`);
		const sceneC = await runScenarioC(adapter.create, args.seed, config);
		result.adapters[adapter.name] = { sceneA, sceneB, sceneC };
	}

	console.log("\n=== Mesh Collision V2 Bench Summary ===");
	for (const [adapterName, metrics] of Object.entries(result.adapters)) {
		console.log(
			`${adapterName}: ` +
				`A[p50=${formatNumber(metrics.sceneA.p50Ms)}ms p95=${formatNumber(metrics.sceneA.p95Ms)}ms] ` +
				`B[qps=${formatNumber(metrics.sceneB.throughputQps)}] ` +
				`C[p50=${formatNumber(metrics.sceneC.p50Ms)}ms p95=${formatNumber(metrics.sceneC.p95Ms)}ms]`
		);
	}

	const comparison = compareWithBaseline(result, baseline);
	if (comparison) {
		console.log("\n=== Relative Speedup vs Baseline ===");
		for (const [adapterName, metrics] of Object.entries(comparison)) {
			console.log(
				`${adapterName}: ` +
					`A-P50=${formatNumber(metrics.sceneA_stepP50_speedup)}x ` +
					`A-P95=${formatNumber(metrics.sceneA_stepP95_speedup)}x ` +
					`B-QPS=${formatNumber(metrics.sceneB_query_speedup)}x ` +
					`C-P50=${formatNumber(metrics.sceneC_stepP50_speedup)}x ` +
					`C-P95=${formatNumber(metrics.sceneC_stepP95_speedup)}x`
			);
		}
	}

	if (args.out) {
		await writeFile(args.out, JSON.stringify(result, null, 2), "utf8");
		console.log(`\nSaved benchmark report: ${args.out}`);
	}
}

await run();
