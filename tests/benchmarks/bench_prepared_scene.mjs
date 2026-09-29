import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// --root runs this exact harness against another checkout without editing it.
const args = new Map(process.argv.slice(2).map((arg) => {
	const separator = arg.indexOf("=");
	return separator < 0 ? [arg, "true"] : [arg.slice(0, separator), arg.slice(separator + 1)];
}));
const root = resolve(args.get("--root") ?? fileURLToPath(new URL("../../", import.meta.url)));
const load = (path) => import(pathToFileURL(resolve(root, path)).href);
const { Scene } = await load("src/core/Scene.ts");
const { Camera } = await load("src/cameras/Camera.ts");
const { Material, AlphaMode } = await load("src/materials/Material.ts");
const { MeshAsset } = await load("src/meshes/MeshAsset.ts");
const { MeshInstance } = await load("src/meshes/MeshInstance.ts");
const { IdGenerator } = await load("src/foundation/IdGenerator.ts");
const { PreparedSceneCache } = await load("src/pipeline/PreparedSceneCache.ts");
const { PreparedScenePacketCache } = await load("src/pipeline/PreparedSceneBuilder.ts");
const { PreparedSceneTileSpatialIndex } = await load("src/pipeline/PreparedSceneSpatialIndex.ts");
const { DEFAULT_INCREMENTAL_RENDERING_OPTIONS } = await load("src/pipeline/incremental.ts");
const { createResolvedPostProcess } = await load("tests/helpers/postprocess.mjs");
const rounds = 5;
const warmup = 20;
const samplesPerRound = args.has("--quick") ? 5 : 20;
const counts = args.get("--count") ? [Number(args.get("--count"))] : [1_000, 10_000];
const queryRect = { x: 150, y: 50, width: 500, height: 350 };

function createFixture(count, shared, changing) {
	// Fixture-local IDs also keep ID-based sort ties identical in filtered reruns.
	// This benchmark runs in its own process; no live renderer shares this counter.
	IdGenerator._counts.clear();
	const scene = new Scene();
	const camera = scene.add(new Camera());
	const materials = new Set();
	const makeMesh = () => MeshAsset.fromFaces(Array.from({ length: 4 }, (_, index) => ({
		material: new Material({ alphaMode: index === 3 ? AlphaMode.Blend : AlphaMode.Opaque }),
		vertices: [{ x: 0, y: 0, z: 0 }, { x: 0.05, y: 0, z: 0 }, { x: 0, y: 0.05, z: 0 }],
	})));
	const sharedMesh = shared ? makeMesh() : null;
	if (sharedMesh) {
		for (let i = 1; i < 3; i++) sharedMesh.primitives[i].material = sharedMesh.primitives[0].material;
	}
	const instances = [];
	let seed = 20260929;
	const random = () => {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		return seed / 0x100000000;
	};
	for (let i = 0; i < count / 4; i++) {
		const mesh = sharedMesh ?? makeMesh();
		for (const primitive of mesh.primitives) materials.add(primitive.material);
		const instance = scene.add(new MeshInstance({ mesh }));
		instance.position.x = (random() - 0.5) * 2;
		instance.position.y = (random() - 0.5) * 2;
		instance.position.z = -5 - random();
		instances.push(instance);
	}
	scene.updateWorldMatrices();
	camera.updateMatrices();
	const cache = new PreparedSceneCache();
	const input = {
		source: { scene, camera, hasActiveAnimations: false },
		viewportWidth: 800, viewportHeight: 450,
		features: {}, postProcess: createResolvedPostProcess(),
		incrementalOptions: { ...DEFAULT_INCREMENTAL_RENDERING_OPTIONS, fullFrameFallbackAreaRatio: 1 },
	};
	let iteration = 0;
	const last = instances.at(-1);
	const initialX = last.position.x;
	return {
		cache, input, instances, materials,
		update() {
			if (changing) {
				last.position.x = initialX + (++iteration % 2) * 0.01;
				last.updateWorldMatrix(scene.root.worldMatrix);
			}
		},
	};
}

function execute(fixture, queries) {
	const result = fixture.cache.build(fixture.input);
	const hits = [];
	if (queries !== "none") hits.push(result.frame.spatialIndex.queryOpaquePackets(queryRect));
	if (queries === "both") hits.push(result.frame.spatialIndex.queryTransparentPackets(queryRect));
	return { result, hits };
}

function fingerprint({ result, hits }) {
	const ordinals = new Map(result.frame.submissions.map((submission, index) => [
		submission.id, index,
	]));
	const packetOrder = (packets) => packets.map((packet) => ordinals.get(packet.submission.id));
	return createHash("sha256").update(JSON.stringify({
		opaque: packetOrder(result.frame.opaquePackets),
		transparent: packetOrder(result.frame.transparentPackets),
		rects: [...result.packetRects].map(([id, rect]) => [ordinals.get(id), rect]),
		dirtyRects: result.dirtyRects,
		dirtyTiles: result.dirtyTiles, forceFullFrame: result.forceFullFrame,
		hits: hits.map(packetOrder),
	})).digest("hex");
}

// Run only after ALL timing cases: replacing accessors/methods can invalidate JIT code
// even after they are restored, so interleaving probes would disturb later cases.
function countWork(fixture, queries) {
	const work = { signatureChecks: 0, materialHashes: 0, matrixHashes: 0, treeBuilds: 0 };
	const restore = [];
	let processingDirtyState = 0;
	const wrap = (owner, key, wrapper) => {
		const original = owner[key];
		owner[key] = wrapper(original);
		restore.push(() => { owner[key] = original; });
	};
	wrap(PreparedScenePacketCache.prototype, "_isSignatureCurrent", (original) => function (...args) {
		work.signatureChecks++;
		return original.apply(this, args);
	});
	wrap(PreparedSceneTileSpatialIndex.prototype, "_buildPacketTree", (original) => function (...args) {
		work.treeBuilds++;
		return original.apply(this, args);
	});
	for (const method of ["_processPacketList", "_processFrameDecals"]) {
		wrap(PreparedSceneCache.prototype, method, (original) => function (...args) {
			processingDirtyState++;
			try { return original.apply(this, args); } finally { processingDirtyState--; }
		});
	}
	const observe = (owner, key, counter) => {
		const descriptor = Object.getOwnPropertyDescriptor(owner, key);
		const value = owner[key];
		Object.defineProperty(owner, key, {
			configurable: descriptor.configurable,
			enumerable: descriptor.enumerable,
			get() { if (processingDirtyState) work[counter]++; return value; },
		});
		restore.push(() => Object.defineProperty(owner, key, descriptor));
	};
	try {
		for (const material of fixture.materials) observe(material, "opacity", "materialHashes");
		for (const instance of fixture.instances) observe(instance.worldMatrix, "elements", "matrixHashes");
		fixture.update();
		const output = execute(fixture, queries);
		return { work, fingerprint: fingerprint(output) };
	} finally {
		for (const undo of restore.reverse()) undo();
	}
}

function quantile(values, fraction) {
	const sorted = values.toSorted((a, b) => a - b);
	return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

const results = [];
for (const count of counts) {
	assert.ok(Number.isInteger(count) && count > 0 && count % 4 === 0);
	for (const shared of [true, false]) {
		for (const changing of [false, true]) {
			for (const queries of ["none", "opaque", "both"]) {
				const name = `${count}/${shared ? "shared" : "unique"}/${changing ? "changing" : "static"}/${queries}`;
				if (args.has("--filter") && !name.includes(args.get("--filter"))) continue;
				Bun.gc(true);
				const fixture = createFixture(count, shared, changing);
				for (let i = 0; i < warmup; i++) { fixture.update(); execute(fixture, queries); }
				const samples = [];
				const roundMedians = [];
				for (let round = 0; round < rounds; round++) {
					Bun.gc(true);
					const batch = [];
					for (let i = 0; i < samplesPerRound; i++) {
						fixture.update();
						const start = performance.now();
						const output = execute(fixture, queries);
						batch.push(performance.now() - start);
						assert.equal(output.result.frame.submissions.length, count);
					}
					samples.push(...batch);
					roundMedians.push(quantile(batch, 0.5));
				}
				const result = {
					name, medianMs: quantile(samples, 0.5), p95Ms: quantile(samples, 0.95),
					roundMedians,
				};
				results.push(result);
				console.log(JSON.stringify(result));
				fixture.cache.reset();
			}
		}
	}
}
for (const result of results) {
	const [count, materials, movement, queries] = result.name.split("/");
	const fixture = createFixture(Number(count), materials === "shared", movement === "changing");
	for (let i = 0; i < 2; i++) { fixture.update(); execute(fixture, queries); }
	Object.assign(result, countWork(fixture, queries));
	console.log(JSON.stringify({ name: result.name, work: result.work, fingerprint: result.fingerprint }));
	fixture.cache.reset();
}
const report = {
	root, runtime: process.versions, rounds, warmup, samplesPerRound,
	gcBetweenRounds: true, probesAfterTiming: true, results,
};
if (args.has("--out")) await writeFile(args.get("--out"), JSON.stringify(report, null, "\t") + "\n");
if (args.has("--baseline")) {
	const baseline = JSON.parse(await readFile(args.get("--baseline"), "utf8"));
	assert.equal(baseline.rounds, rounds);
	assert.equal(baseline.warmup, warmup);
	assert.equal(baseline.samplesPerRound, samplesPerRound);
	assert.equal(baseline.gcBetweenRounds, report.gcBetweenRounds);
	assert.equal(baseline.probesAfterTiming, report.probesAfterTiming);
	for (const result of results) {
		const before = baseline.results.find((item) => item.name === result.name);
		assert.ok(before, `missing baseline: ${result.name}`);
		assert.equal(result.fingerprint, before.fingerprint, `output changed: ${result.name}`);
		const medianChangePercent = (result.medianMs / before.medianMs - 1) * 100;
		console.log(JSON.stringify({
			name: result.name, medianChangePercent, needsRetest: medianChangePercent > 10,
		}));
	}
}
