import { performance } from "node:perf_hooks";
import { readFile, writeFile } from "node:fs/promises";

import { Camera } from "../../src/cameras/Camera.ts";
import { Scene } from "../../src/core/Scene.ts";
import { PBRMaterial } from "../../src/materials/PBRMaterial.ts";
import { MeshAsset } from "../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../src/meshes/MeshInstance.ts";
import {
	PreparedSceneBuilder,
	PreparedScenePacketCache,
} from "../../src/pipeline/PreparedSceneBuilder.ts";

const MESH_COUNT = 5_000;
const MATERIAL_COUNT = 8;
const WARMUP_FRAMES = 30;
const SAMPLE_FRAMES = 120;

function getArg(name) {
	const direct = process.argv.find((arg) => arg.startsWith(`${name}=`));
	if (direct) return direct.slice(name.length + 1);
	const index = process.argv.indexOf(name);
	return index >= 0 ? process.argv[index + 1] ?? null : null;
}

function createMesh(material) {
	return MeshAsset.fromFaces([{
		material,
		vertices: [
			{ x: -0.5, y: -0.5, z: 0, normal: { x: 0, y: 0, z: 1 } },
			{ x: 0.5, y: -0.5, z: 0, normal: { x: 0, y: 0, z: 1 } },
			{ x: 0, y: 0.5, z: 0, normal: { x: 0, y: 0, z: 1 } },
		],
	}]);
}

const materials = Array.from({ length: MATERIAL_COUNT }, (_, index) =>
	new PBRMaterial({ name: `CpuHotpathMaterial${index}` }));
const meshes = materials.map(createMesh);
const scene = new Scene();
const camera = scene.add(new Camera());
for (let index = 0; index < MESH_COUNT; index++) {
	const instance = new MeshInstance({ mesh: meshes[index % meshes.length] });
	instance.position.set(
		((index % 100) - 50) * 0.05,
		(Math.floor(index / 100) - 25) * 0.05,
		-10 - (index % 7) * 0.01,
	);
	scene.add(instance);
}
scene.updateWorldMatrices();
camera.updateMatrices();
const packetCache = new PreparedScenePacketCache();

function sampleFrame(frameIndex) {
	camera.position.x = Math.sin(frameIndex * 0.01) * 0.25;
	scene.updateWorldMatrices();
	camera.updateMatrices();
	packetCache.beginFrame();
	const start = performance.now();
	const frame = PreparedSceneBuilder.build(
		{ scene, camera, hasActiveAnimations: false },
		{ packetCache },
	);
	const elapsedMs = performance.now() - start;
	packetCache.endFrame();
	if (frame.opaquePackets.length !== MESH_COUNT) {
		throw new Error(`Expected ${MESH_COUNT} visible packets, got ${frame.opaquePackets.length}.`);
	}
	return elapsedMs;
}

for (let frame = 0; frame < WARMUP_FRAMES; frame++) sampleFrame(frame);
const samples = [];
for (let frame = 0; frame < SAMPLE_FRAMES; frame++) {
	samples.push(sampleFrame(frame + WARMUP_FRAMES));
}
samples.sort((left, right) => left - right);
const medianMs = samples[Math.floor(samples.length * 0.5)];
const p95Ms = samples[Math.floor(samples.length * 0.95)];
const report = {
	meshCount: MESH_COUNT,
	materialCount: MATERIAL_COUNT,
	warmupFrames: WARMUP_FRAMES,
	sampleFrames: SAMPLE_FRAMES,
	medianMs,
	p95Ms,
	packetCache: packetCache.getDebugStats(),
};
const baselinePath = getArg("--baseline");
if (baselinePath) {
	const baseline = JSON.parse(await readFile(baselinePath, "utf8"));
	const medianSpeedup = baseline.medianMs / medianMs;
	const p95Speedup = baseline.p95Ms / p95Ms;
	report.baseline = { medianSpeedup, p95Speedup };
	if (medianSpeedup < 3 || p95Speedup < 2.5) {
		throw new Error(
			`CPU hotpath target missed: median ${medianSpeedup.toFixed(2)}x, ` +
			`p95 ${p95Speedup.toFixed(2)}x.`,
		);
	}
}
const out = getArg("--out");
if (out) await writeFile(out, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(JSON.stringify(report, null, 2));
