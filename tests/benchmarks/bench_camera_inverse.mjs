import { cpus } from "node:os";
import { writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";

import { Camera } from "../../src/cameras/Camera.ts";
import { Matrix4 } from "../../src/maths/Matrix4.ts";
import { WebGPUFrameBindingCache } from "../../src/backends/webgpu/WebGPUFrameBindingCache.ts";

// CPU measurements only: buffer writes are counted without a GPU or browser.
const sampleCount = 9;
const getterIterations = 100_000;
const frameIterations = 5_000;
let sink = 0;

function argument(name) {
	const index = process.argv.indexOf(name);
	return index < 0 ? undefined : process.argv[index + 1];
}

function createCamera() {
	const camera = new Camera({ aspectRatio: 16 / 9 });
	camera.position.set(3, 4, 5);
	camera.setRotationFromEuler(0.2, 0.4, 0.1);
	return camera;
}

function readInverses(camera) {
	return camera.inverseViewMatrix.elements[0][3]
		+ camera.inverseProjectionMatrix.elements[1][1];
}

function createGetterCase(change, reads = 1) {
	const camera = createCamera();
	const projectionX = camera.projectionMatrix.elements[0][0];
	return {
		step(index) {
			if (change !== "static") {
				camera.viewMatrix.elements[0][3] = (index & 1023) * 0.001;
				camera.viewProjectionMatrix.elements[0][3] = (index & 1023) * 0.002;
			}
			if (change === "all") {
				camera.projectionMatrix.elements[0][0] = projectionX + (index & 1023) * 0.0001;
			}
			let value = 0;
			for (let read = 0; read < reads; read++) value += readInverses(camera);
			return value;
		},
	};
}

function createCaptureCase() {
	return {
		step(index) {
			const camera = new Camera({ fov: 90, aspectRatio: 1 });
			camera.position.set((index & 1023) * 0.001, 2, 5);
			camera.updateMatrices();
			camera.projectionMatrix.applyObliqueClipping({
				normal: { x: 0, y: -1, z: 0 }, constant: -1,
			});
			camera.viewProjectionMatrix = Matrix4.multiply(camera.projectionMatrix, camera.viewMatrix);
			return readInverses(camera);
		},
	};
}

function createFrameCase(moving, scopes = 1) {
	const camera = createCamera();
	const token = {};
	const backend = {
		bytesWritten: 0,
		createBuffer: (descriptor) => ({ ...descriptor, destroy() {} }),
		writeBuffer(buffer, data) { this.bytesWritten += data.byteLength; },
	};
	const registry = {
		getWhiteTexture: () => token, getWhiteSampler: () => token,
		getTextureForSlot: () => token, getSamplerForTexture: () => token,
	};
	const shadow = { atlas: null, transmittanceAtlas: null };
	const caches = Array.from({ length: scopes }, () =>
		new WebGPUFrameBindingCache(backend, null, null, registry, shadow));
	const frame = {
		camera,
		environment: {
			backgroundEnabled: true, lightingEnabled: true,
			backgroundTexture: null, iblTexture: null,
			backgroundStrength: 1, backgroundExposure: 1,
			backgroundTintLinear: { r: 1, g: 1, b: 1 },
		},
	};
	const lighting = {
		ambientColor: [0.1, 0.2, 0.3], directionalLights: [], pointLights: [],
		spotLights: [], areaLights: [], directionalShadows: [], spotShadows: [],
	};
	const environment = {
		shAmbientCoeffs: null, localLightProbeCount: 0, localLightProbes: [],
		irradianceProbeGrid: null, reflectionProbeCount: 0, reflectionProbes: [],
		brdfLUTTexture: null, envSpecularMaxMipLevel: 0,
	};
	const features = {
		enableLighting: true, enableShadows: false, enableClusteredLighting: false,
		postProcess: { getOptions: () => undefined, isEnabled: () => false },
	};
	const options = { temporalStateMode: "disabled" };
	return {
		step(index) {
			if (moving) camera.position.x = (index & 1023) * 0.001;
			// The renderer refreshes matrices before backend frame preparation.
			camera.updateMatrices();
			for (const cache of caches) {
				cache.prepare(frame, lighting, environment, features, 1920, 1080, "mrt", options);
			}
			return backend.bytesWritten;
		},
		destroy() { for (const cache of caches) cache.destroy(); },
	};
}

const definitions = [
	{ name: "getters-static", create: () => createGetterCase("static") },
	{ name: "getters-moving", create: () => createGetterCase("moving") },
	{ name: "getters-all-changing", create: () => createGetterCase("all") },
	{ name: "getters-static-four-reads", create: () => createGetterCase("static", 4) },
	{ name: "getters-moving-four-reads", create: () => createGetterCase("moving", 4) },
	{ name: "capture-one-use", create: createCaptureCase, iterations: 10_000 },
	{ name: "frame-static", create: () => createFrameCase(false), iterations: frameIterations },
	{ name: "frame-moving", create: () => createFrameCase(true), iterations: frameIterations },
	{ name: "frame-moving-four-scopes", create: () => createFrameCase(true, 4),
		iterations: frameIterations },
];

function measureInverseCalls(create) {
	const context = create();
	const original = Matrix4.inverse;
	let calls = 0;
	try {
		context.step(-1);
		Matrix4.inverse = (...args) => { calls++; return original(...args); };
		for (let index = 0; index < 256; index++) sink += context.step(index);
		return calls / 256;
	} finally {
		Matrix4.inverse = original;
		context.destroy?.();
	}
}

const results = [];
for (const definition of definitions) {
	const iterations = definition.iterations ?? getterIterations;
	const context = definition.create();
	const warmupIterations = Math.min(iterations, 20_000);
	for (let index = 0; index < warmupIterations; index++) sink += context.step(index);
	const samplesUs = [];
	for (let sample = 0; sample < sampleCount; sample++) {
		Bun.gc(true);
		let checksum = 0;
		const start = performance.now();
		for (let index = 0; index < iterations; index++) checksum += context.step(index);
		const elapsedMs = performance.now() - start;
		sink += checksum;
		samplesUs.push(elapsedMs * 1000 / iterations);
	}
	context.destroy?.();
	const sorted = [...samplesUs].sort((left, right) => left - right);
	results.push({
		name: definition.name, iterations, warmupIterations, samplesUs,
		medianUs: sorted[Math.floor(sorted.length / 2)],
		minUs: sorted[0], maxUs: sorted.at(-1),
	});
}

// Count separately after all timings so replacing Matrix4.inverse cannot
// invalidate optimized code between measured cases.
for (let index = 0; index < definitions.length; index++) {
	results[index].inverseCallsPerOperation = measureInverseCalls(definitions[index].create);
}

if (!Number.isFinite(sink) || sink === 0) throw new Error("Invalid benchmark checksum.");
const report = {
	label: argument("--label") ?? "current",
	runtime: `Bun ${Bun.version}`, platform: process.platform, arch: process.arch,
	cpu: cpus()[0]?.model, timestamp: new Date().toISOString(),
	sampleCount, inverseGettersPerRead: 2, units: "microseconds per logical operation",
	scope: "CPU only; frame cases include camera updates and frame uniform packing with no-op GPU writes",
	checksum: sink, results,
};
const json = `${JSON.stringify(report, null, 2)}\n`;
const out = argument("--out");
if (out) await writeFile(out, json, "utf8");
console.log(json);
