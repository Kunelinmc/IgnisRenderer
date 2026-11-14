import assert from "node:assert/strict";

import * as postprocess from "../../../src/postprocess/index.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";
import { WebGPUPostProcessRuntime } from "../../../src/backends/webgpu/WebGPUPostProcessRuntime.ts";
import { createPostProcessResourceAccessor } from "../../../src/postprocess/PostProcessResourceAccessor.ts";
import { resolvePostProcessResourceDescriptor } from "../../../src/postprocess/resourceDescriptors.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import { compileWGSLMath } from "../../helpers/wgsl-math.mjs";
import { createPostProcessTestImplementations } from "../../helpers/postprocess.mjs";
import { FakeBackend, FakeEncoder, assertClose } from "../../helpers/webgpu_postprocess_runtime_test_helpers.mjs";

const owners = createPostProcessTestImplementations();

function createPass(options = {}) {
	assert.equal(typeof postprocess.LensFlarePass, "function",
		"HDR highlights need a configurable lens flare pass");
	return new postprocess.LensFlarePass({ enabled: true, options });
}

function createHarness(pass, width = 65, height = 33) {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(createWebGPUComputeFacade(backend));
	const encoder = new FakeEncoder();
	const implementation = owners.get(pass, "webgpu");
	const options = pass.normalizeOptions();
	const declaration = implementation.describeExecution({ options, width, height });
	const texture = (w, h, label) => backend.createTexture({
		width: w, height: h, label, format: "rgba16float", usage: 0,
	});
	const input = texture(width, height, "scene");
	const output = texture(width, height, "output");
	const transients = new Map(declaration.transients.map(({ descriptor }) => {
		const resolved = resolvePostProcessResourceDescriptor(descriptor, width, height);
		return [descriptor.id, texture(resolved.width, resolved.height, descriptor.id)];
	}));
	const resources = createPostProcessResourceAccessor({
		passId: pass.id, declaration, colorInput: input, colorOutput: output,
		getGBuffer: () => { throw new Error("Lens flare must not read a G-buffer"); },
		getHistory: () => { throw new Error("Lens flare must not read history"); },
		getTransient: (id) => transients.get(id) ?? null,
		getShared: () => { throw new Error("Lens flare must not read bloom resources"); },
	});
	return {
		backend, runtime, encoder, implementation, transients, input, output, declaration,
		request: { passId: pass.id, pass, options, declaration },
		context: {
			encoder, targets: { postPing: output, postPong: null },
			shared: runtime, resources, volumetricLightingData: null,
		},
	};
}

function testOptionsAndBackendEligibility() {
	const pass = createPass();
	assert.deepEqual(pass.normalizeOptions(), postprocess.DEFAULT_LENS_FLARE_OPTIONS);
	assert.equal(owners.get(pass, "software"), null);
	assert.equal(owners.get(pass, "webgl"), null);
	assert.equal(pass.alphaContract, "premultiplied");
	assert.deepEqual(pass.colorContract, {
		input: "scene-linear-hdr", output: "scene-linear-hdr",
	});
	assert.equal(pass.schedule.placement, "hdr");
	assert.ok(pass.schedule.order < new postprocess.BloomPass().schedule.order);
	pass.setOptions({
		threshold: NaN, softKnee: -5, intensity: Infinity,
		ghostCount: 100, downsample: -1, ghostSpacing: -1,
		haloRadius: 9, haloWidth: -2, chromaticAberration: 8,
		blurRadius: NaN, edgeFade: 3, ghostRadius: 1, maxGhostScale: -1,
	});
	const options = pass.normalizeOptions();
	assert.equal(options.threshold, 1);
	assert.equal(options.softKnee, 0);
	assert.equal(options.intensity, 0.22);
	assert.equal(options.ghostCount, 8);
	assert.equal(options.downsample, 1);
	assert.equal(options.ghostSpacing, 0);
	assert.equal(options.haloRadius, 1);
	assert.equal(options.haloWidth, 0);
	assert.equal(options.chromaticAberration, 0.05);
	assert.equal(options.blurRadius, 0.75);
	assert.equal(options.edgeFade, 0.5);
	assert.equal(options.ghostRadius, 0.08);
	assert.equal(options.maxGhostScale, 1);
	owners.destroy(pass);
}

function testHDRPlanningWithoutGBuffer() {
	const registry = new postprocess.PostProcessPassRegistry();
	const flare = createPass();
	// Registration order must not change the HDR effect order.
	registry.registerPass(new postprocess.ToneMappingPass({ enabled: true }));
	registry.registerPass(new postprocess.BloomPass({ enabled: true }));
	registry.registerPass(flare);
	const plan = new postprocess.PostProcessPlanner().plan({
		postProcess: registry.createSnapshot("webgpu"), backend: "webgpu",
		frameContext: {
			alphaMode: "premultiplied", attachments: { width: 65, height: 33 },
			incremental: { enabled: false, forceFullFrame: false, firstPass: null },
		},
		gBuffer: {
			width: 65, height: 33, normalSpace: "view", depthEncoding: "linear-view-z",
			channels: {}, worldPosition: { source: "derived", available: false },
		},
		resolveImplementation: (pass) => owners.get(pass, "webgpu"),
	});
	assert.deepEqual(plan.passes.map((entry) => entry.id), ["lens-flare", "bloom", "tonemap"]);
	assert.equal(plan.transientDescriptors.length, 3);
	assert.equal(plan.historyDescriptors.length, 0);
	assert.equal(plan.initialColorDomain, "scene-linear-hdr");
	assert.equal(plan.outputColorDomain, "display-linear");
	owners.destroy();
}

async function testDeclaredDispatchesAndResourceReuse() {
	const pass = createPass();
	const h = createHarness(pass);
	assert.equal(h.declaration.color.output, "new-version");
	assert.equal(h.declaration.gBuffer?.length ?? 0, 0);
	assert.equal(h.declaration.histories?.length ?? 0, 0);
	assert.equal(h.declaration.shared?.length ?? 0, 0);
	assert.equal(h.transients.size, 3);
	for (const entry of h.declaration.transients) {
		assert.equal(entry.descriptor.format, "rgba16float");
		assert.equal(entry.descriptor.widthScale, 0.25);
		assert.deepEqual(entry.uses.map((use) => use.usage).sort(), ["sampled", "storage"]);
	}
	const textureCount = h.backend.textures.length;
	await h.implementation.warmup(h.context);
	assert.equal(h.encoder.calls.length, 0, "warmup must not record a frame");
	assert.equal(h.backend.textures.length, textureCount, "scratch textures are graph-owned");
	assert.deepEqual(await h.implementation.execute(h.request, h.context), { ran: true });
	assert.deepEqual(h.encoder.calls.filter(([name]) => name === "dispatchWorkgroups"), [
		["dispatchWorkgroups", 2, 1, 1], ["dispatchWorkgroups", 2, 1, 1],
		["dispatchWorkgroups", 2, 1, 1], ["dispatchWorkgroups", 2, 1, 1],
		["dispatchWorkgroups", 2, 1, 1], ["dispatchWorkgroups", 2, 1, 1],
		["dispatchWorkgroups", 2, 1, 1],
		["dispatchWorkgroups", 9, 5, 1],
	]);
	assert.deepEqual(h.backend.computePipelines.map((p) => p.desc.compute.entryPoint), [
		"csExtract", "csAperturePrefilterH", "csAperturePrefilterV", "csAperture",
		"csFeatures", "csBlurH", "csBlurV", "csComposite",
	]);
	assert.equal(new Set(h.backend.computePipelines.map((p) => p.desc.compute.module)).size, 1);
	assert.equal(h.backend.writeBufferCalls, 1, "all dispatches share immutable frame parameters");
	const params = h.backend.buffers.find((b) => b.label === "WebGPULensFlareParams");
	assert.equal(params.desc.size, 60);
	assertClose(params.lastWrite[12], 65 / 33);
	assertClose(params.lastWrite[13], 0.035);
	assertClose(params.lastWrite[14], 2);
	const groups = h.backend.bindingGroups.slice();
	const writes = groups.map((group) => group.entries.find((entry) => entry.binding === 3).resource);
	const [bright, features, scratch] = [...h.transients.values()];
	assert.deepEqual(writes, [bright, scratch, features, scratch, features, scratch, features, h.output]);
	const featureGroup = groups[4];
	assert.strictEqual(featureGroup.entries.find((entry) => entry.binding === 0).resource, scratch);
	assert.strictEqual(featureGroup.entries.find((entry) => entry.binding === 4).resource, bright,
		"halos must sample raw highlights independently of the hexagonal aperture");
	for (const group of groups) {
		const source = group.entries.find((entry) => entry.binding === 0).resource;
		assert.notStrictEqual(source, group.entries.find((entry) => entry.binding === 3).resource);
	}
	await h.implementation.execute(h.request, h.context);
	assert.equal(h.backend.bindingGroups.length, groups.length);
	assert.equal(h.backend.computePipelines.length, 8);
	assert.equal(h.backend.textures.length, textureCount);
	owners.destroy(pass);
	assert.ok(groups.every((group) => group.destroyed));
	assert.ok(h.backend.computePipelines.every((pipeline) => pipeline.destroyed));
	assert.ok(h.backend.shaderModules.every((module) => module.destroyed));
	assert.equal(params.destroyed, true);
	assert.ok([...h.transients.values()].every((texture) => !texture.destroyed));
	h.runtime.destroy();
}

async function testSkipInvalidationResizeAndTinyTargets() {
	const pass = createPass({ intensity: 0 });
	const h = createHarness(pass, 1, 1);
	assert.deepEqual(await h.implementation.execute(h.request, h.context), { ran: false });
	assert.equal(h.backend.computePipelines.length, 0);
	assert.equal(h.encoder.calls.length, 0);
	h.request.options = { ...pass.normalizeOptions(), intensity: 1, ghostIntensity: 0, haloIntensity: 0 };
	assert.deepEqual(await h.implementation.execute(h.request, h.context), { ran: false });
	h.request.options = { ...pass.normalizeOptions(), intensity: 1 };
	assert.deepEqual(await h.implementation.execute(h.request, undefined), { ran: false });
	await h.implementation.execute(h.request, h.context);
	assert.ok(h.encoder.calls.filter(([name]) => name === "dispatchWorkgroups")
		.every((call) => call.slice(1).every((dimension) => dimension === 1)));
	const firstPipelines = h.backend.computePipelines.slice();
	h.implementation.invalidate();
	assert.ok(firstPipelines.every((pipeline) => pipeline.destroyed));
	await h.implementation.execute(h.request, h.context);
	assert.equal(h.backend.computePipelines.length, 16);
	const replacement = h.backend.createTexture({
		width: 1, height: 1, format: "rgba16float", usage: 0, label: "resized-features",
	});
	h.transients.set("lens-flare:features", replacement);
	const oldGroups = h.backend.bindingGroups.slice(-8);
	await h.implementation.execute(h.request, h.context);
	assert.ok(oldGroups.some((group) => group.destroyed), "replaced targets must refresh bindings");
	owners.destroy(pass);
	h.runtime.destroy();
}

async function loadOpticalMath() {
	const shader = await ShaderSource.load("webgpu.postprocess.lensFlare");
	const functionNames = [
		"flareThresholdWeight", "flareGhostUv", "flareHaloOffset", "flareUvValid",
		"flareEdgeWeight", "flareComposite", "flareExtract", "flareGhostSourceScale",
		"flareGhostWeight", "flareApertureSteps", "flareApertureAxis", "flareApertureOffset",
		"flareApertureSigma", "flareGaussianWeight", "flareAperturePoint", "flareApertureWeight",
		"flareHaloWeight",
	];
	// Interpret the shader's pure math, excluding texture operations and uniforms.
	const source = [...shader.source.code.matchAll(/fn\s+(\w+)\s*\([^)]*\)\s*->[^{}]+\{[^{}]*\}/g)]
		.filter((match) => functionNames.includes(match[1]))
		.map((match) => match[0]).join("\n");
	return compileWGSLMath(source, functionNames);
}

async function testShaderHighlightAndOpticalMath() {
	const math = await loadOpticalMath();
	assert.equal(math.flareThresholdWeight(0, 0, 0), 0);
	assert.equal(math.flareThresholdWeight(0.4, 1, 0.5), 0);
	assert.equal(math.flareThresholdWeight(1, 1, 0), 0, "zero knee must use a hard threshold");
	assertClose(math.flareThresholdWeight(2, 1, 0), 0.5);
	assert.deepEqual(math.flareExtract([0, 0, 0, 1], 0, 0), [0, 0, 0, 0]);
	assert.deepEqual(math.flareExtract([0.4, 0.4, 0.4, 1], 1, 0.5), [0, 0, 0, 0]);
	assert.deepEqual(math.flareExtract([5, 2, 1, 0], 1, 0), [0, 0, 0, 0]);
	const opaque = math.flareExtract([2, 2, 2, 1], 1, 0);
	const transparent = math.flareExtract([1, 1, 1, 0.5], 1, 0);
	for (let lane = 0; lane < 4; lane++) {
		assertClose(transparent[lane], opaque[lane] * 0.5);
	}
	let previous = 0;
	for (const luma of [0.5, 0.8, 1, 1.2, 2, 10]) {
		const weight = math.flareThresholdWeight(luma, 1, 0.5);
		assert.ok(weight >= previous && weight <= 1);
		previous = weight;
	}
	assert.deepEqual(math.flareGhostUv([0.25, 0.5], 0.5, 0, 2), [0.875, 0.5]);
	assert.deepEqual(math.flareGhostUv([0.25, 0.5], 0.5, 1, 2), [0.75, 0.5]);
	for (const limit of [1, 2, 4]) {
		for (const spacing of [0, 0.3, 0.5, 1]) {
			for (let index = 0; index < 8; index++) {
				const sourceScale = math.flareGhostSourceScale(spacing, index, limit);
				assert.ok(Number.isFinite(sourceScale));
				assert.ok(1 / Math.abs(sourceScale) <= limit + 1e-6);
			}
		}
	}
	const scales = [0, 1, 2, 3].map((i) => math.flareGhostSourceScale(0.3, i, 2));
	assert.equal(new Set(scales).size, 4, "bounded ghosts must retain distinct scales");
	assertClose(math.flareGhostWeight(0.5, 0), math.flareGhostWeight(1, 0) / 4);
	assert.ok(math.flareGhostWeight(1, 3) < math.flareGhostWeight(1, 0));
	assert.equal(math.flareUvValid([-0.01, 0.5]), false);
	assert.equal(math.flareUvValid([1.01, 0.5]), false);
	assert.equal(math.flareUvValid([0.5, 0.5]), true);
	assert.deepEqual(math.flareHaloOffset([0.5, 0.5], 2, 0.35), [0, 0]);
	const horizontal = math.flareHaloOffset([0.8, 0.5], 2, 0.35);
	const vertical = math.flareHaloOffset([0.5, 0.8], 2, 0.35);
	assertClose(horizontal[0] * 2, vertical[1]);
	assert.equal(math.flareHaloWeight([0.5, 0.5], [0.5, 0.5], 2), 0);
	assertClose(math.flareHaloWeight([0.9, 0.5], [0.51, 0.5], 2), 0.025);
	assert.equal(math.flareHaloWeight([0.6, 0.5], [0.9, 0.5], 2), 1);
	assertClose(math.flareHaloWeight([0.7, 0.5], [0.6, 0.5], 2),
		math.flareHaloWeight([0.5, 0.9], [0.5, 0.7], 2));
	assert.equal(math.flareEdgeWeight([0, 0.5], 0.1), 0);
	assert.equal(math.flareEdgeWeight([0.5, 0.5], 0.1), 1);
	assert.equal(math.flareEdgeWeight([0, 0.5], 0), 1);
	assert.deepEqual(math.flareComposite([2, 1, 0, 0.5], [1, 0, 0, 0.5], 0), [2, 1, 0, 0.5]);
	assert.deepEqual(math.flareComposite([0, 0, 0, 0], [2, 1, 0, 0.5], 1), [2, 1, 0, 0.5]);
	assert.deepEqual(math.flareComposite([0, 0, 0, 0], [0, 0, 0, 0], 1), [0, 0, 0, 0]);
	assertClose(math.flareComposite([1, 1, 1, 0.5], [1, 1, 1, 0.5], 1)[3], 0.75);
}

async function testApertureImpulseContinuity() {
	const math = await loadOpticalMath();
	for (const [height, downsample, radius] of [[1080, 1, 0.035], [2160, 4, 0.08]]) {
		const radiusPixels = radius * height / downsample;
		const sigma = math.flareApertureSigma(radiusPixels);
		const extent = 3 * sigma;
		const steps = math.flareApertureSteps(extent);
		assert.ok(extent / steps <= 1, "prefilter tap spacing must stay within one texel");
		assert.ok(sigma + 1e-6 >= radiusPixels / 8, "prefilter sigma must cover half the lattice spacing");
		const center = Math.ceil(radiusPixels + extent + 5);
		const size = center * 2 + 1;
		let image = new Float64Array(size * size);
		image[center * size + center] = 1;
		for (let axisIndex = 0; axisIndex < 3; axisIndex++) {
			const offsets = [];
			if (axisIndex < 2) {
				const axis = math.flareApertureAxis(axisIndex);
				for (let i = -steps; i <= steps; i++) {
					const offset = math.flareApertureOffset(axis, [extent, extent], i, steps);
					offsets.push([...offset, math.flareGaussianWeight(extent * i / steps, sigma)]);
				}
			} else {
				for (let r = -4; r <= 4; r++) {
					for (let q = -4; q <= 4; q++) {
						if (Math.abs(q + r) > 4) continue;
						const point = math.flareAperturePoint(q, r);
						offsets.push([point[0] * radiusPixels, point[1] * radiusPixels,
							math.flareApertureWeight(q, r)]);
					}
				}
			}
			const weightSum = offsets.reduce((sum, entry) => sum + entry[2], 0);
			const output = new Float64Array(image.length);
			// Sample an impulse image using the shader's actual axes and offsets.
			for (let y = 0; y < size; y++) {
				for (let x = 0; x < size; x++) {
					for (const [dx, dy, weight] of offsets) {
						const sx = x + dx;
						const sy = y + dy;
						const ix = Math.floor(sx);
						const iy = Math.floor(sy);
						if (ix < 0 || iy < 0 || ix + 1 >= size || iy + 1 >= size) continue;
						const fx = sx - ix;
						const fy = sy - iy;
						output[y * size + x] += (
							image[iy * size + ix] * (1 - fx) * (1 - fy) +
							image[iy * size + ix + 1] * fx * (1 - fy) +
							image[(iy + 1) * size + ix] * (1 - fx) * fy +
							image[(iy + 1) * size + ix + 1] * fx * fy
						) * weight / weightSum;
					}
				}
			}
			image = output;
		}
		assertClose(image.reduce((sum, value) => sum + value, 0), 1, 0.001);
		const centerValue = image[center * size + center];
		assert.ok(image[center * size + center + Math.round(radiusPixels * 0.45)] > centerValue * 0.85,
			"the aperture must retain a readable hexagonal face rather than a round central peak");
		for (let y = -Math.floor(radiusPixels * 0.55); y <= radiusPixels * 0.55; y++) {
			for (let x = -Math.floor(radiusPixels * 0.55); x <= radiusPixels * 0.55; x++) {
				assert.ok(image[(center + y) * size + center + x] > 0,
					`${height}p/${downsample}: aperture must not leave holes near its center`);
			}
		}
		const verticalEdge = image[(center + Math.round(radiusPixels * 0.8)) * size + center];
		const horizontalEdge = image[center * size + center + Math.round(radiusPixels * 0.8)];
		assert.ok(verticalEdge < horizontalEdge * 0.95,
			"soft hexagons must retain flatter vertical sides and extended horizontal vertices");
	}
}

testOptionsAndBackendEligibility();
testHDRPlanningWithoutGBuffer();
await testDeclaredDispatchesAndResourceReuse();
await testSkipInvalidationResizeAndTinyTargets();
await testShaderHighlightAndOpticalMath();
await testApertureImpulseContinuity();
owners.destroy();
console.log("Lens flare pass tests passed");
