import assert from "node:assert/strict";

import { createWebGPUResourceManager } from "../../helpers/fakes.mjs";
import { WebGPUShadowRuntime } from "../../../src/backends/webgpu/WebGPUShadowRuntime.ts";
import { WebGPUCommandEncoder } from "../../../src/backends/webgpu/WebGPUCommandEncoder.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createHarness() {
	const resourceManager = createWebGPUResourceManager();
	const resources = [];
	const passes = [];
	const submissions = [];
	let encoderCount = 0;
	let finishCount = 0;
	function createResource(desc) {
		const resource = {
			desc,
			destroyCount: 0,
			destroy() { this.destroyCount++; },
		};
		resources.push(resource);
		return resource;
	}
	const nativeEncoder = {
		beginRenderPass(desc) {
			const pass = { desc, viewports: [], draws: [], ended: false };
			passes.push(pass);
			let indexBuffer;
			return {
				setViewport(...viewport) { pass.viewports.push(viewport); },
				setScissorRect() {},
				setBindGroup() {},
				setPipeline() {},
				setVertexBuffer() {},
				setIndexBuffer(buffer) { indexBuffer = buffer; },
				drawIndexed(...args) { pass.draws.push({ indexBuffer, args }); },
				end() { pass.ended = true; },
			};
		},
		finish() {
			finishCount++;
			return { passes };
		},
	};
	const backend = {
		resourceManager,
		device: {
			limits: { maxTextureDimension2D: 8192 },
			createBindGroupLayout: (desc) => desc,
			createPipelineLayout: (desc) => desc,
			createBindGroup: (desc) => desc,
			createBuffer: createResource,
			createCommandEncoder() {
				encoderCount++;
				return nativeEncoder;
			},
		},
		queue: {
			writeBuffer() {},
			submit(buffers) { submissions.push(buffers); },
		},
		createTexture(desc) {
			const texture = createResource(desc);
			return resourceManager.registerTexture({ texture, view: { desc } },
				{ width: desc.width, height: desc.height, format: desc.format },
				() => texture.destroy());
		},
		async createShaderModule(desc) {
			const native = createResource(desc);
			return resourceManager.registerShaderModule(native, desc.label, () => native.destroy());
		},
		async createPipeline(desc) {
			const native = createResource(desc);
			return resourceManager.registerRenderPipeline(native, desc.label, () => native.destroy());
		},
	};
	const borrowedBuffer = { destroyCount: 0, destroy() { this.destroyCount++; } };
	const fallback = resourceManager.registerBuffer(borrowedBuffer, { size: 64 });
	const payload = {
		generation: 0,
		paramsBuffer: fallback,
		jointMatricesBuffer: fallback,
		morphWeightsBuffer: fallback,
	};
	const geometryRequests = [];
	const runtime = new WebGPUShadowRuntime(backend, {
		getGeometry(binding) {
			geometryRequests.push(binding);
			return {
				shadowVertexBindings: [],
				shadowVertexLayouts: [],
				shadowLayoutKey: "static",
				skinProfile: "static",
				indexBuffer: resourceManager.registerBuffer(binding, { size: 12 }),
				indexFormat: "uint32",
				indexCount: 3,
			};
		},
	}, {
		getStaticShadowPayload: () => payload,
		getShadowPayload: () => payload,
	});
	return {
		runtime, resources, passes, submissions, nativeEncoder, borrowedBuffer, geometryRequests,
		resourceManager,
		get encoderCount() { return encoderCount; },
		get finishCount() { return finishCount; },
	};
}

function createFrame() {
	const light = Object.freeze({ id: "sun", type: "directional" });
	const slice = Object.freeze({ resolution: 128, viewProjection: Matrix4.identity() });
	const prepared = Object.freeze({
		light, effectiveTechnique: "single", effectiveResolution: 128,
		slices: Object.freeze([slice]),
	});
	return {
		scene: { lights: [light] },
		shadowPlan: Object.freeze({
			lights: Object.freeze([prepared]), hasRasterWork: true, hasTransmissionWork: true,
		}),
		transient: new Map(),
	};
}

async function testRecordingUsesRuntimeAtlases(useFrameEncoder) {
	const harness = createHarness();
	const { runtime, passes, resourceManager } = harness;
	const caster = createTestDrawPacket({ id: "caster" }).submission;
	const transmitter = createTestDrawPacket({ id: "transmitter" }).submission;
	// Geometry buffers are borrowed, just like the shared animation payload.
	caster.geometry.destroy = () => assert.fail("Destroyed borrowed caster geometry");
	transmitter.geometry.destroy = () => assert.fail("Destroyed borrowed transmitter geometry");
	const packets = {
		shadowCasterSubmissions: [caster], shadowTransmitterSubmissions: [transmitter],
	};
	const encoder = useFrameEncoder ?
		new WebGPUCommandEncoder(harness.nativeEncoder,
			{ resourceManager: harness.resourceManager }, {}) : undefined;
	const lighting = {
		directionalShadows: [{ enabled: true, shadowMapBaseSize: 128, atlasTileSize: 0 }],
		spotShadows: [],
	};
	runtime.prepareAtlas(lighting, 128);
	const atlas = runtime.atlas;
	const transmittanceAtlas = runtime.transmittanceAtlas;
	await runtime.renderShadows(createFrame(), packets, encoder);
	assert.deepEqual(harness.geometryRequests, [caster.geometry, transmitter.geometry]);
	assert.equal(passes.length, 2);
	assert.strictEqual(passes[0].desc.depthStencilAttachment.view, resourceManager.getTexture(atlas).view);
	assert.equal(passes[0].desc.depthStencilAttachment.depthLoadOp, "clear");
	assert.strictEqual(
		passes[1].desc.colorAttachments[0].view, resourceManager.getTexture(transmittanceAtlas).view,
	);
	assert.strictEqual(passes[1].desc.depthStencilAttachment.view, resourceManager.getTexture(atlas).view);
	assert.equal(passes[1].desc.depthStencilAttachment.depthLoadOp, "load");
	assert.deepEqual(passes[1].desc.colorAttachments[0].clearValue, { r: 1, g: 1, b: 1, a: 1 });
	for (const [index, submission] of [caster, transmitter].entries()) {
		assert.deepEqual(passes[index].viewports, [[0, 0, 128, 128, 0, 1]]);
		assert.equal(passes[index].ended, true);
		assert.deepEqual(passes[index].draws, [{
			indexBuffer: submission.geometry, args: [3, 1, 0, 0, 0],
		}]);
	}
	assert.equal(harness.encoderCount, useFrameEncoder ? 0 : 1);
	assert.equal(harness.finishCount, useFrameEncoder ? 0 : 1);
	assert.equal(harness.submissions.length, useFrameEncoder ? 0 : 1);

	const shaders = harness.resources.filter((resource) => resource.desc.language === "wgsl");
	const pipelines = harness.resources.filter((resource) => resource.desc.vertex);
	assert.equal(shaders.length, 1);
	assert.equal(pipelines.length, 2);
	runtime.onShaderRuntimeChanged();
	assert.ok([...shaders, ...pipelines].every((resource) => resource.destroyCount === 1));
	assert.strictEqual(runtime.atlas, atlas);
	assert.strictEqual(runtime.transmittanceAtlas, transmittanceAtlas);
	await runtime.warmup();
	assert.equal(harness.resources.filter((resource) => resource.desc.language === "wgsl").length, 2);
	runtime.destroy();
	runtime.destroy();
	assert.equal(runtime.atlas, null);
	assert.equal(runtime.transmittanceAtlas, null);
	assert.ok(harness.resources.every((resource) => resource.destroyCount === 1));
	assert.equal(harness.borrowedBuffer.destroyCount, 0);
}

async function testInactivePlanAndAtlasReuse() {
	const { runtime, resources, passes, resourceManager } = createHarness();
	const context = createFrame();
	await runtime.renderShadows({ ...context, shadowPlan: { hasRasterWork: false } }, {});
	assert.equal(resources.length, 0);
	assert.equal(passes.length, 0);
	assert.equal(runtime.atlas, null);
	const first = runtime.ensureAtlasForTileSize(128);
	const firstTransmission = runtime.transmittanceAtlas;
	const firstNative = resourceManager.getTexture(first).texture;
	const firstTransmissionNative = resourceManager.getTexture(firstTransmission).texture;
	assert.strictEqual(runtime.ensureAtlasForTileSize(64), first);
	assert.strictEqual(runtime.transmittanceAtlas, firstTransmission);
	const grown = runtime.ensureAtlasForTileSize(256);
	assert.notStrictEqual(grown, first);
	assert.equal(firstNative.destroyCount, 1);
	assert.equal(firstTransmissionNative.destroyCount, 1);
	runtime.destroy();
	assert.ok(resources.every((resource) => resource.destroyCount === 1));
}

async function testForeignAtlasResourcesRejectBeforeEncoding() {
	const harness = createHarness();
	const foreignStore = createWebGPUResourceManager();
	const foreign = foreignStore.registerTexture({ texture: {}, view: {} },
		{ width: 128, height: 128 });
	const owned = harness.runtime.ensureAtlasForTileSize(128);
	const context = createFrame();
	const caster = harness.runtime._casterRenderer;
	await assert.rejects(() => caster.renderAtlas(context, [], [], {
		ensureAtlasForTileSize: () => foreign,
		transmittanceAtlas: null,
		tileSize: 128,
	}), /not owned/i, "foreign depth atlases cannot enter shadow encoding");
	await assert.rejects(() => caster.renderAtlas(context, [], [], {
		ensureAtlasForTileSize: () => owned,
		transmittanceAtlas: foreign,
		tileSize: 128,
	}), /not owned/i, "foreign transmittance atlases cannot enter shadow encoding");
	assert.equal(harness.encoderCount, 0);
	assert.equal(harness.passes.length, 0);
	foreign.destroy();
	harness.runtime.destroy();
}

const previousShaderStage = globalThis.GPUShaderStage;
const previousBufferUsage = globalThis.GPUBufferUsage;
globalThis.GPUShaderStage ??= { VERTEX: 1 };
globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8 };
try {
	await testRecordingUsesRuntimeAtlases(false);
	await testRecordingUsesRuntimeAtlases(true);
	await testInactivePlanAndAtlasReuse();
	await testForeignAtlasResourcesRejectBeforeEncoding();
	console.log("WebGPU shadow runtime behavior tests passed");
} finally {
	if (previousShaderStage === undefined) delete globalThis.GPUShaderStage;
	else globalThis.GPUShaderStage = previousShaderStage;
	if (previousBufferUsage === undefined) delete globalThis.GPUBufferUsage;
	else globalThis.GPUBufferUsage = previousBufferUsage;
}
