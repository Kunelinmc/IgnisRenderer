import assert from "node:assert/strict";
import { WebGPUCustomRenderTargetRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUCustomRenderTargetRuntime.ts";
import { TextureFormat } from "../../../src/backends/types.ts";
import {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../../../src/rendering/CustomRenderTargets.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

class FakeEncoder {
	constructor() {
		this.calls = [];
	}
	beginRenderPass(desc) {
		this.calls.push(["beginRenderPass", desc]);
	}
	endRenderPass() {
		this.calls.push(["endRenderPass"]);
	}
	finish() {
		return {};
	}
}

class FakeBackend {
	constructor(options = {}) {
		this.createdTextures = [];
		this.fallbackFormat = options.fallbackFormat ?? null;
		this.failSampleCount = options.failSampleCount ?? null;
		this.failAfterMatchingAllocations = options.failAfterMatchingAllocations ?? 0;
		this.matchingAllocations = 0;
	}
	createTexture(desc) {
		if (desc.sampleCount === this.failSampleCount) {
			this.matchingAllocations++;
			if (this.matchingAllocations > this.failAfterMatchingAllocations) {
				throw new Error(`test ${desc.sampleCount}x allocation failure`);
			}
		}
		const fallback = this.fallbackFormat?.requested === desc.format ?
			this.fallbackFormat.actual
		:	null;
		const texture = {
			width: desc.width,
			height: desc.height,
			format: fallback ?? desc.format,
			requestedFormat: desc.format,
			formatFallbackReason: fallback ? "test fallback" : undefined,
			descriptor: desc,
			destroyed: false,
			destroy() {
				this.destroyed = true;
			},
		};
		this.createdTextures.push(texture);
		return texture;
	}
	createCommandEncoder() {
		return new FakeEncoder();
	}
	createBuffer() {
		return { size: 4, destroy() {} };
	}
	createSampler(desc = {}) {
		return { label: desc.label };
	}
	createTextureView(texture) {
		return { texture };
	}
	createShaderModule(desc = {}) {
		return { label: desc.label };
	}
	createComputePipeline(desc = {}) {
		return { label: desc.label };
	}
	createPipeline(desc = {}) {
		return { label: desc.label };
	}
	createBindingGroup(desc = {}) {
		return { label: desc.label };
	}
	writeBuffer() {}
	writeTexture() {}
	submit() {}
	registerExternalTexture() {}
	unregisterExternalTexture() {}
	resolveTextureForSlot() {
		return null;
	}
}

class FakeSampleCountResolver {
	constructor(effectiveSampleCount = null) {
		this.effectiveSampleCount = effectiveSampleCount;
		this.fallbacks = new Set();
	}
	resolveDomainSampleCount(domain, requestedSampleCount, formats, constraints = {}) {
		const signature = [
			domain,
			requestedSampleCount,
			formats.slice().sort().join(","),
			constraints.colorAttachmentCount ?? 0,
			constraints.colorAttachmentBytesPerSample ?? 0,
		].join("|");
		const runtimeFallbackActive = this.fallbacks.has(signature);
		return {
			domain,
			requestedSampleCount,
			sampleCount: runtimeFallbackActive
				? 1
				: this.effectiveSampleCount ?? requestedSampleCount,
			signature,
			runtimeFallbackActive,
		};
	}
	fallbackToSingleSample(signature) {
		const size = this.fallbacks.size;
		this.fallbacks.add(signature);
		return this.fallbacks.size !== size;
	}
}

function createContext(passExecute, targetDescriptor = {}) {
	const renderTargets = new RenderTargetRegistrySnapshot([
		{
			id: "gbuf",
			size: { mode: "fixed", width: 32, height: 16 },
			color: [
				{ format: TextureFormat.RGBA8Unorm },
				{ format: TextureFormat.RGBA16Float },
			],
			depth: { format: TextureFormat.Depth32Float },
			sampleCount: 1,
			...targetDescriptor,
		},
	]);
	const customRenderPasses = new CustomRenderPassRegistrySnapshot([
		{
			id: "custom-gbuf",
			target: "gbuf",
			execute: passExecute,
		},
	]);
	return {
		backendProfile: {
			id: "webgpu",
			capabilities: {},
			frameScheduling: "always",
			lighting: {},
		},
		camera: {},
		attachments: { width: 640, height: 480 },
		features: {},
		postProcess: createResolvedPostProcess("webgpu"),
		renderTargets,
		customRenderPasses,
		shadowMaps: new Map(),
		scene: { shadowCasterPackets: [], shadowTransmitterPackets: [] },
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		incremental: { enabled: false, forceFullFrame: true, dirtyRects: [] },
		transient: new Map(),
	};
}

async function testWebGPUCustomTargetExecution() {
	const backend = new FakeBackend();
	const runtime = new WebGPUCustomRenderTargetRuntime(
		backend,
		new FakeSampleCountResolver(),
	);
	let observed = null;
	const context = createContext((passContext) => {
		observed = passContext;
		passContext.encoder.beginRenderPass({
			colorAttachments: passContext.target.color.map((attachment) => ({
				view: attachment.texture,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
			})),
			depthStencilAttachment: {
				view: passContext.target.depth.texture,
				depthLoadOp: "clear",
				depthStoreOp: "store",
				depthClearValue: 1,
			},
		});
		passContext.encoder.endRenderPass();
	});
	runtime.sync(context);

	assert.equal(backend.createdTextures.length, 3);
	assert.equal(backend.createdTextures[0].width, 32);
	assert.equal(backend.createdTextures[1].format, TextureFormat.RGBA16Float);

	const encoder = new FakeEncoder();
	await runtime.executePass(
		{ stage: "custom-gbuf", executor: "backend", enabled: true, dependsOn: [] },
		context,
		encoder
	);
	assert.equal(observed.backend, "webgpu");
	assert.equal(observed.target.color.length, 2);
	assert.equal(observed.target.sampleCount, 1);
	assert.equal(observed.target.color[0].resolveTexture, null);
	assert.equal(observed.target.depth.resolveTexture, null);
	assert.equal(encoder.calls[0][0], "beginRenderPass");
	await assert.rejects(() => runtime.readColor("gbuf", 0), /successful frame/);
	runtime._readbackRuntime = {
		async readTexture(options) {
			return {
				bytes: new Uint8Array(options.width * options.height * 4),
				width: options.width,
				height: options.height,
				format: options.format,
				bytesPerPixel: 4,
				bytesPerRow: options.width * 4,
				toFloat32: () => new Float32Array(),
				toRGBAFloat32: () => new Float32Array(),
				toNormalizedRGBA8Float32: () => new Float32Array(),
			};
		},
		destroy() {},
	};
	runtime.commitFrameState();
	const readback = await runtime.readColor("gbuf", 0, { width: 2, height: 2 });
	assert.equal(readback.origin, "top-left");
	assert.equal(readback.format, TextureFormat.RGBA8Unorm);
	await assert.rejects(() => runtime.readColor("gbuf", 0, { width: 33 }), /between/);
	runtime.destroy();
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

async function testWebGPUMultisampledTargetUsesResolveTextures() {
	const backend = new FakeBackend();
	const runtime = new WebGPUCustomRenderTargetRuntime(
		backend,
		new FakeSampleCountResolver(),
	);
	let observed = null;
	const context = createContext((passContext) => {
		observed = passContext.target;
		passContext.encoder.beginRenderPass({
			colorAttachments: passContext.target.color.map((attachment) => ({
				view: attachment.texture,
				resolveTarget: attachment.resolveTexture,
				loadOp: "clear",
				storeOp: "store",
			})),
			depthStencilAttachment: {
				view: passContext.target.depth.texture,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});
		passContext.encoder.endRenderPass();
	}, { sampleCount: 4 });
	runtime.sync(context);
	assert.equal(backend.createdTextures.length, 5);
	assert.deepEqual(
		backend.createdTextures.map((texture) => texture.descriptor.sampleCount),
		[4, 1, 4, 1, 4],
	);

	const encoder = new FakeEncoder();
	await runtime.executePass(
		{ stage: "custom-gbuf", executor: "backend", enabled: true, dependsOn: [] },
		context,
		encoder,
	);
	assert.equal(observed.sampleCount, 4);
	assert.ok(observed.color.every((attachment) => attachment.resolveTexture));
	assert.equal(observed.depth.resolveTexture, null);
	assert.equal(
		encoder.calls[0][1].colorAttachments[0].resolveTarget,
		observed.color[0].resolveTexture,
	);

	let readTexture = null;
	runtime._readbackRuntime = {
		async readTexture(options) {
			readTexture = options.texture;
			return {
				bytes: new Uint8Array(4),
				width: 1,
				height: 1,
				format: options.format,
				bytesPerPixel: 4,
				bytesPerRow: 4,
				toFloat32: () => new Float32Array(),
				toRGBAFloat32: () => new Float32Array(),
				toNormalizedRGBA8Float32: () => new Float32Array(),
			};
		},
		destroy() {},
	};
	runtime.commitFrameState();
	await runtime.readColor("gbuf", 0, { width: 1, height: 1 });
	assert.equal(readTexture, observed.color[0].resolveTexture);
	runtime.destroy();
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

function testWebGPUCustomTargetCapabilityDowngrade() {
	const backend = new FakeBackend();
	const runtime = new WebGPUCustomRenderTargetRuntime(
		backend,
		new FakeSampleCountResolver(2),
	);
	const context = createContext(() => {}, { sampleCount: 4 });
	runtime.sync(context);
	assert.equal(runtime._targets.get("gbuf").sampleCount, 2);
	assert.deepEqual(
		backend.createdTextures.map((texture) => texture.descriptor.sampleCount),
		[2, 1, 2, 1, 2],
	);
	runtime.destroy();
}

function testWebGPUCustomTargetAllocationFallbackIsDomainScoped() {
	const backend = new FakeBackend({
		failSampleCount: 4,
		failAfterMatchingAllocations: 1,
	});
	const resolver = new FakeSampleCountResolver();
	const runtime = new WebGPUCustomRenderTargetRuntime(backend, resolver);
	const context = createContext(() => {}, { sampleCount: 4 });
	runtime.sync(context);
	assert.equal(runtime._targets.get("gbuf").sampleCount, 1);
	assert.equal(backend.createdTextures[0].destroyed, true);
	assert.equal(resolver.fallbacks.size, 1);
	assert.equal(
		resolver.resolveDomainSampleCount(
			"custom-target:other",
			4,
			["rgba8unorm"],
		).sampleCount,
		4,
	);
	const createdAfterFallback = backend.createdTextures.length;
	runtime.sync(context);
	assert.equal(backend.createdTextures.length, createdAfterFallback);
	const resizedContext = createContext(() => {}, {
		size: { mode: "fixed", width: 64, height: 24 },
		sampleCount: 4,
	});
	runtime.sync(resizedContext);
	assert.equal(runtime._targets.get("gbuf").sampleCount, 1);
	assert.equal(runtime._targets.get("gbuf").width, 64);
	assert.equal(backend.matchingAllocations, 2);
	runtime.sync({
		...resizedContext,
		renderTargets: new RenderTargetRegistrySnapshot(),
	});
	assert.equal(runtime._targets.size, 0);
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
	runtime.destroy();
}

function testWebGPURejectsFormatFallbackTransactionally() {
	const backend = new FakeBackend({
		fallbackFormat: {
			requested: TextureFormat.RGBA16Float,
			actual: TextureFormat.RGBA8Unorm,
		},
	});
	const runtime = new WebGPUCustomRenderTargetRuntime(
		backend,
		new FakeSampleCountResolver(),
	);
	assert.throws(
		() => runtime.sync(createContext(() => {})),
		/requested "rgba16float" but received "rgba8unorm"/
	);
	assert.equal(backend.createdTextures.length, 2);
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

await testWebGPUCustomTargetExecution();
await testWebGPUMultisampledTargetUsesResolveTextures();
testWebGPUCustomTargetCapabilityDowngrade();
testWebGPUCustomTargetAllocationFallbackIsDomainScoped();
testWebGPURejectsFormatFallbackTransactionally();
console.log("WebGPU custom render target tests passed");
