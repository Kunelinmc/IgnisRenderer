import assert from "node:assert/strict";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import {
	createInlineShaderSourceMap,
	ShaderCompileError,
} from "../src/shaders/runtime/index.ts";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
} from "../src/renderers/types.ts";

if (!globalThis.GPUBufferUsage) {
	globalThis.GPUBufferUsage = {
		VERTEX: 1 << 0,
		INDEX: 1 << 1,
		UNIFORM: 1 << 2,
		STORAGE: 1 << 3,
		COPY_SRC: 1 << 4,
		COPY_DST: 1 << 5,
		MAP_READ: 1 << 6,
		MAP_WRITE: 1 << 7,
		QUERY_RESOLVE: 1 << 8,
	};
}

if (!globalThis.GPUTextureUsage) {
	globalThis.GPUTextureUsage = {
		COPY_SRC: 1 << 0,
		COPY_DST: 1 << 1,
		TEXTURE_BINDING: 1 << 2,
		STORAGE_BINDING: 1 << 3,
		RENDER_ATTACHMENT: 1 << 4,
	};
}

class FakeDevice {
	constructor() {
		this.limits = {
			maxBindGroups: 4,
			maxColorAttachments: 8,
			maxColorAttachmentBytesPerSample: 64,
		};
		this.defaultBindGroupLayout = { id: "group0" };
		this.shaderModuleDescs = [];
		this.shaderModuleFailuresRemaining = 0;
		this.shaderCompilationInfoMessages = [];
		this.computePipelineDescs = [];
		this.renderPipelineDescs = [];
		this.pipelineLayouts = [];
		this.bindGroupDescs = [];
		this.samplerDescs = [];
		this.textureDescs = [];
		this.bufferDescs = [];
		this.configureCalls = 0;
		this.supportedSampleCounts = new Set([1, 4]);
	}

	createSampler(desc) {
		this.samplerDescs.push(desc);
		return { desc };
	}

	createShaderModule(desc) {
		this.shaderModuleDescs.push(desc);
		if (this.shaderModuleFailuresRemaining > 0) {
			this.shaderModuleFailuresRemaining--;
			throw new Error("simulated shader module failure");
		}
		return {
			desc,
			getCompilationInfo: async () => ({
				messages: this.shaderCompilationInfoMessages.slice(),
			}),
		};
	}

	createPipelineLayout(desc) {
		const layout = { desc };
		this.pipelineLayouts.push(layout);
		return layout;
	}

	createComputePipeline(desc) {
		this.computePipelineDescs.push(desc);
		const layout0 = this.defaultBindGroupLayout;
		return {
			desc,
			getBindGroupLayout(index) {
				if (index === 0) {
					return layout0;
				}
				throw new Error("no bind group");
			},
		};
	}

	createRenderPipeline(desc) {
		this.renderPipelineDescs.push(desc);
		const layout0 = this.defaultBindGroupLayout;
		return {
			desc,
			getBindGroupLayout(index) {
				if (index === 0) {
					return layout0;
				}
				throw new Error("no bind group");
			},
		};
	}

	createBindGroup(desc) {
		this.bindGroupDescs.push(desc);
		return { desc };
	}

	createTexture(desc) {
		this.textureDescs.push(desc);
		const sampleCount = Math.max(1, Math.floor(desc.sampleCount ?? 1));
		if (!this.supportedSampleCounts.has(sampleCount)) {
			throw new Error(`unsupported sample count: ${sampleCount}`);
		}
		return {
			desc,
			createView() {
				return { desc };
			},
			destroy() {},
		};
	}

	createBuffer(desc) {
		this.bufferDescs.push(desc);
		const bufferState = {
			mapState: desc.mappedAtCreation ? "mapped" : "unmapped",
			_range: new ArrayBuffer(desc.size),
		};
		return {
			get mapState() {
				return bufferState.mapState;
			},
			getMappedRange() {
				if (bufferState.mapState !== "mapped") {
					throw new Error("buffer is not mapped");
				}
				return bufferState._range;
			},
			unmap() {
				bufferState.mapState = "unmapped";
			},
			destroy() {
				bufferState.mapState = "destroyed";
			},
		};
	}

	createCommandEncoder() {
		return {
			finish() {
				return {};
			},
		};
	}

	destroy() {}
}

function createBackend() {
	const backend = new WebGPUBackend();
	const device = new FakeDevice();
	const queueSubmissions = [];
	backend._device = device;
	backend._queue = {
		submit(commands) {
			queueSubmissions.push(commands);
		},
	};
	return { backend, device, queueSubmissions };
}

function createFrameContext(overrides = {}) {
	return {
		camera: {},
		attachments: { width: 4, height: 4 },
		features: {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: true,
			enableSSGI: true,
			enableTAA: true,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			warnings: [],
			ssrOptions: {},
			ssaoOptions: {},
			ssgiOptions: {},
			taaOptions: {},
			volumetricOptions: {},
		},
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
		...overrides,
	};
}

async function testShaderModuleCacheUsesHashKey() {
	const { backend, device } = createBackend();
	const shaderCode = "fn testMain() -> f32 { return 1.0; }";

	const moduleA = await backend.createShaderModule({
		code: shaderCode,
		label: "A",
	});
	const moduleB = await backend.createShaderModule({
		code: `${shaderCode}`,
		label: "B",
	});

	assert.notEqual(moduleA, moduleB);
	assert.equal(moduleA._gpuResource, moduleB._gpuResource);
	assert.equal(device.shaderModuleDescs.length, 1);
	const cacheKeys = Array.from(backend._shaderModuleCache.keys());
	assert.equal(cacheKeys.length, 1);
	assert.notEqual(cacheKeys[0], shaderCode);
	assert.ok(cacheKeys[0].includes("hash:"));
	const entry = backend._shaderModuleCache.values().next().value;
	assert.equal(entry.refCount, 2);
	assert.equal(entry.gpuResource, moduleA._gpuResource);

	moduleA.destroy();
	assert.equal(entry.refCount, 1);
	moduleB.destroy();
	assert.equal(backend._shaderModuleCache.size, 0);
}

async function testShaderModuleRetryWithinSingleRequest() {
	const { backend, device } = createBackend();
	device.shaderModuleFailuresRemaining = 1;
	const shader = await backend.createShaderModule({
		code: "fn retryMain() -> f32 { return 2.0; }",
		label: "RetryShader",
	});
	assert.ok(shader);
	assert.equal(device.shaderModuleDescs.length, 2);
}

async function testShaderModuleCompilationInfoErrorThrowsMappedError() {
	const { backend, device } = createBackend();
	device.shaderCompilationInfoMessages = [
		{
			type: "error",
			message: "simulated shader compile failure",
			lineNum: 3,
			linePos: 2,
		},
	];
	const sourceMap = createInlineShaderSourceMap(
		"line1\nline2\nline3",
		"./parts/test.wgsl",
		"source"
	);
	await assert.rejects(
		() =>
			backend.createShaderModule({
				code: "line1\nline2\nline3",
				sourceMap,
				label: "CompileInfoError",
				language: "wgsl",
				stage: "compute",
			}),
		(error) => {
			assert.ok(error instanceof ShaderCompileError);
			assert.equal(error.backend, "webgpu");
			assert.equal(error.messages[0].line, 3);
			assert.equal(error.messages[0].sourcePath, "./parts/test.wgsl");
			return true;
		}
	);
}

function testSamplerReferenceCounting() {
	const { backend } = createBackend();
	const samplerA = backend.createSampler({});
	const samplerB = backend.createSampler({});

	assert.notEqual(samplerA, samplerB);
	assert.equal(samplerA._gpuResource, samplerB._gpuResource);
	const samplerEntry = backend._samplerCache.values().next().value;
	assert.equal(samplerEntry.refCount, 2);

	samplerA.destroy();
	assert.equal(samplerEntry.refCount, 1);
	samplerB.destroy();
	assert.equal(backend._samplerCache.size, 0);
}

async function testComputePipelineAutoLayoutCaching() {
	const { backend, device } = createBackend();
	const module = await backend.createShaderModule({
		code: "shader compute",
		label: "ComputeShader",
	});
	const desc = {
		label: "ComputePipeline",
		compute: {
			module,
			entryPoint: "csMain",
		},
	};

	const pipelineA = backend.createComputePipeline(desc);
	const pipelineB = backend.createComputePipeline(desc);

	assert.notEqual(pipelineA, pipelineB);
	assert.equal(pipelineA._gpuResource, pipelineB._gpuResource);
	assert.equal(device.pipelineLayouts.length, 0);
	assert.equal(device.computePipelineDescs.length, 1);
	assert.equal(device.computePipelineDescs[0].layout, "auto");
	assert.equal(backend._autoComputePipelineLayoutCache.size, 0);
}

async function testRenderPipelineAutoLayoutCaching() {
	const { backend, device } = createBackend();
	const shader = await backend.createShaderModule({
		code: "shader render",
		label: "RenderShader",
	});
	const desc = {
		label: "RenderPipeline",
		vertex: {
			module: shader,
			entryPoint: "vsMain",
			buffers: [],
		},
		fragment: {
			module: shader,
			entryPoint: "fsMain",
			targets: [{ format: TextureFormat.RGBA8Unorm }],
		},
		depthStencil: {
			format: TextureFormat.Depth24Plus,
			depthWriteEnabled: true,
			depthCompare: "less",
		},
		sampleCount: 4,
	};

	const pipelineA = backend.createPipeline(desc);
	const pipelineB = backend.createPipeline(desc);

	assert.notEqual(pipelineA, pipelineB);
	assert.equal(pipelineA._gpuResource, pipelineB._gpuResource);
	assert.equal(device.pipelineLayouts.length, 0);
	assert.equal(device.renderPipelineDescs.length, 1);
	assert.equal(device.renderPipelineDescs[0].layout, "auto");
	assert.equal(backend._autoRenderPipelineLayoutCache.size, 0);
}

function testBindingGroupCacheUsesHashedKey() {
	const { backend, device } = createBackend();
	const layout = { id: "layout0" };
	const gpuBuffer = { destroy() {} };
	const renderBuffer = {
		size: 16,
		destroy() {},
		_gpuResource: gpuBuffer,
	};

	const groupA = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBuffer }],
	});
	const groupB = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBuffer }],
	});

	assert.equal(groupA, groupB);
	assert.equal(device.bindGroupDescs.length, 1);
	const keys = Array.from(backend._bindingGroupCache.keys());
	assert.equal(keys.length, 1);
	assert.equal(typeof keys[0], "bigint");
}

function testBindingGroupHashCollisionBucketSafety() {
	const { backend, device } = createBackend();
	const originalGetHashKey = backend._getBindingGroupCacheKey;
	backend._getBindingGroupCacheKey = () => 1n;

	const layout = { id: "layout0" };
	const gpuBufferA = { destroy() {} };
	const gpuBufferB = { destroy() {} };
	const renderBufferA = {
		size: 16,
		destroy() {},
		_gpuResource: gpuBufferA,
	};
	const renderBufferB = {
		size: 16,
		destroy() {},
		_gpuResource: gpuBufferB,
	};

	const groupA = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBufferA }],
	});
	const groupB = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBufferB }],
	});
	const groupA2 = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBufferA }],
	});
	const groupB2 = backend.createBindingGroup({
		layout,
		entries: [{ binding: 0, resource: renderBufferB }],
	});

	assert.notEqual(groupA, groupB);
	assert.equal(groupA, groupA2);
	assert.equal(groupB, groupB2);
	assert.equal(device.bindGroupDescs.length, 2);
	const bucket = backend._bindingGroupCache.get(1n);
	assert.equal(bucket.length, 2);

	backend._getBindingGroupCacheKey = originalGetHashKey;
}

function testSetMSAASampleCountClampsAndInvalidates() {
	const { backend } = createBackend();
	let invalidationCount = 0;
	backend._frameExecutor = {
		invalidateFrameTargets() {
			invalidationCount++;
		},
	};
	backend._renderPipelineCache.set("cached", {
		key: "cached",
		label: "cached",
		refCount: 1,
		gpuResource: { getBindGroupLayout() {} },
	});

	backend.setMSAASampleCount(8);
	assert.equal(backend.getMSAASampleCount(), 4);
	assert.equal(invalidationCount, 1);
	assert.equal(backend._renderPipelineCache.size, 0);

	backend.setMSAASampleCount(3);
	assert.equal(backend.getMSAASampleCount(), 1);
	assert.equal(invalidationCount, 2);
}

function testCreateBufferMappedAtCreationExposesUnmap() {
	const { backend } = createBackend();
	const buffer = backend.createBuffer({
		size: 32,
		usage: BufferUsage.CopyDst,
		mappedAtCreation: true,
	});
	assert.equal(typeof buffer.unmap, "function");
	assert.equal(buffer._gpuResource.mapState, "mapped");
	buffer.unmap();
	assert.equal(buffer._gpuResource.mapState, "unmapped");
	buffer.destroy();
	assert.equal(buffer._gpuResource.mapState, "destroyed");
}

function testResizeUsesProvidedDimensions() {
	const { backend, device } = createBackend();
	let invalidateCalls = 0;
	backend._canvas = { width: 1, height: 1 };
	backend._context = {
		configure(config) {
			device.configureCalls++;
			assert.equal(config.device, device);
		},
	};
	backend._frameExecutor = {
		invalidateFrameTargets() {
			invalidateCalls++;
		},
	};
	backend.resize(320.9, 240.2);
	assert.equal(backend.canvas.width, 320);
	assert.equal(backend.canvas.height, 240);
	assert.equal(device.configureCalls, 1);
	assert.equal(invalidateCalls, 1);
}

function testMapBindingResourceRejectsPrimitive() {
	const { backend } = createBackend();
	assert.throws(
		() => backend._mapBindingResource("invalid"),
		/Unsupported WebGPU binding resource/
	);
}

function testCreateTextureClampsPublicDimensions() {
	const { backend, device } = createBackend();
	const texture = backend.createTexture({
		width: 0,
		height: -3,
		format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.TextureBinding,
	});
	assert.equal(texture.width, 1);
	assert.equal(texture.height, 1);
	assert.equal(device.textureDescs.length, 1);
	assert.equal(device.textureDescs[0].size.width, 1);
	assert.equal(device.textureDescs[0].size.height, 1);
}

function testCommandBufferOwnershipAndOneShotSubmit() {
	const { backend, queueSubmissions } = createBackend();
	const encoder = backend.createCommandEncoder();
	const command = encoder.finish();
	backend.submit([command]);
	assert.equal(queueSubmissions.length, 1);

	assert.throws(() => backend.submit([command]), /already been submitted/);

	const { backend: foreignBackend } = createBackend();
	assert.throws(
		() => foreignBackend.submit([command]),
		/does not belong to this WebGPU backend instance/
	);
}

function testPassDependencyValidation() {
	const { backend } = createBackend();
	backend._resources = {
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		endFrame() {},
		destroy() {},
		invalidateFrameTargets() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};

	const context = createFrameContext();
	backend.beginFrame(context);
	assert.ok(
		backend._plannedPassOrder.get("ssao") < backend._plannedPassOrder.get("taa")
	);
	assert.ok(
		backend._plannedPassOrder.get("ssao") < backend._plannedPassOrder.get("ssgi")
	);
	assert.ok(
		backend._plannedPassOrder.get("ssgi") < backend._plannedPassOrder.get("taa")
	);

	assert.throws(
		() =>
			backend.executePass(
				{ stage: "taa", executor: "backend", enabled: true },
				context
			),
		/dependencies: ssgi, ssao/
	);

	backend.executePass(
		{ stage: "ssao", executor: "backend", enabled: true },
		context
	);
	backend.executePass(
		{ stage: "ssgi", executor: "backend", enabled: true },
		context
	);
	assert.doesNotThrow(() =>
		backend.executePass(
			{ stage: "taa", executor: "backend", enabled: true },
			context
		)
	);
}

function testPassPlanAllowsParticleStageBeforeMainOpaque() {
	const { backend } = createBackend();
	backend._resources = {
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		endFrame() {},
		destroy() {},
		invalidateFrameTargets() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {},
	};

	const context = createFrameContext({
		scene: {
			particleSystems: [{ id: "ps-0" }],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
	});
	backend.beginFrame(context);

	assert.ok(
		backend._plannedPassOrder.get("particle-sim") <
			backend._plannedPassOrder.get("main-opaque")
	);
	assert.ok(
		backend._plannedPassOrder.get("main-opaque") <
			backend._plannedPassOrder.get("particles")
	);

	assert.doesNotThrow(() =>
		backend.executePass(
			{ stage: "particle-sim", executor: "backend", enabled: true },
			context
		)
	);
	assert.doesNotThrow(() =>
		backend.executePass(
			{ stage: "main-opaque", executor: "backend", enabled: true },
			context
		)
	);
	assert.doesNotThrow(() =>
		backend.executePass(
			{ stage: "particles", executor: "backend", enabled: true },
			context
		)
	);
}

async function testWarmupAggregatesPhases() {
	const { backend } = createBackend();
	backend._frameExecutor = {
		warmup: async () => ({
			phase: "frame",
			total: 2,
			compiled: 2,
			skipped: 0,
			failed: 0,
			errors: [],
		}),
	};
	backend._resources = {
		warmup: async () => ({
			phase: "resources",
			total: 3,
			compiled: 2,
			skipped: 1,
			failed: 0,
			errors: [],
		}),
	};
	const report = await backend.warmup(createFrameContext());
	assert.equal(report.total, 5);
	assert.equal(report.compiled, 4);
	assert.equal(report.skipped, 1);
	assert.equal(report.failed, 0);
	assert.equal(report.phases.length, 2);
}

async function run() {
	await testShaderModuleCacheUsesHashKey();
	await testShaderModuleRetryWithinSingleRequest();
	await testShaderModuleCompilationInfoErrorThrowsMappedError();
	testSamplerReferenceCounting();
	await testComputePipelineAutoLayoutCaching();
	await testRenderPipelineAutoLayoutCaching();
	testBindingGroupCacheUsesHashedKey();
	testBindingGroupHashCollisionBucketSafety();
	testSetMSAASampleCountClampsAndInvalidates();
	testCreateBufferMappedAtCreationExposesUnmap();
	testResizeUsesProvidedDimensions();
	testMapBindingResourceRejectsPrimitive();
	testCreateTextureClampsPublicDimensions();
	testCommandBufferOwnershipAndOneShotSubmit();
	testPassDependencyValidation();
	testPassPlanAllowsParticleStageBeforeMainOpaque();
	await testWarmupAggregatesPhases();
	console.log("WebGPU backend cache/dependency tests passed");
}

run();
