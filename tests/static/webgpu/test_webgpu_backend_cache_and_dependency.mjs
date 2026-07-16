import assert from "node:assert/strict";
import { Logger } from "../../../src/foundation/Logger.ts";
import { WebGPUBackend } from "../../../src/renderers/WebGPUBackend.ts";
import {
	createInlineShaderSourceMap,
	ShaderCompileError,
} from "../../../src/shaders/runtime/index.ts";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
} from "../../../src/renderers/types.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

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
		this.computePipelineFailuresRemaining = 0;
		this.renderPipelineFailuresRemaining = 0;
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
		throw new Error("synchronous compute pipeline creation is forbidden");
	}

	async createComputePipelineAsync(desc) {
		this.computePipelineDescs.push(desc);
		if (this.computePipelineFailuresRemaining > 0) {
			this.computePipelineFailuresRemaining--;
			throw new Error("simulated compute pipeline failure");
		}
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
		throw new Error("synchronous render pipeline creation is forbidden");
	}

	async createRenderPipelineAsync(desc) {
		this.renderPipelineDescs.push(desc);
		if (this.renderPipelineFailuresRemaining > 0) {
			this.renderPipelineFailuresRemaining--;
			throw new Error("simulated render pipeline failure");
		}
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

function createBackend(options = undefined) {
	const backend = new WebGPUBackend(options);
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
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

function createDeferred() {
	let resolve;
	let reject;
	const promise = new Promise((resolvePromise, rejectPromise) => {
		resolve = resolvePromise;
		reject = rejectPromise;
	});
	return { promise, resolve, reject };
}

async function waitForCondition(predicate, message, count = 32) {
	for (let i = 0; i < count; i++) {
		if (predicate()) {
			return;
		}
		await Promise.resolve();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
	assert.ok(predicate(), message);
}

function createFrameContext(overrides = {}) {
	return {
		viewCamera: {},
		attachments: { width: 4, height: 4 },
		features: {
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess({
			ssao: { enabled: true },
			ssgi: { enabled: true },
			taa: { enabled: true },
		}),
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
		...overrides,
	};
}

async function createCachedRenderPipeline(backend) {
	const shader = await backend.createShaderModule({
		code: "shader render cached",
		label: "CachedRenderShader",
	});
	return backend.createPipeline({
		label: "CachedRenderPipeline",
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
	});
}

function createRendererFramePlan(passes) {
	const dependencyByStage = new Map([
		["shadow", ["prepared-scene-build", "particle-sim"]],
		["reflection", ["prepared-scene-build", "probe-capture"]],
		["main-opaque", ["reflection", "shadow"]],
		["main-transparent", ["main-opaque"]],
		["particles", ["main-transparent"]],
	]);
	const backendPasses = passes.map((pass) =>
		typeof pass === "string" ?
			{
				stage: pass,
				executor: "backend",
				enabled: true,
				dependsOn: dependencyByStage.get(pass) ?? [],
			}
		:	{
				executor: "backend",
				enabled: true,
				dependsOn: dependencyByStage.get(pass.stage) ?? [],
				...pass,
			}
	);
	return {
		stageOrder: [
			...backendPasses.map((pass) => ({
				id: pass.stage,
				kind: "backend-pass",
				dependsOn: [],
			})),
			{ id: "postprocess", kind: "renderer", dependsOn: ["particles"] },
		],
		backendPasses,
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
	let stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.shaderModuleEntries, 1);
	assert.deepEqual(stats.shaderModuleRefCounts, [2]);

	moduleA.destroy();
	stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.deepEqual(stats.shaderModuleRefCounts, [1]);
	moduleB.destroy();
	stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.shaderModuleEntries, 0);
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

async function testStaleShaderModuleCreationRejectsAfterRollback() {
	const { backend, device } = createBackend();
	const compilationInfo = createDeferred();
	device.createShaderModule = (desc) => {
		device.shaderModuleDescs.push(desc);
		return {
			desc,
			getCompilationInfo: () => compilationInfo.promise,
		};
	};

	const shaderPromise = backend.createShaderModule({
		code: "fn staleMain() -> f32 { return 1.0; }",
		label: "StaleShader",
	});
	await waitForCondition(
		() => device.shaderModuleDescs.length === 1,
		"stale shader module creation should start"
	);

	backend._rollbackInitializationState();
	compilationInfo.resolve({ messages: [] });
	await assert.rejects(
		() => shaderPromise,
		/WebGPU shader module creation was invalidated/
	);
	const stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.shaderModuleEntries, 0);
	assert.equal(stats.shaderModuleInFlight, 0);
}

async function testStaleShaderModulePromiseDoesNotClearRecoveredInFlight() {
	const { backend, device: oldDevice } = createBackend();
	const oldCompilationInfo = createDeferred();
	oldDevice.createShaderModule = (desc) => {
		oldDevice.shaderModuleDescs.push(desc);
		return {
			desc,
			getCompilationInfo: () => oldCompilationInfo.promise,
		};
	};

	const shaderDesc = {
		code: "fn recoveredMain() -> f32 { return 1.0; }",
		label: "RecoveredShader",
	};
	const stalePromise = backend.createShaderModule(shaderDesc);
	await waitForCondition(
		() => oldDevice.shaderModuleDescs.length === 1,
		"old shader module creation should start"
	);

	backend._rollbackInitializationState();
	const newDevice = new FakeDevice();
	const newCompilationInfo = createDeferred();
	newDevice.createShaderModule = (desc) => {
		newDevice.shaderModuleDescs.push(desc);
		return {
			desc,
			getCompilationInfo: () => newCompilationInfo.promise,
		};
	};
	backend._device = newDevice;
	backend._queue = { submit() {} };

	const recoveredPromise = backend.createShaderModule(shaderDesc);
	await waitForCondition(
		() => newDevice.shaderModuleDescs.length === 1,
		"recovered shader module creation should start"
	);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.shaderModuleInFlight,
		1
	);

	oldCompilationInfo.resolve({ messages: [] });
	await assert.rejects(
		() => stalePromise,
		/WebGPU shader module creation was invalidated/
	);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.shaderModuleInFlight,
		1
	);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.shaderModuleEntries,
		0
	);

	newCompilationInfo.resolve({ messages: [] });
	const shader = await recoveredPromise;
	assert.ok(shader);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.shaderModuleInFlight,
		0
	);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.shaderModuleEntries,
		1
	);
	assert.strictEqual(shader._gpuResource.desc, newDevice.shaderModuleDescs[0]);
}

function testSamplerReferenceCounting() {
	const { backend } = createBackend();
	const samplerA = backend.createSampler({});
	const samplerB = backend.createSampler({});

	assert.notEqual(samplerA, samplerB);
	assert.equal(samplerA._gpuResource, samplerB._gpuResource);
	let stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.deepEqual(stats.samplerRefCounts, [2]);

	samplerA.destroy();
	stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.deepEqual(stats.samplerRefCounts, [1]);
	samplerB.destroy();
	stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.samplerEntries, 0);
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

	const [pipelineA, pipelineB] = await Promise.all([
		backend.createComputePipeline(desc),
		backend.createComputePipeline(desc),
	]);
	const pipelineC = await backend.createComputePipeline(desc);

	assert.notEqual(pipelineA, pipelineB);
	assert.equal(pipelineA._gpuResource, pipelineB._gpuResource);
	assert.equal(pipelineC._gpuResource, pipelineA._gpuResource);
	assert.equal(device.pipelineLayouts.length, 0);
	assert.equal(device.computePipelineDescs.length, 1);
	assert.equal(device.computePipelineDescs[0].layout, "auto");
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.computePipelineEntries,
		1
	);
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

	const [pipelineA, pipelineB] = await Promise.all([
		backend.createPipeline(desc),
		backend.createPipeline(desc),
	]);
	const pipelineC = await backend.createPipeline(desc);

	assert.notEqual(pipelineA, pipelineB);
	assert.equal(pipelineA._gpuResource, pipelineB._gpuResource);
	assert.equal(pipelineC._gpuResource, pipelineA._gpuResource);
	assert.equal(device.pipelineLayouts.length, 0);
	assert.equal(device.renderPipelineDescs.length, 1);
	assert.equal(device.renderPipelineDescs[0].layout, "auto");
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.renderPipelineEntries,
		1
	);
}

async function testComputePipelineFailureClearsInFlight() {
	const { backend, device } = createBackend();
	const module = await backend.createShaderModule({
		code: "shader compute",
		label: "ComputeShader",
	});
	const desc = {
		label: "ComputePipelineRetry",
		compute: {
			module,
			entryPoint: "csMain",
		},
	};

	device.computePipelineFailuresRemaining = 1;
	await assert.rejects(
		() => backend.createComputePipeline(desc),
		/simulated compute pipeline failure/
	);
	let stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.computePipelineInFlight, 0);
	assert.equal(stats.computePipelineEntries, 0);

	const pipeline = await backend.createComputePipeline(desc);
	assert.ok(pipeline._gpuResource);
	assert.equal(device.computePipelineDescs.length, 2);
	stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.computePipelineEntries, 1);
}

async function testStaleRenderPipelineCreationRejectsAfterRollback() {
	const { backend, device } = createBackend();
	const shader = await backend.createShaderModule({
		code: "shader render",
		label: "RenderShader",
	});
	const desc = {
		label: "RenderPipelineStale",
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
	};
	const deferred = createDeferred();
	device.createRenderPipelineAsync = (pipelineDesc) => {
		device.renderPipelineDescs.push(pipelineDesc);
		return deferred.promise;
	};

	const pipelinePromise = backend.createPipeline(desc);
	await waitForCondition(
		() => backend.getWebGPUCacheDebugStats().pipeline.renderPipelineInFlight === 1,
		"Expected render pipeline creation to be in flight"
	);
	backend._rollbackInitializationState();
	deferred.resolve({
		desc,
		getBindGroupLayout() {
			return device.defaultBindGroupLayout;
		},
	});

	await assert.rejects(
		() => pipelinePromise,
		/WebGPU pipeline creation was invalidated/
	);
	const stats = backend.getWebGPUCacheDebugStats().pipeline;
	assert.equal(stats.renderPipelineEntries, 0);
	assert.equal(stats.renderPipelineInFlight, 0);
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
	const stats = backend.getWebGPUCacheDebugStats().bindingGroups;
	assert.equal(stats.entryCount, 1);
	assert.equal(stats.bucketCount, 1);
}

function testBindingGroupHashCollisionBucketSafety() {
	const { backend, device } = createBackend();
	backend.setBindingGroupHashOverrideForTesting(() => 1n);

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
	const stats = backend.getWebGPUCacheDebugStats().bindingGroups;
	assert.equal(stats.entryCount, 2);
	assert.deepEqual(stats.bucketSizes, [2]);

	backend.setBindingGroupHashOverrideForTesting(null);
}

async function testMSAAConfigurationClampsAndRuntimeFallbackInvalidatesCaches() {
	const { backend } = createBackend({ msaaSampleCount: 8 });
	backend._msaaController.activateDevice();
	await createCachedRenderPipeline(backend);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.renderPipelineEntries,
		1
	);

	assert.equal(backend._msaaController.sampleCount, 4);
	assert.equal(backend._msaaController.fallbackToSingleSample(), true);
	assert.equal(backend._msaaController.sampleCount, 1);
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.renderPipelineEntries,
		0
	);
}

function testMSAAPublicControlIsRemovedAndLegacyOptionFails() {
	const { backend } = createBackend();
	assert.equal(typeof backend.getMSAASampleCount, "undefined");
	assert.equal(typeof backend.setMSAAEnabled, "undefined");
	assert.equal(typeof backend.setMSAASampleCount, "undefined");
	assert.throws(
		() => new WebGPUBackend({ enableMSAA: false }),
		/msaaSampleCount: 1/
	);
	assert.throws(
		() => new WebGPUBackend({ msaaSampleCount: Number.NaN }),
		/finite number/
	);
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
		endFrame() {},
		abortFrame() {},
		invalidateFrameTargets() {
			invalidateCalls++;
		},
	};
	backend.resize({ width: 320.9, height: 240.2 });
	assert.equal(backend.canvas.width, 320);
	assert.equal(backend.canvas.height, 240);
	assert.equal(device.configureCalls, 1);
	assert.equal(invalidateCalls, 1);
}

async function testResizeDuringActiveFrameDefersResourceInvalidation() {
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
		endFrame() {},
		abortFrame() {},
		invalidateFrameTargets() {
			invalidateCalls++;
		},
	};
	backend._frameActive = true;

	backend.resize({ width: 320.9, height: 240.2 });
	assert.equal(backend.canvas.width, 1);
	assert.equal(backend.canvas.height, 1);
	assert.equal(device.configureCalls, 0);
	assert.equal(invalidateCalls, 0);

	await backend.endFrame();
	assert.equal(backend.canvas.width, 320);
	assert.equal(backend.canvas.height, 240);
	assert.equal(device.configureCalls, 1);
	assert.equal(invalidateCalls, 1);
	assert.equal(backend._pendingResize, null);
}

async function testShaderRuntimeChangeDuringActiveFrameDefersInvalidation() {
	const { backend } = createBackend();
	let executorInvalidations = 0;
	let resourceInvalidations = 0;
	backend._frameExecutor = {
		endFrame() {},
		abortFrame() {},
		onShaderRuntimeChanged() {
			executorInvalidations++;
		},
	};
	backend._resources = {
		onShaderRuntimeChanged() {
			resourceInvalidations++;
		},
	};
	await createCachedRenderPipeline(backend);
	backend._frameActive = true;

	backend.shaderRuntime.setMode("warn");
	assert.equal(backend.shaderRuntime.getMode(), "warn");
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.renderPipelineEntries,
		1
	);
	assert.equal(executorInvalidations, 0);
	assert.equal(resourceInvalidations, 0);

	await backend.endFrame();
	assert.equal(
		backend.getWebGPUCacheDebugStats().pipeline.renderPipelineEntries,
		0
	);
	assert.equal(executorInvalidations, 1);
	assert.equal(resourceInvalidations, 1);
	assert.equal(backend._pendingShaderRuntimeInvalidation, false);
}

async function testDeferredResizeInvalidatesFrameTargets() {
	const { backend, device } = createBackend();
	let invalidateCalls = 0;
	backend._canvas = { width: 1, height: 1 };
	backend._context = {
		configure() {
			device.configureCalls++;
		},
	};
	backend._frameExecutor = {
		endFrame() {},
		abortFrame() {},
		invalidateFrameTargets() {
			invalidateCalls++;
		},
	};
	backend._frameActive = true;

	backend.resize({ width: 10, height: 12 });
	await backend.abortFrame();

	assert.equal(backend.canvas.width, 10);
	assert.equal(backend.canvas.height, 12);
	assert.equal(invalidateCalls, 1);
}

async function testPublicDeviceLifecycleMethods() {
	const { backend } = createBackend();
	let resourcesDestroyed = false;
	backend._resources = {
		destroy() {
			resourcesDestroyed = true;
		},
	};

	Logger.configure({ level: "silent", resetOnceKeys: true });
	try {
		backend.onDeviceLost({
			reason: "destroyed",
			message: "simulated loss",
		});
	} finally {
		Logger.reset();
	}

	assert.equal(backend._deviceLost, true);
	assert.equal(backend._deviceLostInfo.message, "simulated loss");
	assert.equal(resourcesDestroyed, true);
	assert.equal(backend.device, null);
	assert.equal(backend.queue, null);

	await assert.rejects(
		() => backend.restore(),
		/cannot restore before a canvas has been initialized/
	);
}

function testAutomaticDeviceLossDestroysPostProcessBeforeRollback() {
	const { backend } = createBackend();
	const calls = [];
	backend._postProcessRuntime = {
		destroy() {
			calls.push("postprocess-runtime");
		},
	};
	backend._frameExecutor = {
		destroy() {
			calls.push("frame-executor");
		},
	};
	backend._resources = {
		destroy() {
			calls.push("resources");
		},
	};

	Logger.configure({ level: "silent", resetOnceKeys: true });
	try {
		backend.onDeviceLost({
			reason: "destroyed",
			message: "simulated loss",
		});
	} finally {
		Logger.reset();
	}

	assert.deepEqual(calls, [
		"postprocess-runtime",
		"frame-executor",
		"resources",
	]);
}

function testMapBindingResourceRejectsPrimitive() {
	const { backend } = createBackend();
	assert.throws(
		() =>
			backend.createBindingGroup({
				layout: { id: "layout-invalid" },
				entries: [{ binding: 0, resource: "invalid" }],
			}),
		/Unsupported binding resource/i
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

function testBackendPlanOmitsRendererOwnedPostProcessStage() {
	const { backend } = createBackend();
	backend._resources = {
		beginFrameResourceLifecycle() {},
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		getPreparedFrameResources() {
			return null;
		},
		endFrame() {},
		abortFrame() {},
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
		framePlan: createRendererFramePlan([
			"particle-sim",
			"main-opaque",
			"particles",
		]),
		scene: {
			particleSystems: [{}],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
	});
	backend.beginFrame(context);
	assert.equal(backend._plannedPasses.has("postprocess"), false);
	assert.equal(backend._plannedPassOrder.has("postprocess"), false);
}

function testPassPlanAllowsParticleStageBeforeMainOpaque() {
	const { backend } = createBackend();
	backend._resources = {
		beginFrameResourceLifecycle() {},
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		getPreparedFrameResources() {
			return null;
		},
		endFrame() {},
		abortFrame() {},
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
		framePlan: createRendererFramePlan([
			"particle-sim",
			"main-opaque",
			"particles",
		]),
		scene: {
			particleSystems: [{ id: "ps-0" }],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
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

async function testAbortFrameClearsPlannerAndDelegatesWithoutEndFrame() {
	const { backend } = createBackend();
	let executorAbortCalls = 0;
	let executorEndCalls = 0;
	let particleEndCalls = 0;
	backend._resources = {
		beginFrameResourceLifecycle() {},
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		getPreparedFrameResources() {
			return null;
		},
		endFrame() {
			executorEndCalls++;
		},
		abortFrame() {
			executorAbortCalls++;
		},
		destroy() {},
		invalidateFrameTargets() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {
			particleEndCalls++;
		},
	};

	const context = createFrameContext({
		framePlan: createRendererFramePlan(["particle-sim", "main-opaque"]),
		scene: {
			particleSystems: [{}],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
	});
	backend.beginFrame(context);
	await backend.executePass(
		{ stage: "particle-sim", executor: "backend", enabled: true },
		context
	);
	assert.equal(backend._plannedPasses.size > 0, true);
	assert.equal(backend._executedPasses.has("particle-sim"), true);

	await backend.abortFrame(new Error("failed frame"));

	assert.equal(executorAbortCalls, 1);
	assert.equal(executorEndCalls, 0);
	assert.equal(particleEndCalls, 1);
	assert.equal(backend._plannedPasses.size, 0);
	assert.equal(backend._plannedPassOrder.size, 0);
	assert.equal(backend._executedPasses.size, 0);
	assert.equal(backend._frameActive, false);
}

async function testEndFrameFailureStillEndsParticleFrameAndClearsPlanner() {
	const { backend } = createBackend();
	const error = new Error("executor end failed");
	let particleEndCalls = 0;
	backend._resources = {
		beginFrameResourceLifecycle() {},
		prepareFrame() {},
	};
	backend._frameExecutor = {
		beginFrame() {},
		executePass() {},
		getPreparedFrameResources() {
			return null;
		},
		endFrame() {
			throw error;
		},
		abortFrame() {},
		destroy() {},
		invalidateFrameTargets() {},
	};
	backend._particleSimulator = {
		beginFrame() {},
		simulate() {},
		emitRenderBatches() {},
		endFrame() {
			particleEndCalls++;
		},
	};

	backend.beginFrame(createFrameContext());
	let caught = null;
	try {
		await backend.endFrame();
	} catch (caughtError) {
		caught = caughtError;
	}

	assert.strictEqual(caught, error);
	assert.equal(particleEndCalls, 1);
	assert.equal(backend._plannedPasses.size, 0);
	assert.equal(backend._plannedPassOrder.size, 0);
	assert.equal(backend._executedPasses.size, 0);
	assert.equal(backend._frameActive, false);
}

async function testWarmupAggregatesPhases() {
	const { backend } = createBackend();
	backend._postProcessRuntime = {
		compileWarmupGraph() {
			return { orderedPasses: [], passes: [] };
		},
	};
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
	await testStaleShaderModuleCreationRejectsAfterRollback();
	await testStaleShaderModulePromiseDoesNotClearRecoveredInFlight();
	testSamplerReferenceCounting();
	await testComputePipelineAutoLayoutCaching();
	await testRenderPipelineAutoLayoutCaching();
	await testComputePipelineFailureClearsInFlight();
	await testStaleRenderPipelineCreationRejectsAfterRollback();
	testBindingGroupCacheUsesHashedKey();
	testBindingGroupHashCollisionBucketSafety();
	await testMSAAConfigurationClampsAndRuntimeFallbackInvalidatesCaches();
	testMSAAPublicControlIsRemovedAndLegacyOptionFails();
	testCreateBufferMappedAtCreationExposesUnmap();
	testResizeUsesProvidedDimensions();
	await testResizeDuringActiveFrameDefersResourceInvalidation();
	await testShaderRuntimeChangeDuringActiveFrameDefersInvalidation();
	await testDeferredResizeInvalidatesFrameTargets();
	await testPublicDeviceLifecycleMethods();
	testAutomaticDeviceLossDestroysPostProcessBeforeRollback();
	testMapBindingResourceRejectsPrimitive();
	testCreateTextureClampsPublicDimensions();
	testCommandBufferOwnershipAndOneShotSubmit();
	testBackendPlanOmitsRendererOwnedPostProcessStage();
	testPassPlanAllowsParticleStageBeforeMainOpaque();
	await testAbortFrameClearsPlannerAndDelegatesWithoutEndFrame();
	await testEndFrameFailureStillEndsParticleFrameAndClearsPlanner();
	await testWarmupAggregatesPhases();
	console.log("WebGPU backend cache/dependency tests passed");
}

run();
