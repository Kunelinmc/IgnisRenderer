import assert from "node:assert/strict";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";

class FakeDevice {
	constructor() {
		this.limits = { maxBindGroups: 4 };
		this.defaultBindGroupLayout = { id: "group0" };
		this.shaderModuleDescs = [];
		this.shaderModuleFailuresRemaining = 0;
		this.computePipelineDescs = [];
		this.renderPipelineDescs = [];
		this.pipelineLayouts = [];
		this.bindGroupDescs = [];
		this.samplerDescs = [];
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
			getCompilationInfo: async () => ({ messages: [] }),
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
	backend.device = device;
	backend.queue = {
		submit() {},
	};
	return { backend, device };
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
			enableTAA: true,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			warnings: [],
			ssrOptions: {},
			ssaoOptions: {},
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

	await backend.createShaderModule({ code: shaderCode, label: "A" });
	await backend.createShaderModule({
		code: `${shaderCode}`,
		label: "B",
	});

	assert.equal(device.shaderModuleDescs.length, 1);
	const cacheKeys = Array.from(backend._shaderModuleCache.keys());
	assert.equal(cacheKeys.length, 1);
	assert.notEqual(cacheKeys[0], shaderCode);
	assert.ok(cacheKeys[0].includes("hash:"));
	const entry = backend._shaderModuleCache.values().next().value;
	assert.equal(typeof entry.module.then, "undefined");
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

function testSamplerReferenceCounting() {
	const { backend } = createBackend();
	const samplerA = backend.createSampler({});
	const samplerB = backend.createSampler({});

	assert.equal(samplerA, samplerB);
	const samplerEntry = backend._samplerCache.values().next().value;
	assert.equal(samplerEntry.refCount, 1);

	samplerA.destroy();
	assert.equal(backend._samplerCache.size, 0);
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

	assert.equal(pipelineA, pipelineB);
	assert.equal(device.pipelineLayouts.length, 0);
	assert.equal(device.computePipelineDescs.length, 1);
	assert.equal(device.computePipelineDescs[0].layout, "auto");
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

function testMapBindingResourceRejectsPrimitive() {
	const { backend } = createBackend();
	assert.throws(
		() => backend._mapBindingResource("invalid"),
		/Unsupported WebGPU binding resource/
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
	assert.ok(backend._plannedPassOrder.get("ssao") < backend._plannedPassOrder.get("taa"));

	assert.throws(
		() =>
			backend.executePass(
				{ stage: "taa", executor: "backend", enabled: true },
				context
			),
		/dependencies: ssao/
	);

	backend.executePass(
		{ stage: "ssao", executor: "backend", enabled: true },
		context
	);
	assert.doesNotThrow(() =>
		backend.executePass(
			{ stage: "taa", executor: "backend", enabled: true },
			context
		)
	);
}

async function run() {
	await testShaderModuleCacheUsesHashKey();
	await testShaderModuleRetryWithinSingleRequest();
	testSamplerReferenceCounting();
	await testComputePipelineAutoLayoutCaching();
	testBindingGroupCacheUsesHashedKey();
	testBindingGroupHashCollisionBucketSafety();
	testMapBindingResourceRejectsPrimitive();
	testPassDependencyValidation();
	console.log("WebGPU backend cache/dependency tests passed");
}

run();
