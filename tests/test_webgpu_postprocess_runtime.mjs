import assert from "node:assert/strict";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";

class FakeBackend {
	constructor() {
		this.samplers = [];
		this.shaderModules = [];
		this.computePipelines = [];
		this.buffers = [];
		this.bindingGroups = [];
		this.bindingGroupDestroyCalls = 0;
		this.bufferDestroyCalls = 0;
	}

	createSampler(desc) {
		const sampler = { label: desc.label, desc };
		this.samplers.push(sampler);
		return sampler;
	}

	async createShaderModule(desc) {
		const module = { label: desc.label, desc };
		this.shaderModules.push(module);
		return module;
	}

	createComputePipeline(desc) {
		const pipeline = { label: desc.label, desc };
		this.computePipelines.push(pipeline);
		return pipeline;
	}

	createBuffer(desc) {
		const buffer = {
			size: desc.size,
			desc,
			destroyed: false,
			destroy: () => {
				if (buffer.destroyed) return;
				buffer.destroyed = true;
				this.bufferDestroyCalls++;
			},
		};
		this.buffers.push(buffer);
		return buffer;
	}

	writeBuffer(buffer, data) {
		buffer.lastWrite = Array.from(data);
	}

	createBindingGroup(desc) {
		const bindingGroup = {
			label: desc.label,
			desc,
			destroyed: false,
			destroy: () => {
				if (bindingGroup.destroyed) return;
				bindingGroup.destroyed = true;
				this.bindingGroupDestroyCalls++;
			},
		};
		this.bindingGroups.push(bindingGroup);
		return bindingGroup;
	}
}

class FakeEncoder {
	constructor() {
		this.calls = [];
	}

	beginComputePass(desc = {}) {
		this.calls.push(["beginComputePass", desc.label ?? null]);
	}

	setComputePipeline(pipeline) {
		this.calls.push(["setComputePipeline", pipeline.label]);
	}

	setBindingGroup(index, group) {
		this.calls.push(["setBindingGroup", index, group.label]);
	}

	dispatchWorkgroups(x, y = 1, z = 1) {
		this.calls.push(["dispatchWorkgroups", x, y, z]);
	}

	endComputePass() {
		this.calls.push(["endComputePass"]);
	}
}

function createTexture(width, height, label) {
	return {
		width,
		height,
		label,
		destroy() {},
	};
}

function assertClose(actual, expected, epsilon = 1e-6) {
	assert.ok(Math.abs(actual - expected) <= epsilon);
}

async function testFXAARuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(17, 9, "scene");
	const postPing = createTexture(17, 9, "ping");
	const postPong = createTexture(17, 9, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executeFXAA(encoder, targets);

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUFXAAShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("fn perceptualLuma"));
	assert.ok(backend.shaderModules[0].desc.code.includes("FXAA_QUALITY"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUFXAAPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUFXAAParams");
	assert.equal(backend.buffers[0].desc.size, 24);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 4);
	assert.equal(
		backend.bindingGroups[0].desc.entries[0].resource,
		sceneColorMain
	);
	assert.equal(backend.bindingGroups[0].desc.entries[3].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 6);
	assertClose(params[0], 1 / 17);
	assertClose(params[1], 1 / 9);
	assertClose(params[2], 0.03125);
	assertClose(params[3], 0.166);
	assertClose(params[4], 0.75);
	assertClose(params[5], 0);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUFXAA"],
		["setComputePipeline", "WebGPUFXAAPipeline"],
		["setBindingGroup", 0, "WebGPUFXAA_Binding"],
		["dispatchWorkgroups", 3, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function testFXAARuntimePingPongsAndCachesResources() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const firstEncoder = new FakeEncoder();
	const secondEncoder = new FakeEncoder();
	const sceneColorMain = createTexture(32, 16, "scene");
	const postPing = createTexture(32, 16, "ping");
	const postPong = createTexture(32, 16, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executeFXAA(firstEncoder, targets);
	await runtime.executeFXAA(secondEncoder, targets);

	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.bindingGroups.length, 2);
	assert.equal(backend.bindingGroups[1].desc.entries[0].resource, postPong);
	assert.equal(backend.bindingGroups[1].desc.entries[3].resource, postPing);
	assert.equal(targets.sceneColor, postPing);
	assert.deepEqual(secondEncoder.calls, [
		["beginComputePass", "WebGPUFXAA"],
		["setComputePipeline", "WebGPUFXAAPipeline"],
		["setBindingGroup", 0, "WebGPUFXAA_Binding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
	]);
}

async function testSSAORuntimeRunsGTAOPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(64, 32, "scene");
	const postPing = createTexture(64, 32, "ping");
	const postPong = createTexture(64, 32, "pong");
	const gNormalRoughMetal = createTexture(64, 32, "g-normal");
	const gMotionDepth = createTexture(64, 32, "g-motion-depth");
	const aoRaw = createTexture(32, 16, "ao-raw");
	const aoBlur = createTexture(32, 16, "ao-blur");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gNormalRoughMetal,
		gMotionDepth,
		aoRaw,
		aoBlur,
	};
	const frameContext = {
		camera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 2,
		},
		features: {
			ssaoOptions: {
				radius: 10,
				bias: 0.2,
				intensity: 1.3,
				samples: 20,
				blurRadius: 3,
				blurSharpness: 12,
			},
		},
	};

	await runtime.executeSSAO(encoder, targets, frameContext);

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUSSAOShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("MAX_DIRECTION_COUNT"));
	assert.ok(backend.shaderModules[0].desc.code.includes("fn csRaw"));
	assert.equal(backend.computePipelines.length, 3);
	assert.equal(backend.computePipelines[0].label, "WebGPUSSAORawPipeline");
	assert.equal(backend.computePipelines[1].label, "WebGPUSSAOBlurPipeline");
	assert.equal(backend.computePipelines[2].label, "WebGPUSSAOCombinePipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUSSAOParams");
	assert.equal(backend.buffers[0].desc.size, 64);
	assert.equal(backend.bindingGroups.length, 4);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, gNormalRoughMetal);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[1].desc.entries[0].resource, aoRaw);
	assert.equal(backend.bindingGroups[1].desc.entries[4].resource, aoBlur);
	assert.equal(backend.bindingGroups[2].desc.entries[0].resource, aoBlur);
	assert.equal(backend.bindingGroups[2].desc.entries[4].resource, aoRaw);
	assert.equal(backend.bindingGroups[3].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[3].desc.entries[1].resource, aoRaw);
	assert.equal(backend.bindingGroups[3].desc.entries[4].resource, postPing);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 16);
	assertClose(params[0], 1 / 64);
	assertClose(params[1], 1 / 32);
	assertClose(params[2], 1 / 32);
	assertClose(params[3], 1 / 16);
	assertClose(params[4], 10);
	assertClose(params[5], 0.2);
	assertClose(params[6], 1.3);
	assertClose(params[7], 20);
	assertClose(params[8], 3);
	assertClose(params[9], 12);
	assertClose(params[10], Math.tan(Math.PI / 6));
	assertClose(params[11], 2);
	assertClose(params[12], 0);
	assertClose(params[13], 1);
	assertClose(params[14], 0);
	assert.ok(params[15] > 0 && params[15] < 1);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUSSAO_Raw"],
		["setComputePipeline", "WebGPUSSAORawPipeline"],
		["setBindingGroup", 0, "WebGPUSSAO_RawBinding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
		["beginComputePass", "WebGPUSSAO_Blur"],
		["setComputePipeline", "WebGPUSSAOBlurPipeline"],
		["setBindingGroup", 0, "WebGPUSSAO_BlurBinding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
		["beginComputePass", "WebGPUSSAO_BlurVertical"],
		["setComputePipeline", "WebGPUSSAOBlurPipeline"],
		["setBindingGroup", 0, "WebGPUSSAO_BlurBindingVertical"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
		["beginComputePass", "WebGPUSSAO_Combine"],
		["setComputePipeline", "WebGPUSSAOCombinePipeline"],
		["setBindingGroup", 0, "WebGPUSSAO_CombineBinding"],
		["dispatchWorkgroups", 8, 4, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPing);
}

async function testInvalidateBindingsDestroysCachedBindingGroups() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(20, 12, "scene");
	const postPing = createTexture(20, 12, "ping");
	const postPong = createTexture(20, 12, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executeFXAA(encoder, targets);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	runtime.invalidateBindings();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

async function testBindingReplacementDestroysStaleBindingGroup() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const firstEncoder = new FakeEncoder();
	const secondEncoder = new FakeEncoder();
	const thirdEncoder = new FakeEncoder();
	const sceneColorMain = createTexture(28, 14, "scene");
	const postPing = createTexture(28, 14, "ping");
	const postPong = createTexture(28, 14, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executeFXAA(firstEncoder, targets);
	await runtime.executeFXAA(secondEncoder, targets);
	await runtime.executeFXAA(thirdEncoder, targets);

	assert.equal(backend.bindingGroups.length, 3);
	assert.equal(backend.bindingGroupDestroyCalls, 1);
}

async function testOnShaderRuntimeChangedDestroysParameterBuffers() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(24, 12, "scene");
	const postPing = createTexture(24, 12, "ping");
	const postPong = createTexture(24, 12, "pong");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
	};

	await runtime.executeFXAA(encoder, targets);
	await runtime.warmupHints([
		"postprocess:ssao",
		"postprocess:taa",
		"postprocess:ssr",
		"postprocess:volumetric",
	]);
	assert.equal(backend.bufferDestroyCalls, 0);
	assert.equal(backend.bindingGroupDestroyCalls, 0);

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(backend.bufferDestroyCalls, 6);
	const destroyedLabels = new Set(
		backend.buffers
			.filter((buffer) => buffer.destroyed)
			.map((buffer) => buffer.desc.label)
	);
	assert.ok(destroyedLabels.has("WebGPUSSAOParams"));
	assert.ok(destroyedLabels.has("WebGPUTAAParams"));
	assert.ok(destroyedLabels.has("WebGPUSSRTraceParams"));
	assert.ok(destroyedLabels.has("WebGPUSSRComposeParams"));
	assert.ok(destroyedLabels.has("WebGPUVolumetricParams"));
	assert.ok(destroyedLabels.has("WebGPUFXAAParams"));

	runtime.onShaderRuntimeChanged();
	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(backend.bufferDestroyCalls, 6);
}

async function run() {
	await testFXAARuntimeUsesDedicatedPipeline();
	await testFXAARuntimePingPongsAndCachesResources();
	await testSSAORuntimeRunsGTAOPipeline();
	await testInvalidateBindingsDestroysCachedBindingGroups();
	await testBindingReplacementDestroysStaleBindingGroup();
	await testOnShaderRuntimeChangedDestroysParameterBuffers();
	console.log("WebGPU postprocess runtime tests passed");
}

await run();
