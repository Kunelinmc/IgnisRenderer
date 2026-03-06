import assert from "node:assert/strict";
import { WebGPUPostProcessRuntime } from "../src/core/backend/webgpu/WebGPUPostProcessRuntime.ts";

class FakeBackend {
	constructor() {
		this.samplers = [];
		this.shaderModules = [];
		this.computePipelines = [];
		this.buffers = [];
		this.bindingGroups = [];
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
			destroy() {},
		};
		this.buffers.push(buffer);
		return buffer;
	}

	writeBuffer(buffer, data) {
		buffer.lastWrite = Array.from(data);
	}

	createBindingGroup(desc) {
		const bindingGroup = { label: desc.label, desc };
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
	assert.ok(
		backend.shaderModules[0].desc.code.includes("fn perceptualLuma")
	);
	assert.ok(backend.shaderModules[0].desc.code.includes("FXAA_QUALITY"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUFXAAPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUFXAAParams");
	assert.equal(backend.buffers[0].desc.size, 24);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 4);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
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

async function run() {
	await testFXAARuntimeUsesDedicatedPipeline();
	await testFXAARuntimePingPongsAndCachesResources();
	console.log("WebGPU postprocess runtime tests passed");
}

await run();
