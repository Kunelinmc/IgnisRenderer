import assert from "node:assert/strict";
import { WebGPUPostProcessRuntime } from "../src/renderers/webgpu/WebGPUPostProcessRuntime.ts";
import {
	FakeBackend,
	FakeEncoder,
	assertClose,
	createTexture,
} from "./helpers/webgpu_postprocess_runtime_test_helpers.mjs";

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

	const result = await runtime.executePass({
		passId: "ssao",
		encoder,
		targets,
		frameContext,
	});
	assert.equal(result.ran, true);

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

async function testSSGIRuntimeUsesDedicatedPipeline() {
	const backend = new FakeBackend();
	const runtime = new WebGPUPostProcessRuntime(backend, () => {});
	const encoder = new FakeEncoder();
	const sceneColorMain = createTexture(32, 16, "scene");
	const postPing = createTexture(32, 16, "ping");
	const postPong = createTexture(32, 16, "pong");
	const gAlbedoAlpha = createTexture(32, 16, "g-albedo");
	const gNormalRoughMetal = createTexture(32, 16, "g-normal");
	const gMotionDepth = createTexture(32, 16, "g-motion-depth");
	const targets = {
		sceneColor: sceneColorMain,
		postPing,
		postPong,
		gAlbedoAlpha,
		gNormalRoughMetal,
		gMotionDepth,
	};
	const frameContext = {
		features: {
			ssgiOptions: {
				radius: 4,
				intensity: 0.5,
				falloff: 1.8,
				depthPhi: 1.4,
				normalPhi: 2.5,
				albedoBoost: 1.2,
				samples: 24,
			},
		},
	};

	const result = await runtime.executePass({
		passId: "ssgi",
		encoder,
		targets,
		frameContext,
	});
	assert.equal(result.ran, true);

	assert.equal(backend.samplers.length, 1);
	assert.equal(backend.shaderModules.length, 1);
	assert.equal(backend.shaderModules[0].label, "WebGPUSSGIShader");
	assert.ok(backend.shaderModules[0].desc.code.includes("SAMPLE_OFFSETS"));
	assert.ok(backend.shaderModules[0].desc.code.includes("MAX_SSGI_SAMPLES"));
	assert.equal(backend.computePipelines.length, 1);
	assert.equal(backend.computePipelines[0].label, "WebGPUSSGIPipeline");
	assert.equal(backend.buffers.length, 1);
	assert.equal(backend.buffers[0].desc.label, "WebGPUSSGIParams");
	assert.equal(backend.buffers[0].desc.size, 48);
	assert.equal(backend.bindingGroups.length, 1);
	assert.equal(backend.bindingGroups[0].desc.entries.length, 7);
	assert.equal(backend.bindingGroups[0].desc.entries[0].resource, sceneColorMain);
	assert.equal(backend.bindingGroups[0].desc.entries[1].resource, gAlbedoAlpha);
	assert.equal(
		backend.bindingGroups[0].desc.entries[2].resource,
		gNormalRoughMetal
	);
	assert.equal(backend.bindingGroups[0].desc.entries[3].resource, gMotionDepth);
	assert.equal(backend.bindingGroups[0].desc.entries[6].resource, postPong);

	const params = backend.buffers[0].lastWrite;
	assert.equal(params.length, 12);
	assertClose(params[0], 1 / 32);
	assertClose(params[1], 1 / 16);
	assertClose(params[2], 4);
	assertClose(params[3], 0.5);
	assertClose(params[4], 1.8);
	assertClose(params[5], 1.4);
	assertClose(params[6], 2.5);
	assertClose(params[7], 1.2);
	assertClose(params[8], 16);
	assertClose(params[9], 0);
	assertClose(params[10], 0);
	assertClose(params[11], 0);

	assert.deepEqual(encoder.calls, [
		["beginComputePass", "WebGPUSSGI"],
		["setComputePipeline", "WebGPUSSGIPipeline"],
		["setBindingGroup", 0, "WebGPUSSGI_Binding"],
		["dispatchWorkgroups", 4, 2, 1],
		["endComputePass"],
	]);
	assert.equal(targets.sceneColor, postPong);
}

async function run() {
	await testSSAORuntimeRunsGTAOPipeline();
	await testSSGIRuntimeUsesDedicatedPipeline();
	console.log("WebGPU postprocess spatial runtime tests passed");
}

await run();
