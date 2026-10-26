import assert from "node:assert/strict";
import {
	WebGPUFrameServiceOwner as WebGPURenderResources
} from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import { submitWebGPUDraws } from "../../../src/backends/webgpu/WebGPUDrawSubmission.ts";
import {
	WebGPUFrameOrchestrator as WebGPUFrameExecutor
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import {
	createWebGPUFrameRuntimeComposition
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameRuntimeComposition.ts";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS,
	WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS,
	WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS
} from "../../../src/backends/webgpu/index.ts";
import {
	BufferUsage
} from "../../../src/backends/types.ts";
import {
	PointLight
} from "../../../src/lights/PointLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	PBRMaterial
} from "../../../src/materials/PBRMaterial.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	MAX_POINT_LIGHTS
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
	WEBGPU_TEXTURE_SLOT,
	WEBGPU_TEXTURE_SLOT_COUNT
} from "../../../src/backends/webgpu/constants.ts";
import {
	createWebGPUComputeFacade
} from "../../../src/backends/webgpu/ComputeFacade.ts";

import {
	FakeWebGPUBackend as FakeBackend
} from "../../helpers/fakes.mjs";

import {
	createFrame,
	createFrameContextWithFeatures,
	createMainFrameOptions,
	createModel,
	createPacket
} from "../../helpers/webgpu-bridge.mjs";
const previousGPUShaderStage = globalThis.GPUShaderStage;
globalThis.GPUShaderStage = {
	...(previousGPUShaderStage ?? {}),
	VERTEX: previousGPUShaderStage?.VERTEX ?? 1,
	FRAGMENT: previousGPUShaderStage?.FRAGMENT ?? 2,
	COMPUTE: previousGPUShaderStage?.COMPUTE ?? 4,
};
ShaderSource.resetConfiguration();
Logger.reset();

function testRenderResourcesConsumeInjectedComputeFacade() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);

	assert.equal(backend.getComputeFacadeCalls, 0);
	assert.equal(
		typeof resources._computeFacade.createComputePipeline,
		"function"
	);

	resources.destroy();
}

function testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend() {
	const backend = new FakeBackend();
	let listenerCount = 0;
	backend.shaderRuntime = {
		revision: 1,
		getMode: () => "strict",
		onDidChange: () => {
			listenerCount++;
			return () => {
				listenerCount--;
			};
		},
	};
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	assert.equal(listenerCount, 0);

	resources.destroy();
	assert.equal(listenerCount, 0);
}

async function testRenderResourcesLogPointLightLimitOnlyOnce() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(
		backend,
		backend,
		createWebGPUComputeFacade(backend),
	);
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: { warn: (...args) => warnings.push(args) },
		resetOnceKeys: true,
	});

	try {
		const model = createModel([new PBRMaterial()]);
		const frame = createFrame(createPacket(model));
		frame.lights = Array.from(
			{ length: MAX_POINT_LIGHTS + 1 },
			() => new PointLight(),
		);
		const context = createFrameContextWithFeatures(
			frame,
			{ enableLighting: true },
			{ clusteredLighting: false },
		);

		resources.prepareFrame(context, createMainFrameOptions());
		resources.prepareFrame(
			context,
			createMainFrameOptions({ temporalStateMode: "reuse" }),
		);

		const pointLimitWarnings = warnings.filter((args) =>
			args.some(
				(value) =>
					typeof value === "string" && value.includes("[webgpu-point-limit]"),
			),
		);
		assert.equal(pointLimitWarnings.length, 1);
	} finally {
		resources.destroy();
		Logger.reset();
	}
}

function testFrameExecutorConsumesComputeFacadeFromHost() {
	const backend = new FakeBackend();
	const resourcesStub = {
		sceneFrameLayout: null,
		getParticleBillboardRenderer() {
			return { async renderParticles() {} };
		},
		createPlanarReflectionDrawResources() {
			return { async getDrawResources() { return null; }, destroy() {} };
		},
		createFrameScope() {
			return {
				prepare() { throw new Error("not used by this test"); },
				destroy() {},
			};
		},
	};
	const sampleCountResolver = {
		resolveDomainSampleCount(domain, requestedSampleCount, formats) {
			return {
				domain,
				requestedSampleCount,
				sampleCount: 1,
				signature: `${domain}|${formats.join(",")}`,
				runtimeFallbackActive: false,
			};
		},
		fallbackToSingleSample() {
			return false;
		},
	};
	const frameRuntime = createWebGPUFrameRuntimeComposition({
		host: backend,
		frameServices: resourcesStub,
		shadowRenderer: {
			renderShadows() {},
		},
		sampleCountResolver,
		warnOnce() {},
	});
	const executor = new WebGPUFrameExecutor(
		backend,
		resourcesStub.createFrameScope(),
		sampleCountResolver,
		1,
		frameRuntime.modules,
	);

	assert.equal(backend.getComputeFacadeCalls, 0);
	assert.equal(typeof executor.getDebugState, "undefined");
}

async function testRenderResourcesUseCopyDstForUploads() {
	const backend = new FakeBackend();
	const model = createModel([
		new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 },
		}),
	]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const frameResources = resources.prepareFrame(
		createFrameContextWithFeatures(
			frame,
			{
				enableLighting: true,
				enableGamma: true,
				enableShadows: true,
			},
			{
				sh: false,
				shadows: true,
				reflection: false,
				environment: false,
				ssao: false,
				taa: false,
				ssr: false,
				volumetric: false,
				fog: false,
				motionBlur: false,
				dof: false,
				bloom: false,
				clusteredLighting: true,
			}
		),
		createMainFrameOptions()
	);

	const draw = await resources.getDrawResources(packet, frameResources, {
		sampleCount: 1,
	});

	assert.ok(draw);
	const firstDraw = draw[0];
	assert.ok(firstDraw);
	assert.equal(
		firstDraw.resolvedInputs.geometry.vertexBuffer,
		firstDraw.vertexBuffer
	);
	assert.equal(
		firstDraw.resolvedInputs.geometry.indexBuffer,
		firstDraw.indexBuffer
	);
	assert.equal(firstDraw.resolvedInputs.materialData.pipelineKey.length > 0, true);
	assert.equal(
		firstDraw.resolvedInputs.textures.length,
		firstDraw.resolvedInputs.materialData.common.textureSlots.length
	);
	assert.equal(
		firstDraw.resolvedInputs.samplers.length,
		firstDraw.resolvedInputs.materialData.common.textureSlots.length
	);
	assert.equal("anisotropyTexture" in firstDraw.resolvedInputs, false);
	assert.ok(firstDraw.resolvedInputs.textures[WEBGPU_TEXTURE_SLOT.ANISOTROPY]);
	assert.deepEqual(
		firstDraw.bindings[0].group.desc.entries.map((entry) => entry.binding),
		[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12],
	);
	assert.ok(
		firstDraw.bindings[0].group.desc.entries.some((entry) => entry.binding === 6)
	);
	assert.ok(
		firstDraw.bindings[0].group.desc.entries.some((entry) => entry.binding === 8)
	);
	for (const binding of [0, 10, 11, 12]) {
		assert.ok(
			firstDraw.bindings[0].group.desc.entries.some(
				(entry) => entry.binding === binding
			)
		);
	}
	assert.equal(
		firstDraw.bindings[1].group.desc.entries.length,
		1 +
			WEBGPU_TEXTURE_SLOT_COUNT +
			WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT +
			9
	);
	assert.ok(
		firstDraw.bindings[1].group.desc.entries.some(
			(entry) => entry.binding === WEBGPU_MODEL_BINDING_SHADER_UNIFORMS
		)
	);
	assert.ok(
		firstDraw.bindings[1].group.desc.entries.some(
			(entry) =>
				entry.binding === 1 + WEBGPU_TEXTURE_SLOT.ANISOTROPY * 2
		)
	);
	assert.deepEqual(
		firstDraw.pipeline.desc.layout.desc.bindGroupLayouts,
		firstDraw.bindings.map((binding) => binding.group.desc.layout)
	);
	assert.equal(
		firstDraw.pipeline.desc.layout.desc.bindGroupLayouts.length,
		3
	);
	assert.equal(firstDraw.pipeline.desc.fragment.targets.length, 5);
	assert.deepEqual(
		firstDraw.pipeline.desc.fragment.targets.map((target) => target.format),
		["rgba16float", "rgba8unorm", "rgba8unorm", "rgba16float", "rgba16float"]
	);
	const modelBindingIndices = firstDraw.bindings[1].group.desc.entries.map(
		(entry) => entry.binding
	);
	assert.ok(modelBindingIndices.includes(29));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_ANIMATION_PARAMS)
	);
	assert.ok(modelBindingIndices.includes(31));
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_JOINT_MATRICES)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_WEIGHTS)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_POSITION)
	);
	assert.ok(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_MORPH_NORMAL)
	);
	assert.ok(
		modelBindingIndices.includes(1 + WEBGPU_TEXTURE_SLOT.ANISOTROPY * 2)
	);
	assert.equal(
		modelBindingIndices.includes(WEBGPU_MODEL_BINDING_STATIC_INSTANCES),
		true
	);
	const sceneVertexLayouts = firstDraw.pipeline.desc.vertex.buffers;
	const sceneVertexAttributes = sceneVertexLayouts.flatMap(
		(layout) => layout.attributes
	);
	assert.deepEqual(sceneVertexLayouts[0].attributes, [
		{ shaderLocation: 0, format: "float32x3", offset: 0 },
	]);
	assert.equal(sceneVertexLayouts[3].arrayStride, 0);
	assert.ok(
		sceneVertexAttributes.some((attribute) => attribute.shaderLocation === 8)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Vertex) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Index) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	assert.ok(
		backend.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Uniform) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	);
	for (const [label, size] of [
		["WebGPUFrameCameraUniforms", WEBGPU_FRAME_CAMERA_UNIFORM_FLOATS * 4],
		["WebGPUFrameLightUniforms", WEBGPU_FRAME_LIGHT_UNIFORM_FLOATS * 4],
		["WebGPUFrameShadowUniforms", WEBGPU_FRAME_SHADOW_UNIFORM_FLOATS * 4],
		["WebGPUFrameEnvironmentUniforms", WEBGPU_FRAME_ENVIRONMENT_UNIFORM_FLOATS * 4],
	]) {
		assert.ok(
			backend.bufferDescs.some(
				(desc) => desc.label === label && desc.size === size
			)
		);
	}
}

async function testStaticBatchSplitDiagnosticsAccumulateAndResetPerFrame() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	try {
		const first = createPacket(createModel([new PBRMaterial()]));
		const different = createPacket(createModel([new PBRMaterial()]));
		const last = {
			...first,
			submission: { ...first.submission, id: `${first.submission.id}:copy` },
		};
		const packets = [first, different, last];
		const frame = createFrame(first);
		frame.opaquePackets = packets;
		const context = createFrameContextWithFeatures(frame, {}, {});
		resources.beginFrameResourceLifecycle();
		const frameResources = resources.prepareFrame(context, createMainFrameOptions());
		const request = {
			encoder: backend.createCommandEncoder(),
			resources,
			frameResources,
			packets,
			resolveDrawOptions: () => ({ sampleCount: 1 }),
		};
		const firstResult = await submitWebGPUDraws(request);
		assert.deepEqual(firstResult.staticBatching, {
			nonAdjacentKeySplits: 1,
			nonContiguousInstanceSplits: 0,
		});
		const anotherScope = resources.prepareFrame(
			context, createMainFrameOptions({ scopeKey: "secondary" }),
		);
		const secondResult = await submitWebGPUDraws({
			...request, frameResources: anotherScope, packets: [first, last],
		});
		assert.deepEqual(secondResult.staticBatching, {
			nonAdjacentKeySplits: 0,
			nonContiguousInstanceSplits: 1,
		});
		const snapshot = resources.getDebugStats().staticBatching;
		assert.equal(snapshot.nonAdjacentKeySplits, 1);
		assert.equal(snapshot.nonContiguousInstanceSplits, 1);
		assert.equal(snapshot.instanceCount, 3);
		const failedEncoder = backend.createCommandEncoder();
		failedEncoder.drawIndexed = () => { throw new Error("Recording failed."); };
		await assert.rejects(
			submitWebGPUDraws({ ...request, encoder: failedEncoder }), /Recording failed/,
		);
		assert.equal(resources.getDebugStats().staticBatching.nonAdjacentKeySplits, 1);
		assert.equal(resources.getDebugStats().staticBatching.nonContiguousInstanceSplits, 1);
		resources.beginFrameResourceLifecycle();
		assert.equal(resources.getDebugStats().staticBatching.nonAdjacentKeySplits, 0);
		assert.equal(resources.getDebugStats().staticBatching.nonContiguousInstanceSplits, 0);
		assert.equal(snapshot.nonAdjacentKeySplits, 1, "Prior diagnostic snapshots stay stable.");
		assert.equal(snapshot.nonContiguousInstanceSplits, 1);
		assert.equal(firstResult.staticBatching.nonContiguousInstanceSplits, 0);
	} finally {
		resources.destroy();
	}
}

function testVolumetricLightingUsesEffectiveSurfaceLights() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	try {
		for (const entry of [
			{ clustered: false, maxLights: 2, pointCount: 16 },
			{ clustered: true, maxLights: 2, pointCount: 2 },
			{ clustered: true, maxLights: 1.9, pointCount: 1 },
			{ clustered: true, maxLights: 0, pointCount: 1 },
			{ clustered: true, maxLights: NaN, sourceCount: 300, pointCount: 256 },
			{ clustered: true, sourceCount: 300, pointCount: 256 },
			{ clustered: true, maxLights: 2000, sourceCount: 1025, pointCount: 1024 },
			{ clustered: true, cameraType: CameraType.Orthographic, pointCount: 16 },
			{ clustered: true, invalidDepthRange: true, pointCount: 16 },
			{ clustered: true, enableLighting: false, pointCount: 0 },
		]) {
			const frame = createFrame(createPacket(createModel([new PBRMaterial()])));
			frame.camera.type = entry.cameraType ?? CameraType.Perspective;
			if (entry.invalidDepthRange) {
				frame.camera.near = 1e20;
				frame.camera.far = 1e20;
			}
			const pointLights = Array.from({ length: entry.sourceCount ?? 17 }, (_, i) => {
				const light = new PointLight({ position: { x: i + 1, y: 0, z: 0 } });
				light.updateWorldMatrix();
				return light;
			});
			frame.lights = [new DirectionalLight(), ...pointLights];
			const context = createFrameContextWithFeatures(frame, {
				enableLighting: entry.enableLighting ?? true,
				enableClusteredLighting: entry.clustered,
				clusteredLightingOptions: { maxLights: entry.maxLights },
			}, { clusteredLighting: true }, {
				postProcess: createResolvedPostProcess({ volumetric: { enabled: true } }),
			});
			const prepared = resources.prepareFrame(context, createMainFrameOptions());
			const lights = prepared.volumetricLightingData?.lights;
			assert.ok(lights, "Enabled volumetric lighting must receive prepared light data.");
			assert.equal(
				lights.filter((light) => light.type === 0).length,
				entry.enableLighting === false ? 0 : 1,
			);
			assert.deepEqual(
				lights.filter((light) => light.type === 1).map((light) => light.position[0]),
				Array.from({ length: entry.pointCount }, (_, i) => i + 1),
			);
			assert.equal(lights.length, entry.pointCount + (entry.enableLighting === false ? 0 : 1));
			resources.abortTemporalFrame();
		}
	} finally {
		resources.destroy();
	}
}

function testVolumetricLightingDataStaysScopedAndClearsWhenDisabled() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));
	try {
		const frame = createFrame(createPacket(createModel([new PBRMaterial()])));
		frame.lights = [new PointLight()];
		const context = createFrameContextWithFeatures(frame, { enableLighting: true }, {}, {
			postProcess: createResolvedPostProcess({ volumetric: { enabled: true } }),
		});
		const main = resources.prepareFrame(context, createMainFrameOptions());
		const captureFrame = createFrame(createPacket(createModel([new PBRMaterial()])));
		captureFrame.lights = [new DirectionalLight()];
		const captureContext = createFrameContextWithFeatures(captureFrame,
			{ enableLighting: true }, {}, { postProcess: context.postProcess });
		const capture = resources.prepareFrame(captureContext,
			createMainFrameOptions({ scopeKey: "capture" }));
		assert.deepEqual(main.volumetricLightingData?.lights.map((light) => light.type), [1]);
		assert.deepEqual(capture.volumetricLightingData?.lights.map((light) => light.type), [0]);
		resources.commitTemporalFrame();
		context.postProcess = createResolvedPostProcess({ volumetric: { enabled: false } });
		const disabled = resources.prepareFrame(context, createMainFrameOptions());
		assert.equal(disabled.volumetricLightingData, null);
		assert.deepEqual(main.volumetricLightingData.lights.map((light) => light.type), [1]);
		assert.deepEqual(capture.volumetricLightingData.lights.map((light) => light.type), [0]);
	} finally {
		resources.destroy();
	}
}

async function run() {
	try {
		await testRenderResourcesConsumeInjectedComputeFacade();
		await testRenderResourcesLeaveShaderRuntimeSubscriptionToBackend();
		await testRenderResourcesLogPointLightLimitOnlyOnce();
		await testFrameExecutorConsumesComputeFacadeFromHost();
		await testRenderResourcesUseCopyDstForUploads();
		await testStaticBatchSplitDiagnosticsAccumulateAndResetPerFrame();
		testVolumetricLightingUsesEffectiveSurfaceLights();
		testVolumetricLightingDataStaysScopedAndClearsWhenDisabled();
		console.log("WebGPU bridge render resources tests passed");
	} finally {
		ShaderSource.resetConfiguration();
		Logger.reset();
		if (previousGPUShaderStage === undefined) {
			delete globalThis.GPUShaderStage;
		} else {
			globalThis.GPUShaderStage = previousGPUShaderStage;
		}
	}
}
await run();
