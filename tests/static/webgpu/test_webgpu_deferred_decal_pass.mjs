import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { Material } from "../../../src/materials/Material.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGPUDeferredDecalPass } from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredDecalPass.ts";
import {
	TextureFormat,
	TextureUsage,
} from "../../../src/backends/types.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function createCamera() {
	return {
		viewProjectionMatrix: Matrix4.identity(),
		getWorldDirection(direction, out) {
			out.x = direction.x;
			out.y = direction.y;
			out.z = direction.z;
			return out;
		},
	};
}

function createDecalPacket(id, material, overrides = {}) {
	return {
		id,
		decal: { id, name: id },
		material,
		worldMatrix: Matrix4.identity(),
		inverseWorldMatrix: Matrix4.identity(),
		normalMatrix: overrides.normalMatrix ?? Matrix4.identity(),
		worldBounds: {
			center: { x: 0, y: 0, z: 0 },
			radius: overrides.radius ?? 10,
		},
		receiverLayerMask: overrides.receiverLayerMask ?? 1,
		priority: overrides.priority ?? 0,
		opacity: overrides.opacity ?? 1,
		edgeFade: overrides.edgeFade ?? 0,
		channelBlendModes: overrides.channelBlendModes ?? {},
		sceneOrder: overrides.sceneOrder ?? 0,
	};
}

function createFrameContext(decalPackets, overrides = {}) {
	return {
		viewCamera: createCamera(),
		attachments: { width: 64, height: 64 },
		features: { warnings: [] },
		postProcess: {},
		shadowMaps: new Map(),
		scene: {
			decalPackets,
			spatialIndex: overrides.spatialIndex ?? null,
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: Matrix4.identity(),
		incremental: {},
		transient: new Map(),
	};
}

function createTargets(backend, width = 64, height = 64) {
	function texture(label, format) {
		return backend.createTexture({
			width,
			height,
			format,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.StorageBinding |
				TextureUsage.CopySrc |
				TextureUsage.CopyDst,
			label,
		});
	}
	return {
		sceneColor: texture("sceneColor", TextureFormat.RGBA16Float),
		sceneColorMain: texture("sceneColorMain", TextureFormat.RGBA16Float),
		depth: texture("depth", TextureFormat.Depth24Plus),
		gAlbedoAlpha: texture("gAlbedoAlpha", TextureFormat.RGBA8Unorm),
		gNormalRoughMetal: texture("gNormalRoughMetal", TextureFormat.RGBA16Float),
		gEmissiveOcclusion: texture("gEmissiveOcclusion", TextureFormat.RGBA16Float),
		gMotionDepth: texture("gMotionDepth", TextureFormat.RGBA16Float),
		gSpecular: texture("gSpecular", TextureFormat.RGBA16Float),
		gCoatSheen: texture("gCoatSheen", TextureFormat.RGBA16Float),
		gSheenReflectance: texture("gSheenReflectance", TextureFormat.RGBA16Float),
		gMaterialExt0: texture("gMaterialExt0", TextureFormat.RGBA16Float),
		gMaterialExt1: texture("gMaterialExt1", TextureFormat.RGBA16Float),
		gMaterialExt2: texture("gMaterialExt2", TextureFormat.RGBA16Float),
		gMaterialExt3: texture("gMaterialExt3", TextureFormat.RGBA16Float),
	};
}

function createResourcesStub(backend) {
	const texture = backend.createTexture({
		width: 1,
		height: 1,
		format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.TextureBinding,
		label: "decal-slot-texture",
	});
	const sampler = backend.createSampler({ label: "decal-slot-sampler" });
	return {
		getGBufferReadLayout() {
			return { id: "gbuffer-read-layout" };
		},
		getGBufferWriteLayout() {
			return { id: "gbuffer-write-layout" };
		},
		getDecalBindGroupLayout() {
			return { id: "decal-layout" };
		},
		getDecalOutputBindGroupLayout() {
			return { id: "decal-output-layout" };
		},
		getDecalBatchBindGroupLayout() {
			return { id: "decal-batch-layout" };
		},
		async getDecalPipeline() {
			return { id: "decal-pipeline" };
		},
		async getDecalBatchPipeline() {
			return { id: "decal-batch-pipeline" };
		},
		getTextureForSlot() {
			return texture;
		},
		async getTextureForSlotAsync() {
			return texture;
		},
		getSamplerForTexture() {
			return sampler;
		},
	};
}

function createPassHarness({ dirtyRects, targets, resources, backend } = {}) {
	const resolvedBackend = backend ?? new FakeWebGPUBackend();
	const encoder = resolvedBackend.createCommandEncoder();
	const resolvedTargets = targets ?? createTargets(resolvedBackend);
	const pass = new WebGPUDeferredDecalPass(
		resolvedBackend,
		resources ?? createResourcesStub(resolvedBackend),
		{
			recordingContext: {
				getEncoder: () => encoder,
				getFrameTargets: () => resolvedTargets,
				getMSAATargets: () => null,
				getTargetWidth: () => resolvedTargets.gAlbedoAlpha.width,
				getTargetHeight: () => resolvedTargets.gAlbedoAlpha.height,
				getTargetMSAASampleCount: () => 1,
				getSceneTargetMode: () => "gbuffer",
				isMRTEnabled: () => true,
				isEarlyZPrepassEnabled: () => true,
				requireFrameResources: () => ({
					decalFrameBinding: { id: "decal-frame-binding" },
				}),
				isIncrementalPartial: () => true,
				resolveDirtyRects: () =>
					dirtyRects ?? [{ x: 4, y: 6, width: 16, height: 16 }],
				selectPacketsForRect: (_context, packets) => packets,
				selectTransparentSubsetForRect: (_context, packets) => packets,
			},
		}
	);
	return { backend: resolvedBackend, encoder, pass };
}

async function testSameMaterialDecalsUseBatchDispatchAndClippedCopy() {
	const material = new Material({ name: "batched" });
	const packets = [
		createDecalPacket("decal-a", material, { sceneOrder: 0 }),
		createDecalPacket("decal-b", material, { sceneOrder: 1 }),
	];
	const { backend, encoder, pass } = createPassHarness();
	const count = await pass.recordDecalPass(createFrameContext(packets));

	assert.equal(count, 1);
	assert.equal(backend.encoderCopyCalls.length, 11);
	assert.deepEqual(backend.encoderCopyCalls[0][0].origin, { x: 4, y: 6 });
	assert.deepEqual(backend.encoderCopyCalls[0][1].origin, { x: 4, y: 6 });
	assert.deepEqual(backend.encoderCopyCalls[0][2], {
		width: 16,
		height: 16,
		depthOrArrayLayers: 1,
	});
	assert.ok(encoder.calls.some((call) => call[0] === "beginComputePass"));
	assert.ok(encoder.calls.some((call) => call[0] === "dispatchWorkgroups"));
	assert.equal(
		encoder.calls.some((call) => call[0] === "beginRenderPass"),
		false
	);
}

async function testMixedMaterialsFallBackInExactOrder() {
	const packets = [
		createDecalPacket("decal-a", new Material({ name: "first" }), {
			sceneOrder: 0,
			opacity: 0.25,
			normalMatrix: [
				[2, 3, 4],
				[5, 6, 7],
				[8, 9, 10],
			],
		}),
		createDecalPacket("decal-b", new Material({ name: "second" }), {
			sceneOrder: 1,
			opacity: 0.75,
		}),
	];
	const { backend, encoder, pass } = createPassHarness({
		dirtyRects: [{ x: 8, y: 10, width: 12, height: 14 }],
	});
	const count = await pass.recordDecalPass(createFrameContext(packets));

	assert.equal(count, 2);
	assert.equal(backend.encoderCopyCalls.length, 22);
	assert.deepEqual(backend.encoderCopyCalls[0][0].origin, { x: 8, y: 10 });
	assert.deepEqual(backend.encoderCopyCalls[0][2], {
		width: 12,
		height: 14,
		depthOrArrayLayers: 1,
	});
	const renderPassLabels = encoder.calls
		.filter((call) => call[0] === "beginRenderPass")
		.map((call) => call[1].label);
	assert.deepEqual(renderPassLabels, [
		"WebGPUDeferredDecal_decal-a",
		"WebGPUDeferredDecal_decal-b",
	]);
	const outputBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUDecalOutputBinding"
	);
	assert.equal(outputBinding.desc.layout.id, "decal-output-layout");
	assert.deepEqual(
		outputBinding.entries.map((entry) => entry.binding),
		[11, 12, 13, 14]
	);
	assert.equal(
		encoder.calls.some((call) => call[0] === "beginComputePass"),
		false
	);
	const materialBindings = backend.bindingGroups.filter((group) =>
		group.label.startsWith("WebGPUDecalMaterialBinding_")
	);
	const uniformBuffers = materialBindings.map(
		(group) => group.entries.find((entry) => entry.binding === 0).resource
	);
	assert.equal(materialBindings.length, 2);
	assert.notEqual(uniformBuffers[0], uniformBuffers[1]);
	assert.deepEqual(
		uniformBuffers.map((buffer) => buffer.label),
		["WebGPUDecalUniform_0", "WebGPUDecalUniform_1"]
	);
	const firstUniform = uniformBuffers[0]._gpuResource.lastWrite.data;
	const secondUniform = uniformBuffers[1]._gpuResource.lastWrite.data;
	assert.deepEqual(Array.from(firstUniform.slice(32, 48)), [
		2, 5, 8, 0,
		3, 6, 9, 0,
		4, 7, 10, 0,
		0, 0, 0, 1,
	]);
	assert.equal(firstUniform[48], 0.25);
	assert.equal(secondUniform[48], 0.75);
}

async function testSeparateBatchDispatchesUseDistinctBuffers() {
	const material = new Material({ name: "multi-dispatch" });
	const packets = [
		createDecalPacket("decal-a", material, { sceneOrder: 0 }),
		createDecalPacket("decal-b", material, { sceneOrder: 1 }),
	];
	const { backend, pass } = createPassHarness({
		dirtyRects: [
			{ x: 4, y: 6, width: 8, height: 8 },
			{ x: 40, y: 42, width: 8, height: 8 },
		],
	});
	const count = await pass.recordDecalPass(createFrameContext(packets));
	const batchBindings = backend.bindingGroups.filter(
		(group) => group.label === "WebGPUDecalBatchBinding"
	);

	assert.equal(count, 2);
	assert.equal(batchBindings.length, 2);
	for (let binding = 0; binding < 4; binding++) {
		assert.notEqual(
			batchBindings[0].entries[binding].resource,
			batchBindings[1].entries[binding].resource
		);
	}
	assert.deepEqual(
		batchBindings.map((group) => group.entries[0].resource.label),
		["WebGPUDecalBatchParams_0", "WebGPUDecalBatchParams_1"]
	);
}

async function testStorageLimitFallsBackWithoutReordering() {
	const backend = new FakeWebGPUBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 4;
	const material = new Material({ name: "storage-limited" });
	const packets = [
		createDecalPacket("decal-a", material, { sceneOrder: 0 }),
		createDecalPacket("decal-b", material, { sceneOrder: 1 }),
	];
	const { encoder, pass } = createPassHarness({ backend });
	const count = await pass.recordDecalPass(createFrameContext(packets));

	assert.equal(count, 2);
	assert.equal(
		encoder.calls.some((call) => call[0] === "beginComputePass"),
		false
	);
	assert.deepEqual(
		encoder.calls
			.filter((call) => call[0] === "beginRenderPass")
			.map((call) => call[1].label),
		["WebGPUDeferredDecal_decal-a", "WebGPUDeferredDecal_decal-b"]
	);
}

async function testReceiverSpatialCullSkipsUnmatchedDecal() {
	const packet = createDecalPacket("culled-decal", new Material(), {
		receiverLayerMask: 1,
	});
	const spatialIndex = {
		queryOpaquePackets() {
			return [
				{
					meshInstance: { renderLayers: 2 },
					worldBounds: {
						center: { x: 0, y: 0, z: 0 },
						radius: 10,
					},
				},
			];
		},
	};
	const { backend, encoder, pass } = createPassHarness();
	const count = await pass.recordDecalPass(
		createFrameContext([packet], { spatialIndex })
	);

	assert.equal(count, 0);
	assert.equal(backend.encoderCopyCalls.length, 0);
	assert.equal(encoder.calls.length, 0);
}

async function testShaderMaterialDecalSourceIsSkipped() {
	const material = new ShaderMaterial({
		vertexShader: "",
		fragmentShader: "",
	});
	const packet = createDecalPacket("shader-decal", material);
	const { backend, encoder, pass } = createPassHarness();
	const count = await pass.recordDecalPass(createFrameContext([packet]));

	assert.equal(count, 0);
	assert.equal(backend.encoderCopyCalls.length, 0);
	assert.equal(encoder.calls.length, 0);
}

function testShaderRotatesAndOrthogonalizesAnisotropyTangent() {
	const source = readFileSync(
		new URL(
			"../../../src/shaders/webgpu/scene/decal.wgsl",
			import.meta.url
		),
		"utf8"
	);
	const rotationCalls = source.match(
		/rotateAnisotropyDirection\(\s*anisotropyDirection,\s*(?:decal|d)\.anisotropyParams\.yz\s*\)/g
	);
	const orthogonalizationCalls = source.match(
		new RegExp(
			"let resolvedAnisotropyTangent = orthogonalizeTangent\\(" +
				"\\s*blendedAnisotropyTangent,\\s*blendedNormal\\s*\\);",
			"g"
		)
	);
	const resolvedEncodes = source.match(
		/encodeNormalForGBuffer\(resolvedAnisotropyTangent\)/g
	);

	assert.match(
		source,
		/direction\.x \* rotation\.x - direction\.y \* rotation\.y/
	);
	assert.match(
		source,
		/direction\.x \* rotation\.y \+ direction\.y \* rotation\.x/
	);
	assert.equal(rotationCalls?.length, 2);
	assert.equal(orthogonalizationCalls?.length, 2);
	assert.equal(resolvedEncodes?.length, 2);
}

await testSameMaterialDecalsUseBatchDispatchAndClippedCopy();
await testMixedMaterialsFallBackInExactOrder();
await testSeparateBatchDispatchesUseDistinctBuffers();
await testStorageLimitFallsBackWithoutReordering();
await testReceiverSpatialCullSkipsUnmatchedDecal();
await testShaderMaterialDecalSourceIsSkipped();
testShaderRotatesAndOrthogonalizesAnisotropyTangent();

console.log("WebGPU deferred decal pass tests passed");
