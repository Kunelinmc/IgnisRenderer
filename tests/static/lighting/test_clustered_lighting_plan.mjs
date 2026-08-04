import assert from "node:assert/strict";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
	DEFAULT_OCCLUSION_CULLING_OPTIONS,
} from "../../../src/pipeline/types.ts";
import { resolveFeatureState } from "../../../src/pipeline/FeatureResolver.ts";
import { CameraType } from "../../../src/cameras/Camera.ts";
import {
	WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT,
} from "../../../src/backends/webgpu/bufferLayouts.ts";
import {
	packClusteredIndexRef,
	unpackClusteredIndexRef,
	packClusterHeaderFlags,
	WebGPUClusteredLightingRuntime,
} from "../../../src/backends/webgpu/WebGPUClusteredLightingRuntime.ts";
import {
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC,
	WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW,
	WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
	WEBGPU_CLUSTERED_INDEX_SHADOW_BIT,
	WEBGPU_CLUSTERED_INDEX_TYPE_MASK,
	WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT,
	WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS,
	WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS,
	WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
	WEBGPU_CLUSTERED_MAX_LIGHTS,
	WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER,
	WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS,
	WEBGPU_CLUSTERED_PARAMS_FLOATS,
	WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS,
} from "../../../src/backends/webgpu/constants.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";

function createCapabilities(clusteredLighting) {
	return {
		sh: false,
		shadows: false,
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
		clusteredLighting,
		oit: false,
		occlusionCulling: true,
	};
}

class ClusteredComputeRecorder {
	constructor() {
		this.buffers = [];
		this.writes = [];
		this.shaderModules = [];
		this.bindGroupLayouts = [];
		this.pipelineLayouts = [];
		this.computePipelines = [];
		this.bindGroups = [];
	}

	createBuffer(desc) {
		const buffer = {
			id: this.buffers.length,
			label: desc.label,
			size: desc.size,
			usage: desc.usage,
			destroyed: false,
			destroy() {
				this.destroyed = true;
			},
		};
		this.buffers.push(buffer);
		return buffer;
	}

	writeBuffer(buffer, data, offset = 0) {
		this.writes.push({ buffer, data, offset });
	}

	async createShaderModule(desc) {
		const module = { label: desc.label, desc };
		this.shaderModules.push(module);
		return module;
	}

	createBindGroupLayout(desc) {
		const layout = { label: desc.label, desc };
		this.bindGroupLayouts.push(layout);
		return layout;
	}

	createPipelineLayout(desc) {
		const layout = { label: desc.label, desc };
		this.pipelineLayouts.push(layout);
		return layout;
	}

	async createComputePipeline(desc) {
		const pipeline = { label: desc.label, desc };
		this.computePipelines.push(pipeline);
		return pipeline;
	}

	createBindingGroup(desc) {
		const group = { label: desc.label, desc };
		this.bindGroups.push(group);
		return group;
	}
}

class ClusteredCommandEncoder {
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
		this.calls.push(["setBindingGroup", index, group.label ?? null]);
	}

	dispatchWorkgroups(x, y = 1, z = 1) {
		this.calls.push(["dispatchWorkgroups", x, y, z]);
	}

	endComputePass() {
		this.calls.push(["endComputePass"]);
	}
}

function createClusteredLight(index) {
	return {
		type: index % 2,
		position: [index, index + 1, -index - 2],
		range: 10,
		direction: [0, -1, 0],
		outerCos: -2,
		innerCos: -2,
		right: [0, 0, 0],
		width: 0,
		up: [0, 0, 0],
		height: 0,
		normal: [0, 1, 0],
		areaScale: 0,
		color: [1, 1, 1],
		castsShadow: false,
		affectsVolumetric: true,
		shadowIndex: 0,
	};
}

function createClusteredAreaLight() {
	return {
		type: WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
		position: [1, 2, 3],
		range: 50,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		right: [1, 0, 0],
		width: 20,
		up: [0, 0, 1],
		height: 10,
		normal: [0, 1, 0],
		areaScale: 200,
		color: [4, 5, 6],
		castsShadow: false,
		affectsVolumetric: false,
		shadowIndex: 0,
	};
}

function countOccurrences(value, needle) {
	return value.split(needle).length - 1;
}

function testClusteredDefaultsAndMerge() {
	const resolved = resolveFeatureState({}, createCapabilities(true), "webgpu");
	assert.equal(resolved.enableClusteredLighting, false);
	assert.deepEqual(resolved.clusteredLightingOptions, {
		tileSizePx: DEFAULT_CLUSTERED_LIGHTING_OPTIONS.tileSizePx,
		zSlices: DEFAULT_CLUSTERED_LIGHTING_OPTIONS.zSlices,
		maxLights: DEFAULT_CLUSTERED_LIGHTING_OPTIONS.maxLights,
		maxLightsPerCluster:
			DEFAULT_CLUSTERED_LIGHTING_OPTIONS.maxLightsPerCluster,
		cullingMode: "gather",
	});

	const merged = resolveFeatureState(
		{
			clusteredLightingOptions: {
				tileSizePx: 32,
				maxLightsPerCluster: 96,
				cullingMode: "scatter",
			},
		},
		createCapabilities(true),
		"webgpu"
	);
	assert.equal(merged.enableClusteredLighting, false);
	assert.deepEqual(merged.clusteredLightingOptions, {
		tileSizePx: 32,
		zSlices: DEFAULT_CLUSTERED_LIGHTING_OPTIONS.zSlices,
		maxLights: DEFAULT_CLUSTERED_LIGHTING_OPTIONS.maxLights,
		maxLightsPerCluster: 96,
		cullingMode: "scatter",
	});
	const webgl = resolveFeatureState(
		{ clusteredLightingOptions: { cullingMode: "scatter" } },
		createCapabilities(true),
		"webgl"
	);
	assert.equal(webgl.clusteredLightingOptions.cullingMode, "scatter");
}

function testClusteredCapabilityGateWarning() {
	const resolved = resolveFeatureState(
		{
			enableClusteredLighting: true,
		},
		createCapabilities(false),
		"webgl"
	);
	assert.equal(resolved.enableClusteredLighting, false);
	assert.ok(
		resolved.warnings.some(
			(warning) => warning.key === "webgl-feature-clustered-lighting"
		)
	);
}

function testOcclusionCullingDefaultsAndCapabilityGate() {
	const resolved = resolveFeatureState(
		{
			enableOcclusionCulling: true,
			occlusionCullingOptions: {
				hysteresisFrames: 4,
			},
		},
		createCapabilities(true),
		"webgpu"
	);
	assert.equal(resolved.enableOcclusionCulling, true);
	assert.deepEqual(resolved.occlusionCullingOptions, {
		...DEFAULT_OCCLUSION_CULLING_OPTIONS,
		hysteresisFrames: 4,
	});

	const unsupported = resolveFeatureState(
		{
			enableOcclusionCulling: true,
		},
		{
			...createCapabilities(true),
			occlusionCulling: false,
		},
		"webgl"
	);
	assert.equal(unsupported.enableOcclusionCulling, false);
	assert.ok(
		unsupported.warnings.some(
			(warning) => warning.key === "webgl-feature-occlusion-culling"
		)
	);
}

function testClusterParamsLayoutWritesLightCount() {
	const writer = WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT.createWriter();
	writer.expectByteLength(
		WEBGPU_CLUSTERED_PARAMS_FLOATS * 4,
		"ClusterGridParams"
	);
	writer.writeU32("screenWidth", 640);
	writer.writeU32("screenHeight", 360);
	writer.writeU32("tilesX", 10);
	writer.writeU32("tilesY", 6);
	writer.writeU32("zSlices", 24);
	writer.writeU32("clusterCount", 1440);
	writer.writeF32("near", 0.1);
	writer.writeF32("far", 100);
	writer.writeF32("logScale", 3);
	writer.writeF32("logBias", 1);
	writer.writeU32("lightCount", 17);
	writer.writeU32("maxLightsPerCluster", 32);

	const data = new Uint32Array(writer.toArrayBuffer());
	assert.equal(data.byteLength, WEBGPU_CLUSTERED_PARAMS_FLOATS * 4);
	assert.equal(data[10], 17);
	assert.equal(data[11], 32);
}

function testRuntimeWritesClampedActiveLightCount() {
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(
		compute,
		{},
		{}
	);
	runtime.prepareFrame(
		{
			camera: {
				type: CameraType.Perspective,
				near: 0.1,
				far: 100,
			},
		},
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				maxLights: 2,
				maxLightsPerCluster: 64,
				tileSizePx: 64,
				zSlices: 24,
			},
		},
		{
			lights: [
				createClusteredLight(0),
				createClusteredLight(1),
				createClusteredLight(2),
			],
		},
		128,
		128
	);

	const paramsWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredParams"
	);
	assert.ok(paramsWrite);
	const params = new Uint32Array(paramsWrite.data);
	assert.equal(params[10], 2);
	assert.equal(params[11], 64);
}

function testRuntimeWritesClusteredAreaSoAData() {
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(
		compute,
		{},
		{}
	);
	runtime.prepareFrame(
		{
			camera: {
				type: CameraType.Perspective,
				near: 0.1,
				far: 100,
			},
		},
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				maxLights: 4,
				maxLightsPerCluster: 8,
				tileSizePx: 64,
				zSlices: 4,
			},
		},
		{
			lights: [createClusteredAreaLight()],
		},
		128,
		128
	);

	const positionWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredPositionRange"
	);
	const colorWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredColorInner"
	);
	const areaWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredAreaPayload"
	);
	const metadataWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredMetadata"
	);
	const cullWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredCullData"
	);
	assert.ok(positionWrite);
	assert.ok(colorWrite);
	assert.ok(areaWrite);
	assert.ok(metadataWrite);
	assert.ok(cullWrite);
	assert.equal(positionWrite.data.byteLength, WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS * 4);
	assert.equal(areaWrite.data.byteLength, WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS * 4);
	assert.equal(metadataWrite.data.byteLength, WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS * 4);
	assert.equal(cullWrite.data.byteLength, WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS * 4);
	const asFloats = (write) => new Float32Array(
		write.data.buffer,
		write.data.byteOffset,
		write.data.byteLength / 4
	);
	const positions = asFloats(positionWrite);
	const colors = asFloats(colorWrite);
	const areas = asFloats(areaWrite);
	const cull = asFloats(cullWrite);
	assert.deepEqual(Array.from(positions), [1, 2, 3, 50]);
	assert.deepEqual(Array.from(colors), [4, 5, 6, -2]);
	assert.deepEqual(Array.from(areas.slice(0, 4)), [1, 0, 0, 20]);
	assert.deepEqual(Array.from(areas.slice(4, 8)), [0, 0, 1, 10]);
	assert.deepEqual(Array.from(areas.slice(8, 12)), [0, 1, 0, 200]);
	assert.ok(Math.abs(cull[3] - (50 + Math.hypot(10, 5))) < 1e-5);
	const metadata = new Uint32Array(
		metadataWrite.data.buffer,
		metadataWrite.data.byteOffset,
		metadataWrite.data.byteLength / 4
	);
	assert.equal(metadata[0], WEBGPU_CLUSTERED_LIGHT_TYPE_AREA);
}

async function testRuntimeDispatchesGatherComputePasses() {
	globalThis.GPUShaderStage ??= { COMPUTE: 4 };
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(
		compute,
		{ label: "scene-layout" },
		{ label: "frame-layout" }
	);
	runtime.prepareFrame(
		{
			camera: {
				type: CameraType.Perspective,
				near: 0.1,
				far: 100,
			},
		},
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				maxLights: 4,
				maxLightsPerCluster: 8,
				tileSizePx: 64,
				zSlices: 4,
			},
		},
		{
			lights: [createClusteredLight(0), createClusteredLight(1)],
		},
		128,
		128
	);

	const encoder = new ClusteredCommandEncoder();
	await runtime.build(encoder, { label: "frame-binding" });

	assert.deepEqual(
		compute.computePipelines.map((pipeline) => pipeline.desc.compute.entryPoint),
		["csClear", "csScatter", "csFinalize", "csGather", "csResolveOverflow"]
	);
	assert.deepEqual(
		encoder.calls
			.filter((call) => call[0] === "beginComputePass")
			.map((call) => call[1]),
		["WebGPUClusteredLightingGather"]
	);
	const sceneBinding = runtime.getSceneBinding();
	assert.equal(sceneBinding.desc.entries.length, 8);
	assert.equal(compute.bindGroupLayouts[0].desc.entries.length, 8);
	assert.deepEqual(
		compute.bindGroupLayouts[0].desc.entries.map((entry) => entry.binding),
		[0, 1, 2, 3, 4, 5, 6, 7]
	);
}

async function testRuntimeRetainsScatterABPath() {
	globalThis.GPUShaderStage ??= { COMPUTE: 4 };
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(compute, {}, {});
	runtime.prepareFrame(
		{ camera: { type: CameraType.Perspective, near: 0.1, far: 100 } },
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				cullingMode: "scatter",
				maxLights: 4,
				maxLightsPerCluster: 8,
				tileSizePx: 64,
				zSlices: 4,
			},
		},
		{ lights: [createClusteredLight(0)] },
		128,
		128
	);
	const encoder = new ClusteredCommandEncoder();
	await runtime.build(encoder, {});
	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "beginComputePass").map((call) => call[1]),
		[
			"WebGPUClusteredLightingClear",
			"WebGPUClusteredLightingScatter",
			"WebGPUClusteredLightingFinalize",
		]
	);
}

async function testRuntimeSkipsStaticCullAndSelectiveUploads() {
	globalThis.GPUShaderStage ??= { COMPUTE: 4 };
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(compute, {}, {});
	const light = createClusteredLight(0);
	const frame = { camera: { type: CameraType.Perspective, near: 0.1, far: 100 } };
	const features = {
		enableLighting: true,
		enableClusteredLighting: true,
		clusteredLightingOptions: {
			maxLights: 4,
			maxLightsPerCluster: 8,
			tileSizePx: 64,
			zSlices: 4,
		},
	};
	runtime.prepareFrame(frame, features, { lights: [light] }, 128, 128);
	await runtime.build(new ClusteredCommandEncoder(), {});
	const writesAfterFirstFrame = compute.writes.length;
	const staticEncoder = new ClusteredCommandEncoder();
	runtime.prepareFrame(frame, features, { lights: [light] }, 128, 128);
	await runtime.build(staticEncoder, {});
	assert.equal(compute.writes.length, writesAfterFirstFrame);
	assert.equal(staticEncoder.calls.length, 0);

	light.color = [2, 2, 2];
	const writesBeforeColor = compute.writes.length;
	const colorEncoder = new ClusteredCommandEncoder();
	runtime.prepareFrame(frame, features, { lights: [light] }, 128, 128);
	await runtime.build(colorEncoder, {});
	assert.deepEqual(
		compute.writes.slice(writesBeforeColor).map((write) => write.buffer.label),
		["WebGPUClusteredColorInner"]
	);
	assert.equal(colorEncoder.calls.length, 0);

	const writesBeforeModeSwitch = compute.writes.length;
	const modeEncoder = new ClusteredCommandEncoder();
	runtime.prepareFrame(
		frame,
		{
			...features,
			clusteredLightingOptions: {
				...features.clusteredLightingOptions,
				cullingMode: "scatter",
			},
		},
		{ lights: [light] },
		128,
		128
	);
	await runtime.build(modeEncoder, {});
	assert.equal(compute.writes.length, writesBeforeModeSwitch);
	assert.deepEqual(
		modeEncoder.calls
			.filter((call) => call[0] === "beginComputePass")
			.map((call) => call[1]),
		[
			"WebGPUClusteredLightingClear",
			"WebGPUClusteredLightingScatter",
			"WebGPUClusteredLightingFinalize",
		]
	);
}

function testRuntimeClampsWebGPULimits() {
	const compute = new ClusteredComputeRecorder();
	const warnings = [];
	const runtime = new WebGPUClusteredLightingRuntime(
		compute,
		{},
		{}
	);
	runtime.onWarn((key, message) => warnings.push({ key, message }));
	const lights = Array.from(
		{ length: WEBGPU_CLUSTERED_MAX_LIGHTS + 1 },
		(_, index) => createClusteredLight(index)
	);
	runtime.prepareFrame(
		{ camera: { type: CameraType.Perspective, near: 0.1, far: 100 } },
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				maxLights: WEBGPU_CLUSTERED_MAX_LIGHTS + 10,
				maxLightsPerCluster: WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER + 10,
				tileSizePx: 64,
				zSlices: 4,
			},
		},
		{ lights },
		64,
		64
	);
	const paramsWrite = compute.writes.find(
		(write) => write.buffer.label === "WebGPUClusteredParams"
	);
	const params = new Uint32Array(paramsWrite.data);
	assert.equal(params[10], WEBGPU_CLUSTERED_MAX_LIGHTS);
	assert.equal(params[11], WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER);
	runtime.prepareFrame(
		{ camera: { type: CameraType.Perspective, near: 0.1, far: 100 } },
		{
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				maxLights: WEBGPU_CLUSTERED_MAX_LIGHTS + 10,
				maxLightsPerCluster: WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER + 10,
			},
		},
		{ lights },
		64,
		64
	);
	assert.equal(
		warnings.filter((warning) => warning.key === "webgpu-clustered-max-lights-limit").length,
		1
	);
	assert.equal(
		warnings.filter(
			(warning) => warning.key === "webgpu-clustered-max-per-cluster-limit"
		).length,
		1
	);
}

function testClusteredIndexBitfieldPackUnpack() {
	const packed = packClusteredIndexRef(0x00abcdef, 1, true, true);
	const unpacked = unpackClusteredIndexRef(packed);
	assert.equal(unpacked.lightIndex, 0x00abcdef & WEBGPU_CLUSTERED_INDEX_LIGHT_MASK);
	assert.equal(unpacked.lightType, 1);
	assert.equal(unpacked.shadowed, true);
	assert.equal(unpacked.volumetric, true);

	const reservedMask =
		~(
			WEBGPU_CLUSTERED_INDEX_LIGHT_MASK |
			WEBGPU_CLUSTERED_INDEX_TYPE_MASK |
			WEBGPU_CLUSTERED_INDEX_SHADOW_BIT |
			WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT
		) >>> 0;
	assert.equal((packed & reservedMask) >>> 0, 0);
}

function testClusterHeaderFlagPack() {
	const flags = packClusterHeaderFlags(true, true, true);
	assert.equal((flags & WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW) !== 0, true);
	assert.equal(
		(flags & WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED) !== 0,
		true
	);
	assert.equal(
		(flags & WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC) !== 0,
		true
	);

	const onlyOverflow = packClusterHeaderFlags(true, false, false);
	assert.equal(onlyOverflow, WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW);
}

async function testClusteredCullShaderUsesActiveCountAndTiling() {
	const shader = (await ShaderSource.load(
		"webgpu.clusteredLightingCull.composite"
	)).code;
	assert.ok(shader.includes("lightCount: u32"));
	assert.ok(shader.includes("maxLightsPerCluster: u32"));
	assert.ok(shader.includes("const CLUSTER_LIGHT_TYPE_AREA: u32 = 2u;"));
	assert.ok(shader.includes("fn activeClusterLightCount() -> u32"));
	assert.ok(shader.includes("arrayLength(&clusterCullData.values)"));
	assert.ok(shader.includes("fn csClear("));
	assert.ok(shader.includes("fn csScatter("));
	assert.ok(shader.includes("fn csFinalize("));
	assert.ok(shader.includes("fn csGather("));
	assert.ok(shader.includes("fn csResolveOverflow("));
	assert.ok(shader.includes("var<workgroup> overflowScores: array<f32, 1024>;"));
	assert.ok(shader.includes("fn scoreIsBetter("));
	assert.ok(shader.includes("let lightIndexA = refA & CLUSTER_LIGHT_INDEX_MASK;"));
	assert.ok(shader.includes("return clusterSliceDepths.depths[min(slice, lastIndex)];"));
	assert.ok(shader.includes("atomicAdd(&clusterHeaders.headers[clusterIndex].count"));
	assert.ok(shader.includes("fn resolveLightClusterRange("));
	assert.ok(!shader.includes("for (var tileBase: u32 = 0u;"));
}

async function testClusteredShadingUsesActiveLightCountGuards() {
	const deferred = (await ShaderSource.load(
		"webgpu.deferredLighting.composite"
	)).code;
	assert.ok(deferred.includes("fn activeClusteredLightCount() -> u32"));
	assert.ok(
		deferred.includes(
			"return min(clusterGrid.lightCount, arrayLength(&clusterPositionRanges.values));"
		)
	);
	assert.equal(
		countOccurrences(
			deferred,
			"let clusterLightCount = activeClusteredLightCount();"
		),
		2
	);
	assert.ok(!deferred.includes("clusterLights.lights"));
	assert.ok(deferred.includes("surface.pixelPosition"));

	for (const part of ["fragmentPbrPoint", "fragmentPhong"]) {
		const source = (
			await ShaderSource.load(`webgpu.scene.part.${part}.composite`)
		).code;
		assert.ok(source.includes("let clusterLightCount = activeClusteredLightCount();"));
	}
	const pointPart = (
		await ShaderSource.load("webgpu.scene.part.fragmentPbrPoint.composite")
	).code;
	assert.equal(countOccurrences(pointPart, "getClusterHeaderForFragment("), 1);
	assert.equal(
		countOccurrences(pointPart, "clusterIndices.indices[clusterHeader.offset"),
		1
	);
	assert.ok(pointPart.includes("evaluateClusteredDirectLightSample"));
	assert.ok(pointPart.includes("clusterRef.lightType == CLUSTER_LIGHT_TYPE_AREA"));
	assert.ok(pointPart.includes("clusterRef.lightType == CLUSTER_LIGHT_TYPE_SPOT"));
	const spotPart = (
		await ShaderSource.load("webgpu.scene.part.fragmentPbrSpot.composite")
	).code;
	assert.ok(spotPart.includes("if (!isClusteredLightingEnabled())"));
	assert.ok(!spotPart.includes("getClusterHeaderForFragment"));
	const areaPart = (
		await ShaderSource.load("webgpu.scene.part.fragmentPbrArea.composite")
	).code;
	assert.ok(areaPart.includes("if (!isClusteredLightingEnabled())"));
	assert.ok(!areaPart.includes("getClusterHeaderForFragment"));
	const scene = (await ShaderSource.load("webgpu.scene.composite")).code;
	assert.equal(countOccurrences(scene, "getClusterHeaderForFragment("), 3);
	assert.equal(countOccurrences(scene, "clusterIndices.indices[clusterHeader.offset"), 2);
}

async function run() {
	testClusteredDefaultsAndMerge();
	testClusteredCapabilityGateWarning();
	testOcclusionCullingDefaultsAndCapabilityGate();
	testClusterParamsLayoutWritesLightCount();
	testRuntimeWritesClampedActiveLightCount();
	testRuntimeWritesClusteredAreaSoAData();
	await testRuntimeDispatchesGatherComputePasses();
	await testRuntimeRetainsScatterABPath();
	await testRuntimeSkipsStaticCullAndSelectiveUploads();
	testRuntimeClampsWebGPULimits();
	testClusteredIndexBitfieldPackUnpack();
	testClusterHeaderFlagPack();
	await testClusteredCullShaderUsesActiveCountAndTiling();
	await testClusteredShadingUsesActiveLightCountGuards();
	console.log("Clustered lighting plan tests passed");
}

await run();
