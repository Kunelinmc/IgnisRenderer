import assert from "node:assert/strict";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
} from "../src/pipeline/types.ts";
import { resolveFeatureState } from "../src/pipeline/FeatureResolver.ts";
import { CameraType } from "../src/cameras/Camera.ts";
import {
	WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT,
} from "../src/renderers/webgpu/bufferLayouts.ts";
import {
	packClusteredIndexRef,
	unpackClusteredIndexRef,
	packClusterHeaderFlags,
	WebGPUClusteredLightingRuntime,
} from "../src/renderers/webgpu/WebGPUClusteredLightingRuntime.ts";
import {
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC,
	WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW,
	WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
	WEBGPU_CLUSTERED_INDEX_SHADOW_BIT,
	WEBGPU_CLUSTERED_INDEX_TYPE_MASK,
	WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT,
	WEBGPU_CLUSTERED_PARAMS_FLOATS,
} from "../src/renderers/webgpu/constants.ts";
import {
	loadClusteredLightingCullShaderComposite,
	loadDeferredLightingShaderComposite,
	loadSceneShaderPartComposite,
} from "../src/shaders/webgpu/shaderSource.ts";

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

	createComputePipeline(desc) {
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
		color: [1, 1, 1],
		castsShadow: false,
		affectsVolumetric: true,
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
	});

	const merged = resolveFeatureState(
		{
			clusteredLightingOptions: {
				tileSizePx: 32,
				maxLightsPerCluster: 96,
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
	});
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
		{},
		() => {}
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
			clusteredLights: [
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

async function testRuntimeDispatchesLightDrivenComputePasses() {
	globalThis.GPUShaderStage ??= { COMPUTE: 4 };
	const compute = new ClusteredComputeRecorder();
	const runtime = new WebGPUClusteredLightingRuntime(
		compute,
		{ label: "scene-layout" },
		{ label: "frame-layout" },
		() => {}
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
			clusteredLights: [createClusteredLight(0), createClusteredLight(1)],
		},
		128,
		128
	);

	const encoder = new ClusteredCommandEncoder();
	await runtime.build(encoder, { label: "frame-binding" });

	assert.deepEqual(
		compute.computePipelines.map((pipeline) => pipeline.desc.compute.entryPoint),
		["csClear", "csScatter", "csFinalize"]
	);
	assert.deepEqual(
		encoder.calls
			.filter((call) => call[0] === "beginComputePass")
			.map((call) => call[1]),
		[
			"WebGPUClusteredLightingClear",
			"WebGPUClusteredLightingScatter",
			"WebGPUClusteredLightingFinalize",
		]
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
	const shader = (await loadClusteredLightingCullShaderComposite()).code;
	assert.ok(shader.includes("lightCount: u32"));
	assert.ok(shader.includes("maxLightsPerCluster: u32"));
	assert.ok(shader.includes("fn activeClusterLightCount() -> u32"));
	assert.ok(
		shader.includes(
			"return min(clusterParams.lightCount, arrayLength(&clusterLights.lights));"
		)
	);
	assert.ok(!shader.includes("let maxLights = arrayLength(&clusterLights.lights);"));
	assert.ok(shader.includes("fn csClear("));
	assert.ok(shader.includes("fn csScatter("));
	assert.ok(shader.includes("fn csFinalize("));
	assert.ok(shader.includes("atomicAdd(&clusterHeaders.headers[clusterIndex].count"));
	assert.ok(shader.includes("fn resolveLightClusterRange("));
	assert.ok(!shader.includes("for (var tileBase: u32 = 0u;"));
}

async function testClusteredShadingUsesActiveLightCountGuards() {
	const deferred = (await loadDeferredLightingShaderComposite()).code;
	assert.ok(deferred.includes("fn activeClusteredLightCount() -> u32"));
	assert.ok(
		deferred.includes(
			"return min(clusterGrid.lightCount, arrayLength(&clusterLights.lights));"
		)
	);
	assert.equal(
		countOccurrences(
			deferred,
			"let clusterLightCount = activeClusteredLightCount();"
		),
		2
	);
	assert.ok(
		!deferred.includes(
			"let clusterLightCount = u32(arrayLength(&clusterLights.lights));"
		)
	);

	for (const part of ["fragmentPbrPoint", "fragmentPbrSpot", "fragmentPhong"]) {
		const source = (await loadSceneShaderPartComposite(part)).code;
		assert.ok(source.includes("let clusterLightCount = activeClusteredLightCount();"));
	}
}

async function run() {
	testClusteredDefaultsAndMerge();
	testClusteredCapabilityGateWarning();
	testClusterParamsLayoutWritesLightCount();
	testRuntimeWritesClampedActiveLightCount();
	await testRuntimeDispatchesLightDrivenComputePasses();
	testClusteredIndexBitfieldPackUnpack();
	testClusterHeaderFlagPack();
	await testClusteredCullShaderUsesActiveCountAndTiling();
	await testClusteredShadingUsesActiveLightCountGuards();
	console.log("Clustered lighting plan tests passed");
}

await run();
