import assert from "node:assert/strict";
import {
	DEFAULT_CLUSTERED_LIGHTING_OPTIONS,
} from "../src/pipeline/types.ts";
import { resolveFeatureState } from "../src/pipeline/FeatureResolver.ts";
import {
	packClusteredIndexRef,
	unpackClusteredIndexRef,
	packClusterHeaderFlags,
} from "../src/renderers/webgpu/WebGPUClusteredLightingRuntime.ts";
import {
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC,
	WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW,
	WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
	WEBGPU_CLUSTERED_INDEX_SHADOW_BIT,
	WEBGPU_CLUSTERED_INDEX_TYPE_MASK,
	WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT,
} from "../src/renderers/webgpu/constants.ts";

function createCapabilities(clusteredLighting) {
	return {
		sh: false,
		shadows: false,
		reflection: false,
		skybox: false,
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

function run() {
	testClusteredDefaultsAndMerge();
	testClusteredCapabilityGateWarning();
	testClusteredIndexBitfieldPackUnpack();
	testClusterHeaderFlagPack();
	console.log("Clustered lighting plan tests passed");
}

run();
