import assert from "node:assert/strict";
import { WebGPUShadowAtlasAllocator } from "../src/renderers/webgpu/WebGPUShadowAtlasAllocator.ts";
import {
	WEBGPU_SHADOW_ATLAS_COLUMNS,
	WEBGPU_SHADOW_ATLAS_ROWS,
} from "../src/renderers/webgpu/constants.ts";

function createBackend(maxTextureDimension2D) {
	const createTextureCalls = [];
	return {
		device: {
			limits: {
				maxTextureDimension2D,
			},
		},
		createTexture(desc) {
			createTextureCalls.push(desc);
			return {
				destroy() {},
			};
		},
		get createTextureCalls() {
			return createTextureCalls;
		},
	};
}

function testAtlasTileSizeIsClampedToDeviceLimit() {
	const backend = createBackend(8192);
	const allocator = new WebGPUShadowAtlasAllocator(backend);

	allocator.ensureAtlasForTileSize(4096);
	assert.equal(allocator.tileSize, 2048);
	assert.equal(backend.createTextureCalls.length, 2);
	assert.equal(
		backend.createTextureCalls[0].width,
		2048 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
	assert.equal(
		backend.createTextureCalls[0].height,
		2048 * WEBGPU_SHADOW_ATLAS_ROWS
	);
	assert.equal(backend.createTextureCalls[0].format, "depth32float");
	assert.equal(backend.createTextureCalls[1].format, "rgba16float");

	allocator.ensureAtlasForTileSize(4096);
	assert.equal(backend.createTextureCalls.length, 2);
}

function testAtlasUsesHigherRequestedDeviceLimit() {
	const backend = createBackend(16384);
	const allocator = new WebGPUShadowAtlasAllocator(backend);

	allocator.ensureAtlasForTileSize(4096);
	assert.equal(allocator.tileSize, 4096);
	assert.equal(backend.createTextureCalls.length, 2);
	assert.equal(
		backend.createTextureCalls[0].width,
		4096 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
	assert.equal(
		backend.createTextureCalls[0].height,
		4096 * WEBGPU_SHADOW_ATLAS_ROWS
	);
	assert.equal(
		backend.createTextureCalls[1].width,
		4096 * WEBGPU_SHADOW_ATLAS_COLUMNS
	);
	assert.equal(
		backend.createTextureCalls[1].height,
		4096 * WEBGPU_SHADOW_ATLAS_ROWS
	);
}

function testPreparePropagatesResolvedAtlasTileSize() {
	const backend = createBackend(8192);
	const allocator = new WebGPUShadowAtlasAllocator(backend);
	const lightingState = {
		directionalShadows: [
			{
				enabled: true,
				shadowMapSize: 4096,
				shadowMapBaseSize: 4096,
				atlasTileSize: 0,
			},
		],
		spotShadows: [
			{
				enabled: true,
				shadowMapSize: 2048,
				shadowMapBaseSize: 2048,
				atlasTileSize: 0,
			},
		],
	};

	allocator.prepare(lightingState, 0);

	assert.equal(lightingState.directionalShadows[0].atlasTileSize, 2048);
	assert.equal(lightingState.spotShadows[0].atlasTileSize, 2048);
	assert.equal(allocator.tileSize, 2048);
}

testAtlasTileSizeIsClampedToDeviceLimit();
testAtlasUsesHigherRequestedDeviceLimit();
testPreparePropagatesResolvedAtlasTileSize();

console.log("test_webgpu_shadow_atlas_allocator: ok");
