import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { WEBGPU_TEXTURE_SLOT } from "../../../src/backends/webgpu/constants.ts";
import { WebGPUTextureRegistry } from "../../../src/backends/webgpu/WebGPUTextureRegistry.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

function createTexture() {
	return new Texture({
		data: new Uint8ClampedArray([255, 255, 255, 255]),
		width: 1,
		height: 1,
	});
}

function testDisposeReleasesOwnedResources() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = createTexture();
	const resource = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR,
	);
	const sampler = registry.getSamplerForTexture(texture);

	texture.dispose();

	assert.equal(resource.destroyed, true);
	assert.equal(sampler.destroyed, true);
	assert.equal(registry._ownedTextures.size, 0);
	assert.equal(registry._ownedSamplers.size, 0);
	assert.equal(backend.textureDestroyCalls, 1);
	assert.equal(backend.samplerDestroyCalls, 1);

	texture.dispose();
	registry.releaseTexture(texture);
	registry.destroy();
	assert.equal(backend.textureDestroyCalls, 1);
	assert.equal(backend.samplerDestroyCalls, 1);
}

function testReleaseDoesNotDestroyExternalTexture() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = createTexture();
	const externalResource = {
		width: 1,
		height: 1,
		format: texture.format,
		destroyed: false,
		destroy() {
			this.destroyed = true;
		},
	};

	registry.registerExternalTexture(texture, externalResource);
	const sampler = registry.getSamplerForTexture(texture);
	registry.releaseTexture(texture);

	assert.equal(externalResource.destroyed, false);
	assert.equal(sampler.destroyed, true);
	assert.equal(registry._ownedTextures.size, 0);
	assert.equal(registry._ownedSamplers.size, 0);
	registry.destroy();
}

function testReplacingOwnedTextureWithExternalResourceReleasesOnlyOwnedResource() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = createTexture();
	const ownedResource = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR,
	);
	const externalResource = {
		width: 1,
		height: 1,
		format: texture.format,
		destroyed: false,
		destroy() {
			this.destroyed = true;
		},
	};

	registry.registerExternalTexture(texture, externalResource);
	texture.dispose();

	assert.equal(ownedResource.destroyed, true);
	assert.equal(externalResource.destroyed, false);
	assert.equal(backend.textureDestroyCalls, 1);
	registry.destroy();
}

async function testDisposeWaitsForPendingMipmapGeneration() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend, backend);
	const texture = new Texture({
		data: new Uint8ClampedArray(2 * 2 * 4).fill(255),
		width: 2,
		height: 2,
	});
	texture.minFilter = "LinearMipmapLinear";
	const resource = registry.getTextureForSlot(
		texture,
		WEBGPU_TEXTURE_SLOT.BASE_COLOR,
	);

	texture.dispose();
	assert.equal(resource.destroyed, false);

	for (let attempt = 0; attempt < 16 && !resource.destroyed; attempt++) {
		await Promise.resolve();
		await new Promise((resolve) => setTimeout(resolve, 0));
	}
	assert.equal(resource.destroyed, true);
	assert.equal(registry._ownedTextures.size, 0);
	registry.destroy();
}

function testFinalizerReleasesAbandonedResources() {
	const OriginalFinalizationRegistry = globalThis.FinalizationRegistry;
	const instances = [];

	class FakeFinalizationRegistry {
		constructor(callback) {
			this.callback = callback;
			this.registrations = [];
			instances.push(this);
		}

		register(target, heldValue, token) {
			this.registrations.push({ target, heldValue, token });
		}

		unregister(token) {
			const previousLength = this.registrations.length;
			this.registrations = this.registrations.filter(
				(registration) => registration.token !== token,
			);
			return this.registrations.length !== previousLength;
		}

		finalizeFirst() {
			const registration = this.registrations.shift();
			assert.ok(registration);
			this.callback(registration.heldValue);
		}
	}

	globalThis.FinalizationRegistry = FakeFinalizationRegistry;
	try {
		const backend = new FakeWebGPUBackend();
		const registry = new WebGPUTextureRegistry(backend, backend);
		const texture = createTexture();
		const resource = registry.getTextureForSlot(
			texture,
			WEBGPU_TEXTURE_SLOT.BASE_COLOR,
		);
		const sampler = registry.getSamplerForTexture(texture);

		assert.equal(instances.length, 1);
		assert.equal(instances[0].registrations.length, 1);
		instances[0].finalizeFirst();

		assert.equal(resource.destroyed, true);
		assert.equal(sampler.destroyed, true);
		assert.equal(registry._ownedTextures.size, 0);
		assert.equal(registry._ownedSamplers.size, 0);

		registry.destroy();
		assert.equal(backend.textureDestroyCalls, 1);
		assert.equal(backend.samplerDestroyCalls, 1);
	} finally {
		globalThis.FinalizationRegistry = OriginalFinalizationRegistry;
	}
}

async function run() {
	testDisposeReleasesOwnedResources();
	testReleaseDoesNotDestroyExternalTexture();
	testReplacingOwnedTextureWithExternalResourceReleasesOnlyOwnedResource();
	await testDisposeWaitsForPendingMipmapGeneration();
	testFinalizerReleasesAbandonedResources();
	console.log("WebGPU texture registry lifecycle tests passed");
}

await run();
