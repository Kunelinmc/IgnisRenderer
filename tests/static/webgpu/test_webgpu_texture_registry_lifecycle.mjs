import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
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

function testUploadsPackedRowsOnlyWhenContentChanges() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const backing = new Uint8Array(143).fill(199);
	const data = backing.subarray(7);
	data.fill(11, 0, 68);
	data.fill(29, 68);
	const texture = new Texture({
		width: 17, height: 2, format: TextureFormat.RGBA8Unorm, data,
	});
	try {
		const resource = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		let writes = backend.writeCalls.filter((call) => call[0] === "writeTexture");
		assert.equal(writes.length, 1);
		assert.equal(writes[0][2].buffer, data.buffer);
		assert.equal(writes[0][2].byteOffset, 7);
		assert.equal(writes[0][2].byteLength, 136);
		assert.equal(writes[0][3].bytesPerRow, 68);
		assert.equal(writes[0][3].rowsPerImage, 2);
		assert.deepEqual(backend.resourceManager.getTexture(resource).texture._bytes, data);
		assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR), resource);
		assert.equal(backend.writeCalls.filter((call) => call[0] === "writeTexture").length, 1);
		data.fill(43, 68);
		texture.markNeedsUpdate();
		assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR), resource);
		writes = backend.writeCalls.filter((call) => call[0] === "writeTexture");
		assert.equal(writes.length, 2);
		assert.deepEqual(backend.resourceManager.getTexture(resource).texture._bytes, data);
	} finally {
		registry.destroy();
	}
}

function testUploadsCompressedMipPhysicalExtents() {
	const backend = new FakeWebGPUBackend();
	const writes = [];
	const writeTexture = backend.queue.writeTexture.bind(backend.queue);
	backend.queue.writeTexture = (destination, data, layout, size) => {
		writes.push({ mipLevel: destination.mipLevel, data, layout, size });
		writeTexture(destination, data, layout, size);
	};
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		format: TextureFormat.BC1RGBAUnorm,
		levels: [
			{ width: 8, height: 8, data: new Uint8Array(32).fill(11) },
			{ width: 4, height: 4, data: new Uint8Array(8).fill(23) },
			{ width: 2, height: 2, data: new Uint8Array(8).fill(37) },
		],
	});
	try {
		registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
		assert.deepEqual(writes.map(({ mipLevel, data, layout, size }) => [
			mipLevel, layout.bytesPerRow, layout.rowsPerImage,
			data.byteLength, size.width, size.height,
		]), [[0, 16, 2, 32, 8, 8], [1, 8, 1, 8, 4, 4], [2, 8, 1, 8, 4, 4]]);
	} finally {
		registry.destroy();
	}
}

function testColorAndDataSamplingKeepIndependentResources() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1,
		data: new Uint8Array([128, 64, 32, 128]),
		format: TextureFormat.RGBA8UnormSrgb,
		colorSpace: "sRGB",
	});
	const color = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	const normal = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.NORMAL);
	assert.equal(color.format, TextureFormat.RGBA8UnormSrgb);
	assert.equal(normal.format, TextureFormat.RGBA8Unorm);
	assert.notEqual(color, normal);
	assert.equal(color.destroyed, false);
	assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR), color);
	texture.dispose();
	assert.equal(color.destroyed, true);
	assert.equal(normal.destroyed, true);
	registry.destroy();
}

function testColorSpaceMutationReplacesSamplingFormatWithoutPixelRevision() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 255]),
		format: TextureFormat.RGBA8UnormSrgb, colorSpace: "sRGB",
	});
	const color = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	const sampler = registry.getSamplerForTexture(texture);
	texture.colorSpace = "Linear";
	assert.equal(registry.isTextureBindingCurrent(texture, color, sampler), false);
	const linear = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	assert.equal(linear.format, TextureFormat.RGBA8Unorm);
	assert.notEqual(linear, color);
	assert.equal(color.destroyed, true);
	registry.destroy();
}

function testExternalTextureRequiresKnownCompatibleSamplingFormat() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, colorSpace: "sRGB",
	});
	assert.throws(() => registry.registerExternalTexture(texture, {
		width: 1, height: 1, destroy() {},
	}), /sampling format/);
	assert.throws(() => registry.registerExternalTexture(texture, {
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm, destroy() {},
	}), /sampling format/);
	registry.destroy();
}

function testSrgbFallbackRejectsLossOfNativeDecoding() {
	const backend = new FakeWebGPUBackend();
	const createTextureResource = backend.createTexture.bind(backend);
	backend.createTexture = (descriptor) => createTextureResource({
		...descriptor, format: TextureFormat.RGBA8Unorm,
	});
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 255]),
		format: TextureFormat.RGBA8UnormSrgb, colorSpace: "sRGB",
	});
	assert.throws(
		() => registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR),
		/sRGB|sampling format/,
	);
	registry.destroy();
}

function testFailedSamplingFormatReplacementPreservesPreviousResource() {
	const backend = new FakeWebGPUBackend();
	const createTextureResource = backend.createTexture.bind(backend);
	backend.createTexture = (descriptor) => createTextureResource({
		...descriptor, format: TextureFormat.RGBA8Unorm,
	});
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 255]),
		colorSpace: "Linear",
	});
	const original = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	texture.colorSpace = "sRGB";
	assert.throws(() => registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR),
		/sRGB.*format/);
	texture.colorSpace = "Linear";
	assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR), original);
	assert.equal(original.destroyed, false);
	registry.destroy();
}

function testCustomDecodeOverrideUsesNativeFormat() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 128]),
		colorSpace: "Linear",
	});
	const raw = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	const decoded = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR, true, true);
	assert.equal(raw.format, TextureFormat.RGBA8Unorm);
	assert.equal(decoded.format, TextureFormat.RGBA8UnormSrgb);
	assert.notEqual(raw, decoded);
	assert.equal(raw.destroyed, false);
	texture.dispose();
	assert.equal(raw.destroyed, true);
	assert.equal(decoded.destroyed, true);
	registry.destroy();
}

function testDisposeReleasesOwnedResources() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
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
	const registry = new WebGPUTextureRegistry(backend);
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
	const registry = new WebGPUTextureRegistry(backend);
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

function testExternalRegistrationReplacesPreviouslyResolvedDataResources() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const texture = new Texture({
		width: 1, height: 1, data: new Uint8Array([128, 64, 32, 255]),
		colorSpace: "Linear",
	});
	const ownedData = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.NORMAL);
	const ownedOverride = registry.getTextureForSlot(
		texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR, true, true,
	);
	const external = {
		width: 1, height: 1, format: TextureFormat.RGBA8Unorm,
		destroyed: false, destroy() { this.destroyed = true; },
	};
	registry.registerExternalTexture(texture, external);
	assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.NORMAL), external);
	assert.equal(registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR), external);
	assert.throws(() => registry.getTextureForSlot(
		texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR, true, true,
	), /sampling format/);
	assert.equal(ownedData.destroyed, true);
	assert.equal(ownedOverride.destroyed, true);
	texture.dispose();
	registry.destroy();
	assert.equal(external.destroyed, false);
}

async function testDisposeWaitsForPendingMipmapGeneration() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
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

async function testDisposeWaitsForAllSamplingRoleMipmaps() {
	const backend = new FakeWebGPUBackend();
	const registry = new WebGPUTextureRegistry(backend);
	const pending = [];
	registry._mipmapGenerator = {
		generate: () => new Promise((resolve) => pending.push(resolve)),
		destroy() {},
	};
	const texture = new Texture({
		width: 2, height: 2, data: new Uint8Array(16).fill(128),
	});
	texture.minFilter = "LinearMipmapLinear";
	const color = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.BASE_COLOR);
	const normal = registry.getTextureForSlot(texture, WEBGPU_TEXTURE_SLOT.NORMAL);
	texture.dispose();
	pending[1]();
	await new Promise((resolve) => setTimeout(resolve, 0));
	assert.equal(color.destroyed, false);
	assert.equal(normal.destroyed, false);
	pending[0]();
	await new Promise((resolve) => setTimeout(resolve, 0));
	assert.equal(color.destroyed, true);
	assert.equal(normal.destroyed, true);
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
		const registry = new WebGPUTextureRegistry(backend);
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
	testUploadsPackedRowsOnlyWhenContentChanges();
	testUploadsCompressedMipPhysicalExtents();
	testFailedSamplingFormatReplacementPreservesPreviousResource();
	testExternalTextureRequiresKnownCompatibleSamplingFormat();
	testCustomDecodeOverrideUsesNativeFormat();
	testColorAndDataSamplingKeepIndependentResources();
	testColorSpaceMutationReplacesSamplingFormatWithoutPixelRevision();
	testSrgbFallbackRejectsLossOfNativeDecoding();
	testDisposeReleasesOwnedResources();
	testReleaseDoesNotDestroyExternalTexture();
	testReplacingOwnedTextureWithExternalResourceReleasesOnlyOwnedResource();
	testExternalRegistrationReplacesPreviouslyResolvedDataResources();
	await testDisposeWaitsForPendingMipmapGeneration();
	await testDisposeWaitsForAllSamplingRoleMipmaps();
	testFinalizerReleasesAbandonedResources();
	console.log("WebGPU texture registry lifecycle tests passed");
}

await run();
