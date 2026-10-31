import assert from "node:assert/strict";

import { createCustomRenderTargetResources } from "../../../src/rendering/CustomRenderTargets.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { TextureUsage } from "../../../src/backends/types.ts";

const descriptor = {
	id: "shared-shadow",
	width: 64,
	height: 32,
	color: [{ format: TextureFormat.RGBA16Float }],
	depth: { format: TextureFormat.Depth32Float },
};

function createBackend({ failAt = 0, failInitialization = false, fallback = false } = {}) {
	const textures = [];
	const events = [];
	return {
		textures, events,
		createTexture(desc) {
			if (textures.length + 1 === failAt) throw new Error("allocation failed");
			const texture = {
				...desc,
				format: fallback ? TextureFormat.RGBA8Unorm : desc.format,
				destroyCount: 0,
				destroy() { this.destroyCount++; events.push("texture"); },
			};
			textures.push(texture);
			return texture;
		},
		initializeTarget(target) {
			assert.equal(target.color[0].texture, textures[0]);
			assert.equal(target.depth.texture, textures.at(-1));
			if (failInitialization) throw new Error("initialization failed");
			events.push("initialize");
			return () => events.push("native-target");
		},
	};
}

function testAttachmentsAndLifetimeAcrossBackendCapabilities() {
	for (const nativeTarget of [false, true]) {
		const backend = createBackend();
		if (!nativeTarget) delete backend.initializeTarget;
		const target = allocate(backend, descriptor);
		assert.equal(target.width, 64);
		assert.equal(target.height, 32);
		assert.equal(target.sampleCount, 1);
		assert.equal(target.color[0].texture.format, TextureFormat.RGBA16Float);
		assert.equal(target.depth.format, TextureFormat.Depth32Float);
		assert.equal(target.color[0].resolveTexture, null);
		assert.equal(target.color[0].texture.usage,
			TextureUsage.RenderAttachment | TextureUsage.TextureBinding |
				TextureUsage.CopySrc | TextureUsage.CopyDst);
		assert.equal(target.depth.texture.usage,
			TextureUsage.RenderAttachment | TextureUsage.TextureBinding | TextureUsage.CopySrc);
		target.destroy();
		target.destroy();
		assert.equal(backend.textures.every((texture) => texture.destroyCount === 1), true);
		assert.deepEqual(backend.events, nativeTarget ?
			["initialize", "native-target", "texture", "texture"] : ["texture", "texture"]);
	}
}

function testAllocationAndInitializationFailuresReleaseAttachments() {
	for (const options of [{ failAt: 2 }, { failInitialization: true }, { fallback: true }]) {
		const backend = createBackend(options);
		assert.throws(() => allocate(backend, descriptor), /failed|requested/);
		assert.equal(backend.textures.length > 0, true);
		assert.equal(backend.textures.every((texture) => texture.destroyCount === 1), true);
	}
}

function testResolvedMultisampleAndReadbackUsage() {
	const backend = createBackend();
	delete backend.initializeTarget;
	const target = allocate(backend, { ...descriptor, sampleCount: 4 });
	assert.equal(target.color[0].texture.sampleCount, 4);
	assert.equal(target.color[0].texture.usage, TextureUsage.RenderAttachment);
	assert.equal(target.color[0].resolveTexture.sampleCount, 1);
	assert.equal(target.color[0].resolveTexture.usage,
		TextureUsage.RenderAttachment | TextureUsage.TextureBinding | TextureUsage.CopySrc | TextureUsage.CopyDst);
	assert.equal(target.depth.texture.usage & TextureUsage.CopySrc, 0);
	target.destroy();
	assert.equal(backend.textures.every((texture) => texture.destroyCount === 1), true);
	for (const invalid of [{ width: 0 }, { height: NaN }, { sampleCount: 1.5 }]) {
		assert.throws(() => allocate(backend, { ...descriptor, ...invalid }), /positive integer/);
	}
	assert.equal(backend.textures.length, 3, "invalid extents do not allocate resources");
}

testAttachmentsAndLifetimeAcrossBackendCapabilities();
testAllocationAndInitializationFailuresReleaseAttachments();
testResolvedMultisampleAndReadbackUsage();
console.log("Cross-backend render target allocation tests passed");

function allocate(backend, descriptor) {
	return createCustomRenderTargetResources(backend, {
		...descriptor,
		size: { mode: "fixed", width: descriptor.width, height: descriptor.height },
	}, descriptor.width, descriptor.height, descriptor.sampleCount ?? 1);
}
