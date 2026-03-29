import assert from "node:assert/strict";
import { SobelNormalMapper } from "../src/addons/SobelNormalMapper.ts";
import { Texture } from "../src/core/Texture.ts";

class FakeCommandEncoder {
	constructor(backend) {
		this._backend = backend;
	}

	beginComputePass(_desc) {}

	setComputePipeline(_pipeline) {}

	setBindingGroup(_index, _group) {}

	dispatchWorkgroups(x, y = 1, z = 1) {
		this._backend.dispatches.push([x, y, z]);
	}

	endComputePass() {}

	finish() {
		return { _backendCommandBuffer: {} };
	}
}

class FakeWebGPUBackend {
	constructor() {
		this.type = "webgpu";
		this.dispatches = [];
		this.submits = 0;
		this.bufferWrites = [];
		this.registeredExternalTextures = [];
		this.unregisteredExternalTextures = [];
		this.createBindingGroupCalls = 0;
		this.destroyCalls = 0;
		this.sourceResource = {
			width: 1,
			height: 1,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	async createShaderModule(desc) {
		return {
			desc,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	createComputePipeline(desc) {
		return {
			desc,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	createBuffer(desc) {
		return {
			size: desc.size,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	createTexture(desc) {
		return {
			width: desc.width,
			height: desc.height,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	createBindingGroup(desc) {
		this.createBindingGroupCalls++;
		return {
			desc,
			destroy: () => {
				this.destroyCalls++;
			},
		};
	}

	createCommandEncoder() {
		return new FakeCommandEncoder(this);
	}

	writeBuffer(_buffer, data) {
		this.bufferWrites.push(Array.from(data));
	}

	submit(commands) {
		this.submits += commands.length;
	}

	getTextureForSlot(_texture, _slotIndex) {
		return this.sourceResource;
	}

	registerExternalTexture(texture, resource, uploadedVersion, mipLevelCount) {
		this.registeredExternalTextures.push({
			texture,
			resource,
			uploadedVersion,
			mipLevelCount,
		});
	}

	unregisterExternalTexture(texture) {
		this.unregisteredExternalTextures.push(texture);
	}
}

class FakeRenderer {
	constructor(backend) {
		this.backend = backend;
		this._events = new Map();
		this.requestRenderCalls = 0;
	}

	on(event, listener) {
		const listeners = this._events.get(event) ?? new Set();
		listeners.add(listener);
		this._events.set(event, listeners);
	}

	off(event, listener) {
		const listeners = this._events.get(event);
		if (!listeners) return;
		listeners.delete(listener);
	}

	emit(event, payload) {
		const listeners = this._events.get(event);
		if (!listeners) return;
		for (const listener of listeners) {
			listener(payload);
		}
	}

	requestRender() {
		this.requestRenderCalls++;
	}
}

async function testInitUpdateAndInvalidationFlow() {
	const backend = new FakeWebGPUBackend();
	const renderer = new FakeRenderer(backend);
	const source = new Texture(new Uint8ClampedArray(4), 15, 9, "sRGB");
	const mapper = new SobelNormalMapper(source);

	await mapper.init(renderer);
	assert.equal(mapper.isInitialized, true);
	assert.equal(backend.registeredExternalTextures.length, 1);
	assert.equal(backend.registeredExternalTextures[0].texture, mapper.normalMap);

	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 1);
	assert.deepEqual(backend.dispatches[0], [2, 2, 1]);
	assert.equal(backend.createBindingGroupCalls, 1);
	assert.equal(mapper.update(), false);
	assert.equal(backend.submits, 1);

	source.markNeedsUpdate();
	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 2);

	mapper.strength = 3.5;
	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 3);
	assert.equal(backend.bufferWrites.length >= 3, true);
}

async function testAttachRunsOnPostAnimationAndDetachStops() {
	const backend = new FakeWebGPUBackend();
	const renderer = new FakeRenderer(backend);
	const source = new Texture(new Uint8ClampedArray(4), 8, 8, "sRGB");
	const mapper = new SobelNormalMapper(source);

	await mapper.attach(renderer);
	assert.equal(mapper.isAttached, true);
	assert.equal(renderer.requestRenderCalls, 1);

	source.markNeedsUpdate();
	renderer.emit("postanimation", {
		now: 16,
		deltaTime: 16,
		scene: null,
		transient: new Map(),
	});
	assert.equal(backend.submits, 1);

	mapper.detach();
	assert.equal(mapper.isAttached, false);
	source.markNeedsUpdate();
	renderer.emit("postanimation", {
		now: 32,
		deltaTime: 16,
		scene: null,
		transient: new Map(),
	});
	assert.equal(backend.submits, 1);
}

async function testDestroyReleasesResources() {
	const backend = new FakeWebGPUBackend();
	const renderer = new FakeRenderer(backend);
	const source = new Texture(new Uint8ClampedArray(4), 4, 4, "sRGB");
	const mapper = new SobelNormalMapper(source);

	await mapper.init(renderer);
	mapper.update();
	mapper.destroy();

	assert.equal(mapper.isInitialized, false);
	assert.equal(backend.unregisteredExternalTextures.length >= 1, true);
	assert.equal(backend.destroyCalls >= 4, true);
	assert.equal(mapper.update(), false);
}

await testInitUpdateAndInvalidationFlow();
await testAttachRunsOnPostAnimationAndDetachStops();
await testDestroyReleasesResources();
console.log("Sobel normal mapper tests passed");
