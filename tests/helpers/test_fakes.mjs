/**
 * @file test_fakes.mjs
 * Centralized fake classes for testing the IgnisRenderer engine.
 * Includes mocks for WebGPU, Workers, Canvas, and Physics engines.
 */

import assert from "node:assert/strict";
import {
	installNoopPostProcessSupport,
} from "./postprocess.mjs";

// --- Worker Fakes ---

/**
 * Mocks the browser's Web Worker API.
 */
import { Texture } from "../../src/core/Texture";

export class FakeWorker {
	constructor(handler) {
		this._handler = handler;
		this._terminated = false;
		this._listeners = {
			message: new Set(),
			error: new Set(),
		};
		this.onmessage = null;
		this.onerror = null;
	}

	addEventListener(type, listener) {
		const set = this._listeners[type];
		if (!set) return;
		set.add(listener);
	}

	removeEventListener(type, listener) {
		const set = this._listeners[type];
		if (!set) return;
		set.delete(listener);
	}

	postMessage(message) {
		if (this._terminated) {
			throw new Error("Cannot postMessage on a terminated FakeWorker");
		}
		queueMicrotask(() => {
			if (this._terminated) return;
			this._handler(message, this);
		});
	}

	emitMessage(data) {
		if (this._terminated) return;
		const event = { data };
		if (typeof this.onmessage === "function") {
			this.onmessage(event);
		}
		for (const listener of this._listeners.message) {
			listener(event);
		}
	}

	emitError(error) {
		if (this._terminated) return;
		const event = {
			message: error?.message ?? String(error),
			error,
		};
		if (typeof this.onerror === "function") {
			this.onerror(event);
		}
		for (const listener of this._listeners.error) {
			listener(event);
		}
	}

	terminate() {
		this._terminated = true;
	}
}

// --- Canvas Fakes ---

/**
 * Mocks CanvasRenderingContext2D.
 */
export class FakeCanvasContext2D {
	constructor(canvas, frameProvider = null) {
		this.canvas = canvas;
		this.fillStyle = "#000000";
		this.strokeStyle = "#000000";
		this.lineWidth = 1;
		this.calls = [];
		this._frameValue = 8;
		this._frameProvider = frameProvider;
	}

	get getImageDataCalls() { return this.calls.filter(c => c[0] === "getImageData").length; }
	get fillRectCalls() { return this.calls.filter(c => c[0] === "fillRect").length; }
	get drawImageCalls() { return this.calls.filter(c => c[0] === "drawImage").length; }

	fillRect(x, y, w, h) { 
		this.calls.push(["fillRect", x, y, w, h]); 
		this._frameValue = (this._frameValue + 16) & 0xff;
	}
	clearRect(x, y, w, h) { this.calls.push(["clearRect", x, y, w, h]); }
	strokeRect(x, y, w, h) { this.calls.push(["strokeRect", x, y, w, h]); }
	drawImage(img, ...args) { 
		this.calls.push(["drawImage", img, ...args]); 
		this._frameValue = (this._frameValue + 8) & 0xff;
	}
	beginPath() { this.calls.push(["beginPath"]); }
	moveTo(x, y) { this.calls.push(["moveTo", x, y]); }
	lineTo(x, y) { this.calls.push(["lineTo", x, y]); }
	stroke() { this.calls.push(["stroke"]); }
	fill() { this.calls.push(["fill"]); }
	getImageData(_x, _y, width, height) { 
		this.calls.push(["getImageData", _x, _y, width, height]);
		if (this._frameProvider) {
			return { data: this._frameProvider(width, height, this.getImageDataCalls) };
		}
		const data = new Uint8ClampedArray(width * height * 4);
		for (let i = 0; i < data.length; i += 4) {
			data[i] = this._frameValue;
			data[i + 1] = this._frameValue;
			data[i + 2] = this._frameValue;
			data[i + 3] = 255;
		}
		return { data };
	}
}

/**
 * Mocks the HTMLCanvasElement.
 */
export class FakeCanvas {
	constructor(width = 1, height = 1) {
		this.width = width;
		this.height = height;
		this.style = {};
		this._context = null;
	}
	getContext(type) {
		if (type === "2d") {
			if (!this._context) this._context = new FakeCanvasContext2D(this);
			return this._context;
		}
		return null;
	}
}

/**
 * Mocks the HTMLVideoElement.
 */
export class FakeVideo {
	constructor({ supportsRVFC = true } = {}) {
		this.readyState = 2; // HAVE_CURRENT_DATA
		this.videoWidth = 2;
		this.videoHeight = 1;
		this.currentTime = 0;
		this._listeners = new Map();
		this._rvfcCallbacks = new Map();
		this._nextRVFCId = 1;
		this.cancelCalls = 0;

		if (supportsRVFC) {
			this.requestVideoFrameCallback = (callback) => {
				const id = this._nextRVFCId++;
				this._rvfcCallbacks.set(id, callback);
				return id;
			};
			this.cancelVideoFrameCallback = (id) => {
				this.cancelCalls++;
				this._rvfcCallbacks.delete(id);
			};
		}
	}

	addEventListener(eventName, callback) {
		const list = this._listeners.get(eventName) ?? [];
		list.push(callback);
		this._listeners.set(eventName, list);
	}

	removeEventListener(eventName, callback) {
		const list = this._listeners.get(eventName) ?? [];
		this._listeners.set(eventName, list.filter((entry) => entry !== callback));
	}

	emit(eventName) {
		const list = this._listeners.get(eventName) ?? [];
		for (const callback of list) callback();
	}

	presentFrame(currentTime) {
		this.currentTime = currentTime;
		const callbacks = Array.from(this._rvfcCallbacks.values());
		this._rvfcCallbacks.clear();
		for (const callback of callbacks) callback(0, {});
	}
}

// --- WebGPU Fakes ---

/**
 * Mocks a GPUBuffer.
 */
export class FakeGPUBuffer {
	constructor(desc) {
		this.size = desc.size;
		this.usage = desc.usage;
		this.label = desc.label;
		this.desc = desc;
		this.destroyed = false;
		this.calls = [];
		this.lastWrite = null;
		this._bytes = new Uint8Array(this.size);
		this._bytes.fill(0);
	}
	get [Symbol.toStringTag]() { return "GPUBuffer"; }
	destroy() { this.destroyed = true; this.calls.push(["destroy"]); }
	mapAsync(_mode, _offset = 0, _size = this.size) { return Promise.resolve(); }
	getMappedRange(offset = 0, size = this.size - offset) { 
		const start = Math.max(0, Math.floor(offset));
		const end = Math.min(this._bytes.length, start + Math.floor(size));
		return this._bytes.buffer.slice(start, end); 
	}
	unmap() { this.calls.push(["unmap"]); }
}

/**
 * Mocks a GPUTexture.
 */
export class FakeGPUTexture {
	constructor(desc) {
		this.width = desc.width;
		this.height = desc.height;
		this.depthOrArrayLayers = desc.depthOrArrayLayers ?? 1;
		this.format = desc.format;
		this.usage = desc.usage;
		this.label = desc.label;
		this.desc = desc;
		this.destroyed = false;
		this.calls = [];
		this.lastWrite = null;
		const bpp = getFakeTextureBytesPerPixel(desc.format);
		this._bytes = new Uint8Array(this.width * this.height * this.depthOrArrayLayers * bpp);
	}
	get [Symbol.toStringTag]() { return "GPUTexture"; }
	createView(desc) { 
		this.calls.push(["createView", desc]);
		return new FakeGPUTextureView(this, desc); 
	}
	destroy() { this.destroyed = true; this.calls.push(["destroy"]); }
}

function getFakeTextureBytesPerPixel(format) {
	return format === "rgba16float" ? 8 : 4;
}

function resolveFakeTextureUploadFormat(texture) {
	if (texture?.data instanceof Float32Array) {
		return "rgba16float";
	}
	for (const mip of texture?.mipmaps ?? []) {
		if (mip instanceof Float32Array) {
			return "rgba16float";
		}
	}
	return "rgba8unorm";
}

/**
 * Mocks a GPUTextureView.
 */
export class FakeGPUTextureView {
	constructor(texture, desc) {
		this.texture = texture;
		this.desc = desc;
	}
	get [Symbol.toStringTag]() { return "GPUTextureView"; }
}

/**
 * Mocks a GPUCommandEncoder / GPURenderPassEncoder / GPUComputePassEncoder.
 */
export class FakeCommandEncoder {
	constructor(backend) {
		this.backend = backend;
		this.calls = [];
		this._ops = [];
	}

	beginComputePass(desc = {}) { this.calls.push(["beginComputePass", desc]); return this; }
	setComputePipeline(pipeline) { this.calls.push(["setComputePipeline", pipeline]); return this; }
	setBindingGroup(index, group) { this.calls.push(["setBindingGroup", index, group]); return this; }
	dispatchWorkgroups(x, y = 1, z = 1) { 
		this.calls.push(["dispatchWorkgroups", x, y, z]); 
		if (this.backend && this.backend.dispatches) this.backend.dispatches.push([x, y, z]);
		return this; 
	}
	endComputePass() { this.calls.push(["endComputePass"]); return this; }

	beginRenderPass(desc = {}) { 
		this.calls.push(["beginRenderPass", desc]); 
		if (this.backend && this.backend.recordedRenderPasses) {
			this.backend.recordedRenderPasses.push(desc);
		}
		return this; 
	}
	setPipeline(pipeline) { this.calls.push(["setPipeline", pipeline]); return this; }
	setScissorRect(x, y, width, height) {
		this.calls.push(["setScissorRect", x, y, width, height]);
		return this;
	}
	setVertexBuffer(index, buffer) { this.calls.push(["setVertexBuffer", index, buffer]); return this; }
	setIndexBuffer(buffer) { this.calls.push(["setIndexBuffer", buffer]); return this; }
	setBindGroup(index, group) { this.calls.push(["setBindGroup", index, group]); return this; }
	setBindingGroup(index, group) { this.calls.push(["setBindGroup", index, group]); return this; } // Compatibility
	draw(v, i, firstV, firstI) { this.calls.push(["draw", v, i, firstV, firstI]); return this; }
	drawIndirect(buffer, offset = 0) {
		this.calls.push(["drawIndirect", buffer, offset]);
		return this;
	}
	drawIndexed(i, instance, firstI, baseV, firstInst) { this.calls.push(["drawIndexed", i, instance, firstI, baseV, firstInst]); return this; }
	endRenderPass() { this.calls.push(["endRenderPass"]); return this; }

	copyTextureToBuffer(src, dst, size) {
		this.calls.push(["copyTextureToBuffer", src, dst, size]);
		this._ops.push(() => {
			const texture = src.texture?._gpuResource || src.texture;
			const buffer = dst.buffer?._gpuResource || dst.buffer;
			if (!texture?._bytes || !buffer?._bytes) return;

			const width = Math.max(1, Math.floor(size.width));
			const height = Math.max(1, Math.floor(size.height));
			const bytesPerPixel = getFakeTextureBytesPerPixel(texture.format);
			const srcBytesPerRow = width * bytesPerPixel;
			const dstBytesPerRow = Math.max(srcBytesPerRow, dst.bytesPerRow || 0);

			for (let y = 0; y < height; y++) {
				const srcOffset = y * srcBytesPerRow;
				const dstOffset = y * dstBytesPerRow;
				buffer._bytes.set(
					texture._bytes.subarray(srcOffset, srcOffset + srcBytesPerRow),
					dstOffset
				);
			}
		});
	}

	copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
		this.calls.push(["copyBufferToBuffer", src, srcOffset, dst, dstOffset, size]);
		this._ops.push(() => {
			const s = src._gpuResource || src;
			const d = dst._gpuResource || dst;
			if (!s._bytes || !d._bytes) return;
			const copied = s._bytes.subarray(srcOffset, srcOffset + size);
			d._bytes.set(copied, dstOffset);
		});
	}

	finish() { 
		const ops = [...this._ops];
		return { 
			label: "GPUCommandBuffer",
			_gpuCommandBuffer: {},
			_ownerToken: {},
			_submitted: false,
			execute: () => { for (const op of ops) op(); }
		}; 
	}
}

/**
 * Mocks the WebGPU Backend.
 */
export class FakeWebGPUBackend {
	constructor() {
		this.type = "webgpu";
		installNoopPostProcessSupport(
			this,
			"webgpu"
		);
		this.canvasFormat = "rgba8unorm";
		this.canvasDepthFormat = "depth24plus";
		this.dispatches = [];
		this.submits = 0;
		this.calls = [];
		this.bufferWrites = [];
		this.textureWrites = [];
		this.registeredExternalTextures = [];
		this.unregisteredExternalTextures = [];
		this.destroyCalls = 0;
		this.createBindingGroupCalls = 0;
		this.createTextureCallsCount = 0;
		this.createTextureCalls = [];
		this.createCommandEncoderCalls = 0;
		this.commandEncoders = [];
		this.recordedRenderPasses = [];
		this.shaderModules = [];
		this.pipelines = [];
		this.computePipelines = [];
		this.renderPipelines = [];
		this.buffers = [];
		this.bindingGroups = [];
		this.bindGroupLayouts = [];
		this.pipelineLayouts = [];
		this.textures = [];
		this.textureViews = [];
		this.samplers = [];
		this.samplerDescs = [];
		this.bufferDescs = [];
		this.getComputeFacadeCalls = 0;
		this.copyCalls = [];
		this.writeCalls = [];
		this.writeBufferCalls = 0;
		this.bindingGroupDestroyCalls = 0;
		this.shaderModuleDestroyCalls = 0;
		this.computePipelineDestroyCalls = 0;
		this.renderPipelineDestroyCalls = 0;
		this.samplerDestroyCalls = 0;
		this.bufferDestroyCalls = 0;
		this.textureDestroyCalls = 0;
		this.failTextureAtCall = null;
		this.failCustomShaderModules = false;
		this.shaderRuntime = null;
		this.warnings = [];
		this._slotTextureCache = new WeakMap();
		this._externalTextureResources = new WeakMap();
		this.canvasColorTexture = {
			width: 1,
			height: 1,
			destroy: () => this.destroyCalls++,
		};
		this.canvasDepthTexture = {
			width: 1,
			height: 1,
			destroy: () => this.destroyCalls++,
		};
		
		this.queue = {
			copyExternalImageToTexture: (...args) => {
				this.copyCalls.push(args);
			},
			writeTexture: (dst, data, layout, size) => {
				this.writeTexture(dst.texture, data, layout, size);
			},
			writeBuffer: (buffer, offset, data) => {
				this.writeBuffer(buffer, data, offset);
			},
			submit: (commands) => {
				this.submit(commands);
			},
			onSubmittedWorkDone: () => Promise.resolve(),
		};

		this.device = {
			limits: {
				maxColorAttachments: 8,
				maxColorAttachmentBytesPerSample: 64,
			},
			bindGroupLayouts: this.bindGroupLayouts,
			pipelineLayouts: this.pipelineLayouts,
			createCommandEncoder: () => this.createCommandEncoder(),
			createBuffer: (desc) => this.createBuffer(desc)._gpuResource,
			createBindGroupLayout: (desc) => {
				const layout = { kind: "bind-group-layout", desc };
				this.bindGroupLayouts.push(layout);
				return layout;
			},
			createPipelineLayout: (desc) => {
				const layout = { kind: "pipeline-layout", desc };
				this.pipelineLayouts.push(layout);
				return layout;
			},
			createShaderModule: (desc) => this.createShaderModule(desc),
			createComputePipeline: (desc) => this.createComputePipeline(desc),
			createRenderPipeline: (desc) => this.createRenderPipeline(desc),
		};
	}

	getMSAASampleCount() { return 1; }
	getCanvasColorTexture() { return this.canvasColorTexture; }
	getCanvasDepthTexture() { return this.canvasDepthTexture; }
	getCanvasRenderTargetSize() {
		return {
			width: Math.max(1, Math.floor(this.canvasColorTexture.width ?? 1)),
			height: Math.max(1, Math.floor(this.canvasColorTexture.height ?? 1)),
		};
	}

	getComputeFacade() {
		this.getComputeFacadeCalls++;
		return this;
	}

	async createShaderModule(desc) {
		if (this.failCustomShaderModules && typeof desc.label === "string" && desc.label.startsWith("WebGPUShaderMaterial")) {
			throw new Error("simulated custom shader module compile failure");
		}
		const module = {
			kind: "shader-module",
			label: desc.label,
			desc,
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.shaderModuleDestroyCalls++;
				module.destroyed = true;
			}
		};
		this.shaderModules.push(module);
		return module;
	}

	createComputePipeline(desc) {
		const pipeline = { 
			kind: "compute-pipeline",
			desc, 
			label: desc.label, 
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.computePipelineDestroyCalls++;
				pipeline.destroyed = true;
			},
			_gpuResource: { getBindGroupLayout: () => {
				const layout = { kind: "bind-group-layout" };
				this.bindGroupLayouts.push(layout);
				return layout;
			} }
		};
		this.pipelines.push(pipeline);
		this.computePipelines.push(pipeline);
		return pipeline;
	}

	createPipeline(desc) {
		return this.createComputePipeline(desc);
	}
	
	createRenderPipeline(desc) {
		const pipeline = {
			kind: "render-pipeline",
			desc,
			label: desc.label,
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.renderPipelineDestroyCalls++;
				pipeline.destroyed = true;
			}
		};
		this.pipelines.push(pipeline);
		this.renderPipelines.push(pipeline);
		return pipeline;
	}

	createBuffer(desc) {
		this.bufferDescs.push(desc);
		const gpuBuffer = new FakeGPUBuffer(desc);
		const buffer = {
			kind: "buffer",
			size: desc.size,
			usage: desc.usage,
			label: desc.label,
			desc,
			_gpuResource: gpuBuffer,
			lastWrite: null,
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.bufferDestroyCalls++;
				buffer.destroyed = true;
				gpuBuffer.destroy();
			}
		};
		this.buffers.push(buffer);
		return buffer;
	}

	createTexture(desc) {
		this.createTextureCallsCount++;
		if (typeof this.failTextureAtCall === "number" && this.createTextureCallsCount >= this.failTextureAtCall) {
			throw new Error("simulated allocation failure");
		}
		this.createTextureCalls.push(desc);
		const gpuTexture = new FakeGPUTexture(desc);
		const texture = {
			kind: "texture",
			width: desc.width,
			height: desc.height,
			usage: desc.usage,
			label: desc.label,
			desc,
			_gpuResource: gpuTexture,
			lastWrite: null,
			_webgpuTexture: {
				texture: gpuTexture,
				view: gpuTexture.createView()
			},
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.textureDestroyCalls++;
				texture.destroyed = true;
				gpuTexture.destroy();
			}
		};
		this.textures.push(texture);
		return texture;
	}

	createTextureView(texture, desc) {
		let view;
		if (!desc && texture._webgpuTexture?.view) {
			view = texture._webgpuTexture.view;
		} else if (desc && texture._webgpuTexture?.texture?.createView) {
			view = texture._webgpuTexture.texture.createView(desc);
		} else if (texture._gpuResource?.createView) {
			view = texture._gpuResource.createView(desc);
		} else {
			view = new FakeGPUTextureView(texture, desc);
		}
		this.textureViews.push(view);
		return view;
	}

	createSampler(desc) {
		this.samplerDescs.push(desc);
		const sampler = { 
			kind: "sampler",
			desc, 
			label: desc.label, 
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.samplerDestroyCalls++;
				sampler.destroyed = true;
			},
			_gpuResource: { [Symbol.toStringTag]: "GPUSampler" }
		};
		this.samplers.push(sampler);
		return sampler;
	}

	createBindingGroup(desc) {
		this.createBindingGroupCalls++;
		const group = { 
			kind: "binding-group",
			desc, 
			label: desc.label, 
			entries: desc.entries,
			destroyed: false,
			destroy: () => {
				this.destroyCalls++;
				this.bindingGroupDestroyCalls++;
				group.destroyed = true;
			} 
		};
		this.bindingGroups.push(group);
		return group;
	}

	writeBuffer(buffer, data, offset = 0) {
		this.writeBufferCalls++;
		const gpuBuffer = buffer._gpuResource || buffer;
		const view = (data instanceof Uint8Array) ? data : 
			(data.buffer ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data));
		
		if (gpuBuffer._bytes && (offset + view.byteLength) <= gpuBuffer._bytes.length) {
			gpuBuffer._bytes.set(view, offset);
		}

		this.bufferWrites.push(Array.from(view));
		this.writeCalls.push(["writeBuffer", buffer, data]);
		
		// Use original data if iterable (e.g. Float32Array), otherwise fallback to bytes (view)
		const lastWrite = (typeof data[Symbol.iterator] === "function") ? Array.from(data) : Array.from(view);
		lastWrite.data = data;
		lastWrite.offset = offset;
		gpuBuffer.lastWrite = lastWrite;
		if (buffer && typeof buffer === "object" && "_gpuResource" in buffer) {
			buffer.lastWrite = lastWrite;
		}
	}

	writeTexture(texture, data, layout, size) {
		const gpuTexture = texture._gpuResource || texture;
		const src = (data instanceof Uint8Array) ? data : 
			(data.buffer ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data));

		if (gpuTexture._bytes) {
			const width = Math.max(1, Math.floor(size.width));
			const height = Math.max(1, Math.floor(size.height));
			const bytesPerPixel = getFakeTextureBytesPerPixel(gpuTexture.format);
			const rowSize = width * bytesPerPixel;
			const bytesPerRow = layout.bytesPerRow ?? rowSize;
			for (let y = 0; y < height; y++) {
				const srcOffset = y * bytesPerRow;
				const dstOffset = y * rowSize;
				if (srcOffset + rowSize <= src.length && dstOffset + rowSize <= gpuTexture._bytes.length) {
					gpuTexture._bytes.set(src.subarray(srcOffset, srcOffset + rowSize), dstOffset);
				}
			}
		}

		this.textureWrites.push({ texture, data: Array.from(src), layout, size });
		this.writeCalls.push(["writeTexture", texture, data, layout, size]);
		
		// Use original data if iterable, otherwise fallback to bytes
		const lastWrite = (typeof data[Symbol.iterator] === "function") ? Array.from(data) : Array.from(src);
		lastWrite.data = data;
		lastWrite.layout = layout;
		lastWrite.size = size;
		gpuTexture.lastWrite = lastWrite;
		if (texture && typeof texture === "object" && "_gpuResource" in texture) {
			texture.lastWrite = lastWrite;
		}
	}


	submit(commands) {
		this.submits += commands ? (commands.length || 1) : 1;
		if (Array.isArray(commands)) {
			for (const cmd of commands) {
				if (typeof cmd.execute === "function") cmd.execute();
			}
		} else if (commands && typeof commands.execute === "function") {
			commands.execute();
		}
	}

	createCommandEncoder() {
		this.createCommandEncoderCalls++;
		const encoder = new FakeCommandEncoder(this);
		this.commandEncoders.push(encoder);
		return encoder;
	}

	getTextureForSlot(texture, slotIndex) {
		if (!texture || typeof texture !== "object") {
			return { kind: "slot-texture", texture, slotIndex };
		}
		const external = this._externalTextureResources.get(texture);
		if (external) {
			return external;
		}
		const width = texture.width | 0;
		const height = texture.height | 0;
		if (width > 0 && height > 0) {
			const format = resolveFakeTextureUploadFormat(texture);
			const cached = this._slotTextureCache.get(texture);
			if (
				cached &&
				cached.width === width &&
				cached.height === height &&
				cached.desc?.format === format
			) {
				return cached;
			}
			if (cached && typeof cached.destroy === "function" && !cached.destroyed) {
				cached.destroy();
			}
			const created = this.createTexture({
				width,
				height,
				format,
				usage: 0,
				label: `SlotTexture_${slotIndex}_${width}x${height}`,
			});
			this._slotTextureCache.set(texture, created);
			return created;
		}
		return { kind: "slot-texture", texture, slotIndex };
	}

	registerExternalTexture(texture, resource) {
		this.registeredExternalTextures.push({ texture, resource });
		if (texture && typeof texture === "object") {
			this._externalTextureResources.set(texture, resource);
		}
		this.calls.push(["registerExternalTexture", texture, resource]);
	}

	unregisterExternalTexture(texture) {
		this.unregisteredExternalTextures.push(texture);
		if (texture && typeof texture === "object") {
			this._externalTextureResources.delete(texture);
		}
		this.calls.push(["unregisterExternalTexture", texture]);
	}

	warn(message, key = "") {
		this.warnings.push({ key, message });
	}
}

// --- Physics Fakes (Rapier) ---

export function createFakeRapierModule() {
	const stats = {
		stepCalls: 0,
		characterControllerCreates: 0,
		characterComputeCalls: 0,
		descriptorCanSleepCalls: [],
		descriptorAdditionalMassCalls: [],
		bodyAdditionalMassCalls: [],
		collisionGroupUpdates: [],
	};
	class FakeRigidBodyDesc {
		constructor(type) {
			this.type = type;
			this.translation = { x: 0, y: 0, z: 0 };
			this.rotation = { x: 0, y: 0, z: 0, w: 1 };
		}
		static dynamic() { return new FakeRigidBodyDesc("dynamic"); }
		static fixed() { return new FakeRigidBodyDesc("fixed"); }
		static kinematicPositionBased() { return new FakeRigidBodyDesc("kinematic"); }
		setTranslation(x, y, z) {
			this.translation = typeof x === "object" ? { ...x } : { x, y, z };
			return this;
		}
		setRotation(q) { this.rotation = { ...q }; return this; }
		setLinvel() { return this; }
		setAngvel() { return this; }
		setCcdEnabled() { return this; }
		setLinearDamping() { return this; }
		setAngularDamping() { return this; }
		setCanSleep(value) {
			stats.descriptorCanSleepCalls.push(Boolean(value));
			return this;
		}
		setAdditionalMass(value) {
			stats.descriptorAdditionalMassCalls.push(Number(value));
			return this;
		}
		setMass(value) {
			stats.descriptorAdditionalMassCalls.push(Number(value));
			return this;
		}
		setEnabledTranslations() { return this; }
		setEnabledRotations() { return this; }
		restrictTranslations() { return this; }
		restrictRotations() { return this; }
	}
	class FakeColliderDesc {
		constructor(kind) { this.kind = kind; }
		static cuboid(x, y, z) { return new FakeColliderDesc({ kind: "box", x, y, z }); }
		static ball(radius) { return new FakeColliderDesc({ kind: "sphere", radius }); }
		static capsule(halfHeight, radius) {
			return new FakeColliderDesc({ kind: "capsule", halfHeight, radius });
		}
		static cylinder(halfHeight, radius) {
			return new FakeColliderDesc({ kind: "cylinder", halfHeight, radius });
		}
		static trimesh(vertices, indices) {
			return new FakeColliderDesc({ kind: "trimesh", vertices, indices });
		}
		setSensor() { return this; }
		setTranslation() { return this; }
		setFriction() { return this; }
		setRestitution() { return this; }
		setDensity() { return this; }
	}
	class FakeRigidBody {
		constructor(desc) {
			this._type = desc.type;
			this._translation = { ...desc.translation };
			this._rotation = { ...desc.rotation };
			this._linvel = { x: 0, y: 0, z: 0 };
		}
		setTranslation(x, y, z) { this._translation = typeof x === "object" ? { ...x } : { x, y, z }; }
		setNextKinematicTranslation(x, y, z) { this.setTranslation(x, y, z); }
		setRotation(q) { this._rotation = { ...q }; }
		setNextKinematicRotation(q) { this.setRotation(q); }
		setLinvel(x, y, z) { this._linvel = typeof x === "object" ? { ...x } : { x, y, z }; }
		setAdditionalMass(value) { stats.bodyAdditionalMassCalls.push(Number(value)); }
		setMass(value) { stats.bodyAdditionalMassCalls.push(Number(value)); }
		linvel() { return { ...this._linvel }; }
		translation() { return { ...this._translation }; }
		rotation() { return { ...this._rotation }; }
		isSleeping() { return false; }
		isCcdEnabled() { return false; }
	}
	class FakeCollider {
		setCollisionGroups(value) {
			stats.collisionGroupUpdates.push(Number(value) >>> 0);
		}
	}

	class FakeCharacterController {
		constructor(offset) {
			this._offset = offset;
			this._computedMovement = { x: 0, y: 0, z: 0 };
			this._grounded = false;
		}
		setOffset(offset) { this._offset = offset; return this; }
		setApplyImpulsesToDynamicBodies() { return this; }
		setMaxSlopeClimbAngle() { return this; }
		setMinSlopeSlideAngle() { return this; }
		enableAutostep() { return this; }
		setAutostep() { return this; }
		enableSnapToGround() { return this; }
		setSnapToGround() { return this; }
		disableAutostep() { return this; }
		disableSnapToGround() { return this; }
		computeColliderMovement(_collider, desired) {
			stats.characterComputeCalls++;
			const d =
				desired && typeof desired === "object" ?
					{
						x: Number.isFinite(desired.x) ? desired.x : 0,
						y: Number.isFinite(desired.y) ? desired.y : 0,
						z: Number.isFinite(desired.z) ? desired.z : 0,
					}
				:	{ x: 0, y: 0, z: 0 };
			this._computedMovement = {
				x: d.x * 0.5,
				y: d.y,
				z: d.z * 0.5,
			};
			this._grounded = d.y <= 0;
		}
		computedMovement() { return { ...this._computedMovement }; }
		computedGrounded() { return this._grounded; }
		free() {}
	}
	class FakeWorld {
		constructor() { this._bodies = new Set(); }
		createRigidBody(desc) {
			const body = new FakeRigidBody(desc);
			this._bodies.add(body);
			return body;
		}
		removeRigidBody(body) { this._bodies.delete(body); }
		createCollider(desc, body) {
			const collider = new FakeCollider();
			collider.desc = desc;
			collider.body = body;
			return collider;
		}
		removeCollider() {}
		createImpulseJoint() { return { id: "joint" }; }
		removeImpulseJoint() {}
		createCharacterController(offset) {
			stats.characterControllerCreates++;
			return new FakeCharacterController(offset);
		}
		step() { stats.stepCalls++; }
		free() {}
	}
	return {
		module: {
			init: async () => {},
			World: FakeWorld,
			Vector3: class { constructor(x, y, z) { this.x = x; this.y = y; this.z = z; } },
			Quaternion: class { constructor(x, y, z, w) { this.x = x; this.y = y; this.z = z; this.w = w; } },
			RigidBodyDesc: FakeRigidBodyDesc,
			ColliderDesc: FakeColliderDesc,
			JointData: { fixed: () => ({}), revolute: () => ({}), spring: () => ({}) },
		},
		stats
	};
}

// --- Physics Fakes (Ammo) ---

export class FakeImageData {
	static instances = [];
	constructor(dataOrWidth, widthOrHeight, maybeHeight) {
		if (dataOrWidth instanceof Uint8ClampedArray) {
			this.data = dataOrWidth;
			this.width = widthOrHeight;
			this.height = maybeHeight || (dataOrWidth.length / (this.width * 4));
			if (this.data.length !== this.width * this.height * 4) {
				throw new RangeError("ImageData source length mismatch.");
			}
		} else {
			this.width = dataOrWidth;
			this.height = widthOrHeight;
			this.data = new Uint8ClampedArray(this.width * this.height * 4);
		}
		FakeImageData.instances.push(this);
	}
}

export function createFakeAmmoModule() {
	const stats = {
		stepCalls: 0,
		addRigidBodyCalls: [],
	};
	
	function readNumberLike(value, key, fallback = 0) {
		if (!value || typeof value !== "object") return fallback;
		const member = value[key];
		if (typeof member === "number") return member;
		if (typeof member === "function") return member.call(value) ?? fallback;
		return fallback;
	}

	class FakeBtVector3 { 
		constructor(x = 0, y = 0, z = 0) { this._x = x; this._y = y; this._z = z; }
		x() { return this._x; } y() { return this._y; } z() { return this._z; }
		setValue(x, y, z) { this._x = x; this._y = y; this._z = z; }
		clone() { return new FakeBtVector3(this._x, this._y, this._z); }
	}
	class FakeBtQuaternion {
		constructor(x = 0, y = 0, z = 0, w = 1) { this._x = x; this._y = y; this._z = z; this._w = w; }
		x() { return this._x; } y() { return this._y; } z() { return this._z; } w() { return this._w; }
		clone() { return new FakeBtQuaternion(this._x, this._y, this._z, this._w); }
	}
	class FakeBtTransform {
		constructor() { this.setIdentity(); }
		setIdentity() { this._origin = new FakeBtVector3(); this._rotation = new FakeBtQuaternion(); }
		setOrigin(o) { this._origin.setValue(o.x(), o.y(), o.z()); }
		setRotation(q) { this._rotation = q; }
		getOrigin() { return this._origin; }
		getRotation() { return this._rotation; }
		clone() { const t = new FakeBtTransform(); t.setOrigin(this._origin); t.setRotation(this._rotation); return t; }
	}
	class FakeBtRigidBody {
		constructor(info) { 
			this._info = info;
			this._transform = new FakeBtTransform(); 
			this._linearVelocity = new FakeBtVector3(); 
			this._active = true;
			this._sleeping = false;
		}
		setWorldTransform(t) { this._transform = t.clone(); }
		getWorldTransform() { return this._transform; }
		activate() { this._active = true; this._sleeping = false; }
		isActive() { return this._active; }
		isSleeping() { return this._sleeping; }
		setLinearVelocity(v) { this._linearVelocity = v; }
		getLinearVelocity() { return this._linearVelocity; }
		setCollisionFlags() {}
		getCollisionFlags() { return 0; }
		setCcdMotionThreshold() {}
		setCcdSweptSphereRadius() {}
		setActivationState() {}
	}
	return {
		module: {
			btVector3: FakeBtVector3,
			btQuaternion: FakeBtQuaternion,
			btTransform: FakeBtTransform,
			btDefaultMotionState: class { 
				constructor(t) { this._t = t; } 
				getWorldTransform(out) { if (out) { out.setOrigin(this._t.getOrigin()); out.setRotation(this._t.getRotation()); } } 
			},
			btRigidBodyConstructionInfo: class {},
			btRigidBody: FakeBtRigidBody,
			btSphereShape: class { calculateLocalInertia(_, out) { out.setValue(0, 0, 0); } },
			btBoxShape: class { calculateLocalInertia(_, out) { out.setValue(0, 0, 0); } },
			btDiscreteDynamicsWorld: class {
				addRigidBody(...args) { stats.addRigidBodyCalls.push(args); }
				removeRigidBody() {}
				setGravity() {}
				stepSimulation() { stats.stepCalls++; }
			},
			btDefaultCollisionConfiguration: class {},
			btCollisionDispatcher: class { constructor() {} },
			btDbvtBroadphase: class {},
			btSequentialImpulseConstraintSolver: class {},
			destroy: () => {},
		},
		stats
	};
}

// --- Renderer Fakes ---

export class FakeRenderer {
	constructor(backend) {
		this.backend = backend;
		this.logger = {
			warn() {},
		};
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

export class FakeDynamicTexture extends Texture {
	constructor(framesToUpdate) {
		super(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1, "sRGB");
		this._framesToUpdate = framesToUpdate;
		this._registerAsDynamicTexture();
	}

	update() {
		if (this._framesToUpdate <= 0) {
			return false;
		}
		this._framesToUpdate--;
		this.markNeedsUpdate();
		return true;
	}
}
