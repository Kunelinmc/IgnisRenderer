import assert from "node:assert/strict";
import { WebGPUCustomRenderTargetRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUCustomRenderTargetRuntime.ts";
import { TextureFormat } from "../../../src/backends/types.ts";
import {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../../../src/rendering/CustomRenderTargets.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

class FakeEncoder {
	constructor() {
		this.calls = [];
	}
	beginRenderPass(desc) {
		this.calls.push(["beginRenderPass", desc]);
	}
	endRenderPass() {
		this.calls.push(["endRenderPass"]);
	}
	finish() {
		return {};
	}
}

class FakeBackend {
	constructor(options = {}) {
		this.createdTextures = [];
		this.fallbackFormat = options.fallbackFormat ?? null;
	}
	createTexture(desc) {
		const fallback = this.fallbackFormat?.requested === desc.format ?
			this.fallbackFormat.actual
		:	null;
		const texture = {
			width: desc.width,
			height: desc.height,
			format: fallback ?? desc.format,
			requestedFormat: desc.format,
			formatFallbackReason: fallback ? "test fallback" : undefined,
			descriptor: desc,
			destroyed: false,
			destroy() {
				this.destroyed = true;
			},
		};
		this.createdTextures.push(texture);
		return texture;
	}
	createCommandEncoder() {
		return new FakeEncoder();
	}
	createBuffer() {
		return { size: 4, destroy() {} };
	}
	createSampler(desc = {}) {
		return { label: desc.label };
	}
	createTextureView(texture) {
		return { texture };
	}
	createShaderModule(desc = {}) {
		return { label: desc.label };
	}
	createComputePipeline(desc = {}) {
		return { label: desc.label };
	}
	createPipeline(desc = {}) {
		return { label: desc.label };
	}
	createBindingGroup(desc = {}) {
		return { label: desc.label };
	}
	writeBuffer() {}
	writeTexture() {}
	submit() {}
	registerExternalTexture() {}
	unregisterExternalTexture() {}
	resolveTextureForSlot() {
		return null;
	}
}

function createContext(passExecute) {
	const renderTargets = new RenderTargetRegistrySnapshot([
		{
			id: "gbuf",
			size: { mode: "fixed", width: 32, height: 16 },
			color: [
				{ format: TextureFormat.RGBA8Unorm },
				{ format: TextureFormat.RGBA16Float },
			],
			depth: { format: TextureFormat.Depth32Float },
			sampleCount: 1,
		},
	]);
	const customRenderPasses = new CustomRenderPassRegistrySnapshot([
		{
			id: "custom-gbuf",
			target: "gbuf",
			execute: passExecute,
		},
	]);
	return {
		backendProfile: {
			id: "webgpu",
			capabilities: {},
			frameScheduling: "always",
			shadow: {},
			lighting: {},
		},
		camera: {},
		attachments: { width: 640, height: 480 },
		features: {},
		postProcess: createResolvedPostProcess("webgpu"),
		renderTargets,
		customRenderPasses,
		shadowMaps: new Map(),
		scene: { shadowCasterPackets: [], shadowTransmitterPackets: [] },
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		incremental: { enabled: false, forceFullFrame: true, dirtyRects: [] },
		transient: new Map(),
	};
}

async function testWebGPUCustomTargetExecution() {
	const backend = new FakeBackend();
	const runtime = new WebGPUCustomRenderTargetRuntime(backend);
	let observed = null;
	const context = createContext((passContext) => {
		observed = passContext;
		passContext.encoder.beginRenderPass({
			colorAttachments: passContext.target.color.map((attachment) => ({
				view: attachment.texture,
				loadOp: "clear",
				storeOp: "store",
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
			})),
			depthStencilAttachment: {
				view: passContext.target.depth.texture,
				depthLoadOp: "clear",
				depthStoreOp: "store",
				depthClearValue: 1,
			},
		});
		passContext.encoder.endRenderPass();
	});
	runtime.sync(context);

	assert.equal(backend.createdTextures.length, 3);
	assert.equal(backend.createdTextures[0].width, 32);
	assert.equal(backend.createdTextures[1].format, TextureFormat.RGBA16Float);

	const encoder = new FakeEncoder();
	await runtime.executePass(
		{ stage: "custom-gbuf", executor: "backend", enabled: true, dependsOn: [] },
		context,
		encoder
	);
	assert.equal(observed.backend, "webgpu");
	assert.equal(observed.target.color.length, 2);
	assert.equal(encoder.calls[0][0], "beginRenderPass");
	await assert.rejects(() => runtime.readColor("gbuf", 0), /successful frame/);
	runtime._readbackRuntime = {
		async readTexture(options) {
			return {
				bytes: new Uint8Array(options.width * options.height * 4),
				width: options.width,
				height: options.height,
				format: options.format,
				bytesPerPixel: 4,
				bytesPerRow: options.width * 4,
				toFloat32: () => new Float32Array(),
				toRGBAFloat32: () => new Float32Array(),
				toNormalizedRGBA8Float32: () => new Float32Array(),
			};
		},
		destroy() {},
	};
	runtime.markFrameCommitted();
	const readback = await runtime.readColor("gbuf", 0, { width: 2, height: 2 });
	assert.equal(readback.origin, "top-left");
	assert.equal(readback.format, TextureFormat.RGBA8Unorm);
	await assert.rejects(() => runtime.readColor("gbuf", 0, { width: 33 }), /between/);
	runtime.destroy();
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

function testWebGPURejectsFormatFallbackTransactionally() {
	const backend = new FakeBackend({
		fallbackFormat: {
			requested: TextureFormat.RGBA16Float,
			actual: TextureFormat.RGBA8Unorm,
		},
	});
	const runtime = new WebGPUCustomRenderTargetRuntime(backend);
	assert.throws(
		() => runtime.sync(createContext(() => {})),
		/requested "rgba16float" but received "rgba8unorm"/
	);
	assert.equal(backend.createdTextures.length, 2);
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

await testWebGPUCustomTargetExecution();
testWebGPURejectsFormatFallbackTransactionally();
console.log("WebGPU custom render target tests passed");
