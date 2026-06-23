import assert from "node:assert/strict";
import { WebGPUCustomRenderTargetRuntime } from "../../../src/renderers/webgpu/rendergraph/WebGPUCustomRenderTargetRuntime.ts";
import { TextureFormat } from "../../../src/renderers/types.ts";
import {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../../../src/renderers/CustomRenderTargets.ts";
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
	constructor() {
		this.createdTextures = [];
	}
	createTexture(desc) {
		const texture = {
			width: desc.width,
			height: desc.height,
			format: desc.format,
			requestedFormat: desc.format,
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
	runtime.destroy();
	assert.equal(backend.createdTextures.every((texture) => texture.destroyed), true);
}

await testWebGPUCustomTargetExecution();
console.log("WebGPU custom render target tests passed");
