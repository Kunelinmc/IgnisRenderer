import assert from "node:assert/strict";
import { WebGPUFrameExecutor } from "../src/renderers/webgpu/WebGPUFrameExecutor.ts";

class FakeBackend {
	constructor() {
		this.device = {
			limits: {
				maxColorAttachments: 8,
				maxColorAttachmentBytesPerSample: 64,
			},
		};
		this.canvasFormat = "rgba8unorm";
		this.createTextureCalls = 0;
		this.createCommandEncoderCalls = 0;
		this.failTextureAtCall = null;
		this.bindingGroupDestroyCalls = 0;
		this.canvasColorTexture = { destroy() {} };
		this.canvasDepthTexture = { destroy() {} };
	}

	getMSAASampleCount() {
		return 1;
	}

	createTexture(desc) {
		this.createTextureCalls++;
		if (
			typeof this.failTextureAtCall === "number" &&
			this.createTextureCalls >= this.failTextureAtCall
		) {
			throw new Error("simulated allocation failure");
		}
		return {
			width: desc.width,
			height: desc.height,
			destroy() {},
		};
	}

	createCommandEncoder() {
		this.createCommandEncoderCalls++;
		return {
			beginRenderPass() {},
			setPipeline() {},
			setBindingGroup() {},
			draw() {},
			endRenderPass() {},
			finish() {
				return {
					_gpuCommandBuffer: {},
					_ownerToken: {},
					_submitted: false,
				};
			},
		};
	}

	submit() {}
	writeBuffer() {}
	createBuffer() {
		return {
			size: 16,
			destroy() {},
		};
	}
	createSampler() {
		return {};
	}
	async createShaderModule() {
		return {};
	}
	createPipeline() {
		return {};
	}

	createComputePipeline(desc) {
		return this.createPipeline(desc);
	}
	createBindingGroup() {
		return {
			destroy: () => {
				this.bindingGroupDestroyCalls++;
			},
		};
	}

	createTextureView(texture, desc) {
		return { texture, desc: desc ?? null };
	}

	getTextureForSlot() {
		return {
			width: 1,
			height: 1,
			destroy() {},
		};
	}

	registerExternalTexture() {}

	unregisterExternalTexture() {}

	getCanvasColorTexture() {
		return this.canvasColorTexture;
	}

	getCanvasDepthTexture() {
		return this.canvasDepthTexture;
	}
}

function createResourcesStub() {
	return {
		sceneFrameLayout: {},
		setSceneTargetMode() {},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getSkyboxResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {},
	};
}

function createModeTrackingResourcesStub() {
	const state = {
		mode: "single",
		modeTransitions: [],
		skyboxModeAtRequest: null,
		drawModeAtRequest: null,
	};
	return {
		sceneFrameLayout: {},
		setSceneTargetMode(mode) {
			state.mode = mode;
			state.modeTransitions.push(mode);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getSkyboxResources() {
			state.skyboxModeAtRequest = state.mode;
			return {
				pipeline: {},
				frameBinding: {},
			};
		},
		async getDrawResources() {
			state.drawModeAtRequest = state.mode;
			return null;
		},
		async renderParticles() {},
		_state: state,
	};
}

function createFrameContext(width, height) {
	return {
		camera: {},
		attachments: { width, height },
		features: {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: true,
			enableTAA: true,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			warnings: [],
			ssrOptions: {},
			ssaoOptions: {},
			taaOptions: {},
			volumetricOptions: {},
		},
		shadowMaps: new Map(),
		scene: {
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
	};
}

async function testZeroSizedFrameSkipsEncoderAndLegacyDepthPath() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	const context = createFrameContext(0, 0);

	executor.beginFrame(context);
	assert.equal(backend.createCommandEncoderCalls, 0);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	await executor.endFrame();
	assert.equal(executor._texturePoolOwners.size, 0);
}

function testFrameTargetAllocationFailureReleasesPartialResources() {
	const backend = new FakeBackend();
	backend.failTextureAtCall = 4;
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());

	assert.throws(
		() => executor._ensureFrameTargets(64, 64, 2, 2),
		/simulated allocation failure/
	);
	assert.equal(executor._texturePoolOwners.size, 0);
	assert.equal(executor._frameTargets, null);
	assert.equal(executor._msaaTargets, null);
}

function testInvalidateFrameTargetsDestroysPresentBinding() {
	const backend = new FakeBackend();
	const executor = new WebGPUFrameExecutor(backend, createResourcesStub());
	executor._presentBinding = {
		destroy() {
			backend.bindingGroupDestroyCalls++;
		},
	};

	executor.invalidateFrameTargets();

	assert.equal(backend.bindingGroupDestroyCalls, 1);
	assert.equal(executor._presentBinding, null);
}

async function testLegacyMainPassForcesSingleSceneTargetMode() {
	const backend = new FakeBackend();
	const resources = createModeTrackingResourcesStub();
	const executor = new WebGPUFrameExecutor(backend, resources);
	const context = createFrameContext(64, 64);
	context.scene.opaquePackets = [{ id: "packet" }];

	executor.beginFrame(context);
	executor._frameTargets = null;

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.equal(resources._state.skyboxModeAtRequest, "single");
	assert.equal(resources._state.drawModeAtRequest, "single");
}

async function run() {
	await testZeroSizedFrameSkipsEncoderAndLegacyDepthPath();
	testFrameTargetAllocationFailureReleasesPartialResources();
	testInvalidateFrameTargetsDestroysPresentBinding();
	await testLegacyMainPassForcesSingleSceneTargetMode();
	console.log("WebGPU frame executor resilience tests passed");
}

await run();
