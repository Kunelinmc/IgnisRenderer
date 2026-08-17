import assert from "node:assert/strict";

import { Texture } from "../../../src/core/Texture.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { IBLPrefilter } from "../../../src/lights/ibl/IBLPrefilter.ts";
import { WebGLPrefilterExecutor } from "../../../src/lights/ibl/WebGLPrefilterExecutor.ts";
import { captureIBLPrefilterSourceRevision } from "../../../src/lights/ibl/IBLPrefilterExecutor.ts";

function createTexture() {
	return new Texture({
		data: new Float32Array([1, 1, 1, 1]),
		width: 1,
		height: 1,
		colorSpace: "HDR",
	});
}

function createRequest(texture = createTexture()) {
	return {
		envMap: texture,
		plan: {
			baseWidth: 1,
			baseHeight: 1,
			mipLevels: [{ level: 0, width: 1, height: 1, roughness: 0 }],
		},
		sourceRevision: captureIBLPrefilterSourceRevision(texture),
	};
}

function createRasterFacade(options = {}) {
	const calls = [];
	const resources = {
		createTexture(desc) {
			const texture = { ...desc, requestedFormat: desc.format, destroy() {} };
			calls.push(["texture", desc.label]);
			return texture;
		},
		createSampler() { return {}; },
		createShaderModule(desc) { return { code: desc.code }; },
		createRenderPipeline(desc) { return { desc }; },
		createBindingGroup(desc) { return { desc }; },
		createBuffer() { throw new Error("not used"); },
		writeTexture(_texture, _data, layout) {
			calls.push(["write", layout.mipLevel ?? 0]);
		},
		async readTexture() {
			return {
				toRGBAFloat32: () => new Float32Array([2, 1, 0.5, 1]),
			};
		},
	};
	const encoder = {
		beginRenderPass() { calls.push(["begin"]); },
		setPipeline() {},
		setBindingGroup() {},
		setViewport() {},
		setUniforms(uniforms) { calls.push(["uniforms", uniforms.length]); },
		draw(count) { calls.push(["draw", count]); },
		endRenderPass() { calls.push(["end"]); },
	};
	return {
		calls,
		facade: {
			getAvailability(request) {
				calls.push(["availability", request]);
				return options.availability ?? {
					state: "ready",
					acceptsRequests: true,
					reason: null,
				};
			},
			async execute(request) {
				calls.push(["execute", request]);
				if (options.error) throw options.error;
				return request.task({
					generation: 1,
					signal: request.signal ?? new AbortController().signal,
					resources,
					encoder,
				});
			},
		},
	};
}

async function testAdapterUsesAuxiliaryRasterCapability() {
	const raster = createRasterFacade();
	const completed = [];
	const request = createRequest();
	request.onMipComplete = (level) => completed.push(level);
	const result = await new WebGLPrefilterExecutor(raster.facade).execute(request);
	assert.equal(result.length, 1);
	assert.deepEqual(Array.from(result[0].data), [2, 1, 0.5, 1]);
	assert.deepEqual(completed, [0]);
	const executeRequest = raster.calls.find(([name]) => name === "execute")[1];
	assert.equal(executeRequest.framePolicy, "between-passes");
	assert.equal(executeRequest.contextLossPolicy, "retain-pending");
	assert.deepEqual(executeRequest.requiredExtensions, ["EXT_color_buffer_float"]);
	assert.deepEqual(executeRequest.alternativeExtensionGroups, [[
		"OES_texture_float_linear",
		"OES_texture_half_float_linear",
	]]);
	assert.ok(raster.calls.some(([name, count]) => name === "draw" && count === 3));
}

async function testAdapterPropagatesAvailabilityAndFailure() {
	const unavailable = createRasterFacade({
		availability: {
			state: "temporarily-unavailable",
			acceptsRequests: true,
			reason: "context lost",
		},
	});
	const executor = new WebGLPrefilterExecutor(unavailable.facade);
	assert.equal(executor.getAvailability().state, "temporarily-unavailable");

	const failed = createRasterFacade({ error: new Error("raster failed") });
	await assert.rejects(
		new WebGLPrefilterExecutor(failed.facade).execute(createRequest()),
		/raster failed/,
	);
}

function createQueuedBackendServices(id, calls) {
	return {
		frame: {},
		auxiliaryRaster: {
			hasExtension: () => true,
			async execute() {
				calls.push(id);
				return [{
					level: 0,
					width: 1,
					height: 1,
					data: new Float32Array([1, 1, 1, 1]),
				}];
			},
			destroy() {},
		},
		restoreContextWorkBaseline() {},
		destroy() {},
	};
}

async function testExplicitLostRequestRestoresWithNewGeneration() {
	const calls = [];
	const backend = new WebGLBackend();
	backend._contextServices = createQueuedBackendServices("old", calls);
	backend._contextWorkQueue.bindContext();
	backend._contextLost = true;
	backend._contextWorkQueue.suspend();
	const request = new IBLPrefilter({ backend }).prefilter(createTexture(), {
		acceleration: "webgl",
		maxMipLevels: 1,
	});
	let settled = false;
	void request.finally(() => { settled = true; });
	await Promise.resolve();
	assert.equal(settled, false);
	backend._contextServices = createQueuedBackendServices("restored", calls);
	backend._contextLost = false;
	backend._contextWorkQueue.bindContext();
	await request;
	assert.deepEqual(calls, ["restored"]);
	backend.destroy();
}

async function testAutoFallsBackWhileContextLost() {
	const calls = [];
	const backend = new WebGLBackend();
	backend._contextServices = createQueuedBackendServices("old", calls);
	backend._contextWorkQueue.bindContext();
	backend._contextLost = true;
	backend._contextWorkQueue.suspend();
	const result = await new IBLPrefilter({ backend }).prefilter(createTexture(), {
		acceleration: "auto",
		maxMipLevels: 1,
	});
	assert.equal(result.mipmaps.length, 1);
	assert.equal(calls.length, 0);
	backend.destroy();
}

await testAdapterUsesAuxiliaryRasterCapability();
await testAdapterPropagatesAvailabilityAndFailure();
await testExplicitLostRequestRestoresWithNewGeneration();
await testAutoFallsBackWhileContextLost();

console.log("WebGL IBL prefilter executor tests passed");
