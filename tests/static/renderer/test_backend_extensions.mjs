import assert from "node:assert/strict";

import {
	RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
	RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
	WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
	createRenderBackendExtensionRegistry,
	resolveOcclusionCullingBackendExtension,
} from "../../../src/backends/BackendExtensions.ts";
import { RendererOcclusionCullingController } from "../../../src/rendering/RendererOcclusionCullingController.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import { attachBackend } from "../../helpers/TestRenderBackend.mjs";

function testRegistryRejectsDuplicateIds() {
	assert.throws(
		() =>
			createRenderBackendExtensionRegistry([
				{
					id: "duplicate",
					insertionPoints: ["renderer:test"],
					api: {},
				},
				{
					id: "duplicate",
					insertionPoints: ["renderer:test"],
					api: {},
				},
			]),
		/Duplicate render backend extension id/
	);
}

function testTypedResolversUseExtensionRegistry() {
	const occlusionApi = {
		getVisibilityProvider: () => ({
			sourceFrameIndex: 1,
			isPacketVisible: () => true,
		}),
	};
	const backend = {
		type: "test",
		extensions: createRenderBackendExtensionRegistry([
			{
				id: RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
				insertionPoints: [RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT],
				api: occlusionApi,
			},
		]),
	};

	assert.equal(resolveOcclusionCullingBackendExtension(backend).api, occlusionApi);
}

function testOcclusionControllerUsesExtensionApi() {
	let resetCalls = 0;
	let providerRequests = 0;
	const provider = {
		sourceFrameIndex: 7,
		isPacketVisible: () => true,
	};
	const backend = {
		type: "test",
		extensions: createRenderBackendExtensionRegistry([
			{
				id: RENDERER_OCCLUSION_CULLING_EXTENSION_ID,
				insertionPoints: [RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT],
				api: {
					getVisibilityProvider: (options) => {
						providerRequests++;
						assert.equal(options.hysteresisFrames, 4);
						return provider;
					},
					resetOcclusionCulling: () => {
						resetCalls++;
					},
				},
			},
		]),
	};
	const controller = new RendererOcclusionCullingController(backend);

	assert.equal(
		controller.getVisibilityProvider({
			enableOcclusionCulling: false,
			occlusionCullingOptions: { hysteresisFrames: 4 },
		}),
		null
	);
	assert.equal(providerRequests, 0);
	assert.equal(
		controller.getVisibilityProvider({
			enableOcclusionCulling: true,
			occlusionCullingOptions: { hysteresisFrames: 4 },
		}),
		provider
	);
	controller.reset();
	assert.equal(providerRequests, 1);
	assert.equal(resetCalls, 1);
}

function testWebGPURegistersExpectedExtensions() {
	const session = attachBackend(new WebGPUBackend());
	const extensions = session.extensions.listExtensions();
	assert.equal(extensions.length, 2);
	assert.deepEqual(
		resolveOcclusionCullingBackendExtension(session).insertionPoints,
		[
			RENDERER_OCCLUSION_VISIBILITY_INSERTION_POINT,
			WEBGPU_OCCLUSION_AFTER_DEPTH_INSERTION_POINT,
		]
	);
}

testRegistryRejectsDuplicateIds();
testTypedResolversUseExtensionRegistry();
testOcclusionControllerUsesExtensionApi();
testWebGPURegistersExpectedExtensions();

console.log("Backend extension registry tests passed");
