import {
	BloomPass,
	ColorFilterPass,
	DepthOfFieldPass,
	FastApproximateAntiAliasingPass,
	FogPass,
	GammaPass,
	MotionBlurPass,
	PostProcessPass,
	PostProcessPassRegistry,
	ScreenSpaceAmbientOcclusionPass,
	ScreenSpaceGlobalIlluminationPass,
	ScreenSpaceRefractionsPass,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	ToneMappingPass,
	VolumetricLightingPass,
	createRenderBackendExtensionRegistry,
} from "../../src/index.ts";
import { BackendPostProcessRuntime } from "../../src/postprocess/BackendPostProcessRuntime.ts";
import { createSyntheticLogicalGBufferBridge } from "../../src/postprocess/GBufferBridge.ts";

export const ALL_POST_PROCESS_PASS_IDS = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"ssrefraction",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"gamma",
];

export const ALL_ENABLED_POST_PROCESS_REQUEST = {
	ssao: { enabled: true },
	ssgi: { enabled: true },
	taa: { enabled: true },
	ssr: { enabled: true },
	ssrefraction: { enabled: true },
	volumetric: { enabled: true },
	fog: { enabled: true, options: { application: "postprocess" } },
	"motion-blur": { enabled: true },
	dof: { enabled: true },
	bloom: { enabled: true },
	tonemap: { enabled: true },
	"color-filter": { enabled: true },
	fxaa: { enabled: true },
	gamma: { enabled: true },
};

const ENGINE_PASS_FACTORIES = {
	ssao: (config) => new ScreenSpaceAmbientOcclusionPass(config),
	ssgi: (config) => new ScreenSpaceGlobalIlluminationPass(config),
	taa: (config) => new TemporalAntiAliasingPass(config),
	ssr: (config) => new ScreenSpaceReflectionsPass(config),
	ssrefraction: (config) => new ScreenSpaceRefractionsPass(config),
	volumetric: (config) => new VolumetricLightingPass(config),
	fog: (config) => new FogPass(config),
	"motion-blur": (config) => new MotionBlurPass(config),
	dof: (config) => new DepthOfFieldPass(config),
	bloom: (config) => new BloomPass(config),
	tonemap: (config) => new ToneMappingPass(config),
	"color-filter": (config) => new ColorFilterPass(config),
	fxaa: (config) => new FastApproximateAntiAliasingPass(config),
	gamma: (config) => new GammaPass(config),
};

class NoopCustomPostProcessPass extends PostProcessPass {
	constructor(id, request, backendType) {
		const supplied = request?.implementation ?? request?.implementations?.[backendType];
		const implementation = supplied ?? {
			describeExecution: () => ({
				color: { access: "read-write", output: "preserve" },
			}),
			execute: () => ({ ran: true }),
		};
		super({
			id,
			schedule: {
				placement: request?.placement,
				order: request?.order,
				incremental: request?.incremental,
			},
			enabled: request?.enabled === true,
			options: request?.options ?? {},
			implementations: {
				[backendType]: () => implementation,
			},
		});
	}
}

export function createPostProcessRegistryFromRequest(
	request = {},
	backendType = "test"
) {
	const registry = new PostProcessPassRegistry();
	for (const [id, value] of Object.entries(request)) {
		const passRequest =
			value && typeof value === "object" ? value : { enabled: value === true };
		const factory = ENGINE_PASS_FACTORIES[id];
		if (factory) {
			registry.registerPass(factory(passRequest));
		} else {
			registry.registerPass(
				new NoopCustomPostProcessPass(id, passRequest, backendType)
			);
		}
	}
	return registry;
}

export function createNoopPostProcessSupport(
	backend = "test"
) {
	const executor = {
		backend,
		createdResources: [],
		destroyedResources: [],
		executedPasses: [],
		createGBufferBridge(context, options = {}) {
			if (options.resourceMode === "synthetic") {
				return createSyntheticLogicalGBufferBridge(context, {
					backend,
					normalSpace: "world",
					depthEncoding: "linear-view-z",
					motionEncoding: "ndc-delta",
				});
			}
			const attachments = context.attachments;
			const width = attachments.width ?? 1;
			const height = attachments.height ?? 1;
			return {
				width,
				height,
				normalSpace: "world",
				depthEncoding: "linear-view-z",
				motionEncoding: "ndc-delta",
				channels: {
					color: {
						semantic: "color",
						width,
						height,
						handle: {
							backend,
							resource: attachments.pixels ?? null,
						},
					},
					depth: {
						semantic: "depth",
						width,
						height,
						handle: {
							backend,
							resource: attachments.depthBuffer ?? null,
						},
					},
					normal: {
						semantic: "normal",
						width,
						height,
						handle: {
							backend,
							resource: attachments.normalBuffer ?? null,
						},
					},
					albedo: {
						semantic: "albedo",
						width,
						height,
						handle: {
							backend,
							resource: attachments.albedoBuffer ?? attachments.pixels ?? null,
						},
					},
					motion: {
						semantic: "motion",
						width,
						height,
						handle: {
							backend,
							resource: attachments.motionBuffer ?? null,
						},
					},
				},
				worldPosition: {
					source: "derived",
					available: Boolean(attachments.depthBuffer),
				},
			};
		},
		createResource(desc) {
			const handle = {
				id: desc.id,
				backend,
				width: desc.width,
				height: desc.height,
				format: desc.format,
				resource: {
					id: desc.id,
					width: desc.width,
					height: desc.height,
					format: desc.format,
					usage: desc.usage,
				},
			};
			this.createdResources.push(handle);
			return handle;
		},
		destroyResource(handle) {
			this.destroyedResources.push(handle);
		},
		executePass(passId) {
			this.executedPasses.push(passId);
			return { ran: true };
		},
	};
	return {
		executor,
		createGBufferBridge: (context) => executor.createGBufferBridge(context),
	};
}

export function createNoopPostProcessAdapter(
	backend = "test"
) {
	const support = createNoopPostProcessSupport(backend);
	return {
		...support,
		adapter: support.executor,
	};
}

export function installNoopPostProcessAdapter(
	target,
	backend = "test"
) {
	const support = createNoopPostProcessAdapter(backend);
	const runtime = new BackendPostProcessRuntime({
		executor: support.executor,
		backend: target,
	});
	support.runtime = runtime;
	if (target.capabilities && typeof target.capabilities === "object") {
		target.capabilities.postProcess = true;
	}
	const originalExecutePass = target.executePass?.bind(target);
	const originalEndFrame = target.endFrame?.bind(target);
	const originalAbortFrame = target.abortFrame?.bind(target);
	target.executePass = async (pass, context) => {
		if (pass.stage === "postprocess") {
			if (Array.isArray(target.executedPasses)) {
				target.executedPasses.push(pass.stage);
			}
			if (Array.isArray(target.executionEvents)) {
				target.executionEvents.push(["backend", pass.stage]);
			}
			await runtime.execute(context);
			return;
		}
		return originalExecutePass?.(pass, context);
	};
	target.endFrame = async () => {
		const result = await originalEndFrame?.();
		runtime.commitFrame();
		return result;
	};
	target.abortFrame = async (error) => {
		await runtime.abortFrame(error);
		return originalAbortFrame?.(error);
	};
	Object.defineProperty(target, "extensions", {
		configurable: true,
		value: createRenderBackendExtensionRegistry([]),
	});
	return support;
}

export function createResolvedPostProcess(
	request = {},
	backendType = "test"
) {
	return createPostProcessRegistryFromRequest(request, backendType).createSnapshot(
		backendType
	);
}

export function createAllEnabledPostProcess(
	request = {},
	backendType = "test"
) {
	return createResolvedPostProcess(
		{
			...ALL_ENABLED_POST_PROCESS_REQUEST,
			...request,
		},
		backendType
	);
}
