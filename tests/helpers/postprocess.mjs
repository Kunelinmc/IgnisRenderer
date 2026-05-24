import {
	BloomPass,
	ColorFilterPass,
	DepthOfFieldPass,
	FastApproximateAntiAliasingPass,
	FogPass,
	GammaPass,
	InteractionOutlinePass,
	MotionBlurPass,
	PostProcessPass,
	PostProcessPassRegistry,
	ScreenSpaceAmbientOcclusionPass,
	ScreenSpaceGlobalIlluminationPass,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	ToneMappingPass,
	VolumetricLightingPass,
} from "../../src/index.ts";

export const ALL_POST_PROCESS_CAPABILITIES = {
	ssao: true,
	ssgi: true,
	taa: true,
	ssr: true,
	volumetric: true,
	fog: true,
	"motion-blur": true,
	dof: true,
	bloom: true,
	tonemap: true,
	"color-filter": true,
	fxaa: true,
	"interaction-outline": true,
	gamma: true,
};

export const ALL_ENABLED_POST_PROCESS_REQUEST = {
	ssao: { enabled: true },
	ssgi: { enabled: true },
	taa: { enabled: true },
	ssr: { enabled: true },
	volumetric: { enabled: true },
	fog: { enabled: true, options: { application: "postprocess" } },
	"motion-blur": { enabled: true },
	dof: { enabled: true },
	bloom: { enabled: true },
	tonemap: { enabled: true },
	"color-filter": { enabled: true },
	fxaa: { enabled: true },
	"interaction-outline": { enabled: true },
	gamma: { enabled: true },
};

const BUILTIN_PASS_FACTORIES = {
	ssao: (config) => new ScreenSpaceAmbientOcclusionPass(config),
	ssgi: (config) => new ScreenSpaceGlobalIlluminationPass(config),
	taa: (config) => new TemporalAntiAliasingPass(config),
	ssr: (config) => new ScreenSpaceReflectionsPass(config),
	volumetric: (config) => new VolumetricLightingPass(config),
	fog: (config) => new FogPass(config),
	"motion-blur": (config) => new MotionBlurPass(config),
	dof: (config) => new DepthOfFieldPass(config),
	bloom: (config) => new BloomPass(config),
	tonemap: (config) => new ToneMappingPass(config),
	"color-filter": (config) => new ColorFilterPass(config),
	fxaa: (config) => new FastApproximateAntiAliasingPass(config),
	"interaction-outline": (config) => new InteractionOutlinePass(config),
	gamma: (config) => new GammaPass(config),
};

class NoopCustomPostProcessPass extends PostProcessPass {
	constructor(id, request, backendType) {
		super({
			id,
			placement: request?.placement,
			order: request?.order,
			enabled: request?.enabled === true,
			options: request?.options ?? {},
			incremental: request?.incremental,
			implementations: {
				[backendType]: request?.implementation ?? {},
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
		const factory = BUILTIN_PASS_FACTORIES[id];
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
	backend = "test",
	capabilities = ALL_POST_PROCESS_CAPABILITIES
) {
	const executor = {
		backend,
		createdResources: [],
		destroyedResources: [],
		executedPasses: [],
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
		capabilities,
		executor,
		createGBufferBridge(context) {
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
	};
}

export function installNoopPostProcessSupport(
	target,
	backend = "test",
	capabilities = ALL_POST_PROCESS_CAPABILITIES
) {
	const support = createNoopPostProcessSupport(backend, capabilities);
	target.postProcessCapabilities = support.capabilities;
	target.postProcessExecutor = support.executor;
	target.createPostProcessGBufferBridge = (context) =>
		support.createGBufferBridge(context);
	return support;
}

export function createResolvedPostProcess(
	request = {},
	capabilities = ALL_POST_PROCESS_CAPABILITIES,
	backendType = "test"
) {
	return createPostProcessRegistryFromRequest(request, backendType).createSnapshot(
		capabilities,
		backendType
	);
}

export function createAllEnabledPostProcess(
	request = {},
	capabilities = ALL_POST_PROCESS_CAPABILITIES,
	backendType = "test"
) {
	return createResolvedPostProcess(
		{
			...ALL_ENABLED_POST_PROCESS_REQUEST,
			...request,
		},
		capabilities,
		backendType
	);
}
