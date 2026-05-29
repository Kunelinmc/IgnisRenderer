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
	registerPostProcessBackendAdapter,
	ScreenSpaceAmbientOcclusionPass,
	ScreenSpaceGlobalIlluminationPass,
	ScreenSpaceReflectionsPass,
	TemporalAntiAliasingPass,
	ToneMappingPass,
	VolumetricLightingPass,
} from "../../src/index.ts";

export const ALL_POST_PROCESS_PASS_IDS = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
];

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
	backend = "test"
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

export function createNoopPostProcessAdapter(
	backend = "test"
) {
	const support = createNoopPostProcessSupport(backend);
	const adapter = {
		backend,
		executor: support.executor,
		createGBufferBridge: (context) => support.createGBufferBridge(context),
	};
	return {
		...support,
		adapter,
	};
}

export function installNoopPostProcessAdapter(
	target,
	backend = "test"
) {
	const support = createNoopPostProcessAdapter(backend);
	registerPostProcessBackendAdapter(target, support.adapter);
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
