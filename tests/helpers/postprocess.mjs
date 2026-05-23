import { resolvePostProcessState } from "../../src/pipeline/PostProcessController.ts";

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

export function createNoopPostProcessSupport(
	backend = "test",
	capabilities = ALL_POST_PROCESS_CAPABILITIES
) {
	const executor = {
		backend,
		capabilities,
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
	return resolvePostProcessState(request, capabilities, backendType);
}

export function createAllEnabledPostProcess(
	request = {},
	capabilities = ALL_POST_PROCESS_CAPABILITIES,
	backendType = "test"
) {
	return resolvePostProcessState(
		{
			...ALL_ENABLED_POST_PROCESS_REQUEST,
			...request,
		},
		capabilities,
		backendType
	);
}
