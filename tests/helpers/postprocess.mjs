import { resolvePostProcessState } from "../../src/pipeline/PostProcess.ts";

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
