import type {
	RenderPipelineStageRegistration,
} from "./RenderPipelineRegistry";

export const DEFAULT_RENDERER_STAGE_IDS = {
	featureResolution: "feature-resolution",
	syncIn: "sync-in",
	animationSim: "animation-sim",
	physicsSim: "physics-sim",
	transformUpdate: "transform-update",
	lodResolve: "lod-resolve",
	csgResolve: "csg-resolve",
	deformationUpdate: "deformation-update",
	preparedSceneBuild: "prepared-scene-build",
	probeCapture: "probe-capture",
	syncOut: "sync-out",
} as const;

export type DefaultRendererStageId =
	(typeof DEFAULT_RENDERER_STAGE_IDS)[keyof typeof DEFAULT_RENDERER_STAGE_IDS];

const rendererStageIds = DEFAULT_RENDERER_STAGE_IDS;

/** Built-in pipeline stage registrations shared by renderer runtimes. */
export const DEFAULT_PIPELINE_STAGES = [
	{
		id: rendererStageIds.featureResolution,
		kind: "renderer",
		dependsOn: [],
	},
	{
		id: rendererStageIds.syncIn,
		kind: "renderer",
		dependsOn: [rendererStageIds.featureResolution],
	},
	{
		id: rendererStageIds.animationSim,
		kind: "renderer",
		dependsOn: [rendererStageIds.syncIn],
		enabled: (context) => context.hasActiveAnimations,
	},
	{
		id: rendererStageIds.physicsSim,
		kind: "renderer",
		dependsOn: [rendererStageIds.animationSim, rendererStageIds.syncIn],
	},
	{
		id: rendererStageIds.transformUpdate,
		kind: "renderer",
		dependsOn: [
			rendererStageIds.physicsSim,
			rendererStageIds.animationSim,
			rendererStageIds.syncIn,
		],
	},
	{
		id: rendererStageIds.lodResolve,
		kind: "renderer",
		dependsOn: [rendererStageIds.transformUpdate],
	},
	{
		id: rendererStageIds.csgResolve,
		kind: "renderer",
		dependsOn: [rendererStageIds.lodResolve],
	},
	{
		id: rendererStageIds.deformationUpdate,
		kind: "renderer",
		dependsOn: [rendererStageIds.csgResolve],
	},
	{
		id: rendererStageIds.preparedSceneBuild,
		kind: "renderer",
		dependsOn: [rendererStageIds.deformationUpdate],
	},
	{
		id: "particle-sim",
		kind: "backend-pass",
		dependsOn: [rendererStageIds.preparedSceneBuild],
		enabled: (context) => context.hasParticleSystems,
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("particle-sim"),
	},
	{
		id: "shadow",
		kind: "backend-pass",
		dependsOn: [rendererStageIds.preparedSceneBuild, "particle-sim"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("shadow"),
	},
	{
		id: rendererStageIds.probeCapture,
		kind: "renderer",
		dependsOn: [rendererStageIds.preparedSceneBuild],
	},
	{
		id: "render-target-views",
		kind: "backend-pass",
		dependsOn: ["particle-sim", "shadow", rendererStageIds.probeCapture],
		shouldRun: ({ frameContext }) =>
			(frameContext?.renderTargetJobs?.size ?? 0) > 0,
	},
	{
		id: "reflection",
		kind: "backend-pass",
		dependsOn: [rendererStageIds.preparedSceneBuild, "render-target-views"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("reflection"),
	},
	{
		id: "main-opaque",
		kind: "backend-pass",
		dependsOn: ["reflection", "shadow"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("main-opaque"),
	},
	{
		id: "main-transparent",
		kind: "backend-pass",
		dependsOn: ["main-opaque"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("main-transparent"),
	},
	{
		id: "particles",
		kind: "backend-pass",
		dependsOn: ["main-transparent"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("particles"),
	},
	{
		id: "postprocess",
		kind: "backend-pass",
		dependsOn: ["particles"],
		shouldRun: ({ requirements }) =>
			requirements.requiredPasses.has("postprocess"),
	},
	{
		id: rendererStageIds.syncOut,
		kind: "renderer",
		dependsOn: ["postprocess"],
	},
] as const satisfies readonly RenderPipelineStageRegistration[];
