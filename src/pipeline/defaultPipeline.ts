import type {
	RenderPipelineStageRegistration,
} from "./RenderPipelineRegistry";

/**
 * Creates the built-in renderer pipeline stage registrations shared by
 * `Renderer` and frame-plan tests.
 *
 * @returns Fresh pipeline stage registration descriptors.
 * @sideEffects None.
 */
export function createDefaultPipelineStages(): RenderPipelineStageRegistration[] {
	return [
		{ id: "feature-resolution", kind: "renderer", dependsOn: [] },
		{ id: "sync-in", kind: "renderer", dependsOn: ["feature-resolution"] },
		{
			id: "animation-sim",
			kind: "renderer",
			dependsOn: ["sync-in"],
			enabled: (context) => context.hasActiveAnimations,
		},
		{
			id: "physics-sim",
			kind: "renderer",
			dependsOn: ["animation-sim", "sync-in"],
		},
		{
			id: "transform-update",
			kind: "renderer",
			dependsOn: ["physics-sim", "animation-sim", "sync-in"],
		},
		{ id: "lod-resolve", kind: "renderer", dependsOn: ["transform-update"] },
		{ id: "csg-resolve", kind: "renderer", dependsOn: ["lod-resolve"] },
		{
			id: "prepared-scene-build",
			kind: "renderer",
			dependsOn: ["csg-resolve"],
		},
		{
			id: "particle-sim",
			kind: "backend-pass",
			dependsOn: ["prepared-scene-build"],
			enabled: (context) => context.hasParticleSystems,
			shouldRun: ({ requirements }) =>
				requirements.requiredPasses.has("particle-sim"),
		},
		{
			id: "shadow",
			kind: "backend-pass",
			dependsOn: ["prepared-scene-build", "particle-sim"],
			shouldRun: ({ requirements }) =>
				requirements.requiredPasses.has("shadow"),
		},
		{
			id: "probe-capture",
			kind: "renderer",
			dependsOn: ["prepared-scene-build"],
		},
		{
			id: "reflection",
			kind: "backend-pass",
			dependsOn: ["prepared-scene-build", "probe-capture"],
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
		{ id: "sync-out", kind: "renderer", dependsOn: ["postprocess"] },
	];
}
