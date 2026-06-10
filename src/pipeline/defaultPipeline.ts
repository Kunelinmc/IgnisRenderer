import type {
	RenderPipelineStageRegistration,
} from "./RenderPipelineRegistry";
import { hasParticleShadowCasters } from "./ParticleShadowVolume";
import { hasPostProcessExecutionPasses } from "../postprocess";

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
		{
			id: "environment-ibl-update",
			kind: "renderer",
			dependsOn: ["feature-resolution"],
		},
		{ id: "sync-in", kind: "renderer", dependsOn: ["environment-ibl-update"] },
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
			shouldRun: ({ frame }) => (frame.particleSystems?.length ?? 0) > 0,
		},
		{
			id: "shadow",
			kind: "backend-pass",
			dependsOn: ["prepared-scene-build", "particle-sim"],
			shouldRun: ({ frame, features }) =>
				features.enableShadows &&
				(frame.shadowCasterPackets.length > 0 ||
					frame.shadowTransmitterPackets.length > 0 ||
					hasParticleShadowCasters(frame.particleSystems)),
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
			shouldRun: ({ frame, features }) =>
				features.enableReflection && frame.reflectivePackets.length > 0,
		},
		{
			id: "main-opaque",
			kind: "backend-pass",
			dependsOn: ["reflection", "shadow"],
		},
		{
			id: "main-transparent",
			kind: "backend-pass",
			dependsOn: ["main-opaque"],
			shouldRun: ({ frame }) => frame.transparentPackets.length > 0,
		},
		{
			id: "particles",
			kind: "backend-pass",
			dependsOn: ["main-transparent"],
			shouldRun: ({ frame }) => (frame.particleSystems?.length ?? 0) > 0,
		},
		{
			id: "postprocess",
			kind: "backend-pass",
			dependsOn: ["particles"],
			shouldRun: ({
				backendCapabilities,
				backendType,
				frameContext,
				postProcess,
			}) =>
				backendCapabilities?.postProcess === true &&
				hasPostProcessExecutionPasses(postProcess, {
					backend: backendType,
					frameContext,
				}),
		},
		{ id: "sync-out", kind: "renderer", dependsOn: ["postprocess"] },
	];
}
