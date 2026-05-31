import type {
	RenderPipelineBackendPassRegistration,
} from "./RenderPipelineRegistry";
import type { RendererStageDefinition } from "./RendererStageGraph";
import { hasParticleShadowCasters } from "./ParticleShadowVolume";

/**
 * Creates the built-in backend pass registrations shared by `Renderer` and
 * frame-plan tests.
 *
 * @returns Fresh backend pass registration descriptors.
 * @sideEffects None.
 */
export function createDefaultBackendPasses(): RenderPipelineBackendPassRegistration[] {
	return [
		{
			id: "particle-sim",
			dependsOn: ["prepared-scene-build"],
			enabled: (context) => context.hasParticleSystems,
			shouldRun: ({ frame }) => (frame.particleSystems?.length ?? 0) > 0,
		},
		{
			id: "shadow",
			dependsOn: ["prepared-scene-build", "particle-sim"],
			shouldRun: ({ frame, features }) =>
				features.enableShadows &&
				(frame.shadowCasterPackets.length > 0 ||
					frame.shadowTransmitterPackets.length > 0 ||
					hasParticleShadowCasters(frame.particleSystems)),
		},
		{
			id: "reflection",
			dependsOn: ["prepared-scene-build", "reflection-probe-capture"],
			shouldRun: ({ frame, features }) =>
				features.enableReflection && frame.reflectivePackets.length > 0,
		},
		{
			id: "main-opaque",
			dependsOn: ["reflection", "shadow"],
		},
		{
			id: "main-transparent",
			dependsOn: ["main-opaque"],
			shouldRun: ({ frame }) => frame.transparentPackets.length > 0,
		},
		{
			id: "particles",
			dependsOn: ["main-transparent"],
			shouldRun: ({ frame }) => (frame.particleSystems?.length ?? 0) > 0,
		},
	];
}

/**
 * Creates the built-in renderer stage graph definitions.
 *
 * @returns Fresh stage definitions in registration order.
 * @sideEffects None.
 */
export function createDefaultRendererStages(): RendererStageDefinition[] {
	return [
		{ id: "feature-resolution", dependsOn: [] },
		{ id: "environment-ibl-update", dependsOn: ["feature-resolution"] },
		{ id: "sync-in", dependsOn: ["environment-ibl-update"] },
		{
			id: "animation-sim",
			dependsOn: ["sync-in"],
			enabled: (context) => context.hasActiveAnimations,
		},
		{ id: "physics-sim", dependsOn: ["animation-sim", "sync-in"] },
		{
			id: "transform-update",
			dependsOn: ["physics-sim", "animation-sim", "sync-in"],
		},
		{ id: "lod-resolve", dependsOn: ["transform-update"] },
		{ id: "csg-resolve", dependsOn: ["lod-resolve"] },
		{ id: "prepared-scene-build", dependsOn: ["csg-resolve"] },
		{
			id: "particle-sim",
			dependsOn: ["prepared-scene-build"],
			enabled: (context) => context.hasParticleSystems,
		},
		{ id: "shadow", dependsOn: ["prepared-scene-build", "particle-sim"] },
		{
			id: "reflection-probe-capture",
			dependsOn: ["prepared-scene-build"],
		},
		{
			id: "reflection",
			dependsOn: ["prepared-scene-build", "reflection-probe-capture"],
		},
		{ id: "main-opaque", dependsOn: ["reflection", "shadow"] },
		{ id: "main-transparent", dependsOn: ["main-opaque"] },
		{ id: "particles", dependsOn: ["main-transparent"] },
		{ id: "postprocess", dependsOn: ["particles"] },
		{ id: "sync-out", dependsOn: ["postprocess"] },
	];
}
