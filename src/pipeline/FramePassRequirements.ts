import { resolveParticleRenderIntent } from "../particles/ParticleRenderIntent";
import { hasParticleShadowCasters } from "./ParticleShadowVolume";
import type {
	FramePassStage,
	PreparedScene,
	ResolvedFeatureState,
} from "./types";

/**
 * Backend-resolved support used by portable frame-pass requirement analysis.
 *
 * @internal `FrameCoordinator` derives this view from the attached backend.
 * Applications should use `Renderer` feature and particle configuration.
 */
export interface FramePassRenderSupport {
	readonly meshParticles: boolean;
	readonly postProcess: boolean;
}

/**
 * Immutable-by-convention backend-pass requirements for one renderer frame.
 *
 * @internal `FrameCoordinator` owns requirement resolution. Applications should
 * render through `Renderer` instead of constructing frame plans directly.
 */
export interface FramePassRequirements {
	readonly requiredPasses: ReadonlySet<FramePassStage>;
}

/**
 * Inputs used to resolve portable backend-pass requirements.
 *
 * @internal `FrameCoordinator` owns this renderer orchestration contract.
 */
export interface FramePassRequirementResolveOptions {
	readonly frame: PreparedScene;
	readonly features: ResolvedFeatureState;
	readonly support: FramePassRenderSupport;
	readonly hasPostProcessWork: boolean;
}

/**
 * Resolves renderer-owned backend-pass requirements without inspecting a
 * backend identifier.
 *
 * @internal `FrameCoordinator` owns this call. Applications should use
 * `Renderer.renderFrame()`.
 */
export function resolveFramePassRequirements(
	options: FramePassRequirementResolveOptions,
): FramePassRequirements {
	const { frame, features, support } = options;
	const requiredPasses = new Set<FramePassStage>();
	const hasParticleSystems = (frame.particleSystems?.length ?? 0) > 0;
	const particleIntent = resolveParticleRenderIntent(frame.particleSystems);

	requiredPasses.add("main-opaque");
	if (hasParticleSystems) {
		requiredPasses.add("particle-sim");
		requiredPasses.add("particles");
	}
	if (
		features.enableShadows &&
		(frame.shadowCasterPackets.length > 0 ||
			frame.shadowTransmitterPackets.length > 0 ||
			hasParticleShadowCasters(frame.particleSystems) ||
			(support.meshParticles &&
				particleIntent.hasShadowCastingMeshTemplates))
	) {
		requiredPasses.add("shadow");
	}
	if (features.enableReflection && frame.reflectivePackets.length > 0) {
		requiredPasses.add("reflection");
	}
	if (
		frame.transparentPackets.length > 0 ||
		(support.meshParticles && particleIntent.hasTransparentMeshTemplates)
	) {
		requiredPasses.add("main-transparent");
	}
	if (support.postProcess && options.hasPostProcessWork) {
		requiredPasses.add("postprocess");
	}

	return { requiredPasses };
}
