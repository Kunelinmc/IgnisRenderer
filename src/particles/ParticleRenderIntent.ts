import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import { isMaterialTransparentPass } from "../materials/transparency";
import type { ParticleSystem } from "./ParticleSystem";

/**
 * Backend-neutral render-pass intent derived from particle templates.
 *
 * @internal The renderer frame requirement resolver owns this analysis.
 * Particle render backends should consume emitted packets instead.
 */
export interface ParticleRenderIntent {
	readonly hasTransparentMeshTemplates: boolean;
	readonly hasShadowCastingMeshTemplates: boolean;
}

/**
 * Describes render passes that visible mesh-particle templates may require.
 *
 * @internal The renderer frame requirement resolver owns this pre-simulation
 * analysis. Applications should configure particle templates through
 * `ParticleSystem`.
 */
export function resolveParticleRenderIntent(
	systems: readonly ParticleSystem[] | null | undefined,
): ParticleRenderIntent {
	let hasTransparentMeshTemplates = false;
	let hasShadowCastingMeshTemplates = false;

	for (const system of systems ?? []) {
		if (system.visible === false) continue;
		for (const template of system.templates ?? []) {
			if (template.shape.kind !== "mesh") continue;
			for (const primitive of template.shape.mesh.primitives) {
				if (primitive.visible === false) continue;
				if (isMaterialTransparentPass(primitive.material)) {
					hasTransparentMeshTemplates = true;
				}
				if (
					(template.castShadows ?? true) &&
					primitive.castShadows !== false &&
					(primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
						DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
				) {
					hasShadowCastingMeshTemplates = true;
				}
				if (
					hasTransparentMeshTemplates &&
					hasShadowCastingMeshTemplates
				) {
					return {
						hasTransparentMeshTemplates,
						hasShadowCastingMeshTemplates,
					};
				}
			}
		}
	}

	return {
		hasTransparentMeshTemplates,
		hasShadowCastingMeshTemplates,
	};
}
