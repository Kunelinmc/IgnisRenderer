import type { ShadowCastingLight } from "../types";
import type {
	ShadowDefinitionSnapshot,
	ShadowFilterMode,
	ShadowProjectionPreference,
} from "./types";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type {
	ParticleMeshRenderBatch,
	ParticleRenderBatch,
} from "../../particles/ParticleRenderBatch";
import type { DrawSubmission } from "../../pipeline/types";

export interface ShadowDiagnostic {
	readonly code:
		| "unsupported-light-type"
		| "projection-fallback"
		| "filter-fallback"
		| "budget-degraded"
		| "budget-disabled"
		| "capability-limit"
		| "transmission-unsupported"
		| "invalid-projection"
		| "custom-kind-deprecated";
	readonly severity: "info" | "warning";
	readonly lightId: string;
	readonly definitionId: string;
	readonly message: string;
}

export interface PreparedShadowSlice {
	readonly index: number;
	readonly resolution: number;
	readonly view: Matrix4;
	readonly projection: Matrix4;
	readonly viewProjection: Matrix4;
	readonly lightDirection: Readonly<IVector3>;
	readonly splitNear: number;
	readonly splitFar: number;
}

export interface PreparedShadowLight {
	/** @internal Backend runtimes use identity to associate light resources. */
	readonly light: ShadowCastingLight;
	readonly lightId: string;
	readonly definition: ShadowDefinitionSnapshot;
	readonly requestedTechnique: ShadowProjectionPreference;
	readonly effectiveTechnique: ShadowProjectionPreference;
	readonly requestedCascadeCount: number;
	readonly effectiveCascadeCount: number;
	readonly requestedResolution: number;
	readonly effectiveResolution: number;
	readonly sampling: ShadowDefinitionSnapshot["sampling"];
	readonly fallbackReason?: ShadowDiagnostic["code"];
	/** Filter authored on the bound definition before planner fallback. */
	readonly requestedFilterMode: ShadowFilterMode;
	/** Filter that backend consumers must execute after planner fallback. */
	readonly effectiveFilterMode: ShadowFilterMode;
	readonly priority: number;
	readonly cost: number;
	readonly score: number;
	readonly slices: readonly PreparedShadowSlice[];
}

export interface ShadowFramePlan {
	readonly revision: number;
	readonly lights: readonly PreparedShadowLight[];
	readonly diagnostics: readonly ShadowDiagnostic[];
	readonly hasRasterWork: boolean;
	readonly hasTransmissionWork: boolean;
}

/** Conservative caster information available before particle simulation. */
export interface ShadowCasterIntent {
	readonly meshSubmissions: readonly DrawSubmission[];
	readonly hasTransparentCasters: boolean;
	readonly hasParticleCasters: boolean;
	readonly particleBounds?: Readonly<{
		readonly center: IVector3;
		readonly radius: number;
	}>;
	readonly estimatedParticleCapacity: number;
}

/** Concrete draw work published after particle simulation. */
export interface ShadowWorkSet {
	readonly casterSubmissions: readonly DrawSubmission[];
	readonly meshParticleBatches: readonly ParticleMeshRenderBatch[];
	readonly billboardBatches: readonly ParticleRenderBatch[];
	readonly transmitterSubmissions: readonly DrawSubmission[];
}

export const EMPTY_SHADOW_FRAME_PLAN: ShadowFramePlan = Object.freeze({
	revision: 0,
	lights: Object.freeze([]),
	diagnostics: Object.freeze([]),
	hasRasterWork: false,
	hasTransmissionWork: false,
});
