import type { ShadowCastingLight } from "../types";
import type {
	ShadowDefinitionSnapshot,
	ShadowFilterMode,
	ShadowProjectionPreference,
	PreparedPagedShadowSettings,
} from "./types";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type {
	ParticleMeshRenderBatch,
	ParticleRenderBatch,
} from "../../particles/ParticleRenderBatch";
import type { DrawPacket } from "../../pipeline/types";

export type ShadowStorageTechnique = "atlas" | "paged" | "atlas-fallback";

export interface ShadowDiagnostic {
	readonly code:
		| "unsupported-light-type"
		| "projection-fallback"
		| "storage-fallback"
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
	/** Logical paged configuration; native page-table state remains backend-private. */
	readonly pagedSettings?: Readonly<PreparedPagedShadowSettings>;
	readonly fallbackReason?: ShadowDiagnostic["code"];
	readonly filterMode: ShadowFilterMode;
	readonly storage: Exclude<ShadowStorageTechnique, "atlas-fallback">;
	readonly priority: number;
	readonly cost: number;
	readonly score: number;
	readonly slices: readonly PreparedShadowSlice[];
}

export interface ShadowRenderJob {
	readonly id: string;
	readonly lightIndex: number;
	readonly technique: ShadowStorageTechnique;
	readonly sliceIndices: readonly number[];
}

export interface ShadowFramePlan {
	readonly revision: number;
	readonly lights: readonly PreparedShadowLight[];
	readonly jobs: readonly ShadowRenderJob[];
	readonly diagnostics: readonly ShadowDiagnostic[];
	readonly hasRasterWork: boolean;
	readonly hasTransmissionWork: boolean;
	readonly hasPagedWork: boolean;
}

/** Conservative caster information available before particle simulation. */
export interface ShadowCasterIntent {
	readonly meshPackets: readonly DrawPacket[];
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
	readonly meshPackets: readonly DrawPacket[];
	readonly meshParticleBatches: readonly ParticleMeshRenderBatch[];
	readonly billboardBatches: readonly ParticleRenderBatch[];
	readonly transmitters: readonly DrawPacket[];
}

export const EMPTY_SHADOW_FRAME_PLAN: ShadowFramePlan = Object.freeze({
	revision: 0,
	lights: Object.freeze([]),
	jobs: Object.freeze([]),
	diagnostics: Object.freeze([]),
	hasRasterWork: false,
	hasTransmissionWork: false,
	hasPagedWork: false,
});
