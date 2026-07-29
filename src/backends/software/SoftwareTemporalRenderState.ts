import type { Matrix4 } from "../../maths/Matrix4";
import { defineTransientKey } from "../../pipeline/types";
import type { TemporalJitterFrameState } from "../cross/TemporalJitterState";

/** @internal Software scene state used to generate temporal motion inputs. */
export interface SoftwareTemporalRenderState extends TemporalJitterFrameState {
	readonly previousViewProjection: Matrix4 | null;
	readonly currentViewProjection: Matrix4;
	readonly previousWorldMatrices: ReadonlyMap<string, Matrix4>;
	readonly currentWorldMatrices: Map<string, Matrix4>;
}

export const SOFTWARE_TEMPORAL_RENDER_STATE_KEY =
	defineTransientKey<SoftwareTemporalRenderState>(
		"backend:software-temporal-render-state",
	);
