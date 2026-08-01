import type { Texture } from "../../core/Texture";
import type { RenderBackendExtensionKey } from "../../backends/BackendExtensions";

export const IBL_PREFILTER_EXECUTOR_EXTENSION_ID =
	"lighting.ibl-prefilter-executor";

export type IBLPrefilterExecutorId =
	| "single-thread"
	| "multi-thread"
	| "webgpu"
	| "webgl";

export interface IBLPrefilterMipData {
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}

export interface IBLPrefilterMipPlan {
	level: number;
	width: number;
	height: number;
	roughness: number;
}

export interface IBLPrefilterPlan {
	baseWidth: number;
	baseHeight: number;
	mipLevels: readonly IBLPrefilterMipPlan[];
}

export interface IBLPrefilterSourceRevision {
	readonly version: number;
	readonly width: number;
	readonly height: number;
	readonly colorSpace: Texture["colorSpace"];
	readonly mipFingerprint: string;
}

export interface IBLPrefilterExecutorAvailability {
	readonly state: "ready" | "temporarily-unavailable" | "unsupported";
	readonly acceptsRequests: boolean;
	readonly reason: string | null;
}

export interface IBLPrefilterExecutionRequest {
	readonly envMap: Texture;
	readonly plan: IBLPrefilterPlan;
	readonly sourceRevision: IBLPrefilterSourceRevision;
	readonly signal?: AbortSignal | null;
	readonly onMipComplete?: (level: number) => void;
}

/**
 * Executes one complete IBL prefilter plan.
 *
 * @internal Owned by the lighting IBL subsystem. Backend implementations must
 * expose this contract through `IBL_PREFILTER_EXECUTOR_EXTENSION`.
 */
export interface IBLPrefilterExecutorLike {
	readonly id: IBLPrefilterExecutorId;
	getAvailability(): IBLPrefilterExecutorAvailability;
	execute(
		request: IBLPrefilterExecutionRequest,
	): IBLPrefilterMipData[] | Promise<IBLPrefilterMipData[]>;
}

/** @internal Typed lighting-owned backend extension key. */
export const IBL_PREFILTER_EXECUTOR_EXTENSION: RenderBackendExtensionKey<
	IBLPrefilterExecutorLike
> = {
	id: IBL_PREFILTER_EXECUTOR_EXTENSION_ID,
};

/** @internal Captures source state that deferred executors must preserve. */
export function captureIBLPrefilterSourceRevision(
	texture: Texture,
): IBLPrefilterSourceRevision {
	return {
		version: texture.version,
		width: texture.width,
		height: texture.height,
		colorSpace: texture.colorSpace,
		mipFingerprint: texture.levels
			.map((level, index) =>
				`${index}:${level.width}x${level.height}:${level.data?.length ?? -1}`,
			)
			.join("|"),
	};
}

/** @internal Rejects deferred work whose source changed before execution. */
export function assertIBLPrefilterSourceRevision(
	texture: Texture,
	revision: IBLPrefilterSourceRevision,
): void {
	const current = captureIBLPrefilterSourceRevision(texture);
	if (
		current.version !== revision.version ||
		current.width !== revision.width ||
		current.height !== revision.height ||
		current.colorSpace !== revision.colorSpace ||
		current.mipFingerprint !== revision.mipFingerprint
	) {
		throw new Error(
			"IBL prefilter source changed while waiting for executor execution.",
		);
	}
}

/** @internal Creates the canonical IBL cancellation error. */
export function createIBLPrefilterAbortError(): Error {
	const error = new Error("IBL prefilter was aborted");
	error.name = "AbortError";
	return error;
}

/** @internal Throws when an IBL request has been cancelled. */
export function assertIBLPrefilterNotAborted(
	signal?: AbortSignal | null,
): void {
	if (signal?.aborted) {
		throw createIBLPrefilterAbortError();
	}
}
