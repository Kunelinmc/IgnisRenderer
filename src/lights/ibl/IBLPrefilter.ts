import { Texture } from "../../core/Texture";
import type { IRenderBackend } from "../../backends/IRenderBackend";
import { TextureFormat } from "../../backends/types";

import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
} from "../runtime/environmentMapRuntime";
import { SingleThreadIBLPrefilterExecutor } from "./IBLPrefilterCPUExecutor";
import {
	assertIBLPrefilterNotAborted,
	captureIBLPrefilterSourceRevision,
	IBL_PREFILTER_EXECUTOR_EXTENSION,
	type IBLPrefilterExecutorId,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
	type IBLPrefilterPlan,
} from "./IBLPrefilterExecutor";
import { MultiThreadIBLPrefilterExecutor } from "./IBLPrefilterWorkerExecutor";

export const IBL_PREFILTER_MAX_SAMPLE_WIDTH = 128;
export const IBL_PREFILTER_MAX_SAMPLE_HEIGHT = 64;
export const IBL_PREFILTER_MAX_MIP_LEVELS = 5;

export type IBLPrefilterAcceleration =
	| "auto"
	| "single-thread"
	| "multi-thread"
	| "webgpu"
	| "webgl";

export interface IBLPrefilterProgress {
	phase: "prefilter";
	completed: number;
	total: number;
	detail?: string;
}

export interface IBLPrefilterServiceOptions {
	backend?: IRenderBackend | null;
}

export interface IBLPrefilterPlanningOptions {
	maxSampleWidth?: number;
	maxSampleHeight?: number;
	maxMipLevels?: number;
}

export interface ResolvedIBLPrefilterOptions {
	maxSampleWidth: number;
	maxSampleHeight: number;
	maxMipLevels: number;
}

export interface IBLPrefilterOptions {
	signal?: AbortSignal | null;
	onProgress?: (progress: IBLPrefilterProgress) => void;
	acceleration?: IBLPrefilterAcceleration;
	workerCount?: number;
	maxSampleWidth?: number;
	maxSampleHeight?: number;
	maxMipLevels?: number;
}

/**
 * Backend-agnostic service for generating specular environment mip chains.
 *
 * @remarks GPU implementations are resolved through the configured backend's
 * generic IBL executor extension. CPU and Worker execution remain lighting
 * subsystem responsibilities.
 */
export class IBLPrefilter {
	private readonly _service: IBLPrefilterServiceOptions;

	/**
	 * Creates a reusable environment IBL prefilter service.
	 *
	 * @param service Optional backend service dependency.
	 * @constraints The backend must outlive calls to `prefilter()`.
	 * @sideEffects None.
	 */
	public constructor(service: IBLPrefilterServiceOptions = {}) {
		if (
			!service ||
			typeof service !== "object" ||
			isRenderBackend(service) ||
			(service.backend != null && !isRenderBackend(service.backend))
		) {
			throw new TypeError(
				"IBLPrefilter requires an IBLPrefilterServiceOptions object.",
			);
		}
		this._service = service;
	}

	/**
	 * Generates a CPU-backed HDR texture containing roughness mip levels.
	 *
	 * @param envMap Source equirectangular or cubemap environment texture.
	 * @param options Execution, sizing, cancellation, and progress settings.
	 * @returns Prefiltered HDR texture.
	 * @sideEffects May schedule Worker or backend-owned GPU work.
	 */
	public async prefilter(
		envMap: Texture,
		options: IBLPrefilterOptions = {},
	): Promise<Texture> {
		assertIBLPrefilterNotAborted(options.signal);
		const sampledEnvironment = ensureEnvironmentTextureEquirect(envMap);
		if (
			!sampledEnvironment ||
			!isTextureReadyForEnvironment(sampledEnvironment)
		) {
			throw new Error(
				"IBL prefilter requires a valid environment texture " +
					"(2D equirect or cubemap).",
			);
		}

		const acceleration = options.acceleration ?? "auto";
		assertIBLPrefilterAcceleration(acceleration);
		const plan = resolveIBLPrefilterWorkPlan(
			sampledEnvironment,
			resolveIBLPrefilterOptions(options),
		);
		const executors = this._createExecutors(options.workerCount);
		const executor = selectIBLPrefilterExecutor(acceleration, executors);
		let completed = 0;
		const mipData = await executor.execute({
			envMap: sampledEnvironment,
			plan,
			sourceRevision:
				captureIBLPrefilterSourceRevision(sampledEnvironment),
			signal: options.signal ?? null,
			onMipComplete: (level) => {
				completed++;
				options.onProgress?.({
					phase: "prefilter",
					completed,
					total: plan.mipLevels.length,
					detail: `mip ${level + 1}/${plan.mipLevels.length}`,
				});
			},
		});
		return assembleIBLPrefilterResult(
			plan.baseWidth,
			plan.baseHeight,
			mipData,
			plan.mipLevels.length,
		);
	}

	private _createExecutors(
		workerCount?: number,
	): readonly IBLPrefilterExecutorLike[] {
		const executors: IBLPrefilterExecutorLike[] = [];
		const backendExecutor = this._service.backend?.extensions
			.getBackendExtension(IBL_PREFILTER_EXECUTOR_EXTENSION);
		if (backendExecutor) executors.push(backendExecutor);
		executors.push(
			new MultiThreadIBLPrefilterExecutor(workerCount),
			new SingleThreadIBLPrefilterExecutor(),
		);
		return executors;
	}
}

export interface PrefilterEnvironmentIBLOptions extends IBLPrefilterOptions {
	service?: IBLPrefilterServiceOptions;
}

/** Prefilters an environment texture with a one-shot service instance. */
export async function prefilterEnvironmentIBL(
	envMap: Texture,
	options: PrefilterEnvironmentIBLOptions = {},
): Promise<Texture> {
	const { service, ...prefilterOptions } = options;
	return new IBLPrefilter(service).prefilter(envMap, prefilterOptions);
}

function assertIBLPrefilterAcceleration(
	acceleration: unknown,
): asserts acceleration is IBLPrefilterAcceleration {
	if (
		acceleration === "auto" ||
		acceleration === "single-thread" ||
		acceleration === "multi-thread" ||
		acceleration === "webgpu" ||
		acceleration === "webgl"
	) {
		return;
	}
	throw new Error(
		`Unsupported IBL prefilter acceleration "${String(acceleration)}".`,
	);
}

function isRenderBackend(source: unknown): source is IRenderBackend {
	if (!source || typeof source !== "object") return false;
	const candidate = source as Partial<IRenderBackend>;
	return !!candidate.profile &&
		typeof candidate.profile.id === "string" &&
		!!candidate.extensions &&
		typeof candidate.extensions.getBackendExtension === "function";
}

export function resolveIBLPrefilterOptions(
	options: IBLPrefilterPlanningOptions = {},
): ResolvedIBLPrefilterOptions {
	return {
		maxSampleWidth: sanitizePrefilterDimension(
			options.maxSampleWidth,
			IBL_PREFILTER_MAX_SAMPLE_WIDTH,
		),
		maxSampleHeight: sanitizePrefilterDimension(
			options.maxSampleHeight,
			IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
		),
		maxMipLevels: sanitizePrefilterDimension(
			options.maxMipLevels,
			IBL_PREFILTER_MAX_MIP_LEVELS,
		),
	};
}

/** @internal Plans output dimensions and roughness levels. */
export function resolveIBLPrefilterWorkPlan(
	envMap: Texture,
	options: ResolvedIBLPrefilterOptions,
): IBLPrefilterPlan {
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(
		envMap,
		options,
	);
	const naturalMipLevels =
		Math.floor(Math.log2(Math.max(baseWidth, baseHeight))) + 1;
	const mipLevelCount = Math.max(
		1,
		Math.min(options.maxMipLevels, naturalMipLevels),
	);
	return {
		baseWidth,
		baseHeight,
		mipLevels: Array.from({ length: mipLevelCount }, (_, level) => ({
			level,
			width: Math.max(1, baseWidth >> level),
			height: Math.max(1, baseHeight >> level),
			roughness: mipLevelCount <= 1 ? 0 : level / (mipLevelCount - 1),
		})),
	};
}

export function resolvePrefilterBaseDimensions(
	envMap: Texture,
	options: Pick<
		IBLPrefilterPlanningOptions,
		"maxSampleWidth" | "maxSampleHeight"
	> = {},
): { baseWidth: number; baseHeight: number } {
	return {
		baseWidth: Math.min(
			envMap.width,
			sanitizePrefilterDimension(
				options.maxSampleWidth,
				IBL_PREFILTER_MAX_SAMPLE_WIDTH,
			),
		),
		baseHeight: Math.min(
			envMap.height,
			sanitizePrefilterDimension(
				options.maxSampleHeight,
				IBL_PREFILTER_MAX_SAMPLE_HEIGHT,
			),
		),
	};
}

function sanitizePrefilterDimension(
	value: number | undefined,
	fallback: number,
): number {
	if (!Number.isFinite(value)) return Math.max(1, Math.floor(fallback));
	return Math.max(1, Math.floor(value as number));
}

/** Selects an executor without branching on backend behavior. */
function selectIBLPrefilterExecutor(
	acceleration: IBLPrefilterAcceleration,
	executors: readonly IBLPrefilterExecutorLike[],
): IBLPrefilterExecutorLike {
	if (acceleration === "auto") {
		const executor = executors.find(
			(candidate) => candidate.getAvailability().state === "ready",
		);
		if (executor) return executor;
		throw new Error("No IBL prefilter executor is available.");
	}

	const executor = executors.find(
		(candidate) => candidate.id ===
			(acceleration as IBLPrefilterExecutorId),
	);
	if (!executor) {
		throw new Error(
			`IBL prefilter executor "${acceleration}" is unavailable.`,
		);
	}
	const availability = executor.getAvailability();
	if (!availability.acceptsRequests) {
		throw new Error(
			availability.reason ??
				`IBL prefilter executor "${acceleration}" is unavailable.`,
		);
	}
	return executor;
}

/** Validates executor output and assembles a CPU-backed HDR texture. */
function assembleIBLPrefilterResult(
	baseWidth: number,
	baseHeight: number,
	mipData: readonly IBLPrefilterMipData[],
	expectedMipLevelCount = mipData.length,
): Texture {
	if (mipData.length !== expectedMipLevelCount || mipData.length === 0) {
		throw new Error(
			`IBL prefilter executor returned ${mipData.length}/` +
				`${expectedMipLevelCount} required mip levels.`,
		);
	}
	const sorted = [...mipData].sort((left, right) => left.level - right.level);
	for (let index = 0; index < sorted.length; index++) {
		const mip = sorted[index];
		const expectedWidth = Math.max(1, baseWidth >> index);
		const expectedHeight = Math.max(1, baseHeight >> index);
		if (
			mip.level !== index ||
			mip.width !== expectedWidth ||
			mip.height !== expectedHeight ||
			mip.data.length !== expectedWidth * expectedHeight * 4
		) {
			throw new Error(
				`IBL prefilter executor returned invalid mip ${index}.`,
			);
		}
	}
	const texture = new Texture({
		data: sorted[0].data,
		width: baseWidth,
		height: baseHeight,
		format: TextureFormat.RGBA16Float,
		colorSpace: "HDR",
		levels: sorted.map((mip) => ({
			data: mip.data,
			width: mip.width,
			height: mip.height,
		})),
		usageHint: "color",
	});
	texture.wrapS = "Repeat";
	texture.wrapT = "Clamp";
	texture.minFilter = sorted.length > 1 ? "LinearMipmapLinear" : "Linear";
	texture.magFilter = "Linear";
	return texture;
}
