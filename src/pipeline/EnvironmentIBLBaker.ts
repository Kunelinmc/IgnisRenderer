import { Texture } from "../core/Texture";
import { Platform } from "../foundation/Platform";
import { LightProbe } from "../lights/LightProbe";
import { globalWorkerScheduler } from "../workers/WorkerScheduler";
import type { WorkerLike } from "../workers/types";
import { postMessageWorkerTransportPlugin } from "../workers/transports";

import {
	ENVIRONMENT_IBL_MAX_MIP_LEVELS,
	prefilterEnvMapCPU,
	projectEquirectTextureToSH,
	resolvePrefilterBaseDimensions,
	buildPrefilteredTexture,
	type EnvironmentIBLPrefilterMipData,
} from "./environmentIblBakeCore";
import { prefilterEnvMapWithWebGPU } from "./environmentIblBakeWebGPU";
import type {
	EnvironmentIBLBakeWorkerTaskPayload,
	EnvironmentIBLBakeWorkerTaskResult,
} from "./workers/environmentIblBakeWorkerProtocol";
import type { WebGPUComputeFacadeSource } from "../renderers/webgpu/computeFacade";

export type EnvironmentIBLBakeAcceleration =
	| "auto"
	| "worker"
	| "cpu"
	| "webgpu";

export interface EnvironmentIBLBakeProgress {
	phase: "project-sh" | "prefilter" | "finalize";
	completed: number;
	total: number;
	detail?: string;
}

export interface EnvironmentIBLBakeOptions {
	signal?: AbortSignal | null;
	onProgress?: (progress: EnvironmentIBLBakeProgress) => void;
	acceleration?: EnvironmentIBLBakeAcceleration;
	workerCount?: number;
	webgpuSource?: WebGPUComputeFacadeSource | null;
}

const DEFAULT_BAKE_POOL_PREFIX = "environment-ibl-bake";

interface EnvironmentIBLBakeWorkerEnvMapPayload {
	width: number;
	height: number;
	colorSpace: Texture["colorSpace"];
	data: Texture["data"];
}

function createAbortError(): Error {
	const error = new Error("Environment IBL bake was aborted");
	error.name = "AbortError";
	return error;
}

function assertNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createAbortError();
}

function emitProgress(
	options: EnvironmentIBLBakeOptions,
	progress: EnvironmentIBLBakeProgress
): void {
	options.onProgress?.(progress);
}

function resolveWorkerCount(requestedCount?: number): number {
	const fallback = Platform.getHardwareConcurrency(4);
	if (!Number.isFinite(requestedCount)) {
		return Math.max(1, fallback);
	}
	return Math.max(1, Math.floor(requestedCount as number));
}

function createBakeWorker(workerIndex: number, poolId: string): WorkerLike {
	const workerCtor =
		(globalThis as typeof globalThis & {
			Worker?: new (...args: any[]) => Worker;
		}).Worker;

	if (typeof workerCtor !== "function") {
		throw new Error(
			`Worker constructor is unavailable for pool "${poolId}" (worker #${workerIndex})`
		);
	}

	return new workerCtor(
		new URL("./workers/environmentIblBake.worker.ts", import.meta.url),
		{
			type: "module",
		}
	) as unknown as WorkerLike;
}

function resolveWorkerPoolId(): string {
	return `${DEFAULT_BAKE_POOL_PREFIX}-${Math.random().toString(36).slice(2)}`;
}

function toWorkerEnvMapPayload(
	envMap: Texture
): EnvironmentIBLBakeWorkerEnvMapPayload {
	return {
		width: envMap.width,
		height: envMap.height,
		colorSpace: envMap.colorSpace,
		data: envMap.data,
	};
}

async function prefilterEnvMapWithWorkers(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	const poolId = resolveWorkerPoolId();
	const workerCount = Math.min(
		resolveWorkerCount(options.workerCount),
		ENVIRONMENT_IBL_MAX_MIP_LEVELS
	);
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const envPayload = toWorkerEnvMapPayload(envMap);

	globalWorkerScheduler.registerPool({
		id: poolId,
		size: workerCount,
		createWorker: (workerIndex, id) => createBakeWorker(workerIndex, id),
		transportPlugins: [postMessageWorkerTransportPlugin],
		defaultTimeoutMs: 0,
	});

	try {
		const tasks: Promise<EnvironmentIBLPrefilterMipData>[] = [];
		for (let level = 0; level < ENVIRONMENT_IBL_MAX_MIP_LEVELS; level++) {
			assertNotAborted(options.signal);
			const payload: EnvironmentIBLBakeWorkerTaskPayload = {
				type: "prefilter-mip",
				envMap: envPayload,
				baseWidth,
				baseHeight,
				maxMipLevels: ENVIRONMENT_IBL_MAX_MIP_LEVELS,
				level,
			};

			const task = globalWorkerScheduler
				.schedule<
					EnvironmentIBLBakeWorkerTaskResult,
					EnvironmentIBLBakeWorkerTaskPayload
				>(
					poolId,
					payload,
					{
						signal: options.signal ?? null,
					}
				)
				.then((result) => {
					if (!result || result.type !== "prefilter-mip") {
						throw new Error(
							"Environment IBL worker returned an invalid response"
						);
					}
					onMipComplete(result.level);
					return {
						level: result.level,
						width: result.width,
						height: result.height,
						data: result.data,
					};
				});

			tasks.push(task);
		}

		const mipData = await Promise.all(tasks);
		return buildPrefilteredTexture(baseWidth, baseHeight, mipData);
	} finally {
		globalWorkerScheduler.unregisterPool(poolId);
	}
}

function prefilterEnvMapOnCPU(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Texture {
	return prefilterEnvMapCPU(envMap, options.signal ?? null, (level) => {
		onMipComplete(level);
	});
}

function canUseWorkerAcceleration(options: EnvironmentIBLBakeOptions): boolean {
	return (
		options.acceleration === "worker" ||
		(options.acceleration !== "cpu" &&
			options.acceleration !== "webgpu" &&
			Platform.hasWorker())
	);
}

function canUseWebGPUAcceleration(options: EnvironmentIBLBakeOptions): boolean {
	if (options.acceleration === "webgpu") {
		return true;
	}
	if (
		options.acceleration === "cpu" ||
		options.acceleration === "worker"
	) {
		return false;
	}
	return !!options.webgpuSource;
}

async function prefilterEnvMapOnWebGPU(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (!options.webgpuSource) {
		throw new Error(
			"WebGPU acceleration was requested for environment IBL baking, but no webgpuSource was provided."
		);
	}
	return prefilterEnvMapWithWebGPU(
		envMap,
		options.webgpuSource,
		options.signal ?? null,
		onMipComplete
	);
}

async function prefilterEnvMap(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (canUseWebGPUAcceleration(options)) {
		try {
			return await prefilterEnvMapOnWebGPU(envMap, options, onMipComplete);
		} catch (error) {
			if (options.acceleration === "webgpu") {
				throw error;
			}
		}
	}

	if (!canUseWorkerAcceleration(options)) {
		if (options.acceleration === "worker") {
			throw new Error(
				"Worker acceleration was requested for environment IBL baking, but Worker API is unavailable."
			);
		}
		return prefilterEnvMapOnCPU(envMap, options, onMipComplete);
	}

	try {
		return await prefilterEnvMapWithWorkers(envMap, options, onMipComplete);
	} catch (error) {
		if (options.acceleration === "worker") {
			throw error;
		}
		return prefilterEnvMapOnCPU(envMap, options, onMipComplete);
	}
}

export async function bakeEnvironmentIBLFromEnvironmentMap(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions = {}
): Promise<LightProbe> {
	assertNotAborted(options.signal);
	const totalProgress = ENVIRONMENT_IBL_MAX_MIP_LEVELS + 2;
	let completed = 0;

	const sh = projectEquirectTextureToSH(envMap, options.signal ?? null);
	completed++;
	emitProgress(options, {
		phase: "project-sh",
		completed,
		total: totalProgress,
	});

	const prefiltered = await prefilterEnvMap(envMap, options, (level) => {
		completed++;
		emitProgress(options, {
			phase: "prefilter",
			completed,
			total: totalProgress,
			detail: `mip ${level + 1}/${ENVIRONMENT_IBL_MAX_MIP_LEVELS}`,
		});
	});

	assertNotAborted(options.signal);
	const probe = new LightProbe(sh);
	probe.prefilteredMap = prefiltered;
	completed++;
	emitProgress(options, {
		phase: "finalize",
		completed,
		total: totalProgress,
	});
	return probe;
}
