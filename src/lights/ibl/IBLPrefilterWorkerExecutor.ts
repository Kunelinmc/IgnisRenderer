import type { Texture } from "../../core/Texture";
import { Platform } from "../../foundation/Platform";
import { globalWorkerScheduler } from "../../workers/WorkerScheduler";
import { postMessageWorkerTransportPlugin } from "../../workers/transports";
import type { WorkerLike } from "../../workers/types";

import type {
	IBLPrefilterWorkerEnvMapPayload,
	IBLPrefilterWorkerTaskPayload,
	IBLPrefilterWorkerTaskResult,
} from "./workers/iblPrefilterWorkerProtocol";
import {
	assertIBLPrefilterNotAborted,
	assertIBLPrefilterSourceRevision,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
} from "./IBLPrefilterExecutor";

const DEFAULT_PREFILTER_POOL_PREFIX = "ibl-prefilter";

/** @internal Lighting-owned Worker scheduler executor. */
export class MultiThreadIBLPrefilterExecutor
	implements IBLPrefilterExecutorLike {
	public readonly id = "multi-thread" as const;
	private readonly _workerCount: number | undefined;

	public constructor(workerCount?: number) {
		this._workerCount = workerCount;
	}

	public getAvailability(): IBLPrefilterExecutorAvailability {
		const available = Platform.hasWorker();
		return {
			state: available ? "ready" : "unsupported",
			acceptsRequests: available,
			reason: available ? null :
				"Multi-thread acceleration was requested for IBL prefiltering, " +
				"but the Worker API is unavailable.",
		};
	}

	public async execute(
		request: IBLPrefilterExecutionRequest,
	): Promise<IBLPrefilterMipData[]> {
		assertIBLPrefilterSourceRevision(
			request.envMap,
			request.sourceRevision,
		);
		assertIBLPrefilterNotAborted(request.signal);
		const poolId = resolveWorkerPoolId();
		const totalMipLevels = request.plan.mipLevels.length;
		const workerCount = Math.min(
			resolveWorkerCount(this._workerCount),
			totalMipLevels,
		);
		const envPayload = toWorkerEnvMapPayload(request.envMap);

		globalWorkerScheduler.registerPool({
			id: poolId,
			size: workerCount,
			createWorker: (workerIndex, id) =>
				createPrefilterWorker(workerIndex, id),
			transportPlugins: [postMessageWorkerTransportPlugin],
			defaultTimeoutMs: 0,
		});

		try {
			const tasks = request.plan.mipLevels.map((mipPlan) => {
				assertIBLPrefilterNotAborted(request.signal);
				const payload: IBLPrefilterWorkerTaskPayload = {
					type: "prefilter-mip",
					envMap: envPayload,
					mipPlan,
				};
				return globalWorkerScheduler
					.schedule<
						IBLPrefilterWorkerTaskResult,
						IBLPrefilterWorkerTaskPayload
					>(poolId, payload, { signal: request.signal ?? null })
					.then((result): IBLPrefilterMipData => {
						if (!result || result.type !== "prefilter-mip") {
							throw new Error(
								"IBL prefilter worker returned an invalid response",
							);
						}
						request.onMipComplete?.(result.level);
						return {
							level: result.level,
							width: result.width,
							height: result.height,
							data: result.data,
						};
					});
			});
			return await Promise.all(tasks);
		} finally {
			globalWorkerScheduler.unregisterPool(poolId);
		}
	}
}

function resolveWorkerCount(requestedCount?: number): number {
	const fallback = Platform.getHardwareConcurrency(4);
	if (!Number.isFinite(requestedCount)) return Math.max(1, fallback);
	return Math.max(1, Math.floor(requestedCount as number));
}

function createPrefilterWorker(workerIndex: number, poolId: string): WorkerLike {
	if (typeof Worker !== "function") {
		throw new Error(
			`Worker constructor is unavailable for pool "${poolId}" ` +
				`(worker #${workerIndex})`,
		);
	}
	return new Worker(
		new URL("./workers/iblPrefilter.worker.ts", import.meta.url),
		{ type: "module" },
	) as unknown as WorkerLike;
}

function resolveWorkerPoolId(): string {
	return `${DEFAULT_PREFILTER_POOL_PREFIX}-${Math.random().toString(36).slice(2)}`;
}

function toWorkerEnvMapPayload(
	envMap: Texture,
): IBLPrefilterWorkerEnvMapPayload {
	return {
		width: envMap.width,
		height: envMap.height,
		colorSpace: envMap.colorSpace,
		data: envMap.data,
	};
}
