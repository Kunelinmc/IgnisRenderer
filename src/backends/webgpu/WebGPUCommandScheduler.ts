import type { ICommandBuffer, ICommandEncoder } from "../ICommandEncoder";
import type { IRenderTexture } from "../types";
import { Logger } from "../../foundation/Logger";
import {
	WEBGPU_COPY_BATCH_SIZE,
	WEBGPU_TIMESTAMP_QUERY_CAPACITY,
} from "./constants";
import { getWebGPUTexture } from "./WebGPUResourceAccess";
import { WebGPUCommandEncoder } from "./WebGPUCommandEncoder";
import type {
	WebGPUCommandBufferInternals,
	WebGPUCommandSchedulerHost,
	WebGPUTimestampWrites,
} from "./WebGPUBackendContracts";

interface TimestampPairEntry {
	label: string;
	startIndex: number;
	endIndex: number;
}

export class WebGPUCommandScheduler {
	private _copyCommandEncoder: GPUCommandEncoder | null = null;
	private _copyPendingCount = 0;
	private _copyFlushScheduled = false;
	private _commandBufferOwnerToken: object = {};
	private _timestampSupported = false;
	private _timestampQuerySet: GPUQuerySet | null = null;
	private _timestampResolveBuffer: GPUBuffer | null = null;
	private _timestampReadBuffer: GPUBuffer | null = null;
	private _timestampQueryCursor = 0;
	private _timestampPairs: TimestampPairEntry[] = [];
	private _timestampReadPending = false;
	private _timestampPeriodNs = 1;
	private _timestampResults = new Map<string, number>();

	constructor(private _host: WebGPUCommandSchedulerHost) {}

	public createCommandEncoder(): ICommandEncoder {
		this._host.assertDeviceOperational("create command encoders");
		const device = this._requireDevice("create command encoders");
		return new WebGPUCommandEncoder(
			device.createCommandEncoder(),
			this._host,
			this._commandBufferOwnerToken
		);
	}

	public copyTextureToTexture(
		source: {
			texture: IRenderTexture;
			origin?: GPUOrigin3D;
			aspect?: GPUTextureAspect;
		},
		destination: {
			texture: IRenderTexture;
			origin?: GPUOrigin3D;
			aspect?: GPUTextureAspect;
		},
		copySize: { width: number; height: number; depthOrArrayLayers?: number }
	): void {
		this._host.assertDeviceOperational("copy textures");
		const commandEncoder = this._getCopyCommandEncoder();
		const sourceTexture = getWebGPUTexture(source.texture).texture;
		const destinationTexture = getWebGPUTexture(destination.texture).texture;

		commandEncoder.copyTextureToTexture(
			{
				texture: sourceTexture,
				origin: source.origin,
				aspect: source.aspect,
			},
			{
				texture: destinationTexture,
				origin: destination.origin,
				aspect: destination.aspect,
			},
			copySize
		);
		this._copyPendingCount++;
		if (this._copyPendingCount >= WEBGPU_COPY_BATCH_SIZE) {
			this.submitPendingCopyCommands();
			return;
		}
		this._scheduleCopyFlush();
	}

	public submit(commands: ICommandBuffer[]): void {
		this._host.assertDeviceOperational("submit command buffers");
		const submitted: GPUCommandBuffer[] = [];
		const copyCommandBuffer = this._flushPendingCopyCommandBuffer();
		if (copyCommandBuffer) {
			submitted.push(copyCommandBuffer);
		}
		for (const command of commands) {
			const internal = this._toInternalCommandBuffer(command);
			internal._submitted = true;
			submitted.push(internal._backendCommandBuffer);
		}
		if (submitted.length <= 0) {
			return;
		}
		const timestampResolve = this._buildTimestampResolveCommand();
		if (timestampResolve) {
			submitted.push(timestampResolve.commandBuffer);
		}
		this._host.runValidationScope("queue.submit", () => {
			this._requireQueue("submit command buffers").submit(submitted);
		});
		if (timestampResolve) {
			this._readTimestampResultsAsync(
				timestampResolve.queryCount,
				timestampResolve.pairs
			);
		}
		this._host.onSubmittedCommandBuffers();
	}

	public submitPendingCopyCommands(): void {
		const commandBuffer = this._flushPendingCopyCommandBuffer();
		if (!commandBuffer || !this._host.queue) {
			return;
		}
		this._host.runValidationScope("queue.submit.copyBatch", () => {
			this._host.queue?.submit([commandBuffer]);
		});
	}

	public createPassTimestampWrites(label: string): WebGPUTimestampWrites | undefined {
		if (!this._timestampSupported || !this._timestampQuerySet) {
			return undefined;
		}
		if (this._timestampQueryCursor + 1 >= WEBGPU_TIMESTAMP_QUERY_CAPACITY) {
			return undefined;
		}
		const startIndex = this._timestampQueryCursor++;
		const endIndex = this._timestampQueryCursor++;
		this._timestampPairs.push({
			label: label || "pass",
			startIndex,
			endIndex,
		});
		return {
			querySet: this._timestampQuerySet,
			beginningOfPassWriteIndex: startIndex,
			endOfPassWriteIndex: endIndex,
		};
	}

	public getTimestampDurationsMs(): ReadonlyMap<string, number> {
		return this._timestampResults;
	}

	public initTimestampResources(): void {
		this._timestampSupported = false;
		this._timestampQuerySet = null;
		this._timestampResolveBuffer = null;
		this._timestampReadBuffer = null;
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		this._timestampResults.clear();
		this._timestampReadPending = false;
		const queue = this._host.queue;
		if (!queue) {
			return;
		}
		const queueWithTimestamp = queue as GPUQueue & {
			getTimestampPeriod?: () => number;
		};
		this._timestampPeriodNs =
			typeof queueWithTimestamp.getTimestampPeriod === "function"
				? queueWithTimestamp.getTimestampPeriod()
				: 1;
		const device = this._host.device;
		if (!device || typeof device.createQuerySet !== "function") {
			return;
		}
		if (
			typeof device.features?.has === "function" &&
			!device.features.has("timestamp-query" as GPUFeatureName)
		) {
			return;
		}
		try {
			this._timestampQuerySet = device.createQuerySet({
				type: "timestamp",
				count: WEBGPU_TIMESTAMP_QUERY_CAPACITY,
			});
			this._timestampResolveBuffer = device.createBuffer({
				label: "WebGPUTimestampResolveBuffer",
				size: WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
			});
			this._timestampReadBuffer = device.createBuffer({
				label: "WebGPUTimestampReadBuffer",
				size: WEBGPU_TIMESTAMP_QUERY_CAPACITY * BigUint64Array.BYTES_PER_ELEMENT,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
			});
			this._timestampSupported = true;
		} catch (error) {
			this._host.reportNonFatalError("init timestamp resources", error);
			this.releaseTimestampResources();
		}
	}

	public releaseTimestampResources(): void {
		this._timestampSupported = false;
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		this._timestampReadPending = false;
		this._timestampResults.clear();
		if (this._timestampReadBuffer) {
			this._tryUnmapBuffer(this._timestampReadBuffer);
			try {
				this._timestampReadBuffer.destroy();
			} catch (error) {
				this._host.reportNonFatalError("timestamp read buffer destroy", error);
			}
			this._timestampReadBuffer = null;
		}
		if (this._timestampResolveBuffer) {
			try {
				this._timestampResolveBuffer.destroy();
			} catch (error) {
				this._host.reportNonFatalError("timestamp resolve buffer destroy", error);
			}
			this._timestampResolveBuffer = null;
		}
		if (this._timestampQuerySet) {
			try {
				this._timestampQuerySet.destroy();
			} catch (error) {
				this._host.reportNonFatalError("timestamp query set destroy", error);
			}
			this._timestampQuerySet = null;
		}
	}

	public reset(): void {
		this._copyCommandEncoder = null;
		this._copyPendingCount = 0;
		this._copyFlushScheduled = false;
		this._commandBufferOwnerToken = {};
		this.releaseTimestampResources();
	}

	private _toInternalCommandBuffer(command: ICommandBuffer): WebGPUCommandBufferInternals {
		const internal = command as
			| (Partial<WebGPUCommandBufferInternals> & {
					_backendCommandBuffer?: unknown;
					_gpuCommandBuffer?: unknown;
			  })
			| null;
		if (!internal || typeof internal !== "object") {
			throw new Error("Invalid command buffer for WebGPU submit().");
		}
		if (!internal._backendCommandBuffer && !internal._gpuCommandBuffer) {
			throw new Error("Invalid command buffer for WebGPU submit().");
		}
		if (!internal._backendCommandBuffer && internal._gpuCommandBuffer) {
			internal._backendCommandBuffer = internal._gpuCommandBuffer as GPUCommandBuffer;
		}
		if (internal._ownerToken !== this._commandBufferOwnerToken) {
			throw new Error("Command buffer does not belong to this WebGPU backend instance.");
		}
		if (internal._submitted) {
			throw new Error("WebGPU command buffer has already been submitted.");
		}
		return internal as WebGPUCommandBufferInternals;
	}

	private _getCopyCommandEncoder(): GPUCommandEncoder {
		const device = this._requireDevice("encode copy commands");
		if (!this._copyCommandEncoder) {
			this._copyCommandEncoder = device.createCommandEncoder({
				label: "WebGPUCopyBatchEncoder",
			});
		}
		return this._copyCommandEncoder;
	}

	private _flushPendingCopyCommandBuffer(): GPUCommandBuffer | null {
		if (!this._copyCommandEncoder || this._copyPendingCount <= 0) {
			this._copyCommandEncoder = null;
			this._copyPendingCount = 0;
			this._copyFlushScheduled = false;
			return null;
		}
		const commandBuffer = this._copyCommandEncoder.finish();
		this._copyCommandEncoder = null;
		this._copyPendingCount = 0;
		this._copyFlushScheduled = false;
		return commandBuffer;
	}

	private _scheduleCopyFlush(): void {
		if (!this._host.isFrameActive()) {
			this.submitPendingCopyCommands();
			return;
		}
		if (this._copyFlushScheduled) {
			return;
		}
		this._copyFlushScheduled = true;
		const scheduleMicrotask =
			typeof queueMicrotask === "function"
				? queueMicrotask
				: (callback: () => void) => {
						void Promise.resolve().then(callback);
					};
		scheduleMicrotask(() => {
			if (!this._copyFlushScheduled) {
				return;
			}
			this.submitPendingCopyCommands();
		});
	}

	private _buildTimestampResolveCommand():
		| {
				commandBuffer: GPUCommandBuffer;
				queryCount: number;
				pairs: TimestampPairEntry[];
		  }
		| undefined {
		if (
			!this._timestampSupported ||
			!this._timestampQuerySet ||
			!this._timestampResolveBuffer ||
			!this._timestampReadBuffer
		) {
			return undefined;
		}
		const queryCount = this._timestampQueryCursor;
		if (queryCount <= 0) {
			return undefined;
		}
		if (this._timestampReadPending) {
			this._dropPendingTimestampSamples();
			return undefined;
		}
		this._tryUnmapBuffer(this._timestampReadBuffer);
		if (this._isBufferMapped(this._timestampReadBuffer)) {
			this._dropPendingTimestampSamples();
			return undefined;
		}
		const pairs = this._timestampPairs.slice();
		const resolveEncoder = this._requireDevice(
			"build timestamp resolve command"
		).createCommandEncoder({
			label: "WebGPUTimestampResolveEncoder",
		});
		resolveEncoder.resolveQuerySet(
			this._timestampQuerySet,
			0,
			queryCount,
			this._timestampResolveBuffer,
			0
		);
		resolveEncoder.copyBufferToBuffer(
			this._timestampResolveBuffer,
			0,
			this._timestampReadBuffer,
			0,
			queryCount * BigUint64Array.BYTES_PER_ELEMENT
		);
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
		return {
			commandBuffer: resolveEncoder.finish(),
			queryCount,
			pairs,
		};
	}

	private _readTimestampResultsAsync(
		queryCount: number,
		pairs: TimestampPairEntry[]
	): void {
		if (this._timestampReadPending || !this._timestampReadBuffer || queryCount <= 0) {
			return;
		}
		this._tryUnmapBuffer(this._timestampReadBuffer);
		if (this._isBufferMapped(this._timestampReadBuffer)) {
			return;
		}
		this._timestampReadPending = true;
		const byteLength = queryCount * BigUint64Array.BYTES_PER_ELEMENT;
		void this._timestampReadBuffer
			.mapAsync(GPUMapMode.READ, 0, byteLength)
			.then(() => {
				if (!this._timestampReadBuffer) {
					return;
				}
				const view = this._timestampReadBuffer.getMappedRange(0, byteLength);
				const data = new BigUint64Array(view.slice(0));
				const result = new Map<string, number>();
				for (let i = 0; i < pairs.length; i++) {
					const pair = pairs[i];
					if (pair.endIndex >= data.length || pair.startIndex >= data.length) {
						continue;
					}
					const start = data[pair.startIndex];
					const end = data[pair.endIndex];
					const deltaTicks = end >= start ? end - start : 0n;
					const durationMs = (Number(deltaTicks) * this._timestampPeriodNs) / 1_000_000;
					result.set(`${pair.label}#${i}`, durationMs);
				}
				this._timestampResults = result;
				this._timestampReadBuffer.unmap();
			})
			.catch((error) => {
				Logger.warn(`WebGPU timestamp readback failed: ${String(error)}`, {
					scope: "WebGPUCommandScheduler",
				});
				if (this._timestampReadBuffer) {
					try {
						this._timestampReadBuffer.unmap();
					} catch (unmapError) {
						this._host.reportNonFatalError("timestamp readback unmap", unmapError);
					}
				}
			})
			.finally(() => {
				this._timestampReadPending = false;
			});
	}

	private _dropPendingTimestampSamples(): void {
		this._timestampQueryCursor = 0;
		this._timestampPairs = [];
	}

	private _isBufferMapped(buffer: GPUBuffer | null): boolean {
		if (!buffer) {
			return false;
		}
		const state = (buffer as GPUBuffer & { mapState?: GPUBufferMapState }).mapState;
		return (state ?? "unmapped") !== "unmapped";
	}

	private _tryUnmapBuffer(buffer: GPUBuffer | null): void {
		if (!this._isBufferMapped(buffer) || !buffer) {
			return;
		}
		try {
			buffer.unmap();
		} catch (error) {
			this._host.reportNonFatalError("buffer.unmap", error);
		}
	}

	private _requireDevice(operation: string): GPUDevice {
		const device = this._host.device;
		if (!device) {
			throw new Error(
				`WebGPU backend is not initialized; cannot ${operation}.`
			);
		}
		return device;
	}

	private _requireQueue(operation: string): GPUQueue {
		const queue = this._host.queue;
		if (!queue) {
			throw new Error(
				`WebGPU backend is not initialized; cannot ${operation}.`
			);
		}
		return queue;
	}
}
