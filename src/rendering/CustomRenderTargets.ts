import type { TextureReadbackResult } from "../backends/IComputeRuntime";
import {
	getTextureFormatInfo,
	TextureFormat,
} from "../core/TextureFormat";
import type { ICommandEncoder } from "../backends/ICommandEncoder";
import type {
	BindingGroupDesc,
	BufferDesc,
	ComputePipelineDesc,
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
	IShaderModule,
	PipelineDesc,
	SamplerDesc,
	ShaderModuleDesc,
	TextureDesc,
} from "../backends/types";
import type { FrameContext } from "../pipeline/types";
import type { Camera } from "../cameras/Camera";
import type { PreparedScene } from "../pipeline/types";
import type { RenderBackendType } from "../backends/IRenderBackend";

export type RenderTargetSizeDescriptor =
	| {
			readonly mode: "canvas-scale";
			readonly scale?: number;
	  }
	| {
			readonly mode: "fixed";
			readonly width: number;
			readonly height: number;
	  };

export interface RenderTargetColorAttachmentDescriptor {
	readonly format: TextureFormat;
	readonly label?: string;
}

export interface RenderTargetDepthAttachmentDescriptor {
	readonly format: TextureFormat;
	readonly label?: string;
}

export interface RenderTargetDescriptor {
	readonly id: string;
	readonly size: RenderTargetSizeDescriptor;
	readonly color: readonly RenderTargetColorAttachmentDescriptor[];
	readonly depth?: RenderTargetDepthAttachmentDescriptor | null;
	/** Finite request normalized by flooring and clamping to at least `1`. */
	readonly sampleCount?: number;
	readonly label?: string;
}

/** Public descriptor used by the handle-centric render-target manager. */
export type RenderTargetCreateDescriptor = Omit<RenderTargetDescriptor, "id">;

export interface RenderTargetSceneViewContent {
	readonly environment?: boolean;
	readonly opaque?: boolean;
	readonly transparent?: boolean;
	readonly particles?: boolean;
	readonly shadows?: "reuse" | "disabled";
}

export interface RenderTargetJobReadbackOptions extends RenderTargetReadbackOptions {
	readonly attachmentIndex?: number;
}

export interface RenderTargetSceneViewJobDescriptor {
	readonly kind: "scene-view";
	readonly camera: Camera;
	readonly content?: RenderTargetSceneViewContent;
	readonly layerMask?: number;
	readonly readback?: RenderTargetJobReadbackOptions | null;
	/** @internal Feature-owned packet exclusion hook. */
	readonly packetFilter?: ((packet: FrameContext["scene"]["opaquePackets"][number]) => boolean) | null;
	/** @internal Feature-owned light id excluded from the prepared view. */
	readonly excludedLightId?: string | null;
	/** @internal Allows renderer-owned cameras outside the scene graph. */
	readonly allowDetachedCamera?: boolean;
}

export interface RenderTargetCustomPassJobDescriptor {
	readonly kind: "custom-pass";
	readonly label?: string;
	readonly execute: (context: CustomRenderPassContext) => void | Promise<void>;
	readonly readback?: RenderTargetJobReadbackOptions | null;
}

export type RenderTargetJobDescriptor =
	| RenderTargetSceneViewJobDescriptor
	| RenderTargetCustomPassJobDescriptor;

export interface RenderTargetJobCompletion {
	readonly jobId: string;
	readonly targetId: string;
	readonly generation: number;
	readonly readback: RenderTargetReadbackResult | null;
}

export interface RenderTargetJobTicket {
	readonly id: string;
	/** Resolves after the writing frame commits and requested readback completes. */
	readonly done: Promise<RenderTargetJobCompletion>;
	/** Cancels queued work and rejects `done`. */
	cancel(): void;
}

export interface RenderTargetJobRegistration {
	readonly id: string;
	destroy(): void;
}

export interface RenderTargetHandle {
	readonly id: string;
	readonly descriptor: RenderTargetDescriptor;
	/** Enqueues one FIFO target job for the next unsealed renderer frame. */
	enqueueJob(descriptor: RenderTargetJobDescriptor): RenderTargetJobTicket;
	/** Registers target work repeated in every eligible renderer frame. */
	registerJob(descriptor: RenderTargetJobDescriptor): RenderTargetJobRegistration;
	/** Reads the latest committed color generation. */
	readColor(
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<RenderTargetReadbackResult>;
	/** Cancels pending work and releases the target. */
	destroy(): void;
}

/** @internal One target job prepared for a renderer frame. */
export interface PreparedRenderTargetJob {
	readonly id: string;
	readonly targetId: string;
	readonly generation: number;
	readonly recurring: boolean;
	readonly descriptor: RenderTargetJobDescriptor;
	readonly scene: PreparedScene | null;
}

/** @internal Immutable job list captured before backend frame sealing. */
export class RenderTargetJobRegistrySnapshot {
	public constructor(
		private readonly _jobs: readonly PreparedRenderTargetJob[] = [],
	) {}

	public getAll(): readonly PreparedRenderTargetJob[] {
		return this._jobs.slice();
	}

	public getForTarget(targetId: string): readonly PreparedRenderTargetJob[] {
		return this._jobs.filter((job) => job.targetId === targetId);
	}

	public get size(): number {
		return this._jobs.length;
	}
}

export interface RenderTargetReadbackOptions {
	readonly width?: number;
	readonly height?: number;
}

export type RenderTargetReadbackOrigin = "top-left" | "bottom-left";

export interface RenderTargetReadbackResult extends TextureReadbackResult {
	readonly origin: RenderTargetReadbackOrigin;
}

export interface CustomRenderTargetAttachment {
	/** Actual texture attached while recording the render pass. */
	readonly texture: IRenderTexture;
	readonly format: TextureFormat;
	/** Single-sample color result, or `null` when no automatic resolve exists. */
	readonly resolveTexture: IRenderTexture | null;
}

export interface CustomRenderTargetExecutionTarget {
	readonly id: string;
	readonly width: number;
	readonly height: number;
	readonly color: readonly CustomRenderTargetAttachment[];
	readonly depth?: CustomRenderTargetAttachment | null;
	readonly sampleCount: number;
}

export interface CustomRenderPassResourceFacade {
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	createShaderModule(desc: ShaderModuleDesc): Promise<IShaderModule> | IShaderModule;
	createRenderPipeline(desc: PipelineDesc): Promise<IRenderPipeline> | IRenderPipeline;
	createComputePipeline?(desc: ComputePipelineDesc): Promise<IComputePipeline> | IComputePipeline;
	createBindingGroup(desc: BindingGroupDesc): IBindingGroup;
}

export interface CustomRenderPassContext {
	readonly backend: RenderBackendType;
	readonly frameContext: FrameContext;
	readonly encoder: ICommandEncoder;
	readonly target: CustomRenderTargetExecutionTarget;
	readonly width: number;
	readonly height: number;
	readonly resources: CustomRenderPassResourceFacade;
}

export class RenderTargetRegistrySnapshot {
	private readonly _targets = new Map<string, RenderTargetDescriptor>();

	public constructor(descriptors: readonly RenderTargetDescriptor[] = []) {
		for (const descriptor of descriptors) {
			const normalized = normalizeRenderTargetDescriptor(descriptor);
			this._targets.set(normalized.id, normalized);
		}
	}

	public get(id: string): RenderTargetDescriptor | null {
		const descriptor = this._targets.get(id);
		return descriptor ? cloneRenderTargetDescriptor(descriptor) : null;
	}

	public has(id: string): boolean {
		return this._targets.has(id);
	}

	public getAll(): readonly RenderTargetDescriptor[] {
		return Array.from(this._targets.values()).map(cloneRenderTargetDescriptor);
	}
}

interface ManagedRenderTargetJob {
	readonly id: string;
	readonly targetId: string;
	readonly descriptor: RenderTargetJobDescriptor;
	readonly recurring: boolean;
	resolve?: (completion: RenderTargetJobCompletion) => void;
	reject?: (error: unknown) => void;
	cancelled: boolean;
	generation: number;
}

interface ManagedRenderTarget {
	readonly descriptor: RenderTargetDescriptor;
	readonly queued: ManagedRenderTargetJob[];
	readonly recurring: Map<string, ManagedRenderTargetJob>;
	destroyed: boolean;
	generation: number;
	committedGeneration: number;
	readonly internal: boolean;
}

export interface RenderTargetManagerOptions {
	readonly supportsJobs: boolean;
	readonly readColor: (
		id: string,
		attachmentIndex: number,
		options?: RenderTargetReadbackOptions,
	) => Promise<RenderTargetReadbackResult>;
	readonly invalidate: (scope: "public" | "internal") => void;
}

/** Handle-centric owner for public and renderer-private render targets. */
export class RenderTargetManager {
	private readonly _targets = new Map<string, ManagedRenderTarget>();
	private readonly _frameJobs = new WeakMap<
		RenderTargetJobRegistrySnapshot,
		ManagedRenderTargetJob[]
	>();
	private readonly _inflight = new Set<ManagedRenderTargetJob>();
	private _nextTargetId = 0;
	private _nextJobId = 0;

	public constructor(private readonly _options: RenderTargetManagerOptions) {}

	/** Creates a persistent renderer-owned render target. */
	public create(descriptor: RenderTargetCreateDescriptor): RenderTargetHandle {
		return this._createTarget(descriptor, false);
	}

	/** @internal Creates a target hidden from the public renderer API. */
	public createInternal(descriptor: RenderTargetCreateDescriptor): RenderTargetHandle {
		return this._createTarget({
			...descriptor,
			label: descriptor.label ?? "RendererInternalRenderTarget",
		}, true);
	}

	private _createTarget(
		descriptor: RenderTargetCreateDescriptor,
		internal: boolean,
	): RenderTargetHandle {
		const id = `render-target-${++this._nextTargetId}`;
		const normalized = normalizeRenderTargetDescriptor({ ...descriptor, id });
		const target: ManagedRenderTarget = {
			descriptor: normalized,
			queued: [],
			recurring: new Map(),
			destroyed: false,
			generation: 0,
			committedGeneration: 0,
			internal,
		};
		this._targets.set(id, target);
		this._invalidateTarget(target);
		return this._createHandle(target);
	}

	/** @internal Captures backend allocation descriptors for frame coordination. */
	public createTargetSnapshot(): RenderTargetRegistrySnapshot {
		return new RenderTargetRegistrySnapshot(
			Array.from(this._targets.values())
				.filter((target) => !target.destroyed)
				.map((target) => target.descriptor),
		);
	}

	/** @internal Prepares queued and recurring target work for one frame. */
	public createJobSnapshot(
		prepareScene: (descriptor: RenderTargetSceneViewJobDescriptor) => PreparedScene,
	): RenderTargetJobRegistrySnapshot {
		const prepared: PreparedRenderTargetJob[] = [];
		const captured: ManagedRenderTargetJob[] = [];
		for (const target of this._targets.values()) {
			if (target.destroyed) continue;
			const jobs: ManagedRenderTargetJob[] = [];
			let blocksLaterWriter = false;
			while (target.queued.length > 0 && !blocksLaterWriter) {
				const queued = target.queued.shift()!;
				jobs.push(queued);
				blocksLaterWriter = queued.descriptor.readback != null;
			}
			if (!blocksLaterWriter) {
				for (const recurring of target.recurring.values()) {
					jobs.push(recurring);
					if (recurring.descriptor.readback != null) break;
				}
			}
			for (const job of jobs) {
				if (job.cancelled) continue;
				job.generation = ++target.generation;
				prepared.push({
					id: job.id,
					targetId: job.targetId,
					generation: job.generation,
					recurring: job.recurring,
					descriptor: job.descriptor,
					scene:
						job.descriptor.kind === "scene-view" ?
							prepareScene(job.descriptor)
						: null,
				});
				captured.push(job);
				this._inflight.add(job);
			}
		}
		const snapshot = new RenderTargetJobRegistrySnapshot(prepared);
		this._frameJobs.set(snapshot, captured);
		return snapshot;
	}

	/** @internal Publishes job generations after a successful backend frame. */
	public async commitFrame(snapshot: RenderTargetJobRegistrySnapshot): Promise<void> {
		const jobs = this._frameJobs.get(snapshot) ?? [];
		this._frameJobs.delete(snapshot);
		for (const job of jobs) {
			this._inflight.delete(job);
			const target = this._targets.get(job.targetId);
			if (target && !job.cancelled) {
				target.committedGeneration = Math.max(
					target.committedGeneration,
					job.generation,
				);
			}
			if (job.recurring || job.cancelled) continue;
			try {
				const request = job.descriptor.readback ?? null;
				const attachmentIndex = request?.attachmentIndex ?? 0;
				const readback = request ?
					await this._options.readColor(job.targetId, attachmentIndex, request)
				: null;
				job.resolve?.({
					jobId: job.id,
					targetId: job.targetId,
					generation: job.generation,
					readback,
				});
			} catch (error) {
				job.reject?.(error);
			}
		}
	}

	/** @internal Rejects failed work or requeues a non-error frame abort. */
	public abortFrame(
		snapshot: RenderTargetJobRegistrySnapshot,
		error?: unknown,
	): void {
		const jobs = this._frameJobs.get(snapshot) ?? [];
		this._frameJobs.delete(snapshot);
		for (let index = jobs.length - 1; index >= 0; index--) {
			const job = jobs[index];
			this._inflight.delete(job);
			if (job.recurring || job.cancelled) continue;
			if (error !== undefined) {
				job.reject?.(error);
				continue;
			}
			const target = this._targets.get(job.targetId);
			if (target && !target.destroyed) target.queued.unshift(job);
		}
	}

	/** Destroys every target owned by this manager. */
	public destroy(): void {
		for (const target of Array.from(this._targets.values())) {
			this._destroyTarget(target, new Error("Renderer render targets were destroyed."));
		}
		this._targets.clear();
	}

	private _createHandle(target: ManagedRenderTarget): RenderTargetHandle {
		const manager = this;
		return {
			id: target.descriptor.id,
			descriptor: cloneRenderTargetDescriptor(target.descriptor),
			enqueueJob(descriptor) {
				manager._assertTargetAlive(target);
				manager._validateJobTarget(target, descriptor);
				const job = manager._createJob(target, descriptor, false);
				let resolve!: (completion: RenderTargetJobCompletion) => void;
				let reject!: (error: unknown) => void;
				const done = new Promise<RenderTargetJobCompletion>((accept, fail) => {
					resolve = accept;
					reject = fail;
				});
				job.resolve = resolve;
				job.reject = reject;
				target.queued.push(job);
				manager._invalidateTarget(target);
				return {
					id: job.id,
					done,
					cancel: () => manager._cancelJob(job),
				};
			},
			registerJob(descriptor) {
				manager._assertTargetAlive(target);
				manager._validateJobTarget(target, descriptor);
				if (descriptor.readback) {
					throw new Error(
						"Recurring render-target jobs cannot request per-frame readback.",
					);
				}
				const job = manager._createJob(target, descriptor, true);
				target.recurring.set(job.id, job);
				manager._invalidateTarget(target);
				return {
					id: job.id,
					destroy: () => {
						job.cancelled = true;
						target.recurring.delete(job.id);
						manager._invalidateTarget(target);
					},
				};
			},
			readColor(attachmentIndex = 0, options) {
				manager._assertTargetAlive(target);
				if (target.committedGeneration <= 0) {
					return Promise.reject(new Error(
						`Render target "${target.descriptor.id}" has no committed generation.`,
					));
				}
				return manager._options.readColor(target.descriptor.id, attachmentIndex, options);
			},
			destroy() {
				manager._destroyTarget(target, new Error(
					`Render target "${target.descriptor.id}" was destroyed.`,
				));
			},
		};
	}

	private _createJob(
		target: ManagedRenderTarget,
		descriptor: RenderTargetJobDescriptor,
		recurring: boolean,
	): ManagedRenderTargetJob {
		return {
			id: `render-target-job-${++this._nextJobId}`,
			targetId: target.descriptor.id,
			descriptor,
			recurring,
			cancelled: false,
			generation: 0,
		};
	}

	private _cancelJob(job: ManagedRenderTargetJob): void {
		if (job.cancelled) return;
		job.cancelled = true;
		const target = this._targets.get(job.targetId);
		if (target) {
			const index = target.queued.indexOf(job);
			if (index >= 0) target.queued.splice(index, 1);
		}
		job.reject?.(new Error(`Render target job "${job.id}" was cancelled.`));
	}

	private _destroyTarget(target: ManagedRenderTarget, error: Error): void {
		if (target.destroyed) return;
		target.destroyed = true;
		this._targets.delete(target.descriptor.id);
		for (const job of [...target.queued, ...target.recurring.values()]) {
			job.cancelled = true;
			job.reject?.(error);
		}
		for (const job of this._inflight) {
			if (job.targetId !== target.descriptor.id) continue;
			job.cancelled = true;
			job.reject?.(error);
			this._inflight.delete(job);
		}
		target.queued.length = 0;
		target.recurring.clear();
		this._invalidateTarget(target);
	}

	private _validateJobTarget(
		target: ManagedRenderTarget,
		descriptor: RenderTargetJobDescriptor,
	): void {
		if (!this._options.supportsJobs) {
			throw new Error("Render-target jobs are unsupported by the active backend.");
		}
		if (descriptor.kind !== "scene-view") return;
		if (
			target.descriptor.color.length !== 1 ||
			target.descriptor.color[0].format !== TextureFormat.RGBA16Float ||
			(target.descriptor.sampleCount ?? 1) !== 1 ||
			(target.descriptor.depth &&
				target.descriptor.depth.format !== TextureFormat.Depth32Float)
		) {
			throw new Error(
				"Scene-view render targets require one rgba16float color attachment, " +
					"sampleCount 1, and optional depth32float depth.",
			);
		}
	}

	private _assertTargetAlive(target: ManagedRenderTarget): void {
		if (target.destroyed) throw new Error("Render target has been destroyed.");
	}

	private _invalidateTarget(target: ManagedRenderTarget): void {
		this._options.invalidate(target.internal ? "internal" : "public");
	}
}

function normalizeRenderTargetDescriptor(
	descriptor: RenderTargetDescriptor
): RenderTargetDescriptor {
	if (!descriptor?.id) {
		throw new Error("Render target id is required.");
	}
	if (!descriptor.color || descriptor.color.length <= 0) {
		throw new Error(`Render target "${descriptor.id}" requires at least one color attachment.`);
	}
	const requestedSampleCount = descriptor.sampleCount ?? 1;
	if (!Number.isFinite(requestedSampleCount)) {
		throw new Error(
			`Render target "${descriptor.id}" sampleCount must be a finite number.`,
		);
	}
	const sampleCount = Math.max(1, Math.floor(requestedSampleCount));
	validateSizeDescriptor(descriptor.id, descriptor.size);
	return {
		id: descriptor.id,
		label: descriptor.label,
		size: { ...descriptor.size } as RenderTargetSizeDescriptor,
		color: descriptor.color.map((attachment, index) => {
			if (!attachment?.format) {
				throw new Error(
					`Render target "${descriptor.id}" color attachment ${index} requires a format.`
				);
			}
			const info = getTextureFormatInfo(attachment.format);
			if (info.formatClass !== "color" || !info.isRenderable) {
				throw new Error(
					`Render target "${descriptor.id}" color attachment ${index} requires a renderable color format.`
				);
			}
			return { ...attachment };
		}),
		depth: normalizeDepthAttachment(descriptor),
		sampleCount,
	};
}

function normalizeDepthAttachment(
	descriptor: RenderTargetDescriptor
): RenderTargetDepthAttachmentDescriptor | null {
	if (!descriptor.depth) {
		return null;
	}
	if (!descriptor.depth.format) {
		throw new Error(`Render target "${descriptor.id}" depth attachment requires a format.`);
	}
	const info = getTextureFormatInfo(descriptor.depth.format);
	if (info.formatClass !== "depth" || !info.isRenderable) {
		throw new Error(
			`Render target "${descriptor.id}" depth attachment requires a renderable depth-only format.`
		);
	}
	return { ...descriptor.depth };
}

function validateSizeDescriptor(id: string, size: RenderTargetSizeDescriptor): void {
	if (!size) {
		throw new Error(`Render target "${id}" size is required.`);
	}
	if (size.mode === "fixed") {
		if (
			!Number.isFinite(size.width) ||
			!Number.isFinite(size.height) ||
			size.width <= 0 ||
			size.height <= 0
		) {
			throw new Error(`Render target "${id}" fixed size must be positive.`);
		}
		return;
	}
	if (size.mode === "canvas-scale") {
		const scale = size.scale ?? 1;
		if (!Number.isFinite(scale) || scale <= 0) {
			throw new Error(`Render target "${id}" canvas scale must be positive.`);
		}
		return;
	}
	throw new Error(`Render target "${id}" has unsupported size mode.`);
}

function cloneRenderTargetDescriptor(
	descriptor: RenderTargetDescriptor
): RenderTargetDescriptor {
	return {
		...descriptor,
		size: { ...descriptor.size } as RenderTargetSizeDescriptor,
		color: descriptor.color.map((attachment) => ({ ...attachment })),
		depth: descriptor.depth ? { ...descriptor.depth } : null,
	};
}


function validateReadbackOptions(
	id: string,
	options: RenderTargetReadbackOptions | undefined
): void {
	for (const [name, value] of [
		["width", options?.width],
		["height", options?.height],
	] as const) {
		if (value !== undefined && (!Number.isInteger(value) || value <= 0)) {
			throw new Error(`Render target "${id}" readback ${name} must be a positive integer.`);
		}
	}
}
