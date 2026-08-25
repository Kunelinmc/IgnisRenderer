import type {
	IRenderBackend,
	RenderBackendAttachContext,
	RenderBackendCompletedFrameCoverage,
	RenderBackendDebugInfo,
	RenderBackendProfile,
	RenderSurfaceSize,
} from "../IRenderBackend";
import type { FrameAttachments, FrameContext, FramePass } from "../../pipeline/types";
import { SoftwareSurfaceRuntime } from "./SoftwareSurfaceRuntime";
import { SoftwarePassExecutor } from "./SoftwarePassExecutor";
import { SoftwareFrameSession } from "./SoftwareFrameSession";
import { Logger } from "../../foundation/Logger";
import { createRenderBackendExtensionRegistry } from "../BackendExtensions";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	displayOutputStatesEqual,
	createSDRDisplayOutputState,
	type DisplayOutputOptions,
	type DisplayOutputState,
	type ResolvedDisplayOutputOptions,
} from "../../rendering/DisplayOutput";
import type { SoftwareBackendOptions } from "./SoftwareBackendContracts";
import { SoftwareDisplayOutputManager } from "./SoftwareDisplayOutputManager";

export type { SoftwareBackendOptions } from "./SoftwareBackendContracts";

type SoftwareBackendState =
	| "detached"
	| "attached"
	| "initializing"
	| "ready"
	| "frame-active"
	| "restoring"
	| "destroyed";

/** CPU scanline rendering backend and renderer lifecycle facade. */
export class SoftwareBackend implements IRenderBackend {
	public readonly extensions = createRenderBackendExtensionRegistry([]);
	public readonly profile: RenderBackendProfile = {
		id: "software",
		capabilities: {
			displayHDR: true,
			sh: true,
			shadows: true,
			reflection: true,
			environment: true,
			postProcess: true,
			meshParticles: false,
			clusteredLighting: false,
			oit: false,
			occlusionCulling: false,
			renderTargets: false,
			renderTargetReadback: false,
		},
		frameScheduling: "on-demand",
		lighting: { localizedProbeMode: "accumulate-globally" },
	};

	private _attachContext: RenderBackendAttachContext | null = null;
	private _state: SoftwareBackendState = "detached";
	private readonly _displayOutput = new SoftwareDisplayOutputManager(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);
	private readonly _surface = new SoftwareSurfaceRuntime(this._displayOutput);
	private _executor: SoftwarePassExecutor | null = null;
	private _session: SoftwareFrameSession | null = null;
	private readonly _options: SoftwareBackendOptions;
	private _displayOutputState = createSDRDisplayOutputState(DEFAULT_DISPLAY_OUTPUT_OPTIONS);
	private _pendingDynamicRangeRefresh = false;
	private _pendingDisplayOutput: {
		requested: ResolvedDisplayOutputOptions;
		resolve: Array<(state: DisplayOutputState) => void>;
		reject: Array<(error: unknown) => void>;
	} | null = null;

	constructor(options: SoftwareBackendOptions = {}) {
		this._options = options;
		this._createRuntime();
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._state !== "detached") {
			throw new Error("SoftwareBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		this._displayOutputState = this._displayOutput.setRequested(context.surface.displayOutput);
		this._state = "attached";
	}

	/** Returns a stable unavailable snapshot because Software owns no GPU device. */
	public getDebugInfo(): RenderBackendDebugInfo {
		return {
			backend: "software",
			api: "software",
			available: false,
			unavailableReason: "Software backend does not use a GPU device.",
		};
	}

	public getDisplayOutputState(): DisplayOutputState {
		return this._displayOutputState;
	}

	public async setDisplayOutput(options: DisplayOutputOptions): Promise<DisplayOutputState> {
		const current = this._displayOutput.setRequested(options);
		if (this._state === "frame-active") {
			return new Promise<DisplayOutputState>((resolve, reject) => {
				if (this._pendingDisplayOutput) {
					this._pendingDisplayOutput.requested = current.requested;
					this._pendingDisplayOutput.resolve.push(resolve);
					this._pendingDisplayOutput.reject.push(reject);
				} else {
					this._pendingDisplayOutput = {
						requested: current.requested,
						resolve: [resolve],
						reject: [reject],
					};
				}
			});
		}
		this._commitDisplayOutput(current);
		return current;
	}

	public async initialize(): Promise<void> {
		if (this._state !== "attached") this._throwForInitializeState();
		this._state = "initializing";
		try {
			const previous = this._displayOutputState;
			this._displayOutputState = this._initializeSurface();
			this._displayOutput.observeDynamicRange(() => {
				this._handleDynamicRangeChange();
			});
			this._ensureRuntime();
			this._state = "ready";
			if (!displayOutputStatesEqual(previous, this._displayOutputState)) {
				this._emitDisplayOutputChange(previous, this._displayOutputState);
			}
			if (
				this._displayOutputState.requested.mode === "hdr" &&
				this._displayOutputState.fallbackReason
			) {
				Logger.warn(
					`[display-hdr-unavailable] SoftwareBackend could not activate HDR: ` +
						`${this._displayOutputState.fallbackReason}.`,
					{ scope: "SoftwareBackend", onceKey: "display-hdr-unavailable" },
				);
			}
		} catch (error) {
			this._state = "attached";
			throw error;
		}
	}

	public async restore(): Promise<void> {
		if (this._state !== "ready") {
			throw new Error(
				`SoftwareBackend.restore() requires the ready state; current state is "${this._state}".`,
			);
		}
		this._state = "restoring";
		try {
			const previousDisplay = this._displayOutputState;
			this._displayOutputState = this._initializeSurface();
			const replacement = this._buildRuntime();
			const previous = this._executor;
			this._executor = replacement.executor;
			this._session = replacement.session;
			previous?.destroy();
			this._state = "ready";
			if (!displayOutputStatesEqual(previousDisplay, this._displayOutputState)) {
				this._emitDisplayOutputChange(previousDisplay, this._displayOutputState);
			}
			this._requireAttachContext().events.emit({ type: "device-restored" });
		} catch (error) {
			this._state = "ready";
			throw error;
		}
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		if (this._state === "detached" || this._state === "destroyed") {
			throw new Error("SoftwareBackend.getAttachments() requires attach() to complete.");
		}
		if (this._state === "frame-active") {
			throw new Error(
				"SoftwareBackend cannot reconfigure attachments while a frame is active.",
			);
		}
		return this._surface.getAttachments(size);
	}

	public resize(size: RenderSurfaceSize): void {
		if (this._state === "detached" || this._state === "destroyed") {
			throw new Error("SoftwareBackend.resize() requires attach() to complete.");
		}
		this._requireSession().resize(size);
	}

	public beginFrame(context: FrameContext): void {
		if (this._state !== "ready") {
			throw new Error(
				`SoftwareBackend.beginFrame() requires the ready state; current state is "${this._state}".`,
			);
		}
		this._state = "frame-active";
		this._requireSession().begin(context);
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		this._requireActiveContext(context, "executePass");
		await this._requireSession().execute(pass, context);
	}

	public skipPass(_pass: FramePass): void {
		const context = this._requireActiveContext(
			this._requireSession().activeContext,
			"skipPass",
		);
		this._requireSession().skip(context);
	}

	public endFrame(): void {
		const session = this._requireSession();
		const context = this._requireActiveContext(session.activeContext, "endFrame");
		session.end(context);
		this._state = "ready";
		this._flushPendingDynamicRangeRefresh();
	}

	/** @internal Returns sanitized post-process graph diagnostics for tests. */
	public getPostProcessGraphDebugState(): unknown {
		return this._executor?.getPostProcessDebugState() ?? null;
	}

	/** @internal Returns active Software temporal transaction state for tests. */
	public getTemporalDebugState(): unknown {
		return this._session?.getTemporalDebugState() ?? null;
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._session?.completedCoverage ?? "full-frame";
	}

	public async abortFrame(error?: unknown): Promise<void> {
		try {
			await this._session?.abort(error);
		} finally {
			if (this._state === "frame-active") this._state = "ready";
			this._flushPendingDynamicRangeRefresh();
		}
	}

	public destroy(): void {
		if (this._state === "destroyed") return;
		const pendingDisplayOutput = this._pendingDisplayOutput;
		this._pendingDisplayOutput = null;
		for (const reject of pendingDisplayOutput?.reject ?? []) {
			reject(new Error("Software backend was destroyed before display output changed."));
		}
		this._session?.reset();
		this._executor?.destroy();
		this._executor = null;
		this._session = null;
		this._displayOutput.destroy();
		this._surface.destroy();
		this._pendingDynamicRangeRefresh = false;
		this._state = "destroyed";
	}

	private _createRuntime(): void {
		const runtime = this._buildRuntime();
		this._executor = runtime.executor;
		this._session = runtime.session;
	}

	private _buildRuntime(): {
		executor: SoftwarePassExecutor;
		session: SoftwareFrameSession;
	} {
		const executor = new SoftwarePassExecutor({
			backend: this,
			backendOptions: this._options,
			getSceneColor: () => this._surface.getSceneColorTarget(),
			getDisplayOutputState: () => this._displayOutputState,
		});
		return {
			executor,
			session: new SoftwareFrameSession(this._surface, executor),
		};
	}

	private _ensureRuntime(): void {
		if (!this._executor || !this._session) this._createRuntime();
	}

	private _initializeSurface(): DisplayOutputState {
		const canvas = this._requireAttachContext().surface.canvas;
		const detection = this._displayOutput.detect(canvas);
		let context: CanvasRenderingContext2D | null = null;
		try {
			context = canvas.getContext("2d", detection.contextSettings);
		} catch {
			// Unsupported HDR context settings must fall back without failing setup.
		}
		if (!context) {
			try {
				context = canvas.getContext("2d", {
					alpha: true,
					willReadFrequently: true,
				});
			} catch {
				// A canvas that cannot create any 2D context remains unavailable.
			}
		}
		const state = this._displayOutput.configure(context);
		this._surface.initialize(context);
		return state;
	}

	private _requireSession(): SoftwareFrameSession {
		if (!this._session) {
			throw new Error("SoftwareBackend runtime is unavailable.");
		}
		return this._session;
	}

	private _requireActiveContext(
		context: FrameContext | null,
		operation: "executePass" | "skipPass" | "endFrame",
	): FrameContext {
		if (this._state !== "frame-active") {
			throw new Error(`SoftwareBackend.${operation}() requires an active frame.`);
		}
		return this._requireSession().requireActive(context, operation);
	}

	private _throwForInitializeState(): never {
		if (this._state === "detached") {
			throw new Error("SoftwareBackend.initialize() requires attach() to complete.");
		}
		throw new Error(
			`SoftwareBackend.initialize() requires the attached state; current state is "${this._state}".`,
		);
	}

	private _requireAttachContext(): RenderBackendAttachContext {
		if (!this._attachContext) {
			throw new Error("SoftwareBackend.attach() must be called before initialize().");
		}
		return this._attachContext;
	}

	private _handleDynamicRangeChange(): void {
		if (this._state === "frame-active") {
			this._pendingDynamicRangeRefresh = true;
			return;
		}
		const current = this._displayOutput.refreshDynamicRange();
		if (!this._commitDisplayOutput(current)) return;
		this._requireAttachContext().events.emit({
			type: "render-invalidated",
			reason: "display-output",
		});
	}

	private _flushPendingDynamicRangeRefresh(): void {
		if (!this._pendingDynamicRangeRefresh && !this._pendingDisplayOutput) return;
		const dynamicRangeChanged = this._pendingDynamicRangeRefresh;
		this._pendingDynamicRangeRefresh = false;
		const pendingDisplayOutput = this._pendingDisplayOutput;
		this._pendingDisplayOutput = null;
		const current = this._displayOutput.refreshDynamicRange();
		const changed = this._commitDisplayOutput(current);
		if (dynamicRangeChanged && changed) {
			this._requireAttachContext().events.emit({
				type: "render-invalidated",
				reason: "display-output",
			});
		}
		for (const resolve of pendingDisplayOutput?.resolve ?? []) resolve(current);
	}

	private _commitDisplayOutput(current: DisplayOutputState): boolean {
		const previous = this._displayOutputState;
		this._displayOutputState = current;
		if (current.requested.mode === "hdr" && current.fallbackReason) {
			Logger.warn(
				`[display-hdr-unavailable] SoftwareBackend could not activate HDR: ` +
					`${current.fallbackReason}.`,
				{ scope: "SoftwareBackend", onceKey: "display-hdr-unavailable" },
			);
		}
		if (displayOutputStatesEqual(previous, current)) return false;
		this._emitDisplayOutputChange(previous, current);
		return true;
	}

	private _emitDisplayOutputChange(
		previous: DisplayOutputState,
		current: DisplayOutputState,
	): void {
		this._requireAttachContext().events.emit({
			type: "display-output-change",
			previous,
			current,
		});
	}
}
