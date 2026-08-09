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
import {
	assertShaderDirectiveProfileRegistryComplete,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
} from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";
import { createRenderBackendExtensionRegistry } from "../BackendExtensions";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	displayOutputStatesEqual,
	resolveSDROnlyDisplayOutput,
	type DisplayOutputOptions,
	type DisplayOutputState,
} from "../../rendering/DisplayOutput";
import type { SoftwareBackendOptions } from "./SoftwareBackendContracts";

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
			displayHDR: false,
			sh: true,
			shadows: true,
			reflection: true,
			environment: true,
			postProcess: true,
			meshParticles: false,
			clusteredLighting: false,
			oit: false,
			occlusionCulling: false,
			customRenderTargets: false,
			customRenderPasses: false,
			renderTargetReadback: false,
		},
		frameScheduling: "on-demand",
		shadow: {
			backendKey: "software",
			supportsFilterModes: ["pcf"],
			lightTypes: {
				directional: {
					projections: ["single", "cascaded"],
					storage: ["atlas"],
					maxLights: 4,
					maxCascadedLights: 1,
				},
				spot: {
					projections: ["single", "cascaded"],
					storage: ["atlas"],
					maxLights: 8,
					maxCascadedLights: 8,
				},
				point: {
					projections: ["single", "cascaded"],
					storage: ["atlas"],
					maxLights: 16,
					maxCascadedLights: 16,
				},
			},
			supportsTransmission: true,
			supportsDirectionalCSM: true,
			supportsSpotCSM: true,
			supportsPointCSM: true,
			maxDynamicShadowCost: 20,
		},
		lighting: { localizedProbeMode: "accumulate-globally" },
	};

	private _attachContext: RenderBackendAttachContext | null = null;
	private _state: SoftwareBackendState = "detached";
	private readonly _surface = new SoftwareSurfaceRuntime();
	private _executor: SoftwarePassExecutor | null = null;
	private _session: SoftwareFrameSession | null = null;
	private readonly _options: SoftwareBackendOptions;
	private _displayOutputState = resolveSDROnlyDisplayOutput(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);

	public constructor(options: SoftwareBackendOptions = {}) {
		assertShaderDirectiveProfileRegistryComplete(
			DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
		);
		this._options = options;
		this._createRuntime();
	}

	public attach(context: RenderBackendAttachContext): void {
		if (this._state !== "detached") {
			throw new Error("SoftwareBackend is already attached to a renderer.");
		}
		this._attachContext = context;
		this._displayOutputState = resolveSDROnlyDisplayOutput(
			context.surface.displayOutput,
		);
		this._surface.attach(context.surface.canvas);
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

	public async setDisplayOutput(
		options: DisplayOutputOptions,
	): Promise<DisplayOutputState> {
		const previous = this._displayOutputState;
		const current = resolveSDROnlyDisplayOutput(options, previous.requested);
		this._displayOutputState = current;
		if (current.fallbackReason === "backend-unsupported") {
			Logger.warn(
				"[display-hdr-unavailable] SoftwareBackend supports SDR presentation only.",
				{ scope: "SoftwareBackend", onceKey: "display-hdr-unavailable" },
			);
		}
		if (!displayOutputStatesEqual(previous, current)) {
			this._requireAttachContext().events.emit({
				type: "display-output-change",
				previous,
				current,
			});
		}
		return current;
	}

	public async initialize(): Promise<void> {
		if (this._state !== "attached") this._throwForInitializeState();
		this._state = "initializing";
		try {
			this._surface.initialize();
			this._ensureRuntime();
			this._state = "ready";
			if (this._displayOutputState.fallbackReason === "backend-unsupported") {
				Logger.warn(
					"[display-hdr-unavailable] SoftwareBackend supports SDR presentation only.",
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
			this._surface.initialize();
			const replacement = this._buildRuntime();
			const previous = this._executor;
			this._executor = replacement.executor;
			this._session = replacement.session;
			previous?.destroy();
			this._state = "ready";
			this._requireAttachContext().events.emit({ type: "device-restored" });
		} catch (error) {
			this._state = "ready";
			throw error;
		}
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		if (this._state === "detached" || this._state === "destroyed") {
			throw new Error(
				"SoftwareBackend.getAttachments() requires attach() to complete.",
			);
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
		}
	}

	public destroy(): void {
		if (this._state === "destroyed") return;
		this._session?.reset();
		this._executor?.destroy();
		this._executor = null;
		this._session = null;
		this._surface.destroy();
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
		});
		return {
			executor,
			session: new SoftwareFrameSession(this._surface, executor),
		};
	}

	private _ensureRuntime(): void {
		if (!this._executor || !this._session) this._createRuntime();
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
			throw new Error(
				"SoftwareBackend.initialize() requires attach() to complete.",
			);
		}
		throw new Error(
			`SoftwareBackend.initialize() requires the attached state; current state is "${this._state}".`,
		);
	}

	private _requireAttachContext(): RenderBackendAttachContext {
		if (!this._attachContext) {
			throw new Error(
				"SoftwareBackend.attach() must be called before initialize().",
			);
		}
		return this._attachContext;
	}
}
