import type {
	IRenderBackend,
	RenderBackendAttachContext,
	RenderBackendCompletedFrameCoverage,
	RenderBackendDebugInfo,
	RenderBackendProfile,
	RenderSurfaceSize,
} from "../IRenderBackend";
import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type DrawPacket,
	type FrameContext,
	type FramePass,
} from "../../pipeline/types";
import { SoftwareSurfaceRuntime } from "./SoftwareSurfaceRuntime";
import { SoftwarePassExecutor } from "./SoftwarePassExecutor";
import { SoftwareFrameRuntime } from "./SoftwareFrameRuntime";
import { SkyboxRenderer } from "./SkyboxRenderer";
import { isShadowCastingLight } from "../../lights";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	mergeParticleShadowBounds,
	resolveParticleShadowCasterBounds,
} from "../../pipeline/ParticleShadowVolume";
import {
	selectCSMDirectionalLights,
	type ShadowBackendCapabilities,
} from "../../pipeline/ShadowMetadata";
import { FrameAttachments } from "../../pipeline/types";
import { TemporalFrameState } from "../cross/TemporalFrameState";
import { SOFTWARE_TEMPORAL_RENDER_STATE_KEY } from "./SoftwareTemporalRenderState";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";
import {
	assertShaderDirectiveProfileRegistryComplete,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
} from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";
import {
	createRenderBackendExtensionRegistry,
} from "../BackendExtensions";
import {
	DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	displayOutputStatesEqual,
	resolveSDROnlyDisplayOutput,
	type DisplayOutputOptions,
	type DisplayOutputState,
} from "../../rendering/DisplayOutput";

export interface SoftwareBackendOptions {
	enableEarlyZPrepass?: boolean;
}

type SoftwarePassHandler = (
	context: FrameContext
) => void | Promise<void>;

type SoftwareBackendState =
	| "detached"
	| "attached"
	| "initializing"
	| "ready"
	| "frame-active"
	| "restoring"
	| "destroyed";

interface SoftwareTemporalCommit {
	previousWorldMatrices: Map<string, FrameContext["worldMatrix"]>;
}

const SOFTWARE_SHADOW_CAPABILITIES: ShadowBackendCapabilities = {
	backendKey: "software",
	supportsSingleMap: true,
	supportsDirectionalCSM: true,
	supportsSpotCSM: true,
	supportsPointCSM: true,
	maxCsmDirectionalLights: 1,
	maxDynamicShadowCost: 20,
};

function resolvePreparedSceneEnvironment(scene: FrameContext["scene"]): {
	backgroundEnabled: boolean;
	lightingEnabled: boolean;
	backgroundTexture: any;
	iblTexture: any;
	backgroundStrength: number;
	backgroundTintLinear: { r: number; g: number; b: number };
	backgroundExposure: number;
} {
	const environment = (scene as { environment?: unknown }).environment as
		| {
				backgroundEnabled?: boolean;
				lightingEnabled?: boolean;
				backgroundTexture?: unknown;
				iblTexture?: unknown;
				backgroundStrength?: number;
				backgroundTintLinear?: { r?: number; g?: number; b?: number };
				backgroundExposure?: number;
		  }
		| undefined;
	return {
		backgroundEnabled: environment?.backgroundEnabled ?? true,
		lightingEnabled: environment?.lightingEnabled ?? true,
		backgroundTexture:
			(environment?.backgroundTexture as any | null | undefined) ?? null,
		iblTexture: (environment?.iblTexture as any | null | undefined) ?? null,
		backgroundStrength:
			typeof environment?.backgroundStrength === "number" ?
				environment.backgroundStrength
			:	1,
		backgroundTintLinear: {
			r:
				typeof environment?.backgroundTintLinear?.r === "number" ?
					environment.backgroundTintLinear.r
				:	1,
			g:
				typeof environment?.backgroundTintLinear?.g === "number" ?
					environment.backgroundTintLinear.g
				:	1,
			b:
				typeof environment?.backgroundTintLinear?.b === "number" ?
					environment.backgroundTintLinear.b
				:	1,
		},
		backgroundExposure:
			typeof environment?.backgroundExposure === "number" ?
				environment.backgroundExposure
			:	1,
	};
}

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
			supportsFilterModes: ["pcf", "vsm"],
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
	private readonly _temporalFrameState = new TemporalFrameState();
	private _previousWorldMatrices = new Map<string, FrameContext["worldMatrix"]>();
	private _postProcessPlan: PostProcessPlan | null = null;
	private readonly _frameRuntime = new SoftwareFrameRuntime();
	private _options: SoftwareBackendOptions;
	private readonly _passHandlers: Map<FramePass["stage"], SoftwarePassHandler>;
	private _displayOutputState = resolveSDROnlyDisplayOutput(
		DEFAULT_DISPLAY_OUTPUT_OPTIONS,
	);

	public constructor(options: SoftwareBackendOptions = {}) {
		assertShaderDirectiveProfileRegistryComplete(DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY);
		this._options = options;
		this._passHandlers = this._createPassHandlers();
		this._ensureRuntime();
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

	/**
	 * Returns software backend diagnostics.
	 *
	 * @returns A stable unavailable snapshot because this backend does not own
	 * a GPU device.
	 * @sideEffects None.
	 */
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
		if (this._state !== "attached") {
			this._throwForInitializeState();
		}
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
			const replacement = this._createExecutor();
			const previous = this._executor;
			this._executor = replacement;
			previous?.destroy();
			this._resetTemporalState();
			this._state = "ready";
			this._requireAttachContext().events.emit({ type: "device-restored" });
		} catch (error) {
			this._state = "ready";
			throw error;
		}
	}

	private _ensureRuntime(): void {
		this._executor ??= this._createExecutor();
	}

	private _createExecutor(): SoftwarePassExecutor {
		return new SoftwarePassExecutor({
			backend: this,
			backendOptions: this._options,
			getCanvasContext: () => this._surface.getCanvasContext(),
		});
	}

	public getAttachments(size: RenderSurfaceSize): FrameAttachments {
		if (this._state === "detached" || this._state === "destroyed") {
			throw new Error("SoftwareBackend.getAttachments() requires attach() to complete.");
		}
		if (this._state === "frame-active") {
			throw new Error("SoftwareBackend cannot reconfigure attachments while a frame is active.");
		}
		return this._surface.getAttachments(size);
	}

	public resize(size: RenderSurfaceSize): void {
		if (this._state === "detached" || this._state === "destroyed") {
			throw new Error("SoftwareBackend.resize() requires attach() to complete.");
		}
		if (this._state === "frame-active") {
			this._frameRuntime.queueResize(size);
			return;
		}
		this._applyResize(size);
	}

	private _applyResize(size: RenderSurfaceSize): void {
		this._surface.resize(size);
		this._executor?.invalidateFrameSized();
		this._resetTemporalState();
	}

	public beginFrame(context: FrameContext): void {
		if (this._state !== "ready") {
			throw new Error(
				`SoftwareBackend.beginFrame() requires the ready state; current state is "${this._state}".`,
			);
		}
		this._state = "frame-active";
		this._frameRuntime.begin(context);
		this._executor?.beginFrame(context);
		this._postProcessPlan =
			this._executor?.planPostProcessFrame(context) ?? null;
		this._prepareTemporalRenderState(context);

		const pixels = context.attachments.pixels!;
		const depthBuffer = context.attachments.depthBuffer!;
		const normalBuffer = context.attachments.normalBuffer;
		const motionBuffer = context.attachments.motionBuffer;
		const frameWidth = context.attachments.width;
		const frameHeight = context.attachments.height;
		const incrementalPartial = this._isIncrementalPartial(context);
		const dirtyRects = this._resolveDirtyRects(context);

		if (!incrementalPartial) {
			const size = pixels.length >> 2;
			for (let i = 0; i < size; i++) {
				const index = i << 2;
				pixels[index] = 0;
				pixels[index + 1] = 0;
				pixels[index + 2] = 0;
				pixels[index + 3] = 255;
			}
			depthBuffer.fill(Infinity);
			normalBuffer?.fill(0);
			motionBuffer?.fill(0);
		} else {
			for (const rect of dirtyRects) {
				const minX = Math.max(0, Math.floor(rect.x));
				const minY = Math.max(0, Math.floor(rect.y));
				const maxX = Math.min(frameWidth, Math.ceil(rect.x + rect.width));
				const maxY = Math.min(frameHeight, Math.ceil(rect.y + rect.height));
				if (minX >= maxX || minY >= maxY) {
					continue;
				}

				for (let y = minY; y < maxY; y++) {
					const rowStart = y * frameWidth;
					for (let x = minX; x < maxX; x++) {
						const pixelIndex = (rowStart + x) << 2;
						pixels[pixelIndex] = 0;
						pixels[pixelIndex + 1] = 0;
						pixels[pixelIndex + 2] = 0;
						pixels[pixelIndex + 3] = 255;
						depthBuffer[rowStart + x] = Infinity;
						if (normalBuffer) {
							const normalIndex = (rowStart + x) * 3;
							normalBuffer[normalIndex] = 0;
							normalBuffer[normalIndex + 1] = 0;
							normalBuffer[normalIndex + 2] = 0;
						}
						if (motionBuffer) {
							const motionIndex = (rowStart + x) * 4;
							motionBuffer[motionIndex] = 0;
							motionBuffer[motionIndex + 1] = 0;
							motionBuffer[motionIndex + 2] = 0;
							motionBuffer[motionIndex + 3] = 0;
						}
					}
				}
			}
		}

		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);
		const shadowCasterBounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds,
		);
		const combinedShadowCasterBounds = mergeParticleShadowBounds(
			shadowCasterBounds,
			resolveParticleShadowCasterBounds(context.scene.particleSystems),
		);
		const selectedCSMLights = selectCSMDirectionalLights(
			shadowLights,
			SOFTWARE_SHADOW_CAPABILITIES.maxCsmDirectionalLights,
		);
		for (const shadowLight of shadowLights) {
			const shadowRenderSet = context.shadowMaps.get(shadowLight);
			if (shadowRenderSet) {
				updateShadowMapMetadata(shadowRenderSet, shadowLight, combinedShadowCasterBounds, {
					camera: context.scene.camera,
					backendCapabilities: SOFTWARE_SHADOW_CAPABILITIES,
					allowCSMDirectionalLights: selectedCSMLights,
					onWarning: (key, message) =>
						Logger.warn(`[${key}] ${message}`, {
							scope: "SoftwareBackend",
							onceKey: key,
						}),
				});
			}
		}

		const environment = resolvePreparedSceneEnvironment(context.scene);
		if (
			!incrementalPartial &&
			context.features.enableEnvironment &&
			environment.backgroundEnabled &&
			environment.backgroundTexture
		) {
			SkyboxRenderer.render(
				environment.backgroundTexture,
				{
					strength: environment.backgroundStrength,
					tintLinear: environment.backgroundTintLinear,
					exposure: environment.backgroundExposure,
				},
				pixels,
				context.viewCamera,
				context.attachments.width,
				context.attachments.height,
			);
		}
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		this._requireActiveContext(context, "executePass");
		if (!this._executor) return;

		if (context.customRenderPasses?.has(pass.stage)) {
			const key = "software-custom-render-targets-unsupported";
			Logger.warn(
				`[${key}] Software backend does not support custom render targets or custom render passes yet; skipping pass "${pass.stage}".`,
				{ scope: "SoftwareBackend", onceKey: key },
			);
			return;
		}

		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			const key = `software-pass-unsupported-${pass.stage}`;
			Logger.warn(
				`[${key}] Software backend does not support pass "${pass.stage}" yet; skipping`,
				{ scope: "SoftwareBackend", onceKey: key },
			);
			return;
		}
		await handler(context);
	}

	public skipPass(_pass: FramePass): void {
		this._requireActiveContext(this._frameRuntime.activeContext, "skipPass");
		// No pass dependency tracking in SoftwareBackend; no-op.
	}

	public endFrame(): void {
		const context = this._requireActiveContext(this._frameRuntime.activeContext, "endFrame");
		this._executor?.endParticleFrame();
		const temporalCommit = this._prepareTemporalRenderCommit(context);

		this._surface.present();

		this._executor?.commitFrame();
		this._temporalFrameState.commitFrame();
		this._commitTemporalRenderState(temporalCommit);
		if (this._canPreserveNonDirtyTiles(context)) {
			this._frameRuntime.complete(true);
		} else {
			this._frameRuntime.complete(false);
		}
		this._postProcessPlan = null;
		this._state = "ready";
		this._flushPendingResize();
	}

	/** @internal Returns sanitized post-process graph diagnostics for backend tests. */
	public getPostProcessGraphDebugState(): unknown {
		return this._executor?.getPostProcessDebugState() ?? null;
	}

	/** @internal Renderer frame-coordination coverage report. */
	public getCompletedFrameCoverage(): RenderBackendCompletedFrameCoverage {
		return this._frameRuntime.completedCoverage;
	}

	public async abortFrame(_error?: unknown): Promise<void> {
		let abortError: unknown = null;
		try {
			await this._executor?.abortFrame(_error);
		} catch (error) {
			abortError = error;
		}
		try {
			this._executor?.endParticleFrame();
		} catch (error) {
			abortError ??= error;
		} finally {
			this._temporalFrameState.abortFrame();
			this._postProcessPlan = null;
			this._frameRuntime.abort();
			if (this._state === "frame-active") {
				this._state = "ready";
			}
			this._flushPendingResize();
		}
		if (abortError) throw abortError;
	}

	private _prepareTemporalRenderState(context: FrameContext): void {
		const snapshot = this._temporalFrameState.beginFrame({
			camera: context.viewCamera,
			width: context.attachments.width,
			height: context.attachments.height,
			frameRequirements: this._postProcessPlan?.frameRequirements ?? {},
			reset: context.incremental.temporalHistoryReset,
		});
		const resetHistory = context.incremental.temporalHistoryReset;
		context.transient.set(SOFTWARE_TEMPORAL_RENDER_STATE_KEY, {
			currentJitter: [
				snapshot.jitterCurrentPrev[0],
				snapshot.jitterCurrentPrev[1],
			],
			previousJitter: [
				snapshot.jitterCurrentPrev[2],
				snapshot.jitterCurrentPrev[3],
			],
			previousViewProjection: snapshot.previousViewProjection,
			currentViewProjection: context.viewCamera.viewProjectionMatrix,
			previousWorldMatrices: resetHistory ? new Map() : this._previousWorldMatrices,
			currentWorldMatrices: new Map(),
		});
	}

	private _prepareTemporalRenderCommit(
		context: FrameContext,
	): SoftwareTemporalCommit | null {
		const state = context.transient.get(SOFTWARE_TEMPORAL_RENDER_STATE_KEY);
		if (!state) {
			return null;
		}
		return {
			previousWorldMatrices: new Map(state.currentWorldMatrices),
		};
	}

	private _commitTemporalRenderState(commit: SoftwareTemporalCommit | null): void {
		if (!commit) return;
		this._previousWorldMatrices = commit.previousWorldMatrices;
	}

	private _resolveParticleDeltaTime(context: FrameContext): number {
		const value = context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY);
		if (typeof value !== "number" || !Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, value);
	}

	private _isIncrementalPartial(context: FrameContext): boolean {
		const incremental = context.incremental;
		return (
			incremental.enabled && !incremental.forceFullFrame && incremental.dirtyRects.length > 0
		);
	}

	private _canPreserveNonDirtyTiles(context: FrameContext): boolean {
		return (
			this._isIncrementalPartial(context) &&
			!context.incremental.temporalHistoryReset &&
			(context.postProcess.getEnabledPasses().length === 0 ||
				this._executor?.completedFramePreservesOutsideDirtyTiles === true)
		);
	}

	private _resolveDirtyRects(
		context: FrameContext,
	): Array<{ x: number; y: number; width: number; height: number }> {
		const width = Math.max(1, context.attachments.width | 0);
		const height = Math.max(1, context.attachments.height | 0);
		if (!this._isIncrementalPartial(context)) {
			return [
				{
					x: 0,
					y: 0,
					width,
					height,
				},
			];
		}
		const result: Array<{ x: number; y: number; width: number; height: number }> = [];
		const incremental = context.incremental;
		for (const rect of incremental.dirtyRects) {
			const minX = Math.max(0, Math.floor(rect.x));
			const minY = Math.max(0, Math.floor(rect.y));
			const maxX = Math.min(width, Math.ceil(rect.x + rect.width));
			const maxY = Math.min(height, Math.ceil(rect.y + rect.height));
			const rectWidth = maxX - minX;
			const rectHeight = maxY - minY;
			if (rectWidth <= 0 || rectHeight <= 0) {
				continue;
			}
			result.push({
				x: minX,
				y: minY,
				width: rectWidth,
				height: rectHeight,
			});
		}
		return result;
	}

	private _resolvePacketsForPass(context: FrameContext, packets: DrawPacket[]): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex || !this._isIncrementalPartial(context)) {
			return packets;
		}
		const dirtyRects = context.incremental.dirtyRects;
		if (dirtyRects.length === 0) {
			return [];
		}
		if (packets === context.scene.opaquePackets) {
			return spatialIndex.queryOpaquePacketsInRects(dirtyRects);
		}
		if (packets === context.scene.transparentPackets) {
			return spatialIndex.queryTransparentPacketsInRects(dirtyRects);
		}
		return packets;
	}

	private _resolveOpaqueReflectivePackets(packets: DrawPacket[]): DrawPacket[] {
		return packets.filter(
			(packet) => packet.material.reflectivity > 0 && packet.material.mirrorPlane !== null,
		);
	}

	public destroy(): void {
		if (this._state === "destroyed") return;
		this._executor?.endParticleFrame();
		this._destroyRuntimeResources();
		this._resetTemporalState();
		this._frameRuntime.clear();
		this._surface.destroy();
		this._state = "destroyed";
	}

	private _destroyRuntimeResources(): void {
		this._executor?.destroy();
		this._executor = null;
	}

	private _createPassHandlers(): Map<FramePass["stage"], SoftwarePassHandler> {
		return new Map<FramePass["stage"], SoftwarePassHandler>([
			[
				"particle-sim",
				(context) => {
					this._executor?.simulateParticles(
						context,
						this._resolveParticleDeltaTime(context),
					);
				},
			],
			[
				"shadow",
				(context) => {
					this._executor?.renderShadows(context);
				},
			],
			[
				"reflection",
				(context) => {
					this._executor?.renderReflections(context);
				},
			],
			[
				"main-opaque",
				async (context) => {
					const packets = this._resolvePacketsForPass(
						context,
						context.scene.opaquePackets,
					);
					await this._executor?.renderOpaque(
						context,
						packets,
						this._resolveOpaqueReflectivePackets(packets),
					);
				},
			],
			[
				"main-transparent",
				async (context) => {
					const packets = this._resolvePacketsForPass(
						context,
						context.scene.transparentPackets,
					);
					await this._executor?.renderTransparent(context, packets);
				},
			],
			[
				"particles",
				(context) => {
					this._executor?.renderParticles(context);
				},
			],
			[
				"postprocess",
				async (context) => {
					if (!this._postProcessPlan) {
						throw new Error("Software post-process plan is unavailable.");
					}
					await this._executor?.executePostProcess(
						context,
						this._postProcessPlan,
					);
				},
			],
		]);
	}

	private _resetTemporalState(): void {
		this._temporalFrameState.reset();
		this._previousWorldMatrices.clear();
		this._postProcessPlan = null;
	}

	private _flushPendingResize(): void {
		const pending = this._frameRuntime.consumePendingResize();
		if (pending) {
			this._applyResize(pending);
		}
	}

	private _requireActiveContext(
		context: FrameContext | null,
		operation: "executePass" | "skipPass" | "endFrame",
	): FrameContext {
		if (this._state !== "frame-active") {
			throw new Error(`SoftwareBackend.${operation}() requires an active frame.`);
		}
		return this._frameRuntime.requireActive(context, operation);
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
}
