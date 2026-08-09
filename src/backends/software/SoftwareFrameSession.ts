import type {
	RenderBackendCompletedFrameCoverage,
	RenderSurfaceSize,
} from "../IRenderBackend";
import type { FrameContext, FramePass } from "../../pipeline/types";
import { TemporalFrameState } from "../cross/TemporalFrameState";
import { SkyboxRenderer } from "./SkyboxRenderer";
import { SoftwarePassExecutor } from "./SoftwarePassExecutor";
import { SoftwareSurfaceRuntime } from "./SoftwareSurfaceRuntime";
import {
	createSoftwareFrameView,
	type SoftwareFrameView,
} from "./SoftwareFrameView";
import type { SoftwareTemporalRenderState } from "./SoftwareTemporalRenderState";
import type { Matrix4 } from "../../maths/Matrix4";

/** @internal Owns the Software backend's active frame transaction. */
export class SoftwareFrameSession {
	private readonly _temporalFrameState = new TemporalFrameState();
	private _previousWorldMatrices = new Map<string, Matrix4>();
	private _activeContext: FrameContext | null = null;
	private _frame: SoftwareFrameView | null = null;
	private _completedCoverage: RenderBackendCompletedFrameCoverage = "full-frame";
	private _pendingResize: RenderSurfaceSize | null = null;

	public constructor(
		private readonly _surface: SoftwareSurfaceRuntime,
		private readonly _executor: SoftwarePassExecutor,
	) {}

	public begin(context: FrameContext): void {
		if (this._activeContext) {
			throw new Error("SoftwareBackend.beginFrame() requires no active frame.");
		}
		this._activeContext = context;
		const postProcessPlan = this._executor.beginCoordinatorFrame(context);
		const snapshot = this._temporalFrameState.beginFrame({
			camera: context.viewCamera,
			width: context.attachments.width,
			height: context.attachments.height,
			frameRequirements: postProcessPlan.frameRequirements ?? {},
			reset: context.incremental.temporalHistoryReset,
		});
		const resetHistory = context.incremental.temporalHistoryReset;
		const temporal: SoftwareTemporalRenderState = {
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
			previousWorldMatrices:
				resetHistory ? new Map() : this._previousWorldMatrices,
			currentWorldMatrices: new Map(),
		};
		this._frame = createSoftwareFrameView(context, temporal);
		this._executor.bindFrame(this._frame);
		this._prepareAttachments(this._frame);
	}

	public execute(pass: FramePass, context: FrameContext): Promise<void> {
		this.requireActive(context, "executePass");
		return this._executor.execute(pass);
	}

	public skip(context: FrameContext): void {
		this.requireActive(context, "skipPass");
	}

	public end(context: FrameContext): void {
		this.requireActive(context, "endFrame");
		const frame = this._requireFrame();
		this._executor.endParticleFrame();
		const temporalCommit = new Map(frame.temporal.currentWorldMatrices);

		this._surface.present();
		this._executor.commitFrame();
		this._temporalFrameState.commitFrame();
		this._previousWorldMatrices = temporalCommit;
		this._completedCoverage = this._canPreserveNonDirtyTiles(context) ?
			"dirty-tiles" : "full-frame";
		this._activeContext = null;
		this._frame = null;
		this._flushPendingResize();
	}

	public async abort(error?: unknown): Promise<void> {
		let abortError: unknown = null;
		try {
			await this._executor.abortFrame(error);
		} catch (caught) {
			abortError = caught;
		} finally {
			this._temporalFrameState.abortFrame();
			this._activeContext = null;
			this._frame = null;
			this._flushPendingResize();
		}
		if (abortError) throw abortError;
	}

	public resize(size: RenderSurfaceSize): void {
		if (this._activeContext) {
			this._pendingResize = { width: size.width, height: size.height };
			return;
		}
		this._applyResize(size);
	}

	public reset(): void {
		this._temporalFrameState.reset();
		this._previousWorldMatrices.clear();
		this._activeContext = null;
		this._frame = null;
		this._pendingResize = null;
		this._completedCoverage = "full-frame";
	}

	public get activeContext(): FrameContext | null {
		return this._activeContext;
	}

	public get completedCoverage(): RenderBackendCompletedFrameCoverage {
		return this._completedCoverage;
	}

	/** @internal Returns the active temporal state for transaction tests. */
	public getTemporalDebugState(): SoftwareTemporalRenderState | null {
		return this._frame?.temporal ?? null;
	}

	public requireActive(
		context: FrameContext | null,
		operation: "executePass" | "skipPass" | "endFrame",
	): FrameContext {
		if (!this._activeContext || !context) {
			throw new Error(`SoftwareBackend.${operation}() requires an active frame.`);
		}
		if (context !== this._activeContext) {
			throw new Error(
				`SoftwareBackend.${operation}() received a foreign frame context.`,
			);
		}
		return context;
	}

	private _prepareAttachments(frame: SoftwareFrameView): void {
		const attachments = frame.attachments;
		if (!frame.incrementalPartial) {
			const size = attachments.pixels.length >> 2;
			for (let i = 0; i < size; i++) {
				const index = i << 2;
				attachments.pixels[index] = 0;
				attachments.pixels[index + 1] = 0;
				attachments.pixels[index + 2] = 0;
				attachments.pixels[index + 3] = 255;
			}
			attachments.depthBuffer.fill(Infinity);
			attachments.normalBuffer?.fill(0);
			attachments.motionBuffer?.fill(0);
		} else {
			for (const region of frame.clipRegions) {
				for (let y = region.minY; y < region.maxYExclusive; y++) {
					const rowStart = y * attachments.width;
					for (let x = region.minX; x < region.maxXExclusive; x++) {
						const pixel = rowStart + x;
						const pixelIndex = pixel << 2;
						attachments.pixels[pixelIndex] = 0;
						attachments.pixels[pixelIndex + 1] = 0;
						attachments.pixels[pixelIndex + 2] = 0;
						attachments.pixels[pixelIndex + 3] = 255;
						attachments.depthBuffer[pixel] = Infinity;
						if (attachments.normalBuffer) {
							const normalIndex = pixel * 3;
							attachments.normalBuffer[normalIndex] = 0;
							attachments.normalBuffer[normalIndex + 1] = 0;
							attachments.normalBuffer[normalIndex + 2] = 0;
						}
						if (attachments.motionBuffer) {
							const motionIndex = pixel << 2;
							attachments.motionBuffer[motionIndex] = 0;
							attachments.motionBuffer[motionIndex + 1] = 0;
							attachments.motionBuffer[motionIndex + 2] = 0;
							attachments.motionBuffer[motionIndex + 3] = 0;
						}
					}
				}
			}
		}

		const environment = frame.scene.environment;
		if (
			!frame.incrementalPartial &&
			frame.features.enableEnvironment &&
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
				attachments.pixels,
				frame.camera,
				attachments.width,
				attachments.height,
				frame.incrementalPartial ? frame.clipRegions : undefined,
			);
		}
	}

	private _canPreserveNonDirtyTiles(context: FrameContext): boolean {
		const frame = this._requireFrame();
		return (
			frame.incrementalPartial &&
			!frame.temporalHistoryReset &&
			(context.postProcess.getEnabledPasses().length === 0 ||
				this._executor.completedFramePreservesOutsideDirtyTiles)
		);
	}

	private _applyResize(size: RenderSurfaceSize): void {
		this._surface.resize(size);
		this._executor.invalidateFrameSized();
		this._temporalFrameState.reset();
		this._previousWorldMatrices.clear();
	}

	private _flushPendingResize(): void {
		const pending = this._pendingResize;
		this._pendingResize = null;
		if (pending) this._applyResize(pending);
	}

	private _requireFrame(): SoftwareFrameView {
		if (!this._frame) {
			throw new Error("SoftwareBackend requires an active Software frame view.");
		}
		return this._frame;
	}
}
