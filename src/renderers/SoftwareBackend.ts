import type { IRenderBackend, RendererBackendBridge } from "./IRenderBackend";
import {
	DEFAULT_TAA_OPTIONS,
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	type DrawPacket,
	type FrameContext,
	type FramePass,
	type TAAOptions,
} from "../pipeline/types";
import type { PostProcessCapabilities } from "../pipeline/PostProcessController";
import type { LogicalGBufferBridge } from "../postprocess";
import { Rasterizer } from "./software/Rasterizer";
import { PostProcessor } from "./software/PostProcessor";
import {
	SoftwarePostProcessExecutor,
	createSoftwareGBufferBridge,
} from "./software/SoftwarePostProcessExecutor";
import { SoftwareMainPass } from "./software/passes/SoftwareMainPass";
import { SoftwareParticlePass } from "./software/passes/SoftwareParticlePass";
import { SoftwareReflectionPass } from "./software/passes/SoftwareReflectionPass";
import { SoftwareShadowPass } from "./software/passes/SoftwareShadowPass";
import type { SoftwarePassLike } from "./software/passes/types";
import { EnvironmentBackgroundRenderer } from "./software/EnvironmentRenderer";
import { isShadowCastingLight } from "../lights";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../pipeline/ShadowMetadata";
import {
	mergeParticleShadowBounds,
	resolveParticleShadowCasterBounds,
} from "../pipeline/ParticleShadowVolume";
import {
	selectCSMDirectionalLights,
	type ShadowBackendCapabilities,
} from "../pipeline/ShadowStrategyRegistry";
import { FrameAttachments } from "../pipeline/types";
import { CameraType } from "../cameras/Camera";
import { TemporalJitterState } from "./temporal/TemporalJitterState";
import {
	SOFTWARE_TAA_RENDER_STATE_KEY,
} from "../postprocess/passes/TemporalAntiAliasingPass";
import { DefaultParticleSimulator } from "../simulation/particles/DefaultParticleSimulator";
import { type SoftwareBackendOptions, type SoftwareRasterMode } from "./software/types";
import { DEFAULT_SOFTWARE_RASTER_MODE } from "./software/constants";
import {
	assertShaderDirectiveProfileRegistryComplete,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
} from "../shaders/runtime";
import { Logger } from "../foundation/Logger";

export type {
	SoftwareBackendOptions,
	SoftwareRasterMode,
	SoftwareTileOptions,
} from "./software/types";

type SoftwarePassHandler = (
	context: FrameContext
) => void | Promise<void>;

const SOFTWARE_SHADOW_CAPABILITIES: ShadowBackendCapabilities = {
	backendKey: "software",
	supportsSingleMap: true,
	supportsDirectionalCSM: true,
	supportsSpotCSM: true,
	supportsPointCSM: true,
	maxCsmDirectionalLights: 1,
	maxDynamicShadowCost: 20,
};

const SOFTWARE_POST_PROCESS_CAPABILITIES: PostProcessCapabilities = {
	ssao: true,
	ssgi: false,
	taa: true,
	ssr: false,
	volumetric: true,
	fog: false,
	"motion-blur": false,
	dof: false,
	bloom: false,
	tonemap: true,
	"color-filter": true,
	fxaa: true,
	"interaction-outline": true,
	gamma: true,
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
	public readonly type = "software";
	public readonly frameScheduling = "on-demand";
	public readonly passExecutors = {
		"animation-sim": "shared",
		"particle-sim": "backend",
	} as const;
	public readonly capabilities = {
		sh: true,
		shadows: true,
		reflection: true,
		environment: true,
		clusteredLighting: false,
		oit: false,
	};
	private readonly _postProcessExecutor = new SoftwarePostProcessExecutor(
		SOFTWARE_POST_PROCESS_CAPABILITIES,
		{
			getPostProcessor: () => this._postProcessor,
			getCanvasContext: () => this._ctx,
		}
	);
	public readonly postProcessCapabilities = SOFTWARE_POST_PROCESS_CAPABILITIES;
	public readonly postProcessExecutor = this._postProcessExecutor;
	public readonly requestedRasterMode: SoftwareRasterMode;

	private _renderer: RendererBackendBridge | null = null;
	private _ctx: CanvasRenderingContext2D | null = null;
	private _rasterizer: Rasterizer | null = null;
	private _mainPass: SoftwareMainPass | null = null;
	private _particlePass: SoftwarePassLike | null = null;
	private _shadowPass: SoftwarePassLike | null = null;
	private _reflectionPass: SoftwarePassLike | null = null;
	private _postProcessor: PostProcessor | null = null;
	private _framePixelsShared = false;
	private _pixels: Uint8ClampedArray | null = null;
	private _depthBuffer: Float32Array | null = null;
	private _normalBuffer: Float32Array | null = null;
	private _motionBuffer: Float32Array | null = null;
	private _temporalJitterState = new TemporalJitterState();
	private _previousViewProjection: FrameContext["camera"]["viewProjectionMatrix"] | null = null;
	private _previousWorldMatrices = new Map<string, FrameContext["worldMatrix"]>();
	private _activeContext: FrameContext | null = null;
	private _frameImageData: ImageData | null = null;
	private _framePixels: Uint8ClampedArray | null = null;
	private _frameWidth = 0;
	private _frameHeight = 0;
	private _particleSimulator: DefaultParticleSimulator | null = null;
	private _offscreenCanvas: OffscreenCanvas | null = null;
	private _offscreenCtx: OffscreenCanvasRenderingContext2D | null = null;
	private _options: SoftwareBackendOptions;
	private _activeRasterMode: SoftwareRasterMode;
	private readonly _passHandlers: Map<FramePass["stage"], SoftwarePassHandler>;

	public constructor(options: SoftwareBackendOptions = {}) {
		assertShaderDirectiveProfileRegistryComplete(DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY);
		this._options = options;
		this.requestedRasterMode = options.rasterMode ?? DEFAULT_SOFTWARE_RASTER_MODE;
		this._activeRasterMode = this.requestedRasterMode;
		this._passHandlers = this._createPassHandlers();
		this._ensureRuntime();
	}

	public get activeRasterMode(): SoftwareRasterMode {
		return this._activeRasterMode;
	}

	/**
	 * Creates the logical G-buffer bridge for software post-process passes.
	 *
	 * @param context Frame context with software color, depth, and normal
	 * buffers.
	 * @returns A logical G-buffer bridge backed by software attachments.
	 * @remarks This method does not allocate backend-owned graph passes or
	 * mutate post-process pass registration.
	 */
	public createPostProcessGBufferBridge(
		context: FrameContext
	): LogicalGBufferBridge {
		return createSoftwareGBufferBridge(context);
	}

	public async init(canvas: HTMLCanvasElement): Promise<void> {
		this._ctx = canvas.getContext("2d");
	}

	public setRenderer(renderer: RendererBackendBridge): void {
		this._renderer = renderer;
		this._ensureRuntime();
	}

	private _ensureRuntime(): void {
		if (this._rasterizer) {
			return;
		}
		this._rasterizer = new Rasterizer();
		this._shadowPass = new SoftwareShadowPass(this._rasterizer);
		this._mainPass = new SoftwareMainPass(this._rasterizer, {
			mode: this.requestedRasterMode,
			tile: this._options.tile,
			enableEarlyZPrepass: this._options.enableEarlyZPrepass,
		});
		this._particlePass = new SoftwareParticlePass();
		this._reflectionPass = new SoftwareReflectionPass(this._rasterizer);
		this._postProcessor = new PostProcessor();
		this._particleSimulator = new DefaultParticleSimulator({
			backendTag: this.type,
		});
		this._syncActiveRasterMode();
	}

	public getAttachments(width: number, height: number): FrameAttachments {
		if (
			!this._pixels ||
			this._pixels.length !== width * height * 4 ||
			!this._depthBuffer ||
			this._depthBuffer.length !== width * height
		) {
			this._pixels = new Uint8ClampedArray(width * height * 4);
			this._depthBuffer = new Float32Array(width * height);
			this._normalBuffer = new Float32Array(width * height * 3);
			this._motionBuffer = new Float32Array(width * height * 4);
		}
		this._frameWidth = width;
		this._frameHeight = height;
		return {
			pixels: this._pixels,
			depthBuffer: this._depthBuffer,
			normalBuffer: this._normalBuffer,
			motionBuffer: this._motionBuffer,
			width,
			height,
		};
	}

	public resize(width: number, height: number): void {
		this._frameImageData = null;
		this._framePixels = null;
		this._framePixelsShared = false;

		if (!this._offscreenCanvas) {
			this._offscreenCanvas = new OffscreenCanvas(width, height);
			this._offscreenCtx = this._offscreenCanvas.getContext(
				"2d",
			) as OffscreenCanvasRenderingContext2D | null;
		} else {
			this._offscreenCanvas.width = width;
			this._offscreenCanvas.height = height;
		}
	}

	public beginFrame(context: FrameContext): void {
		this._activeContext = context;
		this._particleSimulator?.beginFrame(context);
		this._prepareTAARenderState(context);

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
			resolveParticleShadowCasterBounds(context.scene.particleSystems)
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
			EnvironmentBackgroundRenderer.render(
				environment.backgroundTexture,
				{
					strength: environment.backgroundStrength,
					tintLinear: environment.backgroundTintLinear,
					exposure: environment.backgroundExposure,
				},
				pixels,
				context.camera,
				context.attachments.width,
				context.attachments.height,
			);
		}
	}

	public async executePass(pass: FramePass, context: FrameContext): Promise<void> {
		if (!this._renderer || !this._mainPass || !this._reflectionPass) return;

		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			return;
		}
		await handler(context);
	}

	public skipPass(_pass: FramePass): void {
		// No pass dependency tracking in SoftwareBackend; no-op.
	}

	public endFrame(): void {
		this._particleSimulator?.endFrame();
		this._commitTAARenderState();
		this._activeContext = null;

		if (!this._renderer || !this._ctx) return;

		const imageData = this._getFrameImageData(this._renderer);
		if (this._offscreenCtx && this._offscreenCanvas) {
			this._offscreenCtx.putImageData(imageData, 0, 0);
			const bitmap = this._offscreenCanvas.transferToImageBitmap();
			this._ctx.drawImage(bitmap, 0, 0);
			bitmap.close();
		} else {
			this._ctx.putImageData(imageData, 0, 0);
		}
	}

	private _prepareTAARenderState(context: FrameContext): void {
		const taaOptions =
			context.postProcess.getOptions<TAAOptions>("taa") ?? DEFAULT_TAA_OPTIONS;
		const taaEnabled = context.postProcess.isEnabled("taa");
		const jitter = this._temporalJitterState.next({
			enabled: taaEnabled,
			isOrthographic: context.camera.type === CameraType.Orthographic,
			width: context.attachments.width,
			height: context.attachments.height,
			jitterScale: taaOptions.jitterScale ?? DEFAULT_TAA_OPTIONS.jitterScale,
			reset: context.incremental.temporalHistoryReset,
		});
		if (!taaEnabled || context.incremental.temporalHistoryReset) {
			this._previousViewProjection = null;
			this._previousWorldMatrices.clear();
		}
		context.transient.set(SOFTWARE_TAA_RENDER_STATE_KEY, {
			currentJitter: [jitter[0], jitter[1]],
			previousJitter: [jitter[2], jitter[3]],
			previousViewProjection: this._previousViewProjection,
			currentViewProjection: context.camera.viewProjectionMatrix,
			previousWorldMatrices: this._previousWorldMatrices,
			currentWorldMatrices: new Map(),
		});
	}

	private _commitTAARenderState(): void {
		const context = this._activeContext;
		if (!context) {
			return;
		}
		const state = context.transient.get(SOFTWARE_TAA_RENDER_STATE_KEY);
		if (!state) {
			return;
		}
		this._previousViewProjection = context.camera.viewProjectionMatrix.clone();
		this._previousWorldMatrices = new Map(state.currentWorldMatrices);
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

	private _getFrameImageData(renderer: RendererBackendBridge): ImageData {
		const pixels = this._resolveFramePixels(renderer);
		const { width, height } = this._resolveFrameDimensions(renderer, pixels);

		if (
			!this._frameImageData ||
			this._framePixels !== pixels ||
			this._frameImageData.width !== width ||
			this._frameImageData.height !== height
		) {
			this._frameImageData = this._createFrameImageData(pixels, width, height);
			this._framePixels = pixels;
		}

		if (!this._framePixelsShared) {
			this._frameImageData.data.set(pixels);
		}

		return this._frameImageData;
	}

	private _resolveFrameDimensions(
		renderer: RendererBackendBridge,
		pixels: Uint8ClampedArray,
	): { width: number; height: number } {
		const canvasWidth = renderer.canvas.width;
		const canvasHeight = renderer.canvas.height;
		if (pixels.length === canvasWidth * canvasHeight * 4) {
			return {
				width: canvasWidth,
				height: canvasHeight,
			};
		}

		if (
			this._frameWidth > 0 &&
			this._frameHeight > 0 &&
			pixels.length === this._frameWidth * this._frameHeight * 4
		) {
			return {
				width: this._frameWidth,
				height: this._frameHeight,
			};
		}

		if (
			this._frameImageData &&
			pixels.length === this._frameImageData.width * this._frameImageData.height * 4
		) {
			return {
				width: this._frameImageData.width,
				height: this._frameImageData.height,
			};
		}

		const pixelCount = Math.floor(pixels.length / 4);
		return {
			width: Math.max(1, pixelCount),
			height: 1,
		};
	}

	private _resolveFramePixels(renderer: RendererBackendBridge): Uint8ClampedArray {
		const legacyPixels = renderer.pixels;
		const pixels = this._pixels || legacyPixels;

		if (!pixels) {
			throw new Error("Software backend frame buffer is not initialized.");
		}

		return pixels;
	}

	private _createFrameImageData(
		pixels: Uint8ClampedArray,
		width: number,
		height: number,
	): ImageData {
		try {
			const imageData = new ImageData(pixels as ImageDataArray, width, height);
			this._framePixelsShared =
				imageData.data === pixels || imageData.data.buffer === pixels.buffer;
			return imageData;
		} catch {
			const imageData = new ImageData(width, height);
			const copyLength = Math.min(imageData.data.length, pixels.length);
			imageData.data.set(pixels.subarray(0, copyLength));
			this._framePixelsShared = false;
			return imageData;
		}
	}

	public destroy(): void {
		this._mainPass?.destroy();
		this._mainPass = null;
		this._particlePass = null;
		this._shadowPass = null;
		this._reflectionPass = null;
		this._postProcessor = null;
		this._particleSimulator = null;
		this._rasterizer = null;
	}

	private _syncActiveRasterMode(): void {
		const mode = this._mainPass?.getActiveMode();
		this._activeRasterMode = mode ?? this.requestedRasterMode;
	}

	private _createPassHandlers(): Map<FramePass["stage"], SoftwarePassHandler> {
		return new Map<FramePass["stage"], SoftwarePassHandler>([
			["animation-sim", () => {}],
			[
				"particle-sim",
				(context) => {
					this._particleSimulator?.simulate(
						context,
						this._resolveParticleDeltaTime(context),
					);
					this._particleSimulator?.emitRenderBatches(context);
				},
			],
			[
				"shadow",
				(context) => {
					this._shadowPass?.render(context);
				},
			],
			[
				"reflection",
				(context) => {
					this._reflectionPass?.render(context);
				},
			],
			[
				"main-opaque",
				async (context) => {
					if (!this._mainPass) {
						return;
					}
					const packets = this._resolvePacketsForPass(
						context,
						context.scene.opaquePackets,
					);
					await this._mainPass.render(context, packets, false);
					this._syncActiveRasterMode();
				},
			],
			[
				"main-transparent",
				async (context) => {
					if (!this._mainPass) {
						return;
					}
					const packets = this._resolvePacketsForPass(
						context,
						context.scene.transparentPackets,
					);
					await this._mainPass.render(context, packets, true);
					this._syncActiveRasterMode();
				},
			],
			[
				"particles",
				(context) => {
					this._particlePass?.render(context);
				},
			],
			["ssao", () => {}],
			["taa", () => {}],
			["ssr", () => {}],
			[
				"volumetric",
				(context) => {
					if (!this._ctx) {
						return;
					}
					this._postProcessor?.applyVolumetricLight(context, this._ctx);
				},
			],
			[
				"fxaa",
				(context) => {
					if (!this._ctx) {
						return;
					}
					this._postProcessor?.applyFXAA(context, this._ctx);
				},
			],
			[
				"interaction-outline",
				(context) => {
					this._postProcessor?.applyInteractionOutline(context);
				},
			],
			[
				"gamma",
				(context) => {
					if (!this._ctx) {
						return;
					}
					this._postProcessor?.applyGamma(context, this._ctx);
				},
			],
			[
				"tonemap",
				(context) => {
					this._postProcessor?.applyToneMapping(context);
				},
			],
			[
				"color-filter",
				(context) => {
					this._postProcessor?.applyColorFilter(context);
				},
			],
		]);
	}
}
