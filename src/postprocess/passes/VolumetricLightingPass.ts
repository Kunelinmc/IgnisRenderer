import { Matrix4 } from "../../maths/Matrix4";
import { CameraType } from "../../cameras/Camera";
import type { OrthographicCamera } from "../../cameras/OrthographicCamera";
import {
	type DirectionalLight,
	type PointLight,
	type SpotLight,
	LightType,
	isShadowCastingLight,
} from "../../lights";
import { clamp } from "../../maths/Common";
import type { IVector3 } from "../../maths/types";
import type { FrameContext } from "../../pipeline/types";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	createLightContribution,
	evaluateLightContribution,
} from "../../renderers/software/LightEvaluator";
import {
	createSoftwareShadowSampler,
	getSoftwareShadowRuntimeMap,
} from "../../renderers/software/passes/SoftwareShadowPass";
import {
	MAX_EXPOSURE,
	POST_PROCESS_NOISE_REFERENCE_WIDTH,
	VOLUMETRIC_SIGMA_T_SCALE,
} from "../../renderers/constants";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
	WEBGPU_MAX_VOLUMETRIC_LIGHTS as MAX_VOLUMETRIC_LIGHTS,
	WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS as VOLUMETRIC_LIGHT_STRIDE_FLOATS,
} from "../../renderers/webgpu/constants";
import { getWebGPUVolumetricLightLayout } from "../../renderers/webgpu/bufferLayouts";
import type { WebGPUPostProcessFrameTargets } from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type { WebGPULightingState } from "../../renderers/webgpu/types";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import { getRequiredBuiltinPostProcessOrderMetadata } from "../builtinMetadata";
import type {
	PostProcessHistoryDescriptor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
	PostProcessTransientDescriptor,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
export const VOLUMETRIC_LIGHTING_PASS_ID = "volumetric";
export const VOLUMETRIC_LIGHTING_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(VOLUMETRIC_LIGHTING_PASS_ID);
const WEBGPU_HIZ_TRANSIENT_ID = "hiz";
const WEBGPU_HIZ_TRANSIENT_USAGE = ["sampled", "storage"] as const;
const SOFTWARE_VOLUMETRIC_CONSTANTS = Object.freeze({
	SIGMA_T_SCALE: VOLUMETRIC_SIGMA_T_SCALE,
	MIN_RAY_DISTANCE: 0.1,
	MIN_DOWN_SAMPLE: 1,
	MAX_DOWN_SAMPLE: 8,
	MIN_SAMPLES: 1,
	MAX_SAMPLES: 256,
	DEFAULT_DOWN_SAMPLE: 1,
	DEFAULT_SAMPLES: 32,
	MIN_SHADOW_SAMPLE_INTERVAL: 1,
	MAX_SHADOW_SAMPLE_INTERVAL: 32,
	MAX_WEIGHT: 10,
	DEFAULT_WEIGHT: 4,
	MAX_AIR_DENSITY: 10,
	TRANSMITTANCE_EARLY_EXIT: 0.001,
	GRID_SAMPLE_JITTER_STRENGTH: 0.75,
	SCENE_BOUNDS_FADE_START_MULTIPLIER: 1.05,
	SCENE_BOUNDS_FADE_END_MULTIPLIER: 1.8,
	SCENE_DEPTH_LIMIT_MULTIPLIER: 1.6,
	MIN_SCENE_BOUNDS_RADIUS: 1.0,
	TEMPORAL_ACCUMULATION_FACTOR: 0.95,
});

export interface VolumetricOptions {
	/** Ray-march step count. Higher values reduce banding at higher GPU cost. */
	samples?: number;
	/** Software volumetric grid scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Overall scattering contribution added to the scene color. */
	weight?: number;
	/** Exposure multiplier applied to the accumulated light shaft result. */
	exposure?: number;
	/** Participating-media density. Higher values make fog volumes thicker. */
	airDensity?: number;
	/**
	 * Henyey-Greenstein phase anisotropy. Positive values emphasize forward
	 * scattering, negative values emphasize back scattering.
	 */
	anisotropy?: number;
	/** Maximum world-space ray distance sampled from the camera. */
	maxRayDistance?: number;
	/** Fraction of light scattered instead of absorbed, clamped to `[0, 1]`. */
	scatteringAlbedo?: number;
	/** Step interval for shadow lookups. Higher values reduce shadowing cost. */
	shadowSampleInterval?: number;
	/** Software path: whether the provided depth buffer is already linearized. */
	isLinearDepth?: boolean;
	/** Enables depth-adaptive ray marching to spend samples where detail changes. */
	adaptiveSteps?: boolean;
	/** Software path: enables bilateral upscaling for downsampled volumes. */
	useBilateralUpscale?: boolean;
	/** Depth tolerance for bilateral upscale; lower values preserve harder edges. */
	bilateralDepthSigma?: number;
	/** ReSTIR candidate count per pixel. Higher values improve light selection. */
	restirCandidates?: number;
	/** Temporal reservoir blend factor. Higher values stabilize but can ghost. */
	restirTemporalWeight?: number;
	/** Maximum ReSTIR reservoir weight scale to prevent bright outliers. */
	restirScaleClamp?: number;
	/** Allows backend-specific experimental volumetric options. */
	[key: string]: unknown;
}

export const DEFAULT_VOLUMETRIC_OPTIONS: Required<
	Pick<
		VolumetricOptions,
		| "samples"
		| "downsample"
		| "weight"
		| "exposure"
		| "airDensity"
		| "anisotropy"
		| "maxRayDistance"
		| "scatteringAlbedo"
		| "shadowSampleInterval"
		| "isLinearDepth"
		| "adaptiveSteps"
		| "useBilateralUpscale"
		| "bilateralDepthSigma"
		| "restirCandidates"
		| "restirTemporalWeight"
		| "restirScaleClamp"
	>
> = {
	samples: 32,
	downsample: 1,
	weight: 4,
	exposure: 1,
	airDensity: 1,
	anisotropy: 0.2,
	maxRayDistance: 300,
	scatteringAlbedo: 0.9,
	shadowSampleInterval: 2,
	isLinearDepth: true,
	adaptiveSteps: true,
	useBilateralUpscale: true,
	bilateralDepthSigma: 0.05,
	restirCandidates: 8,
	restirTemporalWeight: 0.8,
	restirScaleClamp: 24,
};

export interface SoftwareVolumetricLightingContext {
	readonly canvasContext: CanvasRenderingContext2D | null;
}

export interface WebGPUVolumetricLightingContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly frameBinding?: IBindingGroup;
	readonly lightingState?: WebGPULightingState | null;
	readonly hiZ?: IRenderTexture | null;
	readonly historyRead?: IRenderTexture | null;
	readonly historyWrite?: IRenderTexture | null;
	readonly reservoirHistoryRead?: IRenderTexture | null;
	readonly reservoirHistoryWrite?: IRenderTexture | null;
	readonly motionHistoryRead?: IRenderTexture | null;
	readonly motionHistoryWrite?: IRenderTexture | null;
	publishColorTarget?(texture: IRenderTexture): void;
	writeMotionHistoryFromCurrent?(): void | Promise<void>;
}

interface WebGPUVolumetricResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	lightBuffer: IRenderBuffer | null;
	lightCapacity: number;
	frameIndex: number;
	groupLayout0: GPUBindGroupLayout | null;
	pipelineLayout: GPUPipelineLayout | null;
}

interface CameraBasis {
	right: IVector3;
	up: IVector3;
	backward: IVector3;
}

interface WorldRay extends IVector3 {
	camDirZ: number;
}

type VolumetricLight = DirectionalLight | PointLight | SpotLight;

interface IncrementalDirtyRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

/**
 * Software implementation of volumetric lighting.
 */
export class SoftwareVolumetricLightingImplementation
	implements
		PostProcessPassImplementation<
			SoftwareVolumetricLightingContext,
			VolumetricOptions
		>
{
	public readonly id = "volumetric:software";
	private _prevScatterBuf: Float32Array | null = null;
	private _frameIndex = 0;
	private _scatterGrid: Float32Array | null = null;
	private _visibilityCache: Float32Array | null = null;
	private _scatterBuf: Float32Array | null = null;
	private _lowDepthBuf: Float32Array | null = null;
	private _prevVolumetricBuf: Float32Array | null = null;
	private _prevViewProj: Matrix4 | null = null;

	public execute(
		request: PostProcessPassRequest<VolumetricOptions>,
		_context: SoftwareVolumetricLightingContext | undefined
	): PostProcessPassResult {
		if (
			!request.frameContext.attachments.pixels ||
			!request.frameContext.attachments.depthBuffer
		) {
			return { ran: false };
		}
		this._applyVolumetricLight(request.frameContext, {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...(request.options ?? {}),
		});
		return { ran: true };
	}

	private _getCameraBasis(context: FrameContext): CameraBasis {
		const view = context.camera.viewMatrix.elements;
		return {
			right: { x: view[0][0], y: view[0][1], z: view[0][2] },
			up: { x: view[1][0], y: view[1][1], z: view[1][2] },
			backward: { x: view[2][0], y: view[2][1], z: view[2][2] },
		};
	}

	private _getWorldRayFromPixel(
		px: number,
		py: number,
		w: number,
		h: number,
		basis: CameraBasis,
		context: FrameContext
	): WorldRay {
		const camera = context.camera;

		if (camera.type === CameraType.Orthographic) {
			// In orthographic camera, rays are constant (pointing forward)
			// World forward is -basis.backward
			return {
				x: -basis.backward.x,
				y: -basis.backward.y,
				z: -basis.backward.z,
				camDirZ: -1,
			};
		}

		const fovRad = (camera.fov * Math.PI) / 180;
		const tanHalfFov = Math.tan(fovRad * 0.5);
		const aspect = camera.aspectRatio || w / h;

		const ndcX = ((px + 0.5) / w) * 2 - 1;
		const ndcY = 1 - ((py + 0.5) / h) * 2;

		const cx = ndcX * aspect * tanHalfFov;
		const cy = ndcY * tanHalfFov;
		const cz = -1;
		const invLen = 1.0 / Math.hypot(cx, cy, cz);
		const dirCamX = cx * invLen;
		const dirCamY = cy * invLen;
		const dirCamZ = cz * invLen;

		return {
			x:
				basis.right.x * dirCamX +
				basis.up.x * dirCamY +
				basis.backward.x * dirCamZ,
			y:
				basis.right.y * dirCamX +
				basis.up.y * dirCamY +
				basis.backward.y * dirCamZ,
			z:
				basis.right.z * dirCamX +
				basis.up.z * dirCamY +
				basis.backward.z * dirCamZ,
			camDirZ: dirCamZ,
		};
	}

	private _henyeyGreenstein(cosTheta: number, g: number): number {
		const gg = g * g;
		const denom = Math.pow(1 + gg - 2 * g * cosTheta, 1.5) || 1e-6;
		return (1 - gg) / (4 * Math.PI * denom);
	}

	private _blueNoiseJitter(
		px: number,
		py: number,
		frameIndex: number = 0
	): number {
		const GOLDEN_RATIO = 1.61803398875;
		const a1 = 1.0 / GOLDEN_RATIO;
		const n = px + py * POST_PROCESS_NOISE_REFERENCE_WIDTH + frameIndex;
		return (0.5 + a1 * n) % 1.0;
	}

	private _linearizeDepth(
		depth: number,
		near: number,
		far: number,
		isLinearDepth: boolean = true
	): number {
		if (isLinearDepth || depth === Infinity) return depth;
		return (near * far) / (far - depth * (far - near));
	}

	private _toFiniteNumber(value: unknown, fallback: number): number {
		if (typeof value === "number" && Number.isFinite(value)) return value;
		return fallback;
	}

	private _resolveDirtyRects(context: FrameContext): IncrementalDirtyRect[] {
		const width = Math.max(1, context.attachments.width);
		const height = Math.max(1, context.attachments.height);
		const incremental = context.incremental;
		if (
			!incremental.enabled ||
			incremental.forceFullFrame ||
			incremental.dirtyRects.length === 0
		) {
			return [{
				minX: 0,
				minY: 0,
				maxX: width - 1,
				maxY: height - 1,
			}];
		}
		const dirtyRects: IncrementalDirtyRect[] = [];
		for (const rect of incremental.dirtyRects) {
			const minX = Math.max(0, Math.floor(rect.x));
			const minY = Math.max(0, Math.floor(rect.y));
			const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
			const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
			if (minX > maxX || minY > maxY) {
				continue;
			}
			dirtyRects.push({
				minX,
				minY,
				maxX,
				maxY,
			});
		}
		return dirtyRects;
	}

	private _forEachDirtyRect(
		dirtyRects: IncrementalDirtyRect[],
		callback: (rect: IncrementalDirtyRect) => void
	): void {
		for (const rect of dirtyRects) {
			if (rect.minX > rect.maxX || rect.minY > rect.maxY) {
				continue;
			}
			callback(rect);
		}
	}

	private _samplePreviousVolumetric(
		worldPos: IVector3,
		gridW: number,
		gridH: number,
		prevViewProj: Matrix4,
		prevVolumetricBuf: Float32Array,
		outCol: { r: number; g: number; b: number }
	): boolean {
		const ndc = Matrix4.transformPoint(prevViewProj, {
			x: worldPos.x,
			y: worldPos.y,
			z: worldPos.z,
			w: 1,
		});
		if (Math.abs(ndc.w!) < 1e-6) return false;

		const invW = 1.0 / ndc.w!;
		const nx = ndc.x! * invW;
		const ny = ndc.y! * invW;
		const nz = ndc.z! * invW;

		if (nx < -1 || nx > 1 || ny < -1 || ny > 1 || nz < -1 || nz > 1)
			return false;

		const u = nx * 0.5 + 0.5;
		const v = 0.5 - ny * 0.5;

		const gx = clamp(u * gridW - 0.5, 0, gridW - 1);
		const gy = clamp(v * gridH - 0.5, 0, gridH - 1);

		const x1 = Math.floor(gx);
		const y1 = Math.floor(gy);
		const x2 = Math.min(x1 + 1, gridW - 1);
		const y2 = Math.min(y1 + 1, gridH - 1);

		const tx = gx - x1;
		const ty = gy - y1;

		const i1 = (y1 * gridW + x1) * 3;
		const i2 = (y1 * gridW + x2) * 3;
		const i3 = (y2 * gridW + x1) * 3;
		const i4 = (y2 * gridW + x2) * 3;

		const w1 = (1 - tx) * (1 - ty);
		const w2 = tx * (1 - ty);
		const w3 = (1 - tx) * ty;
		const w4 = tx * ty;

		outCol.r =
			prevVolumetricBuf[i1] * w1 +
			prevVolumetricBuf[i2] * w2 +
			prevVolumetricBuf[i3] * w3 +
			prevVolumetricBuf[i4] * w4;
		outCol.g =
			prevVolumetricBuf[i1 + 1] * w1 +
			prevVolumetricBuf[i2 + 1] * w2 +
			prevVolumetricBuf[i3 + 1] * w3 +
			prevVolumetricBuf[i4 + 1] * w4;
		outCol.b =
			prevVolumetricBuf[i1 + 2] * w1 +
			prevVolumetricBuf[i2 + 2] * w2 +
			prevVolumetricBuf[i3 + 2] * w3 +
			prevVolumetricBuf[i4 + 2] * w4;

		return true;
	}

	private _ensureFloat32Buffer(
		buffer: Float32Array | null,
		size: number
	): Float32Array {
		if (!buffer || buffer.length !== size) {
			return new Float32Array(size);
		}
		return buffer;
	}

	private _computeSceneFalloff(
		distanceSq: number,
		fadeStartSq: number,
		fadeEndSq: number
	): number {
		if (distanceSq <= fadeStartSq) return 1.0;
		if (distanceSq >= fadeEndSq) return 0.0;
		const t = clamp(
			(distanceSq - fadeStartSq) / Math.max(fadeEndSq - fadeStartSq, 1e-6),
			0,
			1
		);
		return 1.0 - t * t * (3.0 - 2.0 * t);
	}

	private _filterScatterBuffer(
		scatterBuf: Float32Array,
		w: number,
		h: number
	): void {
		let temp = this._prevScatterBuf;
		if (!temp || temp.length !== scatterBuf.length) {
			temp = new Float32Array(scatterBuf.length);
			this._prevScatterBuf = temp;
		}

		// 1D tent blur horizontally
		for (let y = 0; y < h; y++) {
			const row = y * w;
			for (let x = 0; x < w; x++) {
				const l = row + Math.max(0, x - 1);
				const c = row + x;
				const r = row + Math.min(w - 1, x + 1);

				const outIdx = c * 3;
				const lIdx = l * 3;
				const cIdx = c * 3;
				const rIdx = r * 3;

				temp[outIdx] =
					(scatterBuf[lIdx] + scatterBuf[cIdx] * 2 + scatterBuf[rIdx]) * 0.25;
				temp[outIdx + 1] =
					(scatterBuf[lIdx + 1] +
						scatterBuf[cIdx + 1] * 2 +
						scatterBuf[rIdx + 1]) *
					0.25;
				temp[outIdx + 2] =
					(scatterBuf[lIdx + 2] +
						scatterBuf[cIdx + 2] * 2 +
						scatterBuf[rIdx + 2]) *
					0.25;
			}
		}

		// 1D tent blur vertically
		for (let y = 0; y < h; y++) {
			const tY = Math.max(0, y - 1);
			const bY = Math.min(h - 1, y + 1);
			for (let x = 0; x < w; x++) {
				const tIdx = (tY * w + x) * 3;
				const cIdx = (y * w + x) * 3;
				const bIdx = (bY * w + x) * 3;

				scatterBuf[cIdx] = (temp[tIdx] + temp[cIdx] * 2 + temp[bIdx]) * 0.25;
				scatterBuf[cIdx + 1] =
					(temp[tIdx + 1] + temp[cIdx + 1] * 2 + temp[bIdx + 1]) * 0.25;
				scatterBuf[cIdx + 2] =
					(temp[tIdx + 2] + temp[cIdx + 2] * 2 + temp[bIdx + 2]) * 0.25;
			}
		}
	}

	private _applyVolumetricLight(
		context: FrameContext,
		options: VolumetricOptions = {}
	): void {
		const depthBuffer = context.attachments.depthBuffer;
		const maxRayDistance = Math.max(
			SOFTWARE_VOLUMETRIC_CONSTANTS.MIN_RAY_DISTANCE,
			this._toFiniteNumber(options.maxRayDistance, 500)
		);

		const { width: w, height: h } = context.attachments;
		const pixels = context.attachments.pixels;
		if (!pixels || !context.attachments.depthBuffer) return;
		const dirtyRects = this._resolveDirtyRects(context);
		if (dirtyRects.length === 0) {
			return;
		}

		const lights = context.scene.lights || [];
		const volLights = lights.filter(
			(light): light is VolumetricLight =>
				light.type === LightType.Directional ||
				light.type === LightType.Point ||
				light.type === LightType.Spot
		);
		if (volLights.length === 0) return;
		const sampleSurface = { position: { x: 0, y: 0, z: 0 } };
		const lightContribution = createLightContribution();

		const camera = context.camera;
		const cameraPos = camera.getWorldPosition();
		const basis = this._getCameraBasis(context);
		const near = camera.near || 0.1;
		const far = Math.min(camera.far || 1000, maxRayDistance);

		// Consolidate options with range protection
		const ds = Math.round(
			clamp(
				this._toFiniteNumber(
					options.downsample,
					SOFTWARE_VOLUMETRIC_CONSTANTS.DEFAULT_DOWN_SAMPLE
				),
				SOFTWARE_VOLUMETRIC_CONSTANTS.MIN_DOWN_SAMPLE,
				SOFTWARE_VOLUMETRIC_CONSTANTS.MAX_DOWN_SAMPLE
			)
		);
		const gridW = Math.ceil(w / ds);
		const gridH = Math.ceil(h / ds);
		const gridD = Math.round(
			clamp(
				this._toFiniteNumber(
					options.samples,
					SOFTWARE_VOLUMETRIC_CONSTANTS.DEFAULT_SAMPLES
				),
				SOFTWARE_VOLUMETRIC_CONSTANTS.MIN_SAMPLES,
				SOFTWARE_VOLUMETRIC_CONSTANTS.MAX_SAMPLES
			)
		);

		const weight = clamp(
			this._toFiniteNumber(options.weight, SOFTWARE_VOLUMETRIC_CONSTANTS.DEFAULT_WEIGHT),
			0,
			SOFTWARE_VOLUMETRIC_CONSTANTS.MAX_WEIGHT
		);
		const exposure = clamp(
			this._toFiniteNumber(options.exposure, 1.0),
			0,
			MAX_EXPOSURE
		);
		const airDensity = clamp(
			this._toFiniteNumber(options.airDensity, 1.0),
			0,
			SOFTWARE_VOLUMETRIC_CONSTANTS.MAX_AIR_DENSITY
		);
		const anisotropy = clamp(
			this._toFiniteNumber(options.anisotropy, 0.4),
			-0.99,
			0.99
		);
		const scatteringAlbedo = clamp(
			this._toFiniteNumber(options.scatteringAlbedo, 0.8),
			0,
			1
		);

		// ... usage continues

		const sigmaT = airDensity * SOFTWARE_VOLUMETRIC_CONSTANTS.SIGMA_T_SCALE;
		const sigmaS = sigmaT * scatteringAlbedo;

		const shadowsEnabled = context.features.enableShadows;
		const shadowSampler = createSoftwareShadowSampler(
			context.shadowMaps,
			getSoftwareShadowRuntimeMap(context.transient),
			{ camera: context.camera }
		);
		const shadowInterval = Math.round(
			clamp(
				this._toFiniteNumber(options.shadowSampleInterval, 1),
				SOFTWARE_VOLUMETRIC_CONSTANTS.MIN_SHADOW_SAMPLE_INTERVAL,
				SOFTWARE_VOLUMETRIC_CONSTANTS.MAX_SHADOW_SAMPLE_INTERVAL
			)
		);

		const sceneBounds = context.scene.sceneBounds;
		const sceneCenter = sceneBounds.center;
		const sceneRadius = Math.max(
			sceneBounds.radius,
			SOFTWARE_VOLUMETRIC_CONSTANTS.MIN_SCENE_BOUNDS_RADIUS
		);
		const sceneFadeStart =
			sceneRadius * SOFTWARE_VOLUMETRIC_CONSTANTS.SCENE_BOUNDS_FADE_START_MULTIPLIER;
		const sceneFadeEnd =
			sceneRadius * SOFTWARE_VOLUMETRIC_CONSTANTS.SCENE_BOUNDS_FADE_END_MULTIPLIER;
		const sceneFadeStartSq = sceneFadeStart * sceneFadeStart;
		const sceneFadeEndSq = sceneFadeEnd * sceneFadeEnd;

		// ... usage continues

		const camToCenter = Math.hypot(
			cameraPos.x - sceneCenter.x,
			cameraPos.y - sceneCenter.y,
			cameraPos.z - sceneCenter.z
		);
		const infinityDepthLimit = clamp(
			camToCenter +
				sceneRadius * SOFTWARE_VOLUMETRIC_CONSTANTS.SCENE_DEPTH_LIMIT_MULTIPLIER,
			near,
			far
		);

		// 1. Light Injection Grid
		this._scatterGrid = this._ensureFloat32Buffer(
			this._scatterGrid,
			gridW * gridH * gridD * 3
		);
		const scatterGrid = this._scatterGrid;

		const lightCount = volLights.length;
		this._visibilityCache = this._ensureFloat32Buffer(
			this._visibilityCache,
			gridW * gridH * lightCount
		);
		const visibilityCache = this._visibilityCache;
		visibilityCache.fill(1.0);

		this._frameIndex++;
		const jitterStrength = ds * SOFTWARE_VOLUMETRIC_CONSTANTS.GRID_SAMPLE_JITTER_STRENGTH;
		const jitterSeedOffsetX = 131;
		const jitterSeedOffsetY = 17;

		for (let z = 0; z < gridD; z++) {
			const zSlice = (z + 0.5) / gridD;
			// Logarithmic distribution for depth slices
			const dist = near * Math.pow(far / near, zSlice);
			const sliceBase = z * gridW * gridH * 3;

			for (let y = 0; y < gridH; y++) {
				const sampleYCenter = (y + 0.5) * ds - 0.5;
				for (let x = 0; x < gridW; x++) {
					const sampleXCenter = (x + 0.5) * ds - 0.5;
					const jitterX =
						(this._blueNoiseJitter(x, y, this._frameIndex) - 0.5) *
						jitterStrength;
					const jitterY =
						(this._blueNoiseJitter(
							x + jitterSeedOffsetX,
							y + jitterSeedOffsetY,
							this._frameIndex
						) -
							0.5) *
						jitterStrength;
					const px = Math.round(clamp(sampleXCenter + jitterX, 0, w - 1));
					const py = Math.round(clamp(sampleYCenter + jitterY, 0, h - 1));
					const ray = this._getWorldRayFromPixel(px, py, w, h, basis, context);

					const ndcX = ((px + 0.5) / w) * 2 - 1;
					const ndcY = 1 - ((py + 0.5) / h) * 2;
					const posView = this._reconstructViewPos(ndcX, ndcY, dist, camera);

					const samplePoint = {
						x:
							cameraPos.x +
							basis.right.x * posView.x +
							basis.up.x * posView.y +
							basis.backward.x * posView.z,
						y:
							cameraPos.y +
							basis.right.y * posView.x +
							basis.up.y * posView.y +
							basis.backward.y * posView.z,
						z:
							cameraPos.z +
							basis.right.z * posView.x +
							basis.up.z * posView.y +
							basis.backward.z * posView.z,
					};
					sampleSurface.position.x = samplePoint.x;
					sampleSurface.position.y = samplePoint.y;
					sampleSurface.position.z = samplePoint.z;

					const sceneDx = samplePoint.x - sceneCenter.x;
					const sceneDy = samplePoint.y - sceneCenter.y;
					const sceneDz = samplePoint.z - sceneCenter.z;
					const sceneFalloff = this._computeSceneFalloff(
						sceneDx * sceneDx + sceneDy * sceneDy + sceneDz * sceneDz,
						sceneFadeStartSq,
						sceneFadeEndSq
					);
					if (sceneFalloff <= 0) {
						const idx = sliceBase + (y * gridW + x) * 3;
						scatterGrid[idx] = 0;
						scatterGrid[idx + 1] = 0;
						scatterGrid[idx + 2] = 0;
						continue;
					}

					let r = 0,
						g = 0,
						b = 0;
					const shouldSampleShadow = z % shadowInterval === 0;
					const cellIndex = y * gridW + x;

					for (let li = 0; li < lightCount; li++) {
						const L = volLights[li];
						const contrib = evaluateLightContribution(
							L,
							sampleSurface,
							lightContribution
						);
						if (!contrib || contrib.type !== "direct" || !contrib.direction)
							continue;
						const lightIntensity = contrib.intensity ?? 1.0;

						const cacheIndex = cellIndex * lightCount + li;
						let vis = visibilityCache[cacheIndex];
						if (shadowsEnabled && isShadowCastingLight(L)) {
							if (shouldSampleShadow || z === 0) {
								// Note: Passing null as normal for volume points to use volume-specific bias
								const shadow = shadowSampler(L, samplePoint, null);
								vis = (shadow.r + shadow.g + shadow.b) / 3;
								visibilityCache[cacheIndex] = vis;
							}
						} else {
							vis = 1.0;
							visibilityCache[cacheIndex] = vis;
						}

						// Fix: viewDotLight direction. ray is Cam->Point, lightDir is Point->LightSource.
						// When looking towards light, they are aligned (dot=1).
						const viewDotLight =
							ray.x * contrib.direction.x +
							ray.y * contrib.direction.y +
							ray.z * contrib.direction.z;
						const phase = this._henyeyGreenstein(
							clamp(viewDotLight, -1, 1),
							anisotropy
						);
						const scatter = phase * sigmaS * weight * sceneFalloff;

						r += contrib.color.r * lightIntensity * vis * scatter;
						g += contrib.color.g * lightIntensity * vis * scatter;
						b += contrib.color.b * lightIntensity * vis * scatter;
					}

					const idx = sliceBase + (y * gridW + x) * 3;
					scatterGrid[idx] = r;
					scatterGrid[idx + 1] = g;
					scatterGrid[idx + 2] = b;
				}
			}
		}

		// 2. Integration along rays
		this._scatterBuf = this._ensureFloat32Buffer(
			this._scatterBuf,
			gridW * gridH * 3
		);
		this._lowDepthBuf = this._ensureFloat32Buffer(
			this._lowDepthBuf,
			gridW * gridH
		);
		const scatterBuf = this._scatterBuf;
		const lowDepthBuf = this._lowDepthBuf;

		const currentViewProj = camera.viewProjectionMatrix;
		const prevViewProj = this._prevViewProj;
		const prevVolBuf = this._prevVolumetricBuf;
		const historyWeight = SOFTWARE_VOLUMETRIC_CONSTANTS.TEMPORAL_ACCUMULATION_FACTOR;
		const tempCol = { r: 0, g: 0, b: 0 };

		for (let y = 0; y < gridH; y++) {
			for (let x = 0; x < gridW; x++) {
				const screenPX = Math.round(clamp((x + 0.5) * ds - 0.5, 0, w - 1));
				const screenPY = Math.round(clamp((y + 0.5) * ds - 0.5, 0, h - 1));
				const depthRaw = depthBuffer[screenPY * w + screenPX];
				const depth = this._linearizeDepth(
					depthRaw,
					near,
					far,
					options.isLinearDepth !== false
				);
				const depthLimit = depth === Infinity ? infinityDepthLimit : depth;

				let accumR = 0,
					accumG = 0,
					accumB = 0;
				let transmittance = 1.0;

				for (let z = 0; z < gridD; z++) {
					const zSlice = (z + 0.5) / gridD;
					const dist = near * Math.pow(far / near, zSlice);
					if (dist > depthLimit) break;

					// Slice thickness in world space
					const nextZSlice = (z + 1.5) / gridD;
					const nextDist = near * Math.pow(far / near, nextZSlice);
					const stepSize = nextDist - dist;

					const idx = (z * gridW * gridH + y * gridW + x) * 3;
					const transStep = Math.exp(-sigmaT * stepSize);

					accumR += scatterGrid[idx] * transmittance * stepSize;
					accumG += scatterGrid[idx + 1] * transmittance * stepSize;
					accumB += scatterGrid[idx + 2] * transmittance * stepSize;

					transmittance *= transStep;
					if (transmittance < SOFTWARE_VOLUMETRIC_CONSTANTS.TRANSMITTANCE_EARLY_EXIT)
						break;
				}

				const bIdx = (y * gridW + x) * 3;
				let finalR = accumR * exposure;
				let finalG = accumG * exposure;
				let finalB = accumB * exposure;

				// Temporal accumulation
				if (
					prevViewProj &&
					prevVolBuf &&
					prevVolBuf.length === scatterBuf.length
				) {
					const ray = this._getWorldRayFromPixel(
						screenPX,
						screenPY,
						w,
						h,
						basis,
						context
					);
					const ndcX = ((screenPX + 0.5) / w) * 2 - 1;
					const ndcY = 1 - ((screenPY + 0.5) / h) * 2;
					const posView = this._reconstructViewPos(
						ndcX,
						ndcY,
						depthLimit,
						camera
					);

					const worldPos = {
						x:
							cameraPos.x +
							basis.right.x * posView.x +
							basis.up.x * posView.y +
							basis.backward.x * posView.z,
						y:
							cameraPos.y +
							basis.right.y * posView.x +
							basis.up.y * posView.y +
							basis.backward.y * posView.z,
						z:
							cameraPos.z +
							basis.right.z * posView.x +
							basis.up.z * posView.y +
							basis.backward.z * posView.z,
					};
					if (
						this._samplePreviousVolumetric(
							worldPos,
							gridW,
							gridH,
							prevViewProj,
							prevVolBuf,
							tempCol
						)
					) {
						finalR = finalR * (1 - historyWeight) + tempCol.r * historyWeight;
						finalG = finalG * (1 - historyWeight) + tempCol.g * historyWeight;
						finalB = finalB * (1 - historyWeight) + tempCol.b * historyWeight;
					}
				}

				scatterBuf[bIdx] = finalR;
				scatterBuf[bIdx + 1] = finalG;
				scatterBuf[bIdx + 2] = finalB;
				lowDepthBuf[y * gridW + x] = depthLimit;
			}
		}

		this._filterScatterBuffer(scatterBuf, gridW, gridH);

		// Store history
		if (
			!this._prevVolumetricBuf ||
			this._prevVolumetricBuf.length !== scatterBuf.length
		) {
			this._prevVolumetricBuf = new Float32Array(scatterBuf.length);
		}
		this._prevVolumetricBuf.set(scatterBuf);
		this._prevViewProj = currentViewProj.clone();

		// 3. Upscale and Combine
		if (options.useBilateralUpscale !== false) {
			this._bilateralUpscale(
				pixels,
				scatterBuf,
				depthBuffer,
				lowDepthBuf,
				w,
				h,
				gridW,
				gridH,
				ds,
				this._toFiniteNumber(options.bilateralDepthSigma, 0.05),
				near,
				far,
				options.isLinearDepth !== false,
				dirtyRects
			);
		} else {
			this._bilinearUpscale(
				pixels,
				scatterBuf,
				w,
				h,
				gridW,
				gridH,
				ds,
				dirtyRects
			);
		}
	}

	private _bilateralUpscale(
		pixels: Uint8ClampedArray,
		scatterBuf: Float32Array,
		depthBuffer: Float32Array,
		lowDepthBuf: Float32Array,
		w: number,
		h: number,
		lowW: number,
		lowH: number,
		ds: number,
		depthSigma: number,
		near: number,
		far: number,
		isLinearDepth: boolean,
		dirtyRects: IncrementalDirtyRect[]
	): void {
		const invSigmaSq2 = 1.0 / (2.0 * depthSigma * depthSigma);
		this._forEachDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const fy = (y + 0.5) / ds - 0.5;
				const ly0 = Math.max(0, Math.floor(fy));
				const ly1 = Math.min(lowH - 1, ly0 + 1);
				const ty = clamp(fy - ly0);
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const fx = (x + 0.5) / ds - 0.5;
					const lx0 = Math.max(0, Math.floor(fx));
					const lx1 = Math.min(lowW - 1, lx0 + 1);
					const tx = clamp(fx - lx0);

					// Fix: ensure currentDepth is also linearized for proper relative difference comparison
					let currentDepth = depthBuffer[y * w + x];
					if (currentDepth <= 0) continue;
					currentDepth = this._linearizeDepth(
						currentDepth,
						near,
						far,
						isLinearDepth
					);

					const idx00 = ly0 * lowW + lx0;
					const idx10 = ly0 * lowW + lx1;
					const idx01 = ly1 * lowW + lx0;
					const idx11 = ly1 * lowW + lx1;
					const d00 = lowDepthBuf[idx00];
					const d10 = lowDepthBuf[idx10];
					const d01 = lowDepthBuf[idx01];
					const d11 = lowDepthBuf[idx11];
					const relDiff00 =
						Math.abs(currentDepth - d00) / Math.max(currentDepth, d00, 1e-6);
					const relDiff10 =
						Math.abs(currentDepth - d10) / Math.max(currentDepth, d10, 1e-6);
					const relDiff01 =
						Math.abs(currentDepth - d01) / Math.max(currentDepth, d01, 1e-6);
					const relDiff11 =
						Math.abs(currentDepth - d11) / Math.max(currentDepth, d11, 1e-6);
					const depthW00 = Math.exp(-relDiff00 * relDiff00 * invSigmaSq2);
					const depthW10 = Math.exp(-relDiff10 * relDiff10 * invSigmaSq2);
					const depthW01 = Math.exp(-relDiff01 * relDiff01 * invSigmaSq2);
					const depthW11 = Math.exp(-relDiff11 * relDiff11 * invSigmaSq2);
					const spatialW00 = (1 - tx) * (1 - ty);
					const spatialW10 = tx * (1 - ty);
					const spatialW01 = (1 - tx) * ty;
					const spatialW11 = tx * ty;
					let w00 = spatialW00 * depthW00;
					let w10 = spatialW10 * depthW10;
					let w01 = spatialW01 * depthW01;
					let w11 = spatialW11 * depthW11;
					const totalWeight = w00 + w10 + w01 + w11;
					if (totalWeight > 1e-6) {
						const invTotal = 1.0 / totalWeight;
						w00 *= invTotal;
						w10 *= invTotal;
						w01 *= invTotal;
						w11 *= invTotal;
					} else {
						w00 = spatialW00;
						w10 = spatialW10;
						w01 = spatialW01;
						w11 = spatialW11;
					}
					const i00 = idx00 * 3;
					const i10 = idx10 * 3;
					const i01 = idx01 * 3;
					const i11 = idx11 * 3;
					const scatterR =
						scatterBuf[i00] * w00 +
						scatterBuf[i10] * w10 +
						scatterBuf[i01] * w01 +
						scatterBuf[i11] * w11;
					const scatterG =
						scatterBuf[i00 + 1] * w00 +
						scatterBuf[i10 + 1] * w10 +
						scatterBuf[i01 + 1] * w01 +
						scatterBuf[i11 + 1] * w11;
					const scatterB =
						scatterBuf[i00 + 2] * w00 +
						scatterBuf[i10 + 2] * w10 +
						scatterBuf[i01 + 2] * w01 +
						scatterBuf[i11 + 2] * w11;
					const idx = (y * w + x) << 2;
					pixels[idx] = Math.min(255, pixels[idx] + scatterR);
					pixels[idx + 1] = Math.min(255, pixels[idx + 1] + scatterG);
					pixels[idx + 2] = Math.min(255, pixels[idx + 2] + scatterB);
					pixels[idx + 3] = 255;
				}
			}
		});
	}

	private _reconstructViewPos(
		ndcX: number,
		ndcY: number,
		zView: number,
		camera: FrameContext["camera"]
	): IVector3 {
		if (camera.type === CameraType.Orthographic) {
			const orthoCam = camera as OrthographicCamera;
			const bounds = orthoCam.getBounds();
			const xView =
				((ndcX + 1) * 0.5) * (bounds.right - bounds.left) + bounds.left;
			const yView =
				((ndcY + 1) * 0.5) * (bounds.top - bounds.bottom) + bounds.bottom;

			return { x: xView, y: yView, z: -zView };
		}

		const fovRad = (camera.fov * Math.PI) / 180;
		const tanHalfFov = Math.tan(fovRad * 0.5);
		const aspect = camera.aspectRatio;

		const xView = ndcX * aspect * tanHalfFov * zView;
		const yView = ndcY * tanHalfFov * zView;

		return { x: xView, y: yView, z: -zView };
	}

	private _bilinearUpscale(
		pixels: Uint8ClampedArray,
		scatterBuf: Float32Array,
		w: number,
		h: number,
		lowW: number,
		lowH: number,
		ds: number,
		dirtyRects: IncrementalDirtyRect[]
	): void {
		this._forEachDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const fy = (y + 0.5) / ds - 0.5;
				const ly0 = Math.max(0, Math.floor(fy));
				const ly1 = Math.min(lowH - 1, ly0 + 1);
				const ty = clamp(fy - ly0);
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const fx = (x + 0.5) / ds - 0.5;
					const lx0 = Math.max(0, Math.floor(fx));
					const lx1 = Math.min(lowW - 1, lx0 + 1);
					const tx = clamp(fx - lx0);
					const i00 = (ly0 * lowW + lx0) * 3;
					const i10 = (ly0 * lowW + lx1) * 3;
					const i01 = (ly1 * lowW + lx0) * 3;
					const i11 = (ly1 * lowW + lx1) * 3;
					const w00 = (1 - tx) * (1 - ty);
					const w10 = tx * (1 - ty);
					const w01 = (1 - tx) * ty;
					const w11 = tx * ty;
					const scatterR =
						scatterBuf[i00] * w00 +
						scatterBuf[i10] * w10 +
						scatterBuf[i01] * w01 +
						scatterBuf[i11] * w11;
					const scatterG =
						scatterBuf[i00 + 1] * w00 +
						scatterBuf[i10 + 1] * w10 +
						scatterBuf[i01 + 1] * w01 +
						scatterBuf[i11 + 1] * w11;
					const scatterB =
						scatterBuf[i00 + 2] * w00 +
						scatterBuf[i10 + 2] * w10 +
						scatterBuf[i01 + 2] * w01 +
						scatterBuf[i11 + 2] * w11;
					const idx = (y * w + x) << 2;
					pixels[idx] = Math.min(255, pixels[idx] + scatterR);
					pixels[idx + 1] = Math.min(255, pixels[idx + 1] + scatterG);
					pixels[idx + 2] = Math.min(255, pixels[idx + 2] + scatterB);
					pixels[idx + 3] = 255;
				}
			}
		});
	}
}
/**
 * WebGPU implementation of volumetric lighting.
 */
export class WebGPUVolumetricLightingImplementation
	implements
		PostProcessPassImplementation<
			WebGPUVolumetricLightingContext,
			VolumetricOptions
		>
{
	public readonly id = "volumetric:webgpu";
	public readonly metadata = {
		context: {
			backend: "webgpu",
			kind: "screen",
			publishColorTarget: true,
			frameBinding: true,
			lightingState: true,
			histories: [
				{ property: "historyRead", historyId: "volumetric", side: "read" },
				{ property: "historyWrite", historyId: "volumetric", side: "write" },
				{
					property: "reservoirHistoryRead",
					historyId: "volumetric-reservoir",
					side: "read",
				},
				{
					property: "reservoirHistoryWrite",
					historyId: "volumetric-reservoir",
					side: "write",
				},
				{ property: "motionHistoryRead", historyId: "motion", side: "read" },
				{ property: "motionHistoryWrite", historyId: "motion", side: "write" },
			],
			transients: [
				{
					property: "hiZ",
					transientId: WEBGPU_HIZ_TRANSIENT_ID,
				},
			],
			motionHistoryCopy: {
				writeProperty: "motionHistoryWrite",
			},
		},
	} as const;
	private _resources = new WeakMap<
		PostProcessSharedContext,
		WebGPUVolumetricResources
	>();
	private _resourceSet = new Set<WebGPUVolumetricResources>();

	public async warmup(
		context: WebGPUVolumetricLightingContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: WebGPUVolumetricLightingContext | undefined
	): Promise<PostProcessPassResult> {
		if (
			!context?.encoder ||
			!context.targets ||
			!context.frameBinding
		) {
			return { ran: false };
		}
		const ran = await this._runVolumetricKernel(request, context);
		if (!ran) {
			return { ran: false };
		}
		await context.writeMotionHistoryFromCurrent?.();
		return {
			ran: true,
			updatedHistoryIds: ["volumetric", "volumetric-reservoir", "motion"],
		};
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("volumetric-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"volumetric pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"volumetric shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"volumetric params buffer"
			);
			resources.shared.destroyManagedResource(
				resources.lightBuffer,
				"volumetric light buffer"
			);
			resources.shared.invalidateBindingsByPrefix("volumetric-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
			resources.lightBuffer = null;
			resources.lightCapacity = 0;
			resources.frameIndex = 0;
			resources.groupLayout0 = null;
			resources.pipelineLayout = null;
		}
		this._resourceSet.clear();
		this._resources = new WeakMap<
			PostProcessSharedContext,
			WebGPUVolumetricResources
		>();
	}

	private async _runVolumetricKernel(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: WebGPUVolumetricLightingContext
	): Promise<boolean> {
		if (request.frameContext.camera.type === CameraType.Orthographic) {
			context.shared.warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for orthographic cameras."
			);
			return false;
		}
		const resources = await this._ensureResources(context.shared);
		const lightCount = this._updateLightBuffer(
			resources,
			context.lightingState ?? null
		);
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!context.hiZ ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params ||
			!resources.lightBuffer ||
			!context.historyRead ||
			!context.historyWrite ||
			!context.reservoirHistoryRead ||
			!context.reservoirHistoryWrite ||
			!context.motionHistoryRead ||
			!context.motionHistoryWrite
		) {
			return false;
		}
		const hiZMips = await context.shared.getHiZHelper().build({
			encoder: context.encoder,
			depth: context.targets.gMotionDepth,
			hiZ: context.hiZ,
		});
		if (hiZMips.length === 0) {
			return false;
		}

		const options = request.options ?? {};
		const samples = Math.max(
			1,
			Math.min(
				128,
				finiteOr(options.samples, DEFAULT_VOLUMETRIC_OPTIONS.samples)
			)
		);
		const weight = Math.max(
			0,
			finiteOr(options.weight, DEFAULT_VOLUMETRIC_OPTIONS.weight)
		);
		const exposure = Math.max(
			0,
			finiteOr(options.exposure, DEFAULT_VOLUMETRIC_OPTIONS.exposure)
		);
		const airDensity = Math.max(
			0.001,
			finiteOr(options.airDensity, DEFAULT_VOLUMETRIC_OPTIONS.airDensity)
		);
		const anisotropy = Math.max(
			-0.95,
			Math.min(
				0.95,
				finiteOr(options.anisotropy, DEFAULT_VOLUMETRIC_OPTIONS.anisotropy)
			)
		);
		const maxRayDistance = Math.max(
			0.1,
			finiteOr(
				options.maxRayDistance,
				DEFAULT_VOLUMETRIC_OPTIONS.maxRayDistance
			)
		);
		const scatteringAlbedo = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options.scatteringAlbedo,
					DEFAULT_VOLUMETRIC_OPTIONS.scatteringAlbedo
				)
			)
		);
		const shadowSampleInterval = Math.max(
			1,
			Math.min(
				32,
				finiteOr(
					options.shadowSampleInterval,
					DEFAULT_VOLUMETRIC_OPTIONS.shadowSampleInterval
				)
			)
		);
		const adaptiveSteps = options.adaptiveSteps === false ? 0 : 1;
		const depthThickness = Math.max(
			0.01,
			finiteOr(
				options.bilateralDepthSigma,
				DEFAULT_VOLUMETRIC_OPTIONS.bilateralDepthSigma
			) * 8
		);
		const maxMip = Math.max(0, hiZMips.length - 1);
		const restirCandidates = Math.max(
			1,
			Math.min(
				64,
				finiteOr(
					options.restirCandidates,
					DEFAULT_VOLUMETRIC_OPTIONS.restirCandidates
				)
			)
		);
		const restirTemporalWeight = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options.restirTemporalWeight,
					DEFAULT_VOLUMETRIC_OPTIONS.restirTemporalWeight
				)
			)
		);
		const restirScaleClamp = Math.max(
			1,
			finiteOr(
				options.restirScaleClamp,
				DEFAULT_VOLUMETRIC_OPTIONS.restirScaleClamp
			)
		);
		resources.frameIndex = (resources.frameIndex + 1) % 4096;

		context.shared.compute.writeBuffer(
			resources.params,
			new Float32Array([
				1 / Math.max(context.targets.sceneColor.width, 1),
				1 / Math.max(context.targets.sceneColor.height, 1),
				samples,
				weight,
				exposure,
				airDensity,
				anisotropy,
				maxRayDistance,
				scatteringAlbedo,
				shadowSampleInterval,
				adaptiveSteps,
				depthThickness,
				maxMip,
				0.75,
				(request.histories.volumetric?.valid ?? false) &&
					(request.histories.motion?.valid ?? false) ?
					1
				:	0,
				lightCount,
				restirCandidates,
				restirTemporalWeight,
				restirScaleClamp,
				resources.frameIndex,
			])
		);

		const target =
			context.targets.sceneColor === context.targets.postPong ?
				context.targets.postPing
			:	context.targets.postPong;
		const binding = context.shared.getCachedBindGroup(
			`volumetric-${target === context.targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: context.targets.sceneColor },
				{ binding: 1, resource: context.targets.gMotionDepth },
				{ binding: 2, resource: context.hiZ },
				{ binding: 3, resource: context.historyRead },
				{ binding: 4, resource: context.motionHistoryRead },
				{ binding: 5, resource: context.shared.sampler },
				{ binding: 6, resource: resources.params },
				{ binding: 7, resource: target },
				{ binding: 8, resource: context.historyWrite },
				{
					binding: 9,
					resource: context.reservoirHistoryRead,
				},
				{
					binding: 10,
					resource: context.reservoirHistoryWrite,
				},
				{ binding: 11, resource: resources.lightBuffer },
			],
			"WebGPUVolumetric_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUVolumetric" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		context.publishColorTarget?.(target);
		return true;
	}

	private _updateLightBuffer(
		resources: WebGPUVolumetricResources,
		lightingState: WebGPULightingState | null
	): number {
		const sourceLights = lightingState?.volumetricLights ?? [];
		const clampedLightCount = Math.min(sourceLights.length, MAX_VOLUMETRIC_LIGHTS);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			resources.shared.warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric ReSTIR clamps light count to ${MAX_VOLUMETRIC_LIGHTS}; extra lights are skipped`
			);
		}
		this._ensureLightBufferCapacity(resources, clampedLightCount);
		if (!resources.lightBuffer) {
			return 0;
		}

		const packedCount = Math.max(1, clampedLightCount);
		const layout = getWebGPUVolumetricLightLayout(packedCount);
		const packed = layout.createWriter();
		packed.expectByteLength(
			packedCount * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			"VolumetricLightBuffer"
		);
		for (let i = 0; i < clampedLightCount; i++) {
			const light = sourceLights[i];
			const isDirectional = light.type === 0;
			const isSpot = light.type === 2;
			packed.writeVec([i, "positionRange"], [
				light.position[0],
				light.position[1],
				light.position[2],
				isDirectional ? -1 : Math.max(light.range, 0.001),
			]);
			packed.writeVec([i, "directionOuter"], [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				isSpot ? light.outerCos : -2,
			]);
			packed.writeVec([i, "colorInner"], [
				light.color[0],
				light.color[1],
				light.color[2],
				isSpot ? light.innerCos : -2,
			]);
		}
		if (clampedLightCount === 0) {
			packed.writeVec([0, "positionRange"], [0, 0, 0, -1]);
		}
		resources.shared.compute.writeBuffer(
			resources.lightBuffer,
			packed.toFloat32Array()
		);
		return clampedLightCount;
	}

	private _ensureLightBufferCapacity(
		resources: WebGPUVolumetricResources,
		lightCount: number
	): void {
		const required = Math.max(1, lightCount);
		if (resources.lightBuffer && resources.lightCapacity >= required) {
			return;
		}
		let capacity = Math.max(1, resources.lightCapacity);
		while (capacity < required) {
			capacity *= 2;
		}
		resources.lightBuffer?.destroy();
		resources.lightBuffer = resources.shared.compute.createBuffer({
			label: "WebGPUVolumetricLights",
			size: capacity * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
		});
		resources.lightCapacity = capacity;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUVolumetricResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				lightBuffer: null,
				lightCapacity: 0,
				frameIndex: 0,
				groupLayout0: null,
				pipelineLayout: null,
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.getHiZHelper().ensureResources();
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.volumetric.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			if (shared.frameBindGroupLayout) {
				resources.groupLayout0 = shared.compute.createBindGroupLayout({
					label: "WebGPUVolumetric_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },
						{
							binding: 6,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{ binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 10,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 11,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "read-only-storage" },
						},
					],
				});
				resources.pipelineLayout = shared.compute.createPipelineLayout({
					label: "WebGPUVolumetric_PipelineLayout",
					bindGroupLayouts: [
						resources.groupLayout0,
						shared.frameBindGroupLayout,
					],
				});
				resources.pipeline = await shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: resources.pipelineLayout,
					compute: { module: resources.module, entryPoint: "csMain" },
				});
			} else {
				resources.pipeline = await shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: resources.module, entryPoint: "csMain" },
				});
			}
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export interface VolumetricLightingPassConfig
	extends Omit<
		PostProcessPassConfig<VolumetricOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical volumetric lighting pass shared by Software and WebGPU.
 */
export class VolumetricLightingPass extends PostProcessPass<
	VolumetricOptions,
	VolumetricOptions
> {
	public constructor(config: VolumetricLightingPassConfig = {}) {
		super({
			...config,
			...VOLUMETRIC_LIGHTING_PASS_ORDER,
			builtIn: true,
			warningLabel: "volumetric effects",
			implementations: {
				software: new SoftwareVolumetricLightingImplementation(),
				webgpu: new WebGPUVolumetricLightingImplementation(),
			},
		});
	}

	public override normalizeOptions(): VolumetricOptions {
		return {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}

	public override getHistoryDescriptors(): readonly PostProcessHistoryDescriptor[] {
		return [
			{ id: "volumetric", usage: DEFAULT_HISTORY_USAGE },
			{ id: "volumetric-reservoir", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: MOTION_HISTORY_USAGE },
		];
	}

	public override getTransientResourceDescriptors(
		request: PostProcessPassResolveRequest<VolumetricOptions>
	): readonly PostProcessTransientDescriptor[] {
		if (request.backend !== "webgpu") {
			return [];
		}
		return [
			{
				id: WEBGPU_HIZ_TRANSIENT_ID,
				usage: WEBGPU_HIZ_TRANSIENT_USAGE,
				mipMode: "full-chain",
			},
		];
	}
}
