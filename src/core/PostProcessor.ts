import type { Renderer } from "./Renderer";
import {
	type DirectionalLight,
	type PointLight,
	type SpotLight,
	LightType,
	isShadowCastingLight,
} from "../lights";
import type { IVector3 } from "../maths/types";
import { Matrix4 } from "../maths/Matrix4";
import {
	PostProcessConstants,
	RenderConstants,
	VolumetricConstants,
} from "./Constants";

export interface PostProcessorLike {
	applyFXAA(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		pixels?: Uint8ClampedArray
	): void;
	applyVolumetricLight(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		pixels?: Uint8ClampedArray,
		depthBuffer?: Float32Array | null,
		options?: VolumetricOptions
	): void;
	applyGamma(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		gamma?: number,
		pixels?: Uint8ClampedArray
	): void;
}

export interface VolumetricOptions {
	mode?: "raymarch" | "radialBlur";
	samples?: number;
	downsample?: number;
	weight?: number;
	exposure?: number;
	airDensity?: number;
	anisotropy?: number;
	maxRayDistance?: number;
	scatteringAlbedo?: number;
	shadowSampleInterval?: number;
	radialSamples?: number;
	radialDensity?: number;
	radialDecay?: number;
	radialWeight?: number;
	radialOffscreenMargin?: number;
	radialLightDistance?: number;
	isLinearDepth?: boolean;
	adaptiveSteps?: boolean;
	useBilateralUpscale?: boolean;
	bilateralDepthSigma?: number;
	[key: string]: unknown;
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

/**
 * PostProcessor handles various image-space effects like FXAA, Volumetric Lighting, and Gamma Correction.
 */
export class PostProcessor implements PostProcessorLike {
	private _gammaLUT: Uint8Array;
	private _lastGamma: number;
	private _prevScatterBuf: Float32Array | null;
	private _frameIndex: number;
	private _fxaaOutput?: Uint8ClampedArray;
	private _lumaBuf?: Float32Array;

	// Temporal accumulation buffers
	private _prevVolumetricBuf: Float32Array | null = null;
	private _prevViewProj: Matrix4 | null = null;

	public renderer: Renderer;

	constructor(renderer: Renderer) {
		this.renderer = renderer;
		this._gammaLUT = new Uint8Array(256);
		this._lastGamma = -1;
		this._prevScatterBuf = null;
		this._prevVolumetricBuf = null;
		this._frameIndex = 0;
	}

	private _getPrimaryDirectionalLight(): DirectionalLight | null {
		const lights = this.renderer.scene?.lights || [];
		let primary: DirectionalLight | null = null;
		let maxIntensity = -Infinity;

		for (const light of lights) {
			if (light.type !== LightType.Directional) continue;
			const intensity = light.intensity ?? 1;
			if (intensity > maxIntensity) {
				maxIntensity = intensity;
				primary = light;
			}
		}

		return primary;
	}

	private _getCameraBasis(): CameraBasis {
		const view = this.renderer.camera.viewMatrix.elements;
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
		basis: CameraBasis
	): WorldRay {
		const camera = this.renderer.camera;
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
		const n = px + py * PostProcessConstants.NOISE_REFERENCE_WIDTH + frameIndex;
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

	private _smoothstep(edge0: number, edge1: number, x: number): number {
		const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
		return t * t * (3 - 2 * t);
	}

	private _clamp(value: number, min: number, max: number): number {
		return Math.max(min, Math.min(max, value));
	}

	private _toFiniteNumber(value: unknown, fallback: number): number {
		if (typeof value === "number" && Number.isFinite(value)) return value;
		return fallback;
	}

	private _computeSceneFalloff(
		distanceSq: number,
		fadeStartSq: number,
		fadeEndSq: number
	): number {
		if (distanceSq <= fadeStartSq) return 1.0;
		if (distanceSq >= fadeEndSq) return 0.0;
		const t = this._clamp(
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

	private _sampleBilinearScalar(
		buffer: Float32Array,
		w: number,
		h: number,
		x: number,
		y: number
	): number {
		const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)));
		const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)));
		const x1 = Math.max(0, Math.min(w - 1, x0 + 1));
		const y1 = Math.max(0, Math.min(h - 1, y0 + 1));
		const tx = this._clamp(x - x0, 0, 1);
		const ty = this._clamp(y - y0, 0, 1);

		const i00 = y0 * w + x0;
		const i10 = y0 * w + x1;
		const i01 = y1 * w + x0;
		const i11 = y1 * w + x1;

		const s00 = buffer[i00];
		const s10 = buffer[i10];
		const s01 = buffer[i01];
		const s11 = buffer[i11];

		return (
			s00 * (1 - tx) * (1 - ty) +
			s10 * tx * (1 - ty) +
			s01 * (1 - tx) * ty +
			s11 * tx * ty
		);
	}

	private _projectWorldPointToScreen(
		point: IVector3,
		w: number,
		h: number
	): { x: number; y: number; ndcX: number; ndcY: number; valid: boolean } {
		const clipPoint = Matrix4.transformPoint(
			this.renderer.camera.viewProjectionMatrix,
			point
		);
		if (clipPoint.w <= RenderConstants.MIN_CLIP_W) {
			return { x: 0, y: 0, ndcX: 0, ndcY: 0, valid: false };
		}

		const invW = 1.0 / clipPoint.w;
		const ndcX = clipPoint.x * invW;
		const ndcY = clipPoint.y * invW;

		return {
			x: (ndcX * 0.5 + 0.5) * w,
			y: (0.5 - ndcY * 0.5) * h,
			ndcX,
			ndcY,
			valid: true,
		};
	}

	private _applyRadialBlurPass(
		source: Float32Array,
		target: Float32Array,
		w: number,
		h: number,
		lightX: number,
		lightY: number,
		samples: number,
		density: number,
		decay: number
	): void {
		target.fill(0);
		const invSampleCount = 1.0 / samples;

		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const deltaX = ((x - lightX) * density) / samples;
				const deltaY = ((y - lightY) * density) / samples;

				let sampleX = x;
				let sampleY = y;
				let illuminationDecay = 1.0;
				let accum = 0;

				for (let i = 0; i < samples; i++) {
					sampleX -= deltaX;
					sampleY -= deltaY;

					if (
						sampleX < 0 ||
						sampleX > w - 1 ||
						sampleY < 0 ||
						sampleY > h - 1
					) {
						break;
					}

					accum +=
						this._sampleBilinearScalar(source, w, h, sampleX, sampleY) *
						illuminationDecay;
					illuminationDecay *= decay;
				}

				target[y * w + x] = accum * invSampleCount;
			}
		}
	}

	private _applyRadialBlurVolumetric(
		pixels: Uint8ClampedArray,
		depthBuffer: Float32Array,
		options: VolumetricOptions,
		w: number,
		h: number
	): void {
		const lights = this.renderer.scene?.lights || [];
		const directionalLights = lights.filter(
			(light): light is DirectionalLight => light.type === LightType.Directional
		);
		if (directionalLights.length === 0) return;

		const ds = Math.round(
			this._clamp(
				this._toFiniteNumber(
					options.downsample,
					VolumetricConstants.DEFAULT_DOWN_SAMPLE
				),
				VolumetricConstants.MIN_DOWN_SAMPLE,
				VolumetricConstants.MAX_DOWN_SAMPLE
			)
		);
		const lowW = Math.ceil(w / ds);
		const lowH = Math.ceil(h / ds);
		const lowPixelCount = lowW * lowH;

		const radialSamples = Math.round(
			this._clamp(
				this._toFiniteNumber(
					options.radialSamples,
					VolumetricConstants.DEFAULT_RADIAL_SAMPLES
				),
				VolumetricConstants.MIN_RADIAL_SAMPLES,
				VolumetricConstants.MAX_RADIAL_SAMPLES
			)
		);
		const radialDensity = this._clamp(
			this._toFiniteNumber(
				options.radialDensity,
				VolumetricConstants.DEFAULT_RADIAL_DENSITY
			),
			VolumetricConstants.MIN_RADIAL_DENSITY,
			VolumetricConstants.MAX_RADIAL_DENSITY
		);
		const radialDecay = this._clamp(
			this._toFiniteNumber(
				options.radialDecay,
				VolumetricConstants.DEFAULT_RADIAL_DECAY
			),
			VolumetricConstants.MIN_RADIAL_DECAY,
			VolumetricConstants.MAX_RADIAL_DECAY
		);
		const radialWeight = this._clamp(
			this._toFiniteNumber(options.radialWeight, options.weight ?? 0.4),
			0,
			VolumetricConstants.MAX_WEIGHT
		);
		const offscreenMargin = this._clamp(
			this._toFiniteNumber(
				options.radialOffscreenMargin,
				VolumetricConstants.DEFAULT_RADIAL_OFFSCREEN_MARGIN
			),
			VolumetricConstants.MIN_RADIAL_OFFSCREEN_MARGIN,
			VolumetricConstants.MAX_RADIAL_OFFSCREEN_MARGIN
		);
		const exposure = this._clamp(
			this._toFiniteNumber(options.exposure, 1.0),
			0,
			PostProcessConstants.MAX_EXPOSURE
		);
		const airDensity = this._clamp(
			this._toFiniteNumber(options.airDensity, 1.0),
			0,
			VolumetricConstants.MAX_AIR_DENSITY
		);
		const anisotropy = this._clamp(
			this._toFiniteNumber(options.anisotropy, 0.4),
			-0.99,
			0.99
		);
		const scatteringAlbedo = this._clamp(
			this._toFiniteNumber(options.scatteringAlbedo, 0.8),
			0,
			1
		);

		const maxRayDistance = Math.max(
			VolumetricConstants.MIN_RAY_DISTANCE,
			this._toFiniteNumber(options.maxRayDistance, 500)
		);

		const camera = this.renderer.camera;
		const near = camera.near || 0.1;
		const far = Math.max(
			near + VolumetricConstants.MIN_RAY_DISTANCE,
			Math.min(camera.far || 1000, maxRayDistance)
		);
		const radialLightDistance = this._clamp(
			this._toFiniteNumber(options.radialLightDistance, far),
			near,
			far
		);
		const isLinearDepth = options.isLinearDepth !== false;

		const sigmaT = airDensity * VolumetricConstants.SIGMA_T_SCALE;
		const sigmaS = sigmaT * scatteringAlbedo;

		const sceneBounds = this.renderer.scene.getBounds();
		const sceneCenter = sceneBounds.center;
		const sceneRadius = Math.max(
			sceneBounds.radius,
			VolumetricConstants.MIN_SCENE_BOUNDS_RADIUS
		);
		const sceneFadeStart =
			sceneRadius * VolumetricConstants.SCENE_BOUNDS_FADE_START_MULTIPLIER;
		const sceneFadeEnd =
			sceneRadius * VolumetricConstants.SCENE_BOUNDS_FADE_END_MULTIPLIER;
		const sceneFadeStartSq = sceneFadeStart * sceneFadeStart;
		const sceneFadeEndSq = sceneFadeEnd * sceneFadeEnd;

		const cameraPos = camera.position;
		const camToCenter = Math.hypot(
			cameraPos.x - sceneCenter.x,
			cameraPos.y - sceneCenter.y,
			cameraPos.z - sceneCenter.z
		);
		const infinityDepthLimit = this._clamp(
			camToCenter +
				sceneRadius * VolumetricConstants.SCENE_DEPTH_LIMIT_MULTIPLIER,
			near,
			far
		);

		const basis = this._getCameraBasis();
		const rayX = new Float32Array(lowPixelCount);
		const rayY = new Float32Array(lowPixelCount);
		const rayZ = new Float32Array(lowPixelCount);
		const pointX = new Float32Array(lowPixelCount);
		const pointY = new Float32Array(lowPixelCount);
		const pointZ = new Float32Array(lowPixelCount);
		const lowDepthBuf = new Float32Array(lowPixelCount);

		for (let y = 0; y < lowH; y++) {
			for (let x = 0; x < lowW; x++) {
				const px = Math.round(this._clamp((x + 0.5) * ds - 0.5, 0, w - 1));
				const py = Math.round(this._clamp((y + 0.5) * ds - 0.5, 0, h - 1));
				const lowIndex = y * lowW + x;

				const ray = this._getWorldRayFromPixel(px, py, w, h, basis);
				rayX[lowIndex] = ray.x;
				rayY[lowIndex] = ray.y;
				rayZ[lowIndex] = ray.z;

				const depthRaw = depthBuffer[py * w + px];
				const linearDepth = this._linearizeDepth(depthRaw, near, far, isLinearDepth);
				const depthLimit =
					linearDepth === Infinity ?
						infinityDepthLimit
					:	this._clamp(linearDepth, near, far);

				lowDepthBuf[lowIndex] = depthLimit;
				pointX[lowIndex] = cameraPos.x + ray.x * depthLimit;
				pointY[lowIndex] = cameraPos.y + ray.y * depthLimit;
				pointZ[lowIndex] = cameraPos.z + ray.z * depthLimit;
			}
		}

		const scatterBuf = new Float32Array(lowPixelCount * 3);
		const visibilityBuf = new Float32Array(lowPixelCount);
		const blurredBuf = new Float32Array(lowPixelCount);

		const ndcBound = 1 + offscreenMargin;
		const shadowsEnabled = this.renderer.params.enableShadows;
		const reusablePoint: IVector3 = { x: 0, y: 0, z: 0 };

		for (const light of directionalLights) {
			const lightContribution = light.computeContribution(cameraPos);
			if (!lightContribution || !lightContribution.direction) continue;

			const lightDirection = lightContribution.direction;
			const lightWorldPoint = {
				x: cameraPos.x + lightDirection.x * radialLightDistance,
				y: cameraPos.y + lightDirection.y * radialLightDistance,
				z: cameraPos.z + lightDirection.z * radialLightDistance,
			};
			const projectedLight = this._projectWorldPointToScreen(lightWorldPoint, w, h);
			if (
				!projectedLight.valid ||
				projectedLight.ndcX < -ndcBound ||
				projectedLight.ndcX > ndcBound ||
				projectedLight.ndcY < -ndcBound ||
				projectedLight.ndcY > ndcBound
			) {
				continue;
			}

			const lightX = projectedLight.x / ds;
			const lightY = projectedLight.y / ds;
			const shadowMap =
				shadowsEnabled && isShadowCastingLight(light) ?
					this.renderer.shadowMaps.get(light) ?? null
				:	null;

			for (let i = 0; i < lowPixelCount; i++) {
				reusablePoint.x = pointX[i];
				reusablePoint.y = pointY[i];
				reusablePoint.z = pointZ[i];

				const sceneDx = reusablePoint.x - sceneCenter.x;
				const sceneDy = reusablePoint.y - sceneCenter.y;
				const sceneDz = reusablePoint.z - sceneCenter.z;
				const sceneFalloff = this._computeSceneFalloff(
					sceneDx * sceneDx + sceneDy * sceneDy + sceneDz * sceneDz,
					sceneFadeStartSq,
					sceneFadeEndSq
				);
				if (sceneFalloff <= 0) {
					visibilityBuf[i] = 0;
					continue;
				}

				let visibility = 1.0;
				if (shadowMap) {
					const shadow = shadowMap.getShadowFactor(reusablePoint, null);
					visibility = (shadow.r + shadow.g + shadow.b) / 3;
				}

				const viewDotLight =
					rayX[i] * lightDirection.x +
					rayY[i] * lightDirection.y +
					rayZ[i] * lightDirection.z;
				const phase = this._henyeyGreenstein(
					this._clamp(viewDotLight, -1, 1),
					anisotropy
				);

				visibilityBuf[i] = visibility * phase * sceneFalloff;
			}

			this._applyRadialBlurPass(
				visibilityBuf,
				blurredBuf,
				lowW,
				lowH,
				lightX,
				lightY,
				radialSamples,
				radialDensity,
				radialDecay
			);

			const lightScaleR = lightContribution.color.r * sigmaS * radialWeight;
			const lightScaleG = lightContribution.color.g * sigmaS * radialWeight;
			const lightScaleB = lightContribution.color.b * sigmaS * radialWeight;

			for (let i = 0; i < lowPixelCount; i++) {
				const blurred = blurredBuf[i];
				if (blurred <= 0) continue;

				const scatterIndex = i * 3;
				scatterBuf[scatterIndex] += blurred * lightScaleR;
				scatterBuf[scatterIndex + 1] += blurred * lightScaleG;
				scatterBuf[scatterIndex + 2] += blurred * lightScaleB;
			}
		}

		for (let i = 0; i < scatterBuf.length; i++) {
			scatterBuf[i] *= exposure;
		}

		this._filterScatterBuffer(scatterBuf, lowW, lowH);

		if (options.useBilateralUpscale !== false) {
			this._bilateralUpscale(
				pixels,
				scatterBuf,
				depthBuffer,
				lowDepthBuf,
				w,
				h,
				lowW,
				lowH,
				ds,
				this._toFiniteNumber(options.bilateralDepthSigma, 0.05),
				near,
				far,
				isLinearDepth
			);
		} else {
			this._bilinearUpscale(pixels, scatterBuf, w, h, lowW, lowH, ds);
		}
	}

	public applyFXAA(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		pixels: Uint8ClampedArray | null = null
	): void {
		const w = canvas.width;
		const h = canvas.height;
		let imageData: ImageData | null = null;

		if (!pixels) {
			imageData = ctx.getImageData(0, 0, w, h);
			pixels = imageData.data;
		}

		if (!this._fxaaOutput || this._fxaaOutput.length !== pixels.length) {
			this._fxaaOutput = new Uint8ClampedArray(pixels.length);
		}
		const output = this._fxaaOutput;
		output.set(pixels);

		const lumaSize = w * h;
		if (!this._lumaBuf || this._lumaBuf.length !== lumaSize) {
			this._lumaBuf = new Float32Array(lumaSize);
		}
		const lumaBuf = this._lumaBuf;

		for (let i = 0, len = pixels.length; i < len; i += 4) {
			lumaBuf[i >> 2] =
				0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
		}

		for (let y = 1; y < h - 1; y++) {
			const row = y * w;
			for (let x = 1; x < w - 1; x++) {
				const i = row + x;
				const idx = i << 2;
				const lCenter = lumaBuf[i];
				const lU = lumaBuf[i - w];
				const lD = lumaBuf[i + w];
				const lL = lumaBuf[i - 1];
				const lR = lumaBuf[i + 1];

				let lMin = Math.min(lCenter, lU, lD, lL, lR);
				let lMax = Math.max(lCenter, lU, lD, lL, lR);
				const lRange = lMax - lMin;

				if (
					lRange <
					Math.max(
						PostProcessConstants.FXAA_EDGE_THRESHOLD_MIN,
						lMax * PostProcessConstants.FXAA_EDGE_THRESHOLD_MULTIPLIER
					)
				) {
					continue;
				}

				const iU = (i - w) << 2;
				const iD = (i + w) << 2;
				const iL = (i - 1) << 2;
				const iR = (i + 1) << 2;

				output[idx] =
					(pixels[idx] + pixels[iU] + pixels[iD] + pixels[iL] + pixels[iR]) *
					0.2;
				output[idx + 1] =
					(pixels[idx + 1] +
						pixels[iU + 1] +
						pixels[iD + 1] +
						pixels[iL + 1] +
						pixels[iR + 1]) *
					0.2;
				output[idx + 2] =
					(pixels[idx + 2] +
						pixels[iU + 2] +
						pixels[iD + 2] +
						pixels[iL + 2] +
						pixels[iR + 2]) *
					0.2;
			}
		}

		if (imageData) {
			imageData.data.set(output);
			ctx.putImageData(imageData, 0, 0);
		} else {
			pixels.set(output);
		}
	}

	public applyVolumetricLight(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		pixels: Uint8ClampedArray | null = null,
		depthBuffer: Float32Array | null = null,
		options: VolumetricOptions = {}
	): void {
		if (!depthBuffer) return;

		const lights = this.renderer.scene?.lights || [];
		const volLights = lights.filter(
			(light): light is VolumetricLight =>
				light.type === LightType.Directional ||
				light.type === LightType.Point ||
				light.type === LightType.Spot
		);
		if (volLights.length === 0) return;

		const w = canvas.width;
		const h = canvas.height;
		let imageData: ImageData | null = null;
		if (!pixels) {
			imageData = ctx.getImageData(0, 0, w, h);
			pixels = imageData.data;
		}

		const volumetricMode = options.mode === "radialBlur" ? "radialBlur" : "raymarch";
		if (volumetricMode === "radialBlur") {
			this._applyRadialBlurVolumetric(pixels, depthBuffer, options, w, h);
			if (imageData) ctx.putImageData(imageData, 0, 0);
			return;
		}

		// Consolidate options with range protection
		const ds = Math.round(
			this._clamp(
				this._toFiniteNumber(
					options.downsample,
					VolumetricConstants.DEFAULT_DOWN_SAMPLE
				),
				VolumetricConstants.MIN_DOWN_SAMPLE,
				VolumetricConstants.MAX_DOWN_SAMPLE
			)
		);
		const gridW = Math.ceil(w / ds);
		const gridH = Math.ceil(h / ds);
		const gridD = Math.round(
			this._clamp(
				this._toFiniteNumber(
					options.samples,
					VolumetricConstants.DEFAULT_SAMPLES
				),
				VolumetricConstants.MIN_SAMPLES,
				VolumetricConstants.MAX_SAMPLES
			)
		);

		const weight = this._clamp(
			this._toFiniteNumber(options.weight, 0.4),
			0,
			VolumetricConstants.MAX_WEIGHT
		);
		const exposure = this._clamp(
			this._toFiniteNumber(options.exposure, 1.0),
			0,
			PostProcessConstants.MAX_EXPOSURE
		);
		const airDensity = this._clamp(
			this._toFiniteNumber(options.airDensity, 1.0),
			0,
			VolumetricConstants.MAX_AIR_DENSITY
		);
		const anisotropy = this._clamp(
			this._toFiniteNumber(options.anisotropy, 0.4),
			-0.99,
			0.99
		);
		const scatteringAlbedo = this._clamp(
			this._toFiniteNumber(options.scatteringAlbedo, 0.8),
			0,
			1
		);

		// Use maxRayDistance consistently to limit total depth
		const maxRayDistance = Math.max(
			VolumetricConstants.MIN_RAY_DISTANCE,
			this._toFiniteNumber(options.maxRayDistance, 500)
		);

		const camera = this.renderer.camera;
		const cameraPos = camera.position;
		const basis = this._getCameraBasis();
		const near = camera.near || 0.1;
		const far = Math.min(camera.far || 1000, maxRayDistance);

		const sigmaT = airDensity * VolumetricConstants.SIGMA_T_SCALE;
		const sigmaS = sigmaT * scatteringAlbedo;

		const shadowsEnabled = this.renderer.params.enableShadows;
		const shadowInterval = Math.round(
			this._clamp(
				this._toFiniteNumber(options.shadowSampleInterval, 1),
				VolumetricConstants.MIN_SHADOW_SAMPLE_INTERVAL,
				VolumetricConstants.MAX_SHADOW_SAMPLE_INTERVAL
			)
		);

		const sceneBounds = this.renderer.scene.getBounds();
		const sceneCenter = sceneBounds.center;
		const sceneRadius = Math.max(
			sceneBounds.radius,
			VolumetricConstants.MIN_SCENE_BOUNDS_RADIUS
		);
		const sceneFadeStart =
			sceneRadius * VolumetricConstants.SCENE_BOUNDS_FADE_START_MULTIPLIER;
		const sceneFadeEnd =
			sceneRadius * VolumetricConstants.SCENE_BOUNDS_FADE_END_MULTIPLIER;
		const sceneFadeStartSq = sceneFadeStart * sceneFadeStart;
		const sceneFadeEndSq = sceneFadeEnd * sceneFadeEnd;

		const camToCenter = Math.hypot(
			cameraPos.x - sceneCenter.x,
			cameraPos.y - sceneCenter.y,
			cameraPos.z - sceneCenter.z
		);
		const infinityDepthLimit = this._clamp(
			camToCenter +
				sceneRadius * VolumetricConstants.SCENE_DEPTH_LIMIT_MULTIPLIER,
			near,
			far
		);

		// 1. Light Injection Grid
		const scatterGrid = new Float32Array(gridW * gridH * gridD * 3);
		const lightCount = volLights.length;
		const visibilityCache = new Float32Array(gridW * gridH * lightCount);
		visibilityCache.fill(1.0);
		this._frameIndex++;
		const jitterStrength = ds * VolumetricConstants.GRID_SAMPLE_JITTER_STRENGTH;
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
					const px = Math.round(this._clamp(sampleXCenter + jitterX, 0, w - 1));
					const py = Math.round(this._clamp(sampleYCenter + jitterY, 0, h - 1));
					const ray = this._getWorldRayFromPixel(px, py, w, h, basis);

					const samplePoint = {
						x: cameraPos.x + ray.x * dist,
						y: cameraPos.y + ray.y * dist,
						z: cameraPos.z + ray.z * dist,
					};

					const sceneDx = samplePoint.x - sceneCenter.x;
					const sceneDy = samplePoint.y - sceneCenter.y;
					const sceneDz = samplePoint.z - sceneCenter.z;
					const sceneFalloff = this._computeSceneFalloff(
						sceneDx * sceneDx + sceneDy * sceneDy + sceneDz * sceneDz,
						sceneFadeStartSq,
						sceneFadeEndSq
					);
					if (sceneFalloff <= 0) continue;

					let r = 0,
						g = 0,
						b = 0;
					const shouldSampleShadow = z % shadowInterval === 0;
					const cellIndex = y * gridW + x;

					for (let li = 0; li < lightCount; li++) {
						const L = volLights[li];
						const contrib = L.computeContribution(samplePoint);
						if (!contrib || contrib.type !== "direct" || !contrib.direction)
							continue;

						const cacheIndex = cellIndex * lightCount + li;
						let vis = visibilityCache[cacheIndex];
						if (shadowsEnabled && isShadowCastingLight(L)) {
							const sm = this.renderer.shadowMaps.get(L);
							if (sm && (shouldSampleShadow || z === 0)) {
								// Note: Passing null as normal for volume points to use volume-specific bias
								const shadow = sm.getShadowFactor(samplePoint, null);
								vis = (shadow.r + shadow.g + shadow.b) / 3;
								visibilityCache[cacheIndex] = vis;
							} else if (!sm) {
								vis = 1.0;
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
							this._clamp(viewDotLight, -1, 1),
							anisotropy
						);
						const scatter = phase * sigmaS * weight * sceneFalloff;

						r += contrib.color.r * vis * scatter;
						g += contrib.color.g * vis * scatter;
						b += contrib.color.b * vis * scatter;
					}

					const idx = sliceBase + (y * gridW + x) * 3;
					scatterGrid[idx] = r;
					scatterGrid[idx + 1] = g;
					scatterGrid[idx + 2] = b;
				}
			}
		}

		// 2. Integration along rays
		const scatterBuf = new Float32Array(gridW * gridH * 3);
		const lowDepthBuf = new Float32Array(gridW * gridH);

		for (let y = 0; y < gridH; y++) {
			for (let x = 0; x < gridW; x++) {
				const screenPX = Math.round(
					this._clamp((x + 0.5) * ds - 0.5, 0, w - 1)
				);
				const screenPY = Math.round(
					this._clamp((y + 0.5) * ds - 0.5, 0, h - 1)
				);
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
					if (transmittance < VolumetricConstants.TRANSMITTANCE_EARLY_EXIT)
						break;
				}

				const bIdx = (y * gridW + x) * 3;
				scatterBuf[bIdx] = accumR * exposure;
				scatterBuf[bIdx + 1] = accumG * exposure;
				scatterBuf[bIdx + 2] = accumB * exposure;
				lowDepthBuf[y * gridW + x] = depthLimit;
			}
		}

		this._filterScatterBuffer(scatterBuf, gridW, gridH);

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
				options.isLinearDepth !== false
			);
		} else {
			this._bilinearUpscale(pixels, scatterBuf, w, h, gridW, gridH, ds);
		}

		if (imageData) ctx.putImageData(imageData, 0, 0);
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
		isLinearDepth: boolean
	): void {
		const invSigmaSq2 = 1.0 / (2.0 * depthSigma * depthSigma);
		for (let y = 0; y < h; y++) {
			const fy = (y + 0.5) / ds - 0.5;
			const ly0 = Math.max(0, Math.floor(fy)),
				ly1 = Math.min(lowH - 1, ly0 + 1),
				ty = Math.max(0, Math.min(1, fy - ly0));
			for (let x = 0; x < w; x++) {
				const fx = (x + 0.5) / ds - 0.5;
				const lx0 = Math.max(0, Math.floor(fx)),
					lx1 = Math.min(lowW - 1, lx0 + 1),
					tx = Math.max(0, Math.min(1, fx - lx0));

				// Fix: ensure currentDepth is also linearized for proper relative difference comparison
				let currentDepth = depthBuffer[y * w + x];
				if (currentDepth <= 0) continue;
				currentDepth = this._linearizeDepth(
					currentDepth,
					near,
					far,
					isLinearDepth
				);

				const idx00 = ly0 * lowW + lx0,
					idx10 = ly0 * lowW + lx1,
					idx01 = ly1 * lowW + lx0,
					idx11 = ly1 * lowW + lx1;
				const d00 = lowDepthBuf[idx00],
					d10 = lowDepthBuf[idx10],
					d01 = lowDepthBuf[idx01],
					d11 = lowDepthBuf[idx11];
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
				const spatialW00 = (1 - tx) * (1 - ty),
					spatialW10 = tx * (1 - ty),
					spatialW01 = (1 - tx) * ty,
					spatialW11 = tx * ty;
				let w00 = spatialW00 * depthW00,
					w10 = spatialW10 * depthW10,
					w01 = spatialW01 * depthW01,
					w11 = spatialW11 * depthW11;
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
				const i00 = idx00 * 3,
					i10 = idx10 * 3,
					i01 = idx01 * 3,
					i11 = idx11 * 3;
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
	}

	private _bilinearUpscale(
		pixels: Uint8ClampedArray,
		scatterBuf: Float32Array,
		w: number,
		h: number,
		lowW: number,
		lowH: number,
		ds: number
	): void {
		for (let y = 0; y < h; y++) {
			const fy = (y + 0.5) / ds - 0.5;
			const ly0 = Math.max(0, Math.floor(fy)),
				ly1 = Math.min(lowH - 1, ly0 + 1),
				ty = Math.max(0, Math.min(1, fy - ly0));
			for (let x = 0; x < w; x++) {
				const fx = (x + 0.5) / ds - 0.5;
				const lx0 = Math.max(0, Math.floor(fx)),
					lx1 = Math.min(lowW - 1, lx0 + 1),
					tx = Math.max(0, Math.min(1, fx - lx0));
				const i00 = (ly0 * lowW + lx0) * 3,
					i10 = (ly0 * lowW + lx1) * 3,
					i01 = (ly1 * lowW + lx0) * 3,
					i11 = (ly1 * lowW + lx1) * 3;
				const w00 = (1 - tx) * (1 - ty),
					w10 = tx * (1 - ty),
					w01 = (1 - tx) * ty,
					w11 = tx * ty;
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
	}

	private _updateGammaLUT(gamma: number): void {
		if (this._lastGamma === gamma) return;
		const invGamma = 1.0 / gamma;
		for (let i = 0; i < 256; i++) {
			this._gammaLUT[i] = Math.pow(i / 255.0, invGamma) * 255.0;
		}
		this._lastGamma = gamma;
	}

	public applyGamma(
		ctx: CanvasRenderingContext2D,
		canvas: HTMLCanvasElement,
		gamma: number = 2.2,
		pixels: Uint8ClampedArray | null = null
	): void {
		const w = canvas.width,
			h = canvas.height;
		let imageData: ImageData | null = null;
		if (!pixels) {
			imageData = ctx.getImageData(0, 0, w, h);
			pixels = imageData.data;
		}
		const safeGamma = this._clamp(
			this._toFiniteNumber(gamma, PostProcessConstants.DEFAULT_GAMMA),
			PostProcessConstants.MIN_GAMMA,
			PostProcessConstants.MAX_GAMMA
		);
		this._updateGammaLUT(safeGamma);
		const lut = this._gammaLUT;
		for (let i = 0; i < pixels.length; i += 4) {
			pixels[i] = lut[pixels[i]];
			pixels[i + 1] = lut[pixels[i + 1]];
			pixels[i + 2] = lut[pixels[i + 2]];
		}
		if (imageData) ctx.putImageData(imageData, 0, 0);
	}
}
