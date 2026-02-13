import type { Renderer, VolumetricOptions } from "./Renderer";
import {
	type DirectionalLight,
	type PointLight,
	type SpotLight,
	LightType,
	isShadowCastingLight,
} from "../lights";
import type { IVector3 } from "../maths/types";
import { Matrix4 } from "../maths/Matrix4";
import { PostProcessConstants, VolumetricConstants } from "./Constants";

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
export class PostProcessor {
	private _gammaLUT: Uint8Array;
	private _lastGamma: number;
	private _prevScatterBuf: Float32Array | null;
	private _frameIndex: number;
	private _fxaaOutput?: Uint8ClampedArray;
	private _lumaBuf?: Float32Array;

	public renderer: Renderer;

	constructor(renderer: Renderer) {
		this.renderer = renderer;
		this._gammaLUT = new Uint8Array(256);
		this._lastGamma = -1;
		this._prevScatterBuf = null;
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

		const ds = Math.round(
			this._clamp(
				this._toFiniteNumber(options.downsample, 2),
				VolumetricConstants.MIN_DOWN_SAMPLE,
				VolumetricConstants.MAX_DOWN_SAMPLE
			)
		);
		const lowW = Math.ceil(w / ds);
		const lowH = Math.ceil(h / ds);
		const baseSampleCount = Math.round(
			this._clamp(
				this._toFiniteNumber(options.samples, 64),
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
		const maxRayDistance = Math.max(
			1,
			this._toFiniteNumber(options.maxRayDistance, 500)
		);
		const scatteringAlbedo = this._clamp(
			this._toFiniteNumber(options.scatteringAlbedo, 0.8),
			0,
			1
		);
		const shadowSampleInterval = Math.round(
			this._clamp(
				this._toFiniteNumber(options.shadowSampleInterval, 4),
				VolumetricConstants.MIN_SHADOW_SAMPLE_INTERVAL,
				VolumetricConstants.MAX_SHADOW_SAMPLE_INTERVAL
			)
		);
		const isLinearDepth = options.isLinearDepth !== false;
		const adaptiveSteps = options.adaptiveSteps !== false;
		const useBilateralUpscale = options.useBilateralUpscale !== false;
		const bilateralDepthSigma = Math.max(
			VolumetricConstants.MIN_BILATERAL_DEPTH_SIGMA,
			this._toFiniteNumber(options.bilateralDepthSigma, 0.05)
		);

		const camera = this.renderer.camera;
		const cameraPos = camera.position;
		const basis = this._getCameraBasis();
		const near = camera.near || 0.1;
		const far = camera.far || 1000;
		const g = anisotropy;
		const sigmaT = airDensity * VolumetricConstants.SIGMA_T_SCALE;
		const sigmaS = sigmaT * scatteringAlbedo;

		const shadowLight = lights.find(isShadowCastingLight);
		const shadowMap =
			shadowLight ? this.renderer.shadowMaps.get(shadowLight) : null;
		const hasShadows =
			this.renderer.params.enableShadows && !!shadowMap?.viewProjectionMatrix;

		const scatterBuf = new Float32Array(lowW * lowH * 3);
		const lowDepthBuf =
			useBilateralUpscale ? new Float32Array(lowW * lowH) : null;
		this._frameIndex++;

		const worldMatrix = this.renderer.params.worldMatrix || Matrix4.identity();
		for (const L of volLights) {
			L.updateWorldMatrix(worldMatrix);
		}

		for (let ly = 0; ly < lowH; ly++) {
			const py = Math.min(h - 1, ly * ds);
			for (let lx = 0; lx < lowW; lx++) {
				const px = Math.min(w - 1, lx * ds);
				let depth = depthBuffer[py * w + px];
				if (depth <= 0) continue;

				depth = this._linearizeDepth(depth, near, far, isLinearDepth);
				const ray = this._getWorldRayFromPixel(px, py, w, h, basis);
				const rayDirZ = Math.max(
					Math.abs(ray.camDirZ),
					VolumetricConstants.MIN_RAY_DIR_Z
				);
				const endDistance =
					depth === Infinity ? maxRayDistance : (
						Math.min(maxRayDistance, depth / rayDirZ)
					);

				if (endDistance <= VolumetricConstants.MIN_RAY_DISTANCE) continue;

				let sampleCount = baseSampleCount;
				if (adaptiveSteps) {
					const distRatio = endDistance / maxRayDistance;
					sampleCount = Math.max(
						VolumetricConstants.MIN_ADAPTIVE_SAMPLE_COUNT,
						Math.round(baseSampleCount * (0.5 + 0.5 * (1 - distRatio)))
					);
				}

				const stepSize = endDistance / sampleCount;
				const transStep = Math.exp(-sigmaT * stepSize);
				let accumR = 0,
					accumG = 0,
					accumB = 0;
				let transmittance = 1.0;
				const jitter = this._blueNoiseJitter(px, py, this._frameIndex);
				let t = jitter * stepSize;
				let cachedVisibility = 1.0;
				let lastShadowSampleStep = -shadowSampleInterval;

				for (let i = 0; i < sampleCount; i++) {
					const samplePoint = {
						x: cameraPos.x + ray.x * t,
						y: cameraPos.y + ray.y * t,
						z: cameraPos.z + ray.z * t,
					};

					for (const L of volLights) {
						const contrib = L.computeContribution(samplePoint);
						if (!contrib || contrib.type !== "direct") continue;
						if (!contrib.direction) continue;
						const lightDir = contrib.direction;

						let visibility = 1.0;
						if (hasShadows && L === shadowLight && shadowMap) {
							if (i - lastShadowSampleStep >= shadowSampleInterval) {
								cachedVisibility = shadowMap.getShadowFactor(
									samplePoint,
									lightDir
								);
								lastShadowSampleStep = i;
							}
							visibility = cachedVisibility;
						}

						const viewDotLight = -(
							ray.x * lightDir.x +
							ray.y * lightDir.y +
							ray.z * lightDir.z
						);
						const hgPhase = this._henyeyGreenstein(
							Math.max(-1, Math.min(1, viewDotLight)),
							g
						);
						const phase = hgPhase * 0.8 + (1.0 / (4 * Math.PI)) * 0.2;
						const scatter =
							visibility * transmittance * sigmaS * phase * weight * stepSize;

						accumR += contrib.color.r * scatter;
						accumG += contrib.color.g * scatter;
						accumB += contrib.color.b * scatter;
					}
					transmittance *= transStep;
					t += stepSize;
					if (transmittance < VolumetricConstants.TRANSMITTANCE_EARLY_EXIT)
						break;
				}

				const bIdx = (ly * lowW + lx) * 3;
				scatterBuf[bIdx] = accumR * exposure;
				scatterBuf[bIdx + 1] = accumG * exposure;
				scatterBuf[bIdx + 2] = accumB * exposure;
				if (lowDepthBuf) lowDepthBuf[ly * lowW + lx] = depth;
			}
		}

		if (useBilateralUpscale && lowDepthBuf) {
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
				bilateralDepthSigma
			);
		} else {
			this._bilinearUpscale(pixels, scatterBuf, w, h, lowW, lowH, ds);
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
		depthSigma: number
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
				const currentDepth = depthBuffer[y * w + x];
				if (currentDepth <= 0) continue;
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
