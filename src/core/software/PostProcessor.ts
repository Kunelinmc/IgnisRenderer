import { Matrix4 } from "../../maths/Matrix4";
import { Vector3 } from "../../maths/Vector3";
import {
	PostProcessConstants,
	VolumetricConstants,
	SSAOConstants,
} from "../constants";
import type { Renderer } from "../Renderer";
import {
	type DirectionalLight,
	type PointLight,
	type SpotLight,
	LightType,
	isShadowCastingLight,
} from "../../lights";
import {
	createLightContribution,
	evaluateLightContribution,
} from "./lighting/LightEvaluator";
import { clamp, linearToSRGB } from "../../maths/Common";
import type { IVector3 } from "../../maths/types";
import { CameraType } from "../../cameras/Camera";
import type { OrthographicCamera } from "../../cameras/OrthographicCamera";
import type {
	SSAOOptions,
	VolumetricOptions,
	FramePassStage,
	FrameContext,
} from "../pipeline/types";

export interface PostProcessorLike {
	applyFXAA(context: FrameContext, ctx: CanvasRenderingContext2D): void;
	applyVolumetricLight(
		context: FrameContext,
		ctx: CanvasRenderingContext2D
	): void;
	applyGamma(context: FrameContext, ctx: CanvasRenderingContext2D): void;
	applySSAO(context: FrameContext): void;
}

// Feature options moved to types.ts

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
	private _sRGBLUT: Uint8Array;
	private _lutBuilt: boolean;
	private _lastGamma: number;
	private _prevScatterBuf: Float32Array | null; // Temp for scatter filter
	private _frameIndex: number;
	private _fxaaOutput?: Uint8ClampedArray;
	private _lumaBuf?: Float32Array;

	private _ssaoKernel: IVector3[] = [];
	private _ssaoNoise: IVector3[] = [];
	private _ssaoBuffer: Float32Array | null = null;
	private _ssaoBlurTemp: Float32Array | null = null;

	// Volumetric lighting persistent buffers
	private _scatterGrid: Float32Array | null = null;
	private _visibilityCache: Float32Array | null = null;
	private _scatterBuf: Float32Array | null = null;
	private _lowDepthBuf: Float32Array | null = null;

	// Temporal accumulation buffers
	private _prevVolumetricBuf: Float32Array | null = null;
	private _prevViewProj: Matrix4 | null = null;

	public renderer: Renderer;

	constructor(renderer: Renderer) {
		this.renderer = renderer;
		this._sRGBLUT = new Uint8Array(256);
		this._lutBuilt = false;
		this._lastGamma = -1;
		this._prevScatterBuf = null;
		this._prevVolumetricBuf = null;
		this._frameIndex = 0;
		this._initSSAOKernel();
	}

	private _initSSAOKernel(): void {
		for (let i = 0; i < SSAOConstants.DEFAULT_SAMPLES; i++) {
			const sample = {
				x: Math.random() * 2 - 1,
				y: Math.random() * 2 - 1,
				z: Math.random(), // Hemisphere
			};
			const isLen = Math.hypot(sample.x, sample.y, sample.z) || 1;
			sample.x /= isLen;
			sample.y /= isLen;
			sample.z /= isLen;

			// Scale samples to be more grouped towards the origin
			let scale = i / SSAOConstants.DEFAULT_SAMPLES;
			scale = 0.1 + scale * scale * (1.0 - 0.1);
			sample.x *= scale;
			sample.y *= scale;
			sample.z *= scale;

			this._ssaoKernel.push(sample);
		}

		// Noise texture for SSAO rotations
		for (
			let i = 0;
			i < SSAOConstants.NOISE_SIZE * SSAOConstants.NOISE_SIZE;
			i++
		) {
			const noise = {
				x: Math.random() * 2 - 1,
				y: Math.random() * 2 - 1,
				z: 0.0,
			};
			const isLen = Math.hypot(noise.x, noise.y, noise.z) || 1;
			this._ssaoNoise.push({ x: noise.x / isLen, y: noise.y / isLen, z: 0.0 });
		}
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

	private _toFiniteNumber(value: unknown, fallback: number): number {
		if (typeof value === "number" && Number.isFinite(value)) return value;
		return fallback;
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

	private _sampleBilinear(
		pixels: Uint8ClampedArray,
		w: number,
		h: number,
		x: number,
		y: number,
		outCol: { r: number; g: number; b: number; a: number }
	): void {
		const sx = clamp(x, 0, w - 1);
		const sy = clamp(y, 0, h - 1);
		const x1 = Math.floor(sx);
		const y1 = Math.floor(sy);
		const x2 = Math.min(x1 + 1, w - 1);
		const y2 = Math.min(y1 + 1, h - 1);

		const tx = sx - x1;
		const ty = sy - y1;

		const i1 = (y1 * w + x1) << 2;
		const i2 = (y1 * w + x2) << 2;
		const i3 = (y2 * w + x1) << 2;
		const i4 = (y2 * w + x2) << 2;

		const w1 = (1 - tx) * (1 - ty);
		const w2 = tx * (1 - ty);
		const w3 = (1 - tx) * ty;
		const w4 = tx * ty;

		outCol.r =
			pixels[i1] * w1 + pixels[i2] * w2 + pixels[i3] * w3 + pixels[i4] * w4;
		outCol.g =
			pixels[i1 + 1] * w1 +
			pixels[i2 + 1] * w2 +
			pixels[i3 + 1] * w3 +
			pixels[i4 + 1] * w4;
		outCol.b =
			pixels[i1 + 2] * w1 +
			pixels[i2 + 2] * w2 +
			pixels[i3 + 2] * w3 +
			pixels[i4 + 2] * w4;
		outCol.a =
			pixels[i1 + 3] * w1 +
			pixels[i2 + 3] * w2 +
			pixels[i3 + 3] * w3 +
			pixels[i4 + 3] * w4;
	}

	private _getLumaBilinear(
		luma: Float32Array,
		w: number,
		h: number,
		x: number,
		y: number
	): number {
		const sx = clamp(x, 0, w - 1);
		const sy = clamp(y, 0, h - 1);
		const x1 = Math.floor(sx);
		const y1 = Math.floor(sy);
		const x2 = Math.min(x1 + 1, w - 1);
		const y2 = Math.min(y1 + 1, h - 1);

		const tx = sx - x1;
		const ty = sy - y1;

		const i1 = y1 * w + x1;
		const i2 = y1 * w + x2;
		const i3 = y2 * w + x1;
		const i4 = y2 * w + x2;

		return (
			luma[i1] * (1 - tx) * (1 - ty) +
			luma[i2] * tx * (1 - ty) +
			luma[i3] * (1 - tx) * ty +
			luma[i4] * tx * ty
		);
	}

	public applyFXAA(context: FrameContext, ctx: CanvasRenderingContext2D): void {
		const { width: w, height: h } = context.attachments;
		let pixels = context.attachments.pixels;
		let imageData: ImageData | null = null;

		if (!pixels) {
			imageData = ctx.getImageData(0, 0, w, h);
			pixels = imageData.data;
		}

		if (!this._fxaaOutput || this._fxaaOutput.length !== pixels.length) {
			this._fxaaOutput = new Uint8ClampedArray(pixels.length);
		}
		const output = this._fxaaOutput;

		const lumaSize = w * h;
		if (!this._lumaBuf || this._lumaBuf.length !== lumaSize) {
			this._lumaBuf = new Float32Array(lumaSize);
		}
		const luma = this._lumaBuf;

		// 1. Calculate Perceptual Luma (0.0 - 1.0)
		for (let i = 0, len = pixels.length; i < len; i += 4) {
			const r = pixels[i] / 255.0;
			const g = pixels[i + 1] / 255.0;
			const b = pixels[i + 2] / 255.0;
			// Rec. 709 luma with square root for perceptual linear-to-gamma approximation
			luma[i >> 2] = Math.sqrt(0.2126 * r + 0.7152 * g + 0.0722 * b);
		}

		const thresholdMin = PostProcessConstants.FXAA_EDGE_THRESHOLD_MIN;
		const thresholdMul = PostProcessConstants.FXAA_EDGE_THRESHOLD_MULTIPLIER;
		const subpixQual = PostProcessConstants.FXAA_SUBPIX_QUALITY;
		const qual = PostProcessConstants.FXAA_QUALITY;
		const iterations = qual.length;

		const outCol = { r: 0, g: 0, b: 0, a: 0 };

		for (let y = 0; y < h; y++) {
			const row = y * w;
			for (let x = 0; x < w; x++) {
				const i = row + x;
				const idx = i << 2;
				const L = luma[i];

				// Neighbor lumas
				const Ln = y > 0 ? luma[i - w] : L;
				const Ls = y < h - 1 ? luma[i + w] : L;
				const Le = x < w - 1 ? luma[i + 1] : L;
				const Lw = x > 0 ? luma[i - 1] : L;

				const Lmin = Math.min(L, Ln, Ls, Le, Lw);
				const Lmax = Math.max(L, Ln, Ls, Le, Lw);
				const Lrange = Lmax - Lmin;

				if (Lrange < Math.max(thresholdMin, Lmax * thresholdMul)) {
					output[idx] = pixels[idx];
					output[idx + 1] = pixels[idx + 1];
					output[idx + 2] = pixels[idx + 2];
					output[idx + 3] = pixels[idx + 3];
					continue;
				}

				// Edge detection - proceding with FXAA 3.11 logic
				const Lnw = y > 0 && x > 0 ? luma[i - w - 1] : Ln;
				const Lne = y > 0 && x < w - 1 ? luma[i - w + 1] : Ln;
				const Lsw = y < h - 1 && x > 0 ? luma[i + w - 1] : Ls;
				const Lse = y < h - 1 && x < w - 1 ? luma[i + w + 1] : Ls;

				const LedgesRU = Ln + Ls;
				const LedgesLV = Le + Lw;

				// Subpixel aliasing removal
				let Lfiltered = 2.0 * (LedgesRU + LedgesLV);
				Lfiltered += Lne + Lnw + Lse + Lsw;
				Lfiltered /= 12.0;
				const subpixOffset1 = Math.abs(Lfiltered - L);
				const subpixOffset2 = clamp(subpixOffset1 / Lrange, 0.0, 1.0);
				const subpixOffsetFinal =
					(-2.0 * subpixOffset2 + 3.0) * subpixOffset2 * subpixOffset2;
				const subpixOffset = subpixOffsetFinal * subpixOffsetFinal * subpixQual;

				// Edge orientation
				const edgeHorz =
					Math.abs(-2.0 * Lw + Lnw + Lsw) +
					Math.abs(-2.0 * L + Ln + Ls) * 2.0 +
					Math.abs(-2.0 * Le + Lne + Lse);
				const edgeVert =
					Math.abs(-2.0 * Ln + Lnw + Lne) +
					Math.abs(-2.0 * L + Lw + Le) * 2.0 +
					Math.abs(-2.0 * Ls + Lsw + Lse);
				const isHorz = edgeHorz >= edgeVert;

				// Select step direction
				const L1 = isHorz ? Ln : Lw;
				const L2 = isHorz ? Ls : Le;
				const gradient1 = Math.abs(L1 - L);
				const gradient2 = Math.abs(L2 - L);

				const is1Steeper = gradient1 >= gradient2;
				const gradientScaled = 0.25 * Math.max(gradient1, gradient2);

				const stepSign = is1Steeper ? -1 : 1;
				const Ledge = is1Steeper ? (L1 + L) * 0.5 : (L2 + L) * 0.5;

				// Span Search
				let posN_x = x,
					posN_y = y;
				if (isHorz) posN_y += stepSign * 0.5;
				else posN_x += stepSign * 0.5;

				let posP_x = posN_x,
					posP_y = posN_y;
				const off_x = isHorz ? 1 : 0;
				const off_y = isHorz ? 0 : 1;

				let doneN = false,
					doneP = false;
				let lEndN = 0,
					lEndP = 0;
				for (let j = 0; j < iterations; j++) {
					if (!doneN) {
						lEndN = this._getLumaBilinear(luma, w, h, posN_x, posN_y);
						if (Math.abs(lEndN - Ledge) >= gradientScaled) doneN = true;
					}
					if (!doneP) {
						lEndP = this._getLumaBilinear(luma, w, h, posP_x, posP_y);
						if (Math.abs(lEndP - Ledge) >= gradientScaled) doneP = true;
					}
					if (doneN && doneP) break;
					if (!doneN) {
						posN_x -= off_x * qual[j];
						posN_y -= off_y * qual[j];
					}
					if (!doneP) {
						posP_x += off_x * qual[j];
						posP_y += off_y * qual[j];
					}
				}

				const distN = isHorz ? x - posN_x : y - posN_y;
				const distP = isHorz ? posP_x - x : posP_y - y;

				const isNDistSmaller = distN < distP;
				const distMin = Math.min(distN, distP);
				const lEndMin = isNDistSmaller ? lEndN : lEndP;

				const lDiff = L - Ledge;
				const isLPositive = lDiff >= 0;
				const isEndPositive = lEndMin - Ledge >= 0;
				const reachedProperly = isEndPositive !== isLPositive;

				let edgeOffset = -distMin / (distN + distP) + 0.5;
				if (!reachedProperly) edgeOffset = 0.0;

				const pixelOffset = Math.max(subpixOffset, edgeOffset);

				// Sample final color
				let finalX = x,
					finalY = y;
				if (isHorz) finalY += stepSign * pixelOffset;
				else finalX += stepSign * pixelOffset;

				this._sampleBilinear(pixels, w, h, finalX, finalY, outCol);
				output[idx] = outCol.r;
				output[idx + 1] = outCol.g;
				output[idx + 2] = outCol.b;
				output[idx + 3] = outCol.a;
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
		context: FrameContext,
		ctx: CanvasRenderingContext2D
	): void {
		const depthBuffer = context.attachments.depthBuffer;
		const options = context.features.volumetricOptions || {};
		const maxRayDistance = Math.max(
			VolumetricConstants.MIN_RAY_DISTANCE,
			this._toFiniteNumber(options.maxRayDistance, 500)
		);

		const { width: w, height: h } = context.attachments;
		const pixels = context.attachments.pixels;
		if (!pixels || !context.attachments.depthBuffer) return;

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

		let imageData: ImageData | null = null;

		const camera = context.camera;
		const cameraPos = camera.position;
		const basis = this._getCameraBasis(context);
		const near = camera.near || 0.1;
		const far = Math.min(camera.far || 1000, maxRayDistance);

		// Consolidate options with range protection
		const ds = Math.round(
			clamp(
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
			clamp(
				this._toFiniteNumber(
					options.samples,
					VolumetricConstants.DEFAULT_SAMPLES
				),
				VolumetricConstants.MIN_SAMPLES,
				VolumetricConstants.MAX_SAMPLES
			)
		);

		const weight = clamp(
			this._toFiniteNumber(options.weight, VolumetricConstants.DEFAULT_WEIGHT),
			0,
			VolumetricConstants.MAX_WEIGHT
		);
		const exposure = clamp(
			this._toFiniteNumber(options.exposure, 1.0),
			0,
			PostProcessConstants.MAX_EXPOSURE
		);
		const airDensity = clamp(
			this._toFiniteNumber(options.airDensity, 1.0),
			0,
			VolumetricConstants.MAX_AIR_DENSITY
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

		const sigmaT = airDensity * VolumetricConstants.SIGMA_T_SCALE;
		const sigmaS = sigmaT * scatteringAlbedo;

		const shadowsEnabled = this.renderer.features.enableShadows;
		const shadowInterval = Math.round(
			clamp(
				this._toFiniteNumber(options.shadowSampleInterval, 1),
				VolumetricConstants.MIN_SHADOW_SAMPLE_INTERVAL,
				VolumetricConstants.MAX_SHADOW_SAMPLE_INTERVAL
			)
		);

		const sceneBounds = context.scene.sceneBounds;
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

		// ... usage continues

		const camToCenter = Math.hypot(
			cameraPos.x - sceneCenter.x,
			cameraPos.y - sceneCenter.y,
			cameraPos.z - sceneCenter.z
		);
		const infinityDepthLimit = clamp(
			camToCenter +
				sceneRadius * VolumetricConstants.SCENE_DEPTH_LIMIT_MULTIPLIER,
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
					const px = Math.round(clamp(sampleXCenter + jitterX, 0, w - 1));
					const py = Math.round(clamp(sampleYCenter + jitterY, 0, h - 1));
					const ray = this._getWorldRayFromPixel(px, py, w, h, basis, context);

					const ndcX = ((px + 0.5) / w) * 2 - 1;
					const ndcY = 1 - ((py + 0.5) / h) * 2;
					const posView = this._reconstructViewPos(ndcX, ndcY, dist);

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
		const historyWeight = VolumetricConstants.TEMPORAL_ACCUMULATION_FACTOR;
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
					if (transmittance < VolumetricConstants.TRANSMITTANCE_EARLY_EXIT)
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
					const posView = this._reconstructViewPos(ndcX, ndcY, depthLimit);

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

	public applySSAO(context: FrameContext): void {
		const depthBuffer = context.attachments.depthBuffer;
		const normalBuffer = context.attachments.normalBuffer;
		const options = context.features.ssaoOptions || {};
		if (!depthBuffer || !normalBuffer) return;

		const w = context.attachments.width;
		const h = context.attachments.height;
		const radius = options.radius ?? SSAOConstants.DEFAULT_RADIUS;
		const bias = options.bias ?? SSAOConstants.DEFAULT_BIAS;
		const intensity = options.intensity ?? SSAOConstants.DEFAULT_INTENSITY;

		if (!this._ssaoBuffer || this._ssaoBuffer.length !== w * h) {
			this._ssaoBuffer = new Float32Array(w * h);
		}
		const ssaoBuffer = this._ssaoBuffer;

		const camera = context.camera;
		const projection = camera.projectionMatrix.elements;
		const near = camera.near;
		const far = camera.far;

		// 1. SSAO calculation
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const idx = y * w + x;
				const originDepth = depthBuffer[idx];
				if (originDepth === Infinity || originDepth <= 0) {
					ssaoBuffer[idx] = 1.0;
					continue;
				}

				// Reconstruct view-space position
				const ndcX = (x / w) * 2 - 1;
				const ndcY = 1 - (y / h) * 2;
				const posView = this._reconstructViewPos(ndcX, ndcY, originDepth);

				const nIdx = idx * 3;
				const normal = {
					x: normalBuffer[nIdx],
					y: normalBuffer[nIdx + 1],
					z: normalBuffer[nIdx + 2],
				};

				// Random rotation from noise
				const noiseIdx =
					(y % SSAOConstants.NOISE_SIZE) * SSAOConstants.NOISE_SIZE +
					(x % SSAOConstants.NOISE_SIZE);
				const randomVec = this._ssaoNoise[noiseIdx];

				// Gram-Schmidt process to create tangent basis
				const tangent = {
					x: randomVec.x - normal.x * Vector3.dot(randomVec, normal),
					y: randomVec.y - normal.y * Vector3.dot(randomVec, normal),
					z: randomVec.z - normal.z * Vector3.dot(randomVec, normal),
				};
				const tangentLen = Math.hypot(tangent.x, tangent.y, tangent.z) || 1;
				tangent.x /= tangentLen;
				tangent.y /= tangentLen;
				tangent.z /= tangentLen;

				const bitangent = Vector3.cross(normal, tangent);
				const TBN = [
					[tangent.x, bitangent.x, normal.x],
					[tangent.y, bitangent.y, normal.y],
					[tangent.z, bitangent.z, normal.z],
				];

				let occlusion = 0;
				for (let i = 0; i < SSAOConstants.DEFAULT_SAMPLES; i++) {
					const sample = this._ssaoKernel[i];
					// World to view space sample
					const sampleViewOffset = {
						x:
							TBN[0][0] * sample.x +
							TBN[0][1] * sample.y +
							TBN[0][2] * sample.z,
						y:
							TBN[1][0] * sample.x +
							TBN[1][1] * sample.y +
							TBN[1][2] * sample.z,
						z:
							TBN[2][0] * sample.x +
							TBN[2][1] * sample.y +
							TBN[2][2] * sample.z,
					};

					const samplePos = {
						x: posView.x + sampleViewOffset.x * radius,
						y: posView.y + sampleViewOffset.y * radius,
						z: posView.z + sampleViewOffset.z * radius,
					};

					// Project sample position to screen space
					let screenX: number, screenY: number;
					if (camera.type === CameraType.Orthographic) {
						const ndcX = projection[0][0] * samplePos.x + projection[0][3];
						const ndcY = projection[1][1] * samplePos.y + projection[1][3];
						screenX = Math.round((ndcX * 0.5 + 0.5) * w - 0.5);
						screenY = Math.round((0.5 - ndcY * 0.5) * h - 0.5);
					} else {
						const offsetNDC = {
							x:
								(projection[0][0] * samplePos.x +
									projection[0][2] * samplePos.z) /
								-samplePos.z,
							y:
								(projection[1][1] * samplePos.y +
									projection[1][2] * samplePos.z) /
								-samplePos.z,
						};
						screenX = Math.round((offsetNDC.x * 0.5 + 0.5) * w - 0.5);
						screenY = Math.round((0.5 - offsetNDC.y * 0.5) * h - 0.5);
					}

					if (screenX >= 0 && screenX < w && screenY >= 0 && screenY < h) {
						const sampleDepth = depthBuffer[screenY * w + screenX];
						const samplePosDepth = -samplePos.z;
						const rangeCheck =
							Math.abs(originDepth - sampleDepth) < radius ? 1.0 : 0.0;
						occlusion +=
							(sampleDepth <= samplePosDepth - bias ? 1.0 : 0.0) * rangeCheck;
					}
				}

				occlusion =
					1.0 - (occlusion / SSAOConstants.DEFAULT_SAMPLES) * intensity;
				ssaoBuffer[idx] = occlusion;
			}
		}

		// 2. Blur SSAO to reduce noise
		this._ssaoBlur(ssaoBuffer, w, h);

		// 3. Apply to pixels
		const pixels = context.attachments.pixels;
		for (let i = 0, len = w * h; i < len; i++) {
			const factor = ssaoBuffer[i];
			const idx = i << 2;
			pixels[idx] *= factor;
			pixels[idx + 1] *= factor;
			pixels[idx + 2] *= factor;
		}
	}

	private _reconstructViewPos(
		ndcX: number,
		ndcY: number,
		zView: number
	): IVector3 {
		const camera = this.renderer.camera;

		if (camera.type === CameraType.Orthographic) {
			const orthoCam = camera as OrthographicCamera;
			const halfHeight = orthoCam.size / 2;
			const halfWidth = halfHeight * camera.aspectRatio;

			const xView = ndcX * halfWidth;
			const yView = ndcY * halfHeight;

			return { x: xView, y: yView, z: -zView };
		}

		const fovRad = (camera.fov * Math.PI) / 180;
		const tanHalfFov = Math.tan(fovRad * 0.5);
		const aspect = camera.aspectRatio;

		const xView = ndcX * aspect * tanHalfFov * zView;
		const yView = ndcY * tanHalfFov * zView;

		return { x: xView, y: yView, z: -zView };
	}

	private _ssaoBlur(buffer: Float32Array, w: number, h: number): void {
		this._ssaoBlurTemp = this._ensureFloat32Buffer(
			this._ssaoBlurTemp,
			buffer.length
		);
		const temp = this._ssaoBlurTemp;
		temp.set(buffer);

		const blurSize = 2;
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				let sum = 0;
				let count = 0;
				for (let dy = -blurSize; dy <= blurSize; dy++) {
					for (let dx = -blurSize; dx <= blurSize; dx++) {
						const nx = x + dx;
						const ny = y + dy;
						if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
							sum += temp[ny * w + nx];
							count++;
						}
					}
				}
				buffer[y * w + x] = sum / count;
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

	private _buildSRGBLUT(gamma: number): void {
		if (this._lutBuilt && this._lastGamma === gamma) return;

		const isStandardSRGB = Math.abs(gamma - 2.2) < 0.001;
		const invGamma = 1.0 / gamma;

		for (let i = 0; i < 256; i++) {
			const x = i / 255.0;
			if (isStandardSRGB) {
				this._sRGBLUT[i] = Math.round(linearToSRGB(x) * 255.0);
			} else {
				this._sRGBLUT[i] = Math.round(Math.pow(x, invGamma) * 255.0);
			}
		}
		this._lutBuilt = true;
		this._lastGamma = gamma;
	}

	public applyGamma(
		context: FrameContext,
		ctx: CanvasRenderingContext2D
	): void {
		const w = context.attachments.width,
			h = context.attachments.height;
		const gamma = context.features.enableGamma
			? PostProcessConstants.DEFAULT_GAMMA
			: 1.0; // Simplification, usually from features
		let pixels = context.attachments.pixels;
		let imageData: ImageData | null = null;

		this._buildSRGBLUT(gamma);
		const lut = this._sRGBLUT;
		for (let i = 0; i < pixels.length; i += 4) {
			pixels[i] = lut[pixels[i]];
			pixels[i + 1] = lut[pixels[i + 1]];
			pixels[i + 2] = lut[pixels[i + 2]];
		}
		if (imageData) ctx.putImageData(imageData, 0, 0);
	}
}
