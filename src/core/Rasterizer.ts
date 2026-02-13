import { Material } from "../materials/Material";
import { CoreConstants, PostProcessConstants } from "./Constants";
import {
	FlatLitShader,
	LitShader,
	PBRStrategy,
	BlinnPhongStrategy,
	PhongEvaluator,
	PBREvaluator,
	UnlitShader,
	type IShader,
	type ShaderContext,
	type IMaterialEvaluator,
	type FragmentInput,
	type PhongSurfaceProperties,
	type PBRSurfaceProperties,
} from "../shaders";
import type { ShadowMap } from "../utils/ShadowMapping";
import type { Renderer } from "./Renderer";
import type { ProjectedVertex, ProjectedFace } from "./types";

export interface RasterizerLike {
	drawTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		isTransparent?: boolean,
		overrideSize?: { width: number; height: number }
	): void;
	drawDepthTriangle(pts: ProjectedVertex[], shadowMap: ShadowMap): void;
	drawTransmissionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		shadowMap: ShadowMap
	): void;
}

interface CachedVertex {
	x: number;
	y: number;
	z: number;
	iz: number;
	worldOx: number;
	worldOy: number;
	worldOz: number;
	normalOx: number;
	normalOy: number;
	normalOz: number;
	uO: number;
	vO: number;
}

interface EdgeInterpolationResult {
	x: number;
	iz: number;
	worldOx: number;
	worldOy: number;
	worldOz: number;
	normalOx: number;
	normalOy: number;
	normalOz: number;
	uO: number;
	vO: number;
}

/**
 * Rasterizer handles the scanline conversion of projected triangles to pixels.
 *
 * CORE CONVENTIONS:
 * - Depth Buffer: Stores linear camera-space depth (z-distance) for standard Z-buffering.
 * - Perspective Correction: Attributes are multiplied by 1/w before interpolation and recovered per-pixel.
 * - Shading: Supports Flat, Gouraud, Phong, and PBR shading models.
 */
export class Rasterizer implements RasterizerLike {
	private _renderer: Renderer;
	private _vertsCache: CachedVertex[];
	private _defaultMaterial: Material;

	// Shader & Evaluator cache
	private _phongEvaluator: PhongEvaluator;
	private _pbrEvaluator: PBREvaluator;
	private _shaderCache: Map<string, IShader> = new Map();

	// Pre-allocated objects for zero-allocation rendering
	private _edgeRes1: EdgeInterpolationResult = this._createEdgeRes();
	private _edgeRes2: EdgeInterpolationResult = this._createEdgeRes();
	private _fragmentInput: FragmentInput = {
		zCam: 0,
		worldX: 0,
		worldY: 0,
		worldZ: 0,
		normalX: 0,
		normalY: 0,
		normalZ: 0,
		u: 0,
		v: 0,
	};

	constructor(renderer: Renderer) {
		this._renderer = renderer;
		this._defaultMaterial = new Material();
		this._vertsCache = Array.from({ length: 3 }, () => ({
			x: 0,
			y: 0,
			z: 0,
			iz: 0,
			worldOx: 0,
			worldOy: 0,
			worldOz: 0,
			normalOx: 0,
			normalOy: 0,
			normalOz: 0,
			uO: 0,
			vO: 0,
		}));

		this._phongEvaluator = new PhongEvaluator(this._defaultMaterial);
		this._pbrEvaluator = new PBREvaluator(this._defaultMaterial);
	}

	private _createEdgeRes(): EdgeInterpolationResult {
		return {
			x: 0,
			iz: 0,
			worldOx: 0,
			worldOy: 0,
			worldOz: 0,
			normalOx: 0,
			normalOy: 0,
			normalOz: 0,
			uO: 0,
			vO: 0,
		};
	}

	private _getShader(shading: string, material: Material): IShader {
		const isPBR = shading === "PBR" || material.type === "PBR";
		const evaluator = isPBR ? this._pbrEvaluator : this._phongEvaluator;
		evaluator.setMaterial(material);

		const key = `${shading}_${isPBR ? "PBR" : "Phong"}`;
		let shader = this._shaderCache.get(key);

		if (!shader) {
			if (isPBR) {
				if (shading === "Unlit") {
					shader = new UnlitShader(evaluator);
				} else {
					shader = new LitShader(
						new PBRStrategy(),
						evaluator as IMaterialEvaluator<PBRSurfaceProperties>
					);
				}
			} else {
				const strategy = new BlinnPhongStrategy();
				if (shading === "Unlit") {
					shader = new UnlitShader(evaluator);
				} else if (shading === "Flat") {
					shader = new FlatLitShader(
						strategy,
						evaluator as IMaterialEvaluator<PhongSurfaceProperties>
					);
				} else {
					shader = new LitShader(
						strategy,
						evaluator as IMaterialEvaluator<PhongSurfaceProperties>
					);
				}
			}
			this._shaderCache.set(key, shader!);
		} else {
			shader.setEvaluator(evaluator);
		}

		return shader!;
	}

	public drawDepthTriangle(pts: ProjectedVertex[], shadowMap: ShadowMap): void {
		const { size, buffer } = shadowMap;

		let [vTop, vMid, vBot] = pts;
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(0, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(size - 1, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			let leftX, leftZ, rightX, rightZ;

			if (py < vMid.y) {
				const dy1 = vMid.y - vTop.y;
				const t1 = dy1 === 0 ? 0 : (py - vTop.y) / dy1;
				leftX = vTop.x + (vMid.x - vTop.x) * t1;
				leftZ = vTop.z + (vMid.z - vTop.z) * t1;

				const dy2 = vBot.y - vTop.y;
				const t2 = dy2 === 0 ? 0 : (py - vTop.y) / dy2;
				rightX = vTop.x + (vBot.x - vTop.x) * t2;
				rightZ = vTop.z + (vBot.z - vTop.z) * t2;
			} else {
				const dy1 = vBot.y - vMid.y;
				const t1 = dy1 === 0 ? 0 : (py - vMid.y) / dy1;
				leftX = vMid.x + (vBot.x - vMid.x) * t1;
				leftZ = vMid.z + (vBot.z - vMid.z) * t1;

				const dy2 = vBot.y - vTop.y;
				const t2 = dy2 === 0 ? 0 : (py - vTop.y) / dy2;
				rightX = vTop.x + (vBot.x - vTop.x) * t2;
				rightZ = vTop.z + (vBot.z - vTop.z) * t2;
			}

			if (leftX > rightX) {
				[leftX, rightX] = [rightX, leftX];
				[leftZ, rightZ] = [rightZ, leftZ];
			}

			const startX = Math.max(0, Math.ceil(leftX - 0.5));
			const endX = Math.min(size - 1, Math.floor(rightX - 0.5));
			if (endX < startX) continue;

			const spanWidth = rightX - leftX;
			const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);
			const dz = (rightZ - leftZ) * spanInv;
			const dx = startX + 0.5 - leftX;
			let z = leftZ + dx * dz;

			const row = y * size;
			for (let x = startX; x <= endX; x++) {
				const idx = row + x;
				if (z < buffer[idx]) {
					buffer[idx] = z;
				}
				z += dz;
			}
		}
	}

	public drawTransmissionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		shadowMap: ShadowMap
	): void {
		const { size, buffer, transmissionBuffer } = shadowMap;
		const material = face.material;
		if (!material || !transmissionBuffer) return;

		// Extract material color and opacity
		let r = 1,
			g = 1,
			b = 1;
		const opacity = material.opacity ?? 1;

		if (material.type === "PBR") {
			const pbr = material as any;
			r = pbr.albedo.r / 255;
			g = pbr.albedo.g / 255;
			b = pbr.albedo.b / 255;
		} else if (material.type === "Phong") {
			const phong = material as any;
			r = phong.diffuse.r / 255;
			g = phong.diffuse.g / 255;
			b = phong.diffuse.b / 255;
		}

		// Calculate transmission multiplier: Color * Opacity + White * (1 - Opacity)
		// This means: more opaque -> more color; more transparent -> more white passes
		const transR = r * opacity + (1 - opacity);
		const transG = g * opacity + (1 - opacity);
		const transB = b * opacity + (1 - opacity);

		let [vTop, vMid, vBot] = pts;
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(0, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(size - 1, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			let leftX, leftZ, rightX, rightZ;

			if (py < vMid.y) {
				const dy1 = vMid.y - vTop.y;
				const t1 = dy1 === 0 ? 0 : (py - vTop.y) / dy1;
				leftX = vTop.x + (vMid.x - vTop.x) * t1;
				leftZ = vTop.z + (vMid.z - vTop.z) * t1;

				const dy2 = vBot.y - vTop.y;
				const t2 = dy2 === 0 ? 0 : (py - vTop.y) / dy2;
				rightX = vTop.x + (vBot.x - vTop.x) * t2;
				rightZ = vTop.z + (vBot.z - vTop.z) * t2;
			} else {
				const dy1 = vBot.y - vMid.y;
				const t1 = dy1 === 0 ? 0 : (py - vMid.y) / dy1;
				leftX = vMid.x + (vBot.x - vMid.x) * t1;
				leftZ = vMid.z + (vBot.z - vMid.z) * t1;

				const dy2 = vBot.y - vTop.y;
				const t2 = dy2 === 0 ? 0 : (py - vTop.y) / dy2;
				rightX = vTop.x + (vBot.x - vTop.x) * t2;
				rightZ = vTop.z + (vBot.z - vTop.z) * t2;
			}

			if (leftX > rightX) {
				[leftX, rightX] = [rightX, leftX];
				[leftZ, rightZ] = [rightZ, leftZ];
			}

			const startX = Math.max(0, Math.ceil(leftX - 0.5));
			const endX = Math.min(size - 1, Math.floor(rightX - 0.5));
			if (endX < startX) continue;

			const spanWidth = rightX - leftX;
			const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);
			const dz = (rightZ - leftZ) * spanInv;
			const dx = startX + 0.5 - leftX;
			let z = leftZ + dx * dz;

			const row = y * size;
			for (let x = startX; x <= endX; x++) {
				const idx = row + x;
				// IMPORTANT: Transparent objects Only attenuate light if they are IN FRONT of the opaque depth
				// and they are in front of the light (z > 0 in light space)
				if (z < buffer[idx]) {
					const cIdx = idx * 3;
					transmissionBuffer[cIdx] *= transR;
					transmissionBuffer[cIdx + 1] *= transG;
					transmissionBuffer[cIdx + 2] *= transB;
				}
				z += dz;
			}
		}
	}

	private _fillEdgeRes(
		res: EdgeInterpolationResult,
		vA: CachedVertex,
		vB: CachedVertex,
		y: number
	): void {
		const dy = vB.y - vA.y;
		const t = dy === 0 ? 0 : (y - vA.y) / dy;
		res.x = vA.x + (vB.x - vA.x) * t;
		res.iz = vA.iz + (vB.iz - vA.iz) * t;
		res.worldOx = vA.worldOx + (vB.worldOx - vA.worldOx) * t;
		res.worldOy = vA.worldOy + (vB.worldOy - vA.worldOy) * t;
		res.worldOz = vA.worldOz + (vB.worldOz - vA.worldOz) * t;
		res.normalOx = vA.normalOx + (vB.normalOx - vA.normalOx) * t;
		res.normalOy = vA.normalOy + (vB.normalOy - vA.normalOy) * t;
		res.normalOz = vA.normalOz + (vB.normalOz - vA.normalOz) * t;
		res.uO = vA.uO + (vB.uO - vA.uO) * t;
		res.vO = vA.vO + (vB.vO - vA.vO) * t;
	}

	public drawTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		isTransparent: boolean = false,
		overrideSize?: { width: number; height: number }
	): void {
		const width = overrideSize?.width ?? this._renderer.canvas.width;
		const height = overrideSize?.height ?? this._renderer.canvas.height;
		const depthBuffer = this._renderer.depthBuffer;
		const material = face.material ?? this._defaultMaterial;

		if (!depthBuffer) return;

		const verts = this._vertsCache;
		const shadingModel = material.shading || "Flat";
		const isLightingEnabled = this._renderer.params.enableLighting !== false;
		const shading = isLightingEnabled ? shadingModel : "Unlit";

		const shader = this._getShader(shading, material);
		const shaderContext: ShaderContext = {
			renderer: this._renderer,
			cameraPos: this._renderer.camera.position,
			lights: this._renderer.scene.lights,
			worldMatrix: this._renderer.params.worldMatrix,
			shAmbientCoeffs: this._renderer.shAmbientCoeffs,
			enableShadows: !!this._renderer.params.enableShadows,
			enableSH: !!this._renderer.params.enableSH,
			enableLighting: isLightingEnabled,
			gamma: PostProcessConstants.DEFAULT_GAMMA,
		};
		shader.initialize(face, shaderContext);

		const camPos = this._renderer.camera.position;
		let isCameraOnFrontSide = true;
		if (material.mirrorPlane) {
			const p = material.mirrorPlane;
			const dist =
				camPos.x * p.normal.x +
				camPos.y * p.normal.y +
				camPos.z * p.normal.z +
				p.constant;
			isCameraOnFrontSide = dist > 0;
		}

		for (let i = 0; i < 3; i++) {
			const p = pts[i];
			const world = p.world ?? { x: 0, y: 0, z: 0 };
			const normal = p.normal ?? face.normal ?? { x: 0, y: 0, z: 1 };
			const iz = p.w;

			const v = verts[i];
			v.x = p.x;
			v.y = p.y;
			v.z = p.z;
			v.iz = iz;
			v.worldOx = world.x * iz;
			v.worldOy = world.y * iz;
			v.worldOz = world.z * iz;
			v.normalOx = normal.x * iz;
			v.normalOy = normal.y * iz;
			v.normalOz = normal.z * iz;
			v.uO = (p.u ?? 0) * iz;
			v.vO = (p.v ?? 0) * iz;
		}

		let [vTop, vMid, vBot] = [verts[0], verts[1], verts[2]];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(0, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(height - 1, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			let left = this._edgeRes1;
			let right = this._edgeRes2;

			if (py < vMid.y) {
				this._fillEdgeRes(left, vTop, vMid, py);
				this._fillEdgeRes(right, vTop, vBot, py);
			} else {
				this._fillEdgeRes(left, vMid, vBot, py);
				this._fillEdgeRes(right, vTop, vBot, py);
			}

			if (left.x > right.x) {
				const tmp = left;
				left = right;
				right = tmp;
			}

			const startX = Math.max(0, Math.ceil(left.x - 0.5));
			const endX = Math.min(width - 1, Math.floor(right.x - 0.5));
			if (endX < startX) continue;

			const spanWidth = right.x - left.x;
			const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);

			const diz = (right.iz - left.iz) * spanInv;
			const dWorldOx = (right.worldOx - left.worldOx) * spanInv;
			const dWorldOy = (right.worldOy - left.worldOy) * spanInv;
			const dWorldOz = (right.worldOz - left.worldOz) * spanInv;
			const dNormalOx = (right.normalOx - left.normalOx) * spanInv;
			const dNormalOy = (right.normalOy - left.normalOy) * spanInv;
			const dNormalOz = (right.normalOz - left.normalOz) * spanInv;
			const duO = (right.uO - left.uO) * spanInv;
			const dvO = (right.vO - left.vO) * spanInv;

			const dx = startX + 0.5 - left.x;
			let iz = left.iz + dx * diz;
			let worldOx = left.worldOx + dx * dWorldOx;
			let worldOy = left.worldOy + dx * dWorldOy;
			let worldOz = left.worldOz + dx * dWorldOz;
			let normalOx = left.normalOx + dx * dNormalOx;
			let normalOy = left.normalOy + dx * dNormalOy;
			let normalOz = left.normalOz + dx * dNormalOz;
			let uO = left.uO + dx * duO;
			let vO = left.vO + dx * dvO;

			const bufRow = y * width;
			const input = this._fragmentInput;

			for (let x = startX; x <= endX; x++) {
				const bufIdx = bufRow + x;
				const safeIz =
					Math.abs(iz) > CoreConstants.EPSILON ? iz
					: iz >= 0 ? CoreConstants.EPSILON
					: -CoreConstants.EPSILON;
				const zCam = 1 / safeIz;

				if (zCam > 0 && zCam < depthBuffer[bufIdx]) {
					input.zCam = zCam;
					input.worldX = worldOx * zCam;
					input.worldY = worldOy * zCam;
					input.worldZ = worldOz * zCam;
					input.normalX = normalOx * zCam;
					input.normalY = normalOy * zCam;
					input.normalZ = normalOz * zCam;
					input.u = uO * zCam;
					input.v = vO * zCam;

					let finalColor = shader.shade(input);

					if (
						finalColor &&
						material.reflectivity > 0 &&
						material.mirrorPlane &&
						isCameraOnFrontSide
					) {
						const p = material.mirrorPlane;
						const key = `${p.normal.x},${p.normal.y},${p.normal.z},${p.constant}`;
						const refBuffer =
							this._renderer.reflectionRenderer.reflectionBuffers.get(key);
						if (refBuffer) {
							// 1. Distortion (Ripples)
							let offsetX = 0;
							let offsetY = 0;
							if (material.distortion > 0) {
								const time = (this._renderer.lastTime || 0) * 0.002;
								const freq = 0.5;
								const dist = material.distortion * 5;
								offsetX =
									Math.sin(input.worldX * freq + time) *
									Math.cos(input.worldZ * freq + time) *
									dist;
								offsetY =
									Math.cos(input.worldX * freq + time) *
									Math.sin(input.worldZ * freq + time) *
									dist;
							}

							// Sample from reflection buffer with coordinate scaling
							let refX = Math.floor((x + offsetX) * (refBuffer.width / width));
							let refY = Math.floor(
								(y + offsetY) * (refBuffer.height / height)
							);

							// Clamp to buffer bounds
							refX = Math.max(0, Math.min(refBuffer.width - 1, refX));
							refY = Math.max(0, Math.min(refBuffer.height - 1, refY));

							const refIdx = (refY * refBuffer.width + refX) << 2;
							const refData = refBuffer.imageData.data;

							// 2. Fresnel effect
							let reflectivity = Math.max(
								0,
								Math.min(1, material.reflectivity)
							);

							if (material.fresnel) {
								// View vector V = normalize(cameraPos - worldPos)
								const vx = camPos.x - input.worldX;
								const vy = camPos.y - input.worldY;
								const vz = camPos.z - input.worldZ;
								const vLen =
									Math.sqrt(vx * vx + vy * vy + vz * vz) ||
									CoreConstants.EPSILON;
								const nx = input.normalX;
								const ny = input.normalY;
								const nz = input.normalZ;
								const nLen =
									Math.sqrt(nx * nx + ny * ny + nz * nz) ||
									CoreConstants.EPSILON;

								// Dot product (N dot V)
								const dot = Math.abs(
									(vx * nx + vy * ny + vz * nz) / (vLen * nLen)
								);
								// Fresnel Schlick: R = R0 + (1-R0)(1-cos)^5
								// We simplify: f = (1 - dot)^3
								const fresnelFactor = Math.pow(1.0 - dot, 3);
								reflectivity *= 0.1 + 0.9 * fresnelFactor;
							}

							const invRef = 1 - reflectivity;
							finalColor = {
								r: finalColor.r * invRef + refData[refIdx] * reflectivity,
								g: finalColor.g * invRef + refData[refIdx + 1] * reflectivity,
								b: finalColor.b * invRef + refData[refIdx + 2] * reflectivity,
							};
						}
					}

					if (finalColor) {
						const idx = bufIdx << 2;
						if (!isTransparent) {
							pixels[idx] = finalColor.r;
							pixels[idx + 1] = finalColor.g;
							pixels[idx + 2] = finalColor.b;
							pixels[idx + 3] = CoreConstants.OPAQUE_ALPHA;
							depthBuffer[bufIdx] = zCam;
						} else {
							const faceAlpha = face.color?.a ?? 1;
							const shaderAlpha = shader.getOpacity();
							const alpha = Math.max(0, Math.min(1, faceAlpha * shaderAlpha));
							const invA = 1 - alpha;
							pixels[idx] = finalColor.r * alpha + pixels[idx] * invA;
							pixels[idx + 1] = finalColor.g * alpha + pixels[idx + 1] * invA;
							pixels[idx + 2] = finalColor.b * alpha + pixels[idx + 2] * invA;
							pixels[idx + 3] = CoreConstants.OPAQUE_ALPHA;
						}
					}
				}

				iz += diz;
				worldOx += dWorldOx;
				worldOy += dWorldOy;
				worldOz += dWorldOz;
				normalOx += dNormalOx;
				normalOy += dNormalOy;
				normalOz += dNormalOz;
				uO += duO;
				vO += dvO;
			}
		}

		if (material.wireframe) {
			this._drawWireframe(pts, face, pixels, isTransparent, overrideSize);
		}
	}

	private _drawWireframe(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		isTransparent: boolean = false,
		overrideSize?: { width: number; height: number }
	): void {
		const width = overrideSize?.width ?? this._renderer.canvas.width;
		const height = overrideSize?.height ?? this._renderer.canvas.height;
		const depthBuffer = this._renderer.depthBuffer;
		const material = face.material ?? this._defaultMaterial;

		if (!depthBuffer) return;

		const wireColor = { r: 255, g: 255, b: 255 };
		const alpha =
			isTransparent ?
				Math.max(0, Math.min(1, face.color?.a ?? material.opacity ?? 1))
			:	1;

		const drawLine = (p0: ProjectedVertex, p1: ProjectedVertex) => {
			const x0 = p0.x,
				y0 = p0.y,
				iz0 = p0.w;
			const x1 = p1.x,
				y1 = p1.y,
				iz1 = p1.w;

			const dx = Math.abs(x1 - x0);
			const dy = Math.abs(y1 - y0);
			const steps = Math.max(dx, dy);
			if (steps === 0) return;

			const xInc = (x1 - x0) / steps;
			const yInc = (y1 - y0) / steps;
			const izInc = (iz1 - iz0) / steps;

			let x = x0,
				y = y0,
				iz = iz0;

			for (let i = 0; i <= steps; i++) {
				const px = Math.floor(x);
				const py = Math.floor(y);

				if (px >= 0 && px < width && py >= 0 && py < height) {
					const bufIdx = py * width + px;
					const safeIz =
						Math.abs(iz) > CoreConstants.EPSILON ? iz
						: iz >= 0 ? CoreConstants.EPSILON
						: -CoreConstants.EPSILON;
					const zCam = 1 / safeIz;

					if (
						zCam > 0 &&
						zCam < depthBuffer[bufIdx] + CoreConstants.WIREFRAME_DEPTH_BIAS
					) {
						const idx = bufIdx << 2;
						pixels[idx] = wireColor.r;
						pixels[idx + 1] = wireColor.g;
						pixels[idx + 2] = wireColor.b;
						pixels[idx + 3] = alpha * CoreConstants.MAX_CHANNEL_VALUE;
					}
				}
				x += xInc;
				y += yInc;
				iz += izInc;
			}
		};

		for (let i = 0; i < pts.length; i++) {
			drawLine(pts[i], pts[(i + 1) % pts.length]);
		}
	}
}
