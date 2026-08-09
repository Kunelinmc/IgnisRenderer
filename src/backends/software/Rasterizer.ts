import {
	AlphaMode,
	Material,
} from "../../materials/Material";
import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import { CoreConstants } from "./constants";
import {
	type ShaderContext,
	type FragmentInput,
} from "../../shaders";
import { hasSoftwareTextureData } from "../../shaders/software/textureSampling";
import { clamp } from "../../maths/Common";
import type { SceneLight } from "../../lights";
import type { ProjectedVertex, ProjectedFace } from "../../core/types";
import {
	type IVector3,
	type SHCoefficients,
} from "../../maths/types";
import type { Texture } from "../../core/Texture";
import type { SoftwareShadowRenderTarget } from "./SoftwareShadowContracts";
import type { TemporalJitterFrameState } from "../cross/TemporalJitterState";
import { SoftwareTriangleInterpolator } from "./Interpolator";
import {
	SoftwareMaterialRuntime,
	type SoftwareFragmentProgram,
} from "./SoftwareMaterialRuntime";
import type { DecalPacket } from "../../pipeline/types";

export interface RasterizerLike {
	prepareFragmentProgram(
		face: ProjectedFace,
		context: RasterizerContext,
		isTransparent?: boolean,
		decalPackets?: readonly DecalPacket[]
	): SoftwareFragmentProgram;
	drawTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		context: RasterizerContext,
		program: SoftwareFragmentProgram,
		isTransparent?: boolean
	): void;
	drawCameraDepthTriangle(
		pts: ProjectedVertex[],
		context: RasterizerContext
	): void;
	drawDepthTriangle(
		pts: ProjectedVertex[],
		shadowTarget: SoftwareShadowRenderTarget,
		material?: Material
	): void;
	drawTransmissionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		shadowTarget: SoftwareShadowRenderTarget
	): void;
}

/**
 * Rasterizer handles the scanline conversion of projected triangles to pixels.
 *
 * CORE CONVENTIONS:
 * - Depth Buffer: Stores linear camera-space depth (z-distance) for standard Z-buffering.
 * - Perspective Correction: Attributes are multiplied by 1/w before interpolation and recovered per-pixel.
 * - Shading: Supports Flat, Gouraud, Phong, and PBR shading models.
 */
export interface RasterizerContext {
	width: number;
	height: number;
	depthBuffer: Float32Array;
	earlyDepthBuffer?: Float32Array | null;
	clipRect?: {
		minX: number;
		minY: number;
		maxX: number;
		maxY: number;
	} | null;
	normalBuffer?: Float32Array | null;
	motionBuffer?: Float32Array | null;
	temporal?: TemporalJitterFrameState & {
		previousViewProjection: Matrix4 | null;
	};
	camera: {
		position: IVector3;
		viewMatrix: Matrix4;
	};
	lights: SceneLight[];
	sampleShadow?: ShaderContext["sampleShadow"];
	shAmbientCoeffs: SHCoefficients | null;
	environmentSpecularTexture?: Texture | null;
	enableLighting: boolean;
	enableSH: boolean;
	enableShadows: boolean;
}

/**
 * Rasterizer handles the scanline conversion of projected triangles to pixels.
 */
export class Rasterizer implements RasterizerLike {
	private _interpolator: SoftwareTriangleInterpolator =
		new SoftwareTriangleInterpolator();
	private _materialRuntime: SoftwareMaterialRuntime =
		new SoftwareMaterialRuntime();

	/** @internal Material evaluator owned by the Software raster service. */
	public get materialRuntime(): SoftwareMaterialRuntime {
		return this._materialRuntime;
	}

	// Pre-allocated objects for zero-allocation rendering
	private _fragmentInput: FragmentInput = {
		zCam: 0,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 0 },
		tangent: { x: 0, y: 0, z: 0, w: 0 },
		u: 0,
		v: 0,
		u2: 0,
		v2: 0,
		u3: 0,
		v3: 0,
		u4: 0,
		v4: 0,
	};

	/**
	 * @internal
	 * Owned by the Software rasterization subsystem; backend orchestration should
	 * use `SoftwareMainPass`. Prepares the material program for an immediately
	 * following triangle draw. Software passes must not retain the returned program
	 * across another prepare call because shaders and surface modifiers are reusable
	 * mutable state.
	 */
	public prepareFragmentProgram(
		face: ProjectedFace,
		context: RasterizerContext,
		isTransparent: boolean = false,
		decalPackets?: readonly DecalPacket[]
	): SoftwareFragmentProgram {
		return this._materialRuntime.prepareFragmentProgram(
			face,
			context,
			isTransparent,
			decalPackets
		);
	}

	public drawDepthTriangle(
		pts: ProjectedVertex[],
		shadowTarget: SoftwareShadowRenderTarget,
		material?: Material
	): void {
		const size = shadowTarget.size;
		const buffer = shadowTarget.depthBuffer;
		const alphaMode = material?.alphaMode;
		const maskTexture =
			(
				alphaMode === AlphaMode.Mask &&
				material?.map &&
				hasSoftwareTextureData(material.map) &&
				material.map.width > 0 &&
				material.map.height > 0
			) ?
				material.map
		:	null;
		const useMask = maskTexture !== null;
		const alphaCutoff = material?.alphaCutoff ?? 0.5;

		let [vTop, vMid, vBot] = pts;
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(0, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(size - 1, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		if (!useMask) {
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
			return;
		}

		const pTop = {
			x: vTop.x,
			y: vTop.y,
			z: vTop.z,
			iz: vTop.w,
			uO: (vTop.u ?? 0) * vTop.w,
			vO: (vTop.v ?? 0) * vTop.w,
		};
		const pMid = {
			x: vMid.x,
			y: vMid.y,
			z: vMid.z,
			iz: vMid.w,
			uO: (vMid.u ?? 0) * vMid.w,
			vO: (vMid.v ?? 0) * vMid.w,
		};
		const pBot = {
			x: vBot.x,
			y: vBot.y,
			z: vBot.z,
			iz: vBot.w,
			uO: (vBot.u ?? 0) * vBot.w,
			vO: (vBot.v ?? 0) * vBot.w,
		};

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			let leftX, leftZ, leftIz, leftUO, leftVO;
			let rightX, rightZ, rightIz, rightUO, rightVO;

			if (py < pMid.y) {
				const dy1 = pMid.y - pTop.y;
				const t1 = dy1 === 0 ? 0 : (py - pTop.y) / dy1;
				leftX = pTop.x + (pMid.x - pTop.x) * t1;
				leftZ = pTop.z + (pMid.z - pTop.z) * t1;
				leftIz = pTop.iz + (pMid.iz - pTop.iz) * t1;
				leftUO = pTop.uO + (pMid.uO - pTop.uO) * t1;
				leftVO = pTop.vO + (pMid.vO - pTop.vO) * t1;

				const dy2 = pBot.y - pTop.y;
				const t2 = dy2 === 0 ? 0 : (py - pTop.y) / dy2;
				rightX = pTop.x + (pBot.x - pTop.x) * t2;
				rightZ = pTop.z + (pBot.z - pTop.z) * t2;
				rightIz = pTop.iz + (pBot.iz - pTop.iz) * t2;
				rightUO = pTop.uO + (pBot.uO - pTop.uO) * t2;
				rightVO = pTop.vO + (pBot.vO - pTop.vO) * t2;
			} else {
				const dy1 = pBot.y - pMid.y;
				const t1 = dy1 === 0 ? 0 : (py - pMid.y) / dy1;
				leftX = pMid.x + (pBot.x - pMid.x) * t1;
				leftZ = pMid.z + (pBot.z - pMid.z) * t1;
				leftIz = pMid.iz + (pBot.iz - pMid.iz) * t1;
				leftUO = pMid.uO + (pBot.uO - pMid.uO) * t1;
				leftVO = pMid.vO + (pBot.vO - pMid.vO) * t1;

				const dy2 = pBot.y - pTop.y;
				const t2 = dy2 === 0 ? 0 : (py - pTop.y) / dy2;
				rightX = pTop.x + (pBot.x - pTop.x) * t2;
				rightZ = pTop.z + (pBot.z - pTop.z) * t2;
				rightIz = pTop.iz + (pBot.iz - pTop.iz) * t2;
				rightUO = pTop.uO + (pBot.uO - pTop.uO) * t2;
				rightVO = pTop.vO + (pBot.vO - pTop.vO) * t2;
			}

			if (leftX > rightX) {
				[leftX, rightX] = [rightX, leftX];
				[leftZ, rightZ] = [rightZ, leftZ];
				[leftIz, rightIz] = [rightIz, leftIz];
				[leftUO, rightUO] = [rightUO, leftUO];
				[leftVO, rightVO] = [rightVO, leftVO];
			}

			const startX = Math.max(0, Math.ceil(leftX - 0.5));
			const endX = Math.min(size - 1, Math.floor(rightX - 0.5));
			if (endX < startX) continue;

			const spanWidth = rightX - leftX;
			const spanInv = 1.0 / (spanWidth || CoreConstants.EPSILON);
			const dz = (rightZ - leftZ) * spanInv;
			const diz = (rightIz - leftIz) * spanInv;
			const duO = (rightUO - leftUO) * spanInv;
			const dvO = (rightVO - leftVO) * spanInv;
			const dx = startX + 0.5 - leftX;
			let z = leftZ + dx * dz;
			let iz = leftIz + dx * diz;
			let uO = leftUO + dx * duO;
			let vO = leftVO + dx * dvO;

			const row = y * size;
			for (let x = startX; x <= endX; x++) {
				const idx = row + x;
				if (z < buffer[idx]) {
					const safeIz =
						Math.abs(iz) > CoreConstants.EPSILON ? iz
						: iz >= 0 ? CoreConstants.EPSILON
						: -CoreConstants.EPSILON;
					const invIz = 1 / safeIz;
					const u = uO * invIz;
					const v = vO * invIz;
					const alpha = this._materialRuntime.sampleAlphaMask(
						material,
						u,
						v
					);
					if (alpha >= alphaCutoff) {
						buffer[idx] = z;
					}
				}
				z += dz;
				iz += diz;
				uO += duO;
				vO += dvO;
			}
		}
	}

	public drawTransmissionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		shadowTarget: SoftwareShadowRenderTarget
	): void {
		const size = shadowTarget.size;
		const buffer = shadowTarget.depthBuffer;
		const transmissionBuffer = shadowTarget.transmissionBuffer;
		const material = face.material;
		if (!material || !transmissionBuffer) return;

		const transmittance = resolveMaterialShadowTransmittance(material);
		const transR = transmittance.r;
		const transG = transmittance.g;
		const transB = transmittance.b;

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

	public drawCameraDepthTriangle(
		pts: ProjectedVertex[],
		context: RasterizerContext
	): void {
		const { width, height } = context;
		const depthTarget = context.earlyDepthBuffer ?? context.depthBuffer;
		if (!depthTarget) return;
		const clipRect = context.clipRect;
		const clipMinX =
			clipRect ? Math.max(0, Math.floor(clipRect.minX)) : 0;
		const clipMinY =
			clipRect ? Math.max(0, Math.floor(clipRect.minY)) : 0;
		const clipMaxX =
			clipRect ? Math.min(width - 1, Math.floor(clipRect.maxX)) : width - 1;
		const clipMaxY =
			clipRect ?
				Math.min(height - 1, Math.floor(clipRect.maxY))
			:	height - 1;
		if (
			clipMinX > clipMaxX ||
			clipMinY > clipMaxY ||
			width <= 0 ||
			height <= 0
		) {
			return;
		}

		const interpolator = this._interpolator;
		const verts = interpolator.prepareCameraDepth(pts);

		let [vTop, vMid, vBot] = [verts[0], verts[1], verts[2]];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(clipMinY, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(clipMaxY, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			interpolator.sampleScanlineEdges(vTop, vMid, vBot, py);
			const left = interpolator.left;
			const right = interpolator.right;

			const startX = Math.max(clipMinX, Math.ceil(left.x - 0.5));
			const endX = Math.min(clipMaxX, Math.floor(right.x - 0.5));
			if (endX < startX) continue;

			const span = interpolator.depthSpan;
			span.setup(left, right, startX);
			const rowStart = y * width;

			for (let x = startX; x <= endX; x++) {
				const bufIdx = rowStart + x;
				if (span.computeDepth()) {
					if (
						span.zCamValue > 0 &&
						span.zCamValue < depthTarget[bufIdx]
					) {
						depthTarget[bufIdx] = span.zCamValue;
					}
				}
				span.advance();
			}
		}
	}

	public drawTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		context: RasterizerContext,
		program: SoftwareFragmentProgram,
		isTransparent: boolean = false
	): void {
		const { width, height, depthBuffer } = context;
		const earlyDepthBuffer = context.earlyDepthBuffer ?? null;
		const clipRect = context.clipRect;
		const clipMinX =
			clipRect ? Math.max(0, Math.floor(clipRect.minX)) : 0;
		const clipMinY =
			clipRect ? Math.max(0, Math.floor(clipRect.minY)) : 0;
		const clipMaxX =
			clipRect ? Math.min(width - 1, Math.floor(clipRect.maxX)) : width - 1;
		const clipMaxY =
			clipRect ?
				Math.min(height - 1, Math.floor(clipRect.maxY))
			:	height - 1;
		const material = program.material;

		if (!depthBuffer) return;
		if (
			clipMinX > clipMaxX ||
			clipMinY > clipMaxY ||
			width <= 0 ||
			height <= 0
		) {
			return;
		}
		const viewMat = context.camera.viewMatrix;

		const interpolator = this._interpolator;
		const shouldWriteDepth = program.shouldWriteDepth;

		const verts = interpolator.prepareFragment(pts, face);

		let [vTop, vMid, vBot] = [verts[0], verts[1], verts[2]];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(clipMinY, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(clipMaxY, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			interpolator.sampleScanlineEdges(vTop, vMid, vBot, py);
			const left = interpolator.left;
			const right = interpolator.right;

			const startX = Math.max(clipMinX, Math.ceil(left.x - 0.5));
			const endX = Math.min(clipMaxX, Math.floor(right.x - 0.5));
			if (endX < startX) continue;

			const span = interpolator.fragmentSpan;
			span.setup(left, right, startX);

			const bufRow = y * width;
			const input = this._fragmentInput;

			for (let x = startX; x <= endX; x++) {
				const bufIdx = bufRow + x;

				// Use w for early z-test check, but final shade depth uses linear depth
				if (span.computeDepth()) {
					if (span.zCamValue > 0) {
						const earlyDepthValue =
							earlyDepthBuffer ? earlyDepthBuffer[bufIdx] : depthBuffer[bufIdx];
						const passedEarlyDepth =
							earlyDepthBuffer ?
								span.zCamValue <= earlyDepthValue + CoreConstants.EPSILON
							:	span.zCamValue < earlyDepthValue;
						if (!passedEarlyDepth) {
							span.advance();
							continue;
						}

						span.writeFragmentInput(input);

						const finalOutput = program.shade(input);
						let finalColor = finalOutput?.color;
						const shadedDepth = finalOutput?.depth ?? span.zCamValue;

						if (
							finalColor &&
							shadedDepth > 0 &&
							shadedDepth < depthBuffer[bufIdx]
						) {
							if (shouldWriteDepth) {
								depthBuffer[bufIdx] = shadedDepth;
							}
							const idx = bufIdx << 2;
							if (!isTransparent) {
								pixels[idx] = finalColor.r;
								pixels[idx + 1] = finalColor.g;
								pixels[idx + 2] = finalColor.b;
								pixels[idx + 3] = 255;

								if (context.normalBuffer) {
									const nIdx = bufIdx * 3;
									const surfaceNormal =
										program.getSurfaceNormal() ?? input.normal;
									const nView = Matrix4.transformNormal(
										viewMat,
										surfaceNormal
									);
									const nLen = Math.hypot(nView.x, nView.y, nView.z) || 1;
									context.normalBuffer[nIdx] = nView.x / nLen;
									context.normalBuffer[nIdx + 1] = nView.y / nLen;
									context.normalBuffer[nIdx + 2] = nView.z / nLen;
								}
								if (context.motionBuffer) {
									this._writeMotionDepth(
										context,
										bufIdx,
										x,
										y,
										{
											x: span.previousWorldOx * span.zCam,
											y: span.previousWorldOy * span.zCam,
											z: span.previousWorldOz * span.zCam,
										},
										shadedDepth
									);
								}
							} else {
								const faceAlpha = face.color?.a ?? 1;
								const shaderAlpha = program.getOpacity();
								const alpha = clamp(faceAlpha * shaderAlpha);
								const invA = 1 - alpha;
								pixels[idx] = finalColor.r * alpha + pixels[idx] * invA;
								pixels[idx + 1] = finalColor.g * alpha + pixels[idx + 1] * invA;
								pixels[idx + 2] = finalColor.b * alpha + pixels[idx + 2] * invA;
								pixels[idx + 3] = CoreConstants.OPAQUE_ALPHA;
							}
						}
					}
				}

				span.advance();
			}
		}

		if (material.wireframe) {
			this._drawWireframe(pts, face, pixels, context, isTransparent);
		}
	}

	private _writeMotionDepth(
		context: RasterizerContext,
		pixelIndex: number,
		x: number,
		y: number,
		previousWorld: IVector3,
		depth: number
	): void {
		const motionBuffer = context.motionBuffer;
		if (!motionBuffer) {
			return;
		}
		const currentNdcX = ((x + 0.5) / Math.max(1, context.width)) * 2 - 1;
		const currentNdcY = 1 - ((y + 0.5) / Math.max(1, context.height)) * 2;
		let previousNdcX = currentNdcX;
		let previousNdcY = currentNdcY;
		if (context.temporal?.previousViewProjection) {
			const previousClip = Matrix4.transformPoint(
				context.temporal.previousViewProjection,
				previousWorld
			);
			const previousW =
				Math.abs(previousClip.w ?? 0) > CoreConstants.EPSILON ?
					previousClip.w!
				:	(previousClip.w ?? 0) >= 0 ?
					CoreConstants.EPSILON
				:	-CoreConstants.EPSILON;
			previousNdcX =
				previousClip.x / previousW + context.temporal.previousJitter[0];
			previousNdcY =
				previousClip.y / previousW + context.temporal.previousJitter[1];
		}
		const offset = pixelIndex << 2;
		motionBuffer[offset] = currentNdcX - previousNdcX;
		motionBuffer[offset + 1] = currentNdcY - previousNdcY;
		motionBuffer[offset + 2] = depth;
		motionBuffer[offset + 3] = 0;
	}

	private _drawWireframe(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		context: RasterizerContext,
		isTransparent: boolean = false
	): void {
		const { width, height, depthBuffer } = context;
		const clipRect = context.clipRect;
		const clipMinX =
			clipRect ? Math.max(0, Math.floor(clipRect.minX)) : 0;
		const clipMinY =
			clipRect ? Math.max(0, Math.floor(clipRect.minY)) : 0;
		const clipMaxX =
			clipRect ? Math.min(width - 1, Math.floor(clipRect.maxX)) : width - 1;
		const clipMaxY =
			clipRect ?
				Math.min(height - 1, Math.floor(clipRect.maxY))
			:	height - 1;
		const material = this._materialRuntime.resolveMaterial(face.material);

		if (!depthBuffer) return;
		if (clipMinX > clipMaxX || clipMinY > clipMaxY) return;

		const wireColor = { r: 255, g: 255, b: 255 };
		const alpha =
			isTransparent ? clamp(face.color?.a ?? material.opacity ?? 1) : 1;

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

				if (
					px >= clipMinX &&
					px <= clipMaxX &&
					py >= clipMinY &&
					py <= clipMaxY
				) {
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
