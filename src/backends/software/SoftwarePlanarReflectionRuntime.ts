import type { MirrorPlane } from "../../materials/Material";
import { AlphaMode } from "../../materials/Material";
import { materialUsesTransmission } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import { Plane } from "../../maths/Plane";
import {
	defineTransientKey,
	type DrawPacket,
	type FrameContext,
	type TransientStore,
} from "../../pipeline/types";
import type { ProjectedFace, ProjectedVertex } from "../../core/types";
import { sampleSoftwareTextureAlpha } from "../../shaders/software/textureSampling";
import { SkyboxRenderer } from "./SkyboxRenderer";
import {
	SoftwareTriangleInterpolator,
	type SoftwareFragmentSpan,
} from "./Interpolator";
import { Projector, type SoftwareProjectionView } from "./Projector";
import type { Rasterizer, RasterizerContext } from "./Rasterizer";
import { CoreConstants, RenderConstants } from "./constants";
import {
	createSoftwareShadowSampler,
	getSoftwareShadowRuntimeMap,
} from "./passes/SoftwareShadowPass";
import { resolveLegacyShadowMaps } from "../../pipeline/shadows/LegacyShadowPlanAdapter";

export const SOFTWARE_PLANAR_REFLECTION_RUNTIME_KEY =
	defineTransientKey<SoftwarePlanarReflectionRuntime>(
		"software-planar-reflection-runtime"
	);

export interface SoftwarePlanarReflectionBuffer {
	imageData: ImageData;
	width: number;
	height: number;
}

interface PlaneAggregateInfo {
	plane: Plane;
}

interface SoftwareCompositeClipRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

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

function resolveCompositeClipRects(
	context: FrameContext
): SoftwareCompositeClipRect[] {
	const width = Math.max(1, Math.floor(context.attachments.width));
	const height = Math.max(1, Math.floor(context.attachments.height));
	if (
		!context.incremental.enabled ||
		context.incremental.forceFullFrame ||
		context.incremental.dirtyRects.length === 0
	) {
		return [{
			minX: 0,
			minY: 0,
			maxX: width - 1,
			maxY: height - 1,
		}];
	}

	const rects: SoftwareCompositeClipRect[] = [];
	for (const rect of context.incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
		const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
		if (minX <= maxX && minY <= maxY) {
			rects.push({ minX, minY, maxX, maxY });
		}
	}
	return rects;
}

export function resolveSoftwarePlanarReflectionPlaneKey(
	plane: MirrorPlane | Plane | null | undefined
): string | null {
	if (!plane) {
		return null;
	}
	return `${plane.normal.x},${plane.normal.y},${plane.normal.z},${plane.constant}`;
}

export function getSoftwarePlanarReflectionRuntime(
	transient: TransientStore
): SoftwarePlanarReflectionRuntime | null {
	return transient.get(SOFTWARE_PLANAR_REFLECTION_RUNTIME_KEY) ?? null;
}

export function setSoftwarePlanarReflectionRuntime(
	transient: TransientStore,
	runtime: SoftwarePlanarReflectionRuntime
): void {
	transient.set(SOFTWARE_PLANAR_REFLECTION_RUNTIME_KEY, runtime);
}

export class SoftwarePlanarReflectionRuntime {
	private _rasterizer: Rasterizer;
	private _depthBuffer: Float32Array | null = null;
	private _planesPool: Map<string, Plane> = new Map();
	private _imageDataPool: Map<string, ImageData[]> = new Map();
	private _compositeInterpolator: SoftwareTriangleInterpolator =
		new SoftwareTriangleInterpolator();

	public reflectionBuffers: Map<string, SoftwarePlanarReflectionBuffer> =
		new Map();

	public resolutionScale: number = 0.5;

	public constructor(rasterizer: Rasterizer) {
		this._rasterizer = rasterizer;
	}

	public render(context: FrameContext): void {
		const planeInfos = this._collectPlaneInfos(context);

		if (planeInfos.size === 0) {
			this._clearBuffers();
			return;
		}

		const { width, height } = context.attachments;
		if (width <= 0 || height <= 0) {
			this._clearBuffers();
			return;
		}

		const scaledWidth = Math.max(1, Math.floor(width * this.resolutionScale));
		const scaledHeight = Math.max(1, Math.floor(height * this.resolutionScale));

		for (const [key, info] of planeInfos) {
			const buffer = this._prepareBuffer(key, scaledWidth, scaledHeight);
			this._renderReflectionForPlane(info.plane, buffer, context);
		}

		this._cleanupStaleResources(planeInfos);
	}

	public composite(context: FrameContext, packets: DrawPacket[]): void {
		if (
			!context.features.enableReflection ||
			packets.length <= 0 ||
			this.reflectionBuffers.size <= 0
		) {
			return;
		}
		const pixels = context.attachments.pixels;
		const depthBuffer = context.attachments.depthBuffer;
		const width = context.attachments.width;
		const height = context.attachments.height;
		if (!pixels || !depthBuffer || width <= 0 || height <= 0) {
			return;
		}

		const clipRects = resolveCompositeClipRects(context);
		if (clipRects.length <= 0) {
			return;
		}

		for (const packet of packets) {
			const material = packet.material;
			const reflectivity = Math.max(
				0,
				Math.min(1, material.reflectivity ?? 0)
			);
			if (
				reflectivity <= 0 ||
				!material.mirrorPlane ||
				material.alphaMode === AlphaMode.Blend
			) {
				continue;
			}

			const plane = material.mirrorPlane;
			const cameraPosition = context.viewCamera.getWorldPosition();
			const cameraDistance =
				cameraPosition.x * plane.normal.x +
				cameraPosition.y * plane.normal.y +
				cameraPosition.z * plane.normal.z +
				plane.constant;
			if (cameraDistance <= 0) {
				continue;
			}

			const key = resolveSoftwarePlanarReflectionPlaneKey(plane);
			const buffer = key ? this.reflectionBuffers.get(key) : null;
			if (!buffer) {
				continue;
			}

			const faces = Projector.projectPacket(packet, context);
			for (const face of faces) {
				const projected = face.projected;
				for (let i = 1; i < projected.length - 1; i++) {
					const triangle: [ProjectedVertex, ProjectedVertex, ProjectedVertex] = [
						projected[0],
						projected[i],
						projected[i + 1],
					];
					for (const clipRect of clipRects) {
						this._compositeTriangle(
							triangle,
							face,
							pixels,
							depthBuffer,
							width,
							height,
							clipRect,
							buffer,
							reflectivity
						);
					}
				}
			}
		}
	}

	private _collectPlaneInfos(
		context: FrameContext
	): Map<string, PlaneAggregateInfo> {
		const infos = new Map<string, PlaneAggregateInfo>();

		for (const packet of context.scene.reflectivePackets) {
			const material = packet.material;
			const key = resolveSoftwarePlanarReflectionPlaneKey(
				material?.mirrorPlane
			);
			if (!material || material.reflectivity <= 0 || !key) {
				continue;
			}

			let info = infos.get(key);
			if (!info) {
				if (!this._planesPool.has(key)) {
					this._planesPool.set(
						key,
						new Plane(material.mirrorPlane!.normal, material.mirrorPlane!.constant)
					);
				}
				info = {
					plane: this._planesPool.get(key)!,
				};
				infos.set(key, info);
			}
		}
		return infos;
	}

	private _prepareBuffer(
		key: string,
		width: number,
		height: number
	): SoftwarePlanarReflectionBuffer {
		let buffer = this.reflectionBuffers.get(key);

		if (buffer && (buffer.width !== width || buffer.height !== height)) {
			this._releaseImageDataToPool(
				buffer.width,
				buffer.height,
				buffer.imageData
			);
			buffer = undefined;
		}

		if (!buffer) {
			const imageData =
				this._getImageDataFromPool(width, height) ||
				new ImageData(width, height);
			buffer = { imageData, width, height };
			this.reflectionBuffers.set(key, buffer);
		}

		return buffer;
	}

	private _clearBuffers(): void {
		for (const buffer of this.reflectionBuffers.values()) {
			this._releaseImageDataToPool(
				buffer.width,
				buffer.height,
				buffer.imageData
			);
		}
		this.reflectionBuffers.clear();
	}

	private _cleanupStaleResources(
		activePlanes: Map<string, PlaneAggregateInfo>
	): void {
		for (const [key, buffer] of this.reflectionBuffers.entries()) {
			if (!activePlanes.has(key)) {
				this._releaseImageDataToPool(
					buffer.width,
					buffer.height,
					buffer.imageData
				);
				this.reflectionBuffers.delete(key);
			}
		}
		for (const key of this._planesPool.keys()) {
			if (!activePlanes.has(key)) {
				this._planesPool.delete(key);
			}
		}
	}

	private _renderReflectionForPlane(
		plane: Plane,
		buffer: SoftwarePlanarReflectionBuffer,
		context: FrameContext
	): void {
		const pixels = buffer.imageData.data;
		const environment = resolvePreparedSceneEnvironment(context.scene);
		const sourceCamera = context.viewCamera;
		const originalCameraPosition = sourceCamera.getWorldPosition();
		const reflectMat = Matrix4.reflection(plane);
		const mirroredPosition = Matrix4.transformPoint(reflectMat, originalCameraPosition);
		const mirrorViewMatrix = Matrix4.multiply(sourceCamera.viewMatrix, reflectMat);
		const mirrorProjMatrix = sourceCamera.projectionMatrix.clone();
		const isCameraAbove =
			plane.normal.x * originalCameraPosition.x +
				plane.normal.y * originalCameraPosition.y +
				plane.normal.z * originalCameraPosition.z +
				plane.constant >
			0;
		const clipPlaneNormal = Matrix4.transformDirection(mirrorViewMatrix, plane.normal);
		let clipPlaneConstant = plane.distanceToPoint(mirroredPosition);
		if (!isCameraAbove) {
			clipPlaneNormal.x *= -1;
			clipPlaneNormal.y *= -1;
			clipPlaneNormal.z *= -1;
			clipPlaneConstant *= -1;
		}
		mirrorProjMatrix.applyObliqueClipping({
			normal: clipPlaneNormal,
			constant: clipPlaneConstant,
		});
		const projectionView: SoftwareProjectionView = {
			camera: {
				type: sourceCamera.type,
				near: sourceCamera.near,
				position: mirroredPosition,
				viewMatrix: mirrorViewMatrix,
				projectionMatrix: mirrorProjMatrix,
				viewProjectionMatrix: Matrix4.multiply(mirrorProjMatrix, mirrorViewMatrix),
			},
			width: buffer.width,
			height: buffer.height,
			temporalState: null,
			trackTemporalHistory: false,
		};

		if (
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
				{
					type: sourceCamera.type,
					fov: sourceCamera.fov,
					aspectRatio: sourceCamera.aspectRatio,
					viewMatrix: mirrorViewMatrix,
				},
				buffer.width,
				buffer.height
			);
		} else {
			pixels.fill(0);
			for (let i = 3; i < pixels.length; i += 4) {
				pixels[i] = RenderConstants.REFLECTION_BUFFER_ALPHA;
			}
		}

		{
			const bufferSize = buffer.width * buffer.height;
			if (!this._depthBuffer || this._depthBuffer.length !== bufferSize) {
				this._depthBuffer = new Float32Array(bufferSize);
			}
			const depthBuffer = this._depthBuffer;
			depthBuffer.fill(Infinity);

			const opaqueFaces: ProjectedFace[] = [];
			const transparentFaces: ProjectedFace[] = [];
			const planeKey = resolveSoftwarePlanarReflectionPlaneKey(plane);
			const packets = context.scene.opaquePackets.concat(
				context.scene.transparentPackets
			);

			for (const packet of packets) {
				const faces = Projector.projectPacket(
					packet,
					context,
					true,
					buffer,
					projectionView,
				);

				for (const face of faces) {
					if (
						resolveSoftwarePlanarReflectionPlaneKey(
							face.material?.mirrorPlane
						) === planeKey
					) {
						continue;
					}

					const facePos = face.center || face.projected[0].world;
					if (facePos) {
						const dist =
							plane.normal.x * facePos.x +
							plane.normal.y * facePos.y +
							plane.normal.z * facePos.z +
							plane.constant;
						if (isCameraAbove ? dist < 0 : dist > 0) continue;
					}

					const alpha = face.color?.a ?? face.material?.opacity ?? 1;
					if (alpha < 0.1) continue;
					const explicitAlphaMode = face.material?.alphaMode;
					const alphaMode = explicitAlphaMode || AlphaMode.Opaque;
					const transmissionTransparent =
						face.material ? materialUsesTransmission(face.material) : false;
					if (
						alphaMode === AlphaMode.Blend ||
						transmissionTransparent ||
						(explicitAlphaMode === undefined &&
							alpha < RenderConstants.REFLECTION_TRANSPARENT_THRESHOLD)
					) {
						transparentFaces.push(face);
					} else {
						opaqueFaces.push(face);
					}
				}
			}

			for (const face of opaqueFaces) {
				const projected = face.projected;
				for (let i = 1; i < projected.length - 1; i++) {
					this._drawReflectionTriangle(
						[projected[0], projected[i], projected[i + 1]],
						face,
						pixels,
						depthBuffer,
						buffer,
						context,
						projectionView,
						false
					);
				}
			}

			transparentFaces.sort((a, b) => b.depthInfo.avg - a.depthInfo.avg);

			for (const face of transparentFaces) {
				const projected = face.projected;
				for (let i = 1; i < projected.length - 1; i++) {
					this._drawReflectionTriangle(
						[projected[0], projected[i], projected[i + 1]],
						face,
						pixels,
						depthBuffer,
						buffer,
						context,
						projectionView,
						true
					);
				}
			}
		}
	}

	private _drawReflectionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		depthBuffer: Float32Array,
		overrideSize: { width: number; height: number },
		context: FrameContext,
		projectionView: SoftwareProjectionView,
		isTransparent: boolean
	): void {
		const runtimeMap = getSoftwareShadowRuntimeMap(context.transient);
		const reflectionCamera = projectionView.camera;
		const sampleShadow = createSoftwareShadowSampler(
			resolveLegacyShadowMaps(context.shadowPlan),
			runtimeMap,
			{
				camera: {
					position: reflectionCamera.position,
					viewMatrix: reflectionCamera.viewMatrix,
				},
			}
		);
		const environment = resolvePreparedSceneEnvironment(context.scene);

		const rasterizerContext: RasterizerContext = {
			width: overrideSize.width,
			height: overrideSize.height,
			depthBuffer,
			camera: {
				position: reflectionCamera.position || context.viewCamera.getWorldPosition(),
				viewMatrix: reflectionCamera.viewMatrix,
			},
			lights: context.scene.lights,
			shadowMaps: resolveLegacyShadowMaps(context.shadowPlan),
			sampleShadow,
			shAmbientCoeffs: context.shAmbientCoeffs,
			environmentSpecularTexture:
				environment.lightingEnabled ?
					environment.iblTexture
				:	null,
			enableLighting: context.features.enableLighting,
			enableSH: context.features.enableSH,
			enableShadows: context.features.enableShadows,
		};
		const program = this._rasterizer.prepareFragmentProgram(
			face,
			rasterizerContext,
			isTransparent
		);

		this._rasterizer.drawTriangle(
			pts,
			face,
			pixels,
			rasterizerContext,
			program,
			isTransparent
		);
	}

	private _compositeTriangle(
		pts: [ProjectedVertex, ProjectedVertex, ProjectedVertex],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		depthBuffer: Float32Array,
		width: number,
		height: number,
		clipRect: SoftwareCompositeClipRect,
		buffer: SoftwarePlanarReflectionBuffer,
		reflectivity: number
	): void {
		const interpolator = this._compositeInterpolator;
		const verts = interpolator.prepareFragment(pts, face);
		const material = face.material;

		let [vTop, vMid, vBot] = [verts[0], verts[1], verts[2]];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];
		if (vMid.y > vBot.y) [vMid, vBot] = [vBot, vMid];
		if (vTop.y > vMid.y) [vTop, vMid] = [vMid, vTop];

		const minY = Math.max(clipRect.minY, Math.ceil(vTop.y - 0.5));
		const maxY = Math.min(clipRect.maxY, Math.floor(vBot.y - 0.5));
		if (minY > maxY) return;

		const useMask = material?.alphaMode === AlphaMode.Mask;
		const inverseReflectivity = 1 - reflectivity;
		const scaleX = buffer.width / width;
		const scaleY = buffer.height / height;
		const reflectionData = buffer.imageData.data;

		for (let y = minY; y <= maxY; y++) {
			const py = y + 0.5;
			interpolator.sampleScanlineEdges(vTop, vMid, vBot, py);
			const left = interpolator.left;
			const right = interpolator.right;

			const startX = Math.max(clipRect.minX, Math.ceil(left.x - 0.5));
			const endX = Math.min(clipRect.maxX, Math.floor(right.x - 0.5));
			if (endX < startX) continue;

			const span = interpolator.fragmentSpan;
			span.setup(left, right, startX);
			const row = y * width;

			for (let x = startX; x <= endX; x++) {
				const idx = row + x;
				if (span.computeDepth() && span.zCamValue > 0) {
					if (span.zCamValue <= depthBuffer[idx] + CoreConstants.EPSILON) {
						if (!useMask || this._passesAlphaMask(material!, span)) {
							let refX = Math.floor(x * scaleX);
							let refY = Math.floor(y * scaleY);
							refX = Math.max(0, Math.min(buffer.width - 1, refX));
							refY = Math.max(0, Math.min(buffer.height - 1, refY));

							const pixelIdx = idx << 2;
							const refIdx = (refY * buffer.width + refX) << 2;
							pixels[pixelIdx] =
								pixels[pixelIdx] * inverseReflectivity +
								reflectionData[refIdx] * reflectivity;
							pixels[pixelIdx + 1] =
								pixels[pixelIdx + 1] * inverseReflectivity +
								reflectionData[refIdx + 1] * reflectivity;
							pixels[pixelIdx + 2] =
								pixels[pixelIdx + 2] * inverseReflectivity +
								reflectionData[refIdx + 2] * reflectivity;
							pixels[pixelIdx + 3] = CoreConstants.OPAQUE_ALPHA;
						}
					}
				}
				span.advance();
			}
		}
	}

	private _passesAlphaMask(
		material: NonNullable<ProjectedFace["material"]>,
		span: SoftwareFragmentSpan
	): boolean {
		const alpha =
			sampleSoftwareTextureAlpha(
				material.map,
				span.uO * span.zCam,
				span.vO * span.zCam
			) * (material.opacity ?? 1);
		return alpha >= (material.alphaCutoff ?? 0.5);
	}

	private _getImageDataFromPool(
		width: number,
		height: number
	): ImageData | null {
		const key = `${width},${height}`;
		const pool = this._imageDataPool.get(key);
		if (pool && pool.length > 0) return pool.pop()!;
		return null;
	}

	private _releaseImageDataToPool(
		width: number,
		height: number,
		imageData: ImageData
	): void {
		const key = `${width},${height}`;
		let pool = this._imageDataPool.get(key);
		if (!pool) {
			pool = [];
			this._imageDataPool.set(key, pool);
		}
		pool.push(imageData);
	}
}
