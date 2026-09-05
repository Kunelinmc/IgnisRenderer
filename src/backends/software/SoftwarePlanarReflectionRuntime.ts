import { AlphaMode } from "../../materials/Material";
import { materialUsesTransmission } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import { Plane } from "../../maths/Plane";
import {
	type DrawPacket,
} from "../../pipeline/types";
import type { ProjectedFace, ProjectedVertex } from "../../core/types";
import { sampleSoftwareTextureAlpha } from "../../shaders/software/textureSampling";
import { SkyboxRenderer } from "./SkyboxRenderer";
import {
	SoftwareTriangleInterpolator,
	type SoftwareFragmentSpan,
} from "./Interpolator";
import { Projector, type SoftwareProjectionView } from "./Projector";
import type { Rasterizer } from "./Rasterizer";
import type { SoftwarePassContext } from "./SoftwareFrameServices";
import type {
	SoftwareFrameView,
} from "./SoftwareFrameView";
import {
	resolveSoftwarePlanarReflectionPlaneKey,
	SoftwareReflectionPlanner,
	type SoftwareReflectionPlaneInfo,
} from "./SoftwareReflectionPlanner";
import {
	SoftwareReflectionResources,
	type SoftwarePlanarReflectionBuffer,
} from "./SoftwareReflectionResources";
import {
	SoftwareReflectionRenderer,
	type SoftwareMirroredPlaneRender,
} from "./SoftwareReflectionRenderer";
import {
	SoftwareReflectionCompositor,
	type SoftwareCompositeClipRect,
	type SoftwareReflectionTriangleComposite,
} from "./SoftwareReflectionCompositor";
import { createSoftwareRasterizerContext } from "./SoftwareRasterContextFactory";

const DEPTH_EPSILON = 1e-6;
const REFLECTION_TRANSPARENT_THRESHOLD = 0.99;

export type { SoftwarePlanarReflectionBuffer } from "./SoftwareReflectionResources";
export { resolveSoftwarePlanarReflectionPlaneKey } from "./SoftwareReflectionPlanner";

export class SoftwarePlanarReflectionRuntime {
	private _rasterizer: Rasterizer;
	private _depthBuffer: Float32Array | null = null;
	private readonly _planner = new SoftwareReflectionPlanner();
	private readonly _resources: SoftwareReflectionResources;
	private readonly _renderer = new SoftwareReflectionRenderer();
	private readonly _compositor = new SoftwareReflectionCompositor();
	private readonly _renderPlane: SoftwareMirroredPlaneRender = (plane, buffer, context) =>
		this._renderReflectionForPlane(plane, buffer, context);
	private readonly _compositeReflectionTriangle: SoftwareReflectionTriangleComposite = (
		...args
	) => this._compositeTriangle(...args);
	private _compositeInterpolator: SoftwareTriangleInterpolator =
		new SoftwareTriangleInterpolator();

	public get reflectionBuffers(): Map<string, SoftwarePlanarReflectionBuffer> {
		return this._resources.buffers;
	}

	public resolutionScale: number = 0.5;

	constructor(
		rasterizer: Rasterizer,
		resources: SoftwareReflectionResources = new SoftwareReflectionResources(),
	) {
		this._rasterizer = rasterizer;
		this._resources = resources;
	}

	public render(context: SoftwarePassContext): void {
		const planes = this._collectPlaneInfos(context.frame);
		const rendered = this._renderer.render(
			context,
			planes,
			this._resources,
			this.resolutionScale,
			this._renderPlane,
		);
		if (rendered) this._planner.trim(planes);
		else this._planner.clear();
	}

	public composite(context: SoftwarePassContext, packets: DrawPacket[]): void {
		this._compositor.composite(
			context,
			packets,
			this.reflectionBuffers,
			this._compositeReflectionTriangle,
		);
	}

	private _collectPlaneInfos(frame: SoftwareFrameView): Map<string, SoftwareReflectionPlaneInfo> {
		return this._planner.collect(frame);
	}

	private _renderReflectionForPlane(
		plane: Plane,
		buffer: SoftwarePlanarReflectionBuffer,
		context: SoftwarePassContext,
	): void {
		const frame = context.frame;
		const pixels = buffer.color;
		const environment = frame.scene.environment;
		const sourceCamera = frame.camera;
		const originalCameraPosition = sourceCamera.position;
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
				pixels,
				{
					type: sourceCamera.type,
					fov: sourceCamera.fov,
					aspectRatio: sourceCamera.aspectRatio,
					viewMatrix: mirrorViewMatrix,
				},
				buffer.width,
				buffer.height,
			);
		} else {
			pixels.fill(0);
			for (let i = 3; i < pixels.length; i += 4) {
				pixels[i] = 1;
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
			const packets = frame.scene.opaquePackets.concat(frame.scene.transparentPackets);

			for (const packet of packets) {
				const faces = Projector.projectPacket(packet, frame, true, buffer, projectionView);

				for (const face of faces) {
					if (
						resolveSoftwarePlanarReflectionPlaneKey(face.material?.mirrorPlane) ===
						planeKey
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

					const alpha = face.material.opacity;
					if (alpha < 0.1) continue;
					const explicitAlphaMode = face.material?.alphaMode;
					const alphaMode = explicitAlphaMode || AlphaMode.Opaque;
					const transmissionTransparent = face.material
						? materialUsesTransmission(face.material)
						: false;
					if (
						alphaMode === AlphaMode.Blend ||
						transmissionTransparent ||
						(explicitAlphaMode === undefined &&
							alpha < REFLECTION_TRANSPARENT_THRESHOLD)
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
						false,
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
						true,
					);
				}
			}
		}
	}

	private _drawReflectionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Float32Array,
		depthBuffer: Float32Array,
		overrideSize: { width: number; height: number },
		context: SoftwarePassContext,
		projectionView: SoftwareProjectionView,
		isTransparent: boolean,
	): void {
		const frame = context.frame;
		const reflectionCamera = projectionView.camera;
		const sampleShadow = context.services.shadow.samplerFactory({
			position: reflectionCamera.position,
			viewMatrix: reflectionCamera.viewMatrix,
		});
		const rasterizerContext = createSoftwareRasterizerContext(context, {
			width: overrideSize.width,
			height: overrideSize.height,
			depthBuffer,
			camera: {
				position: reflectionCamera.position || frame.camera.position,
				viewMatrix: reflectionCamera.viewMatrix,
			},
			sampleShadow,
			includeFrameAttachments: false,
		});
		const program = this._rasterizer.prepareFragmentProgram(
			face,
			rasterizerContext,
			isTransparent,
		);

		this._rasterizer.drawTriangle(pts, face, pixels, rasterizerContext, program, isTransparent);
	}

	private _compositeTriangle(
		pts: [ProjectedVertex, ProjectedVertex, ProjectedVertex],
		face: ProjectedFace,
		pixels: Float32Array,
		depthBuffer: Float32Array,
		width: number,
		height: number,
		clipRect: SoftwareCompositeClipRect,
		buffer: SoftwarePlanarReflectionBuffer,
		reflectivity: number,
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
		const reflectionData = buffer.color;

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
					if (span.zCamValue <= depthBuffer[idx] + DEPTH_EPSILON) {
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
							pixels[pixelIdx + 3] = 1;
						}
					}
				}
				span.advance();
			}
		}
	}

	private _passesAlphaMask(
		material: NonNullable<ProjectedFace["material"]>,
		span: SoftwareFragmentSpan,
	): boolean {
		const alpha =
			sampleSoftwareTextureAlpha(material.map, span.uO * span.zCam, span.vO * span.zCam) *
			(material.opacity ?? 1);
		return alpha >= (material.alphaCutoff ?? 0.5);
	}
}
