import { Matrix4 } from "../../../maths/Matrix4";
import { Plane } from "../../../maths/Plane";
import { Projector } from "./Projector";
import { RenderConstants } from "./constants";
import type { FrameContext } from "../../pipeline/types";
import type { ProjectedFace, ProjectedVertex } from "../../types";
import type { Rasterizer } from "./Rasterizer";
import { SkyboxRenderer } from "./SkyboxRenderer";
import { AlphaMode } from "../../../materials/Material";

interface ReflectionBuffer {
	imageData: ImageData;
	width: number;
	height: number;
}

interface PlaneAggregateInfo {
	plane: Plane;
}

export class ReflectionRenderer {
	private _rasterizer: Rasterizer;
	private _depthBuffer: Float32Array | null = null;
	private _planesPool: Map<string, Plane> = new Map();
	private _imageDataPool: Map<string, ImageData[]> = new Map();

	public reflectionBuffers: Map<string, ReflectionBuffer> = new Map();

	// Allows scaling the resolution of reflection buffers for performance vs quality tradeoff
	public resolutionScale: number = 0.5;

	constructor(rasterizer: Rasterizer) {
		this._rasterizer = rasterizer;
	}

	public render(context: FrameContext): void {
		// 1. Collect all unique mirror planes and their aggregate filter settings
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

		// 2. Render and process each plane
		for (const [key, info] of planeInfos) {
			const buffer = this._prepareBuffer(key, scaledWidth, scaledHeight);

			// Render reflection
			this._renderReflectionForPlane(info.plane, buffer, context);
		}

		// 3. Cleanup stale buffers and planes
		this._cleanupStaleResources(planeInfos);
	}

	private _collectPlaneInfos(
		context: FrameContext
	): Map<string, PlaneAggregateInfo> {
		const infos = new Map<string, PlaneAggregateInfo>();

		for (const model of context.scene.models) {
			for (const primitive of model.primitives) {
				const material = primitive.material;
				if (material && material.mirrorPlane) {
					const p = material.mirrorPlane;
					const key = `${p.normal.x},${p.normal.y},${p.normal.z},${p.constant}`;

					let info = infos.get(key);
					if (!info) {
						if (!this._planesPool.has(key)) {
							this._planesPool.set(key, new Plane(p.normal, p.constant));
						}
						info = {
							plane: this._planesPool.get(key)!,
						};
						infos.set(key, info);
					}
				}
			}
		}
		return infos;
	}

	private _prepareBuffer(
		key: string,
		width: number,
		height: number
	): ReflectionBuffer {
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
		buffer: ReflectionBuffer,
		context: FrameContext
	): void {
		const pixels = buffer.imageData.data;

		if (context.features.enableSkybox && context.scene.skybox) {
			SkyboxRenderer.render(
				context.scene.skybox,
				pixels,
				context.camera,
				buffer.width,
				buffer.height
			);
		} else {
			pixels.fill(0); // Clear
			for (let i = 3; i < pixels.length; i += 4) {
				pixels[i] = RenderConstants.REFLECTION_BUFFER_ALPHA;
			}
		}

		// Backup camera state
		const camera = context.camera;
		const originalViewMatrix = camera.viewMatrix;
		const originalProjectionMatrix = camera.projectionMatrix;
		const originalViewProjMatrix = camera.viewProjectionMatrix;
		const originalCameraPosition = {
			x: camera.position.x,
			y: camera.position.y,
			z: camera.position.z,
		};

		// 1. Calculate Reflection Matrix
		const reflectMat = Matrix4.reflection(plane);
		const mirroredPosition = Matrix4.transformPoint(
			reflectMat,
			originalCameraPosition
		);

		// 2. Set Mirror Camera: ViewMirror = ViewMain * R
		const mirrorViewMatrix = Matrix4.multiply(originalViewMatrix, reflectMat);
		camera.viewMatrix = mirrorViewMatrix;

		// 3. Oblique Near Plane Clipping
		const mirrorProjMatrix = originalProjectionMatrix.clone();
		const isCameraAbove =
			plane.normal.x * originalCameraPosition.x +
				plane.normal.y * originalCameraPosition.y +
				plane.normal.z * originalCameraPosition.z +
				plane.constant >
			0;

		const clipPlaneNormal = Matrix4.transformDirection(
			mirrorViewMatrix,
			plane.normal
		);
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

		camera.projectionMatrix = mirrorProjMatrix;
		camera.viewProjectionMatrix = Matrix4.multiply(
			mirrorProjMatrix,
			mirrorViewMatrix
		);
		camera.position.copy(mirroredPosition);

		try {
			const bufferSize = buffer.width * buffer.height;
			if (!this._depthBuffer || this._depthBuffer.length !== bufferSize) {
				this._depthBuffer = new Float32Array(bufferSize);
			}
			const depthBuffer = this._depthBuffer;
			depthBuffer.fill(Infinity);

			const opaqueFaces: ProjectedFace[] = [];
			const transparentFaces: ProjectedFace[] = [];

			// Render scene with mirrored camera
			for (const model of context.scene.models) {
				const faces = Projector.projectModel(model, context, true, buffer);

				for (const face of faces) {
					// skip if same plane
					if (face.material && face.material.mirrorPlane) {
						const mp = face.material.mirrorPlane;
						if (
							mp.normal.x === plane.normal.x &&
							mp.normal.y === plane.normal.y &&
							mp.normal.z === plane.normal.z &&
							mp.constant === plane.constant
						) {
							continue;
						}
					}

					// Only reflect objects on the same side as the camera
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
					if (
						alphaMode === AlphaMode.Blend ||
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
						true
					);
				}
			}
		} finally {
			// Restore camera
			camera.viewMatrix = originalViewMatrix;
			camera.projectionMatrix = originalProjectionMatrix;
			camera.viewProjectionMatrix = originalViewProjMatrix;
			camera.position.copy(originalCameraPosition);
		}
	}

	private _drawReflectionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		depthBuffer: Float32Array,
		overrideSize: { width: number; height: number },
		context: FrameContext,
		isTransparent: boolean
	): void {
		const rasterizerContext = {
			width: overrideSize.width,
			height: overrideSize.height,
			depthBuffer,
			camera: {
				position: context.camera.position,
				viewMatrix: context.camera.viewMatrix,
			},
			lights: context.scene.lights,
			shadowMaps: context.shadowMaps,
			shAmbientCoeffs: context.shAmbientCoeffs,
			features: {
				enableLighting: context.features.enableLighting,
				enableSH: context.features.enableSH,
				enableShadows: context.features.enableShadows,
				enableGamma: context.features.enableGamma,
				enableReflection: context.features.enableReflection,
				worldMatrix: context.worldMatrix,
			},
		};

		this._rasterizer.drawTriangle(
			pts,
			face,
			pixels,
			rasterizerContext,
			isTransparent
		);
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
