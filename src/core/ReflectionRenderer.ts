import { Matrix4 } from "../maths/Matrix4";
import { Plane } from "../maths/Plane";
import { Projector } from "./Projector";
import { RenderConstants } from "./Constants";
import type { Renderer } from "./Renderer";
import type { ProjectedFace, ProjectedVertex } from "./types";

interface ReflectionBuffer {
	imageData: ImageData;
	width: number;
	height: number;
}

interface PlaneAggregateInfo {
	plane: Plane;
}

export class ReflectionRenderer {
	private _renderer: Renderer;
	private _depthBuffer: Float32Array | null = null;
	private _planesPool: Map<string, Plane> = new Map();
	private _imageDataPool: Map<string, ImageData[]> = new Map();

	public reflectionBuffers: Map<string, ReflectionBuffer> = new Map();

	// Allows scaling the resolution of reflection buffers for performance vs quality tradeoff
	public resolutionScale: number = 0.5;

	constructor(renderer: Renderer) {
		this._renderer = renderer;
	}

	public render(): void {
		// 1. Collect all unique mirror planes and their aggregate filter settings
		const planeInfos = this._collectPlaneInfos();

		if (planeInfos.size === 0) {
			this._clearBuffers();
			return;
		}

		const { width, height } = this._renderer.canvas;
		const scaledWidth = Math.floor(width * this.resolutionScale);
		const scaledHeight = Math.floor(height * this.resolutionScale);

		// 2. Render and process each plane
		for (const [key, info] of planeInfos) {
			const buffer = this._prepareBuffer(key, scaledWidth, scaledHeight);

			// Render reflection
			this._renderReflectionForPlane(info.plane, buffer);
		}

		// 3. Cleanup stale buffers and planes
		this._cleanupStaleResources(planeInfos);
	}

	private _collectPlaneInfos(): Map<string, PlaneAggregateInfo> {
		const infos = new Map<string, PlaneAggregateInfo>();

		for (const model of this._renderer.scene.models) {
			for (const face of model.faces) {
				const material = face.material;
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
		buffer: ReflectionBuffer
	): void {
		const renderer = this._renderer;
		const pixels = buffer.imageData.data;
		pixels.fill(0); // Clear
		for (let i = 3; i < pixels.length; i += 4) {
			pixels[i] = RenderConstants.REFLECTION_BUFFER_ALPHA;
		}

		// Backup camera state
		const originalViewMatrix = renderer.camera.viewMatrix;
		const originalProjectionMatrix = renderer.camera.projectionMatrix;
		const originalViewProjMatrix = renderer.camera.viewProjectionMatrix;
		const originalCameraPosition = {
			x: renderer.camera.position.x,
			y: renderer.camera.position.y,
			z: renderer.camera.position.z,
		};

		// 1. Calculate Reflection Matrix
		const reflectMat = Matrix4.reflection(plane);
		const mirroredPosition = Matrix4.transformPoint(
			reflectMat,
			originalCameraPosition
		);

		// 2. Set Mirror Camera: ViewMirror = ViewMain * R
		const mirrorViewMatrix = Matrix4.multiply(originalViewMatrix, reflectMat);
		renderer.camera.viewMatrix = mirrorViewMatrix;

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

		renderer.camera.projectionMatrix = mirrorProjMatrix;
		renderer.camera.viewProjectionMatrix = Matrix4.multiply(
			mirrorProjMatrix,
			mirrorViewMatrix
		);
		renderer.camera.position.copy(mirroredPosition);

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
			for (const model of renderer.scene.models) {
				const faces = Projector.projectModel(model, renderer, true, buffer);

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
					const alphaMode = explicitAlphaMode || "OPAQUE";
					if (
						alphaMode === "BLEND" ||
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
						true
					);
				}
			}
		} finally {
			// Restore camera
			renderer.camera.viewMatrix = originalViewMatrix;
			renderer.camera.projectionMatrix = originalProjectionMatrix;
			renderer.camera.viewProjectionMatrix = originalViewProjMatrix;
			renderer.camera.position.copy(originalCameraPosition);
		}
	}

	private _drawReflectionTriangle(
		pts: ProjectedVertex[],
		face: ProjectedFace,
		pixels: Uint8ClampedArray,
		depthBuffer: Float32Array,
		overrideSize: { width: number; height: number },
		isTransparent: boolean
	): void {
		const oldDepth = this._renderer.depthBuffer;
		this._renderer.depthBuffer = depthBuffer;
		try {
			this._renderer.rasterizer.drawTriangle(
				pts,
				face,
				pixels,
				isTransparent,
				overrideSize
			);
		} finally {
			this._renderer.depthBuffer = oldDepth;
		}
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
