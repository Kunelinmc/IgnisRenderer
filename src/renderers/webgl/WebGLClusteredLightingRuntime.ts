import { CameraType } from "../../cameras/Camera";
import { Logger } from "../../foundation/Logger";
import { Matrix4 } from "../../maths/Matrix4";
import type { FrameContext } from "../../pipeline/types";
import type {
	WebGLPointLight,
	WebGLSpotLight,
	WebGLShadowData,
	WebGLLightState,
} from "./WebGLLightCollector";

interface ClusteredRuntimeOptions {
	tileSizePx: number;
	zSlices: number;
	maxLights: number;
	maxLightsPerCluster: number;
}

interface ClusteredLightRecord {
	type: 0 | 1;
	position: [number, number, number];
	range: number;
	direction: [number, number, number];
	outerCos: number;
	innerCos: number;
	color: [number, number, number];
	castsShadow: boolean;
	shadowIndex: number;
}

interface ClusteredLightRange {
	tileXMin: number;
	tileXMax: number;
	tileYMin: number;
	tileYMax: number;
	sliceMin: number;
	sliceMax: number;
}

export interface WebGLClusteredLightingState {
	enabled: boolean;
	screenWidth: number;
	screenHeight: number;
	tilesX: number;
	tilesY: number;
	zSlices: number;
	maxLightsPerCluster: number;
	logScale: number;
	logBias: number;
	headerTexture: WebGLTexture | null;
	headerTexWidth: number;
	headerTexHeight: number;
	indexTexture: WebGLTexture | null;
	indexTexWidth: number;
	indexTexHeight: number;
	lightTexture: WebGLTexture | null;
	lightTexWidth: number;
	lightTexHeight: number;
}

const DEFAULT_CLUSTER_OPTIONS: ClusteredRuntimeOptions = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
};

const CLUSTER_FLAG_OVERFLOW = 1;

export class WebGLClusteredLightingRuntime {
	private _gl: WebGL2RenderingContext;
	private _headerTexture: WebGLTexture | null = null;
	private _indexTexture: WebGLTexture | null = null;
	private _lightTexture: WebGLTexture | null = null;
	private _state: WebGLClusteredLightingState = {
		enabled: false,
		screenWidth: 1,
		screenHeight: 1,
		tilesX: 1,
		tilesY: 1,
		zSlices: 1,
		maxLightsPerCluster: 1,
		logScale: 0,
		logBias: 0,
		headerTexture: null,
		headerTexWidth: 1,
		headerTexHeight: 1,
		indexTexture: null,
		indexTexWidth: 1,
		indexTexHeight: 1,
		lightTexture: null,
		lightTexWidth: 1,
		lightTexHeight: 1,
	};

	constructor(gl: WebGL2RenderingContext) {
		this._gl = gl;
	}

	public getState(): WebGLClusteredLightingState {
		return {
			...this._state,
			headerTexture: this._headerTexture,
			indexTexture: this._indexTexture,
			lightTexture: this._lightTexture,
		};
	}

	public prepare(
		context: FrameContext,
		lights: WebGLLightState,
		maxTextureSize: number
	): void {
		const width = Math.max(1, Math.floor(context.attachments.width));
		const height = Math.max(1, Math.floor(context.attachments.height));
		const requested = context.features.clusteredLightingOptions ?? {};
		const options = this._resolveOptions(requested);
		const isPerspective = context.camera.type === CameraType.Perspective;

		if (!context.features.enableClusteredLighting || !context.features.enableLighting) {
			this._disable(width, height, options);
			return;
		}
		if (!isPerspective) {
			Logger.warn(
				"WebGL clustered lighting only supports perspective cameras; falling back to legacy forward lights",
				{
					scope: "WebGLClusteredLightingRuntime",
					onceKey: "webgl-clustered-perspective-only",
				}
			);
			this._disable(width, height, options);
			return;
		}

		const near = Math.max(0.05, finiteOr(context.camera.near, 0.1));
		const far = Math.max(near + 1e-3, finiteOr(context.camera.far, near + 1));
		const logDenom = Math.log(far) - Math.log(near);
		if (logDenom <= 1e-6) {
			Logger.warn(
				"WebGL clustered lighting requires a valid perspective depth range; falling back to legacy forward lights",
				{
					scope: "WebGLClusteredLightingRuntime",
					onceKey: "webgl-clustered-invalid-depth-range",
				}
			);
			this._disable(width, height, options);
			return;
		}

		const tileSizePx = options.tileSizePx;
		const tilesX = Math.max(1, Math.ceil(width / tileSizePx));
		const tilesY = Math.max(1, Math.ceil(height / tileSizePx));
		const zSlices = options.zSlices;
		const clusterCount = Math.max(1, tilesX * tilesY * zSlices);
		const logScale = zSlices / logDenom;
		const logBias = (-Math.log(near) * zSlices) / logDenom;

		const sourceLights = this._collectClusteredLights(lights);
		const maxLights = Math.max(1, Math.floor(options.maxLights));
		const maxLightsPerCluster = Math.max(1, Math.floor(options.maxLightsPerCluster));
		const activeLights = sourceLights.slice(0, maxLights);
		if (sourceLights.length > activeLights.length) {
			Logger.warn(
				`WebGL clustered lighting clamps lights to ${maxLights}; extra lights are skipped`,
				{
					scope: "WebGLClusteredLightingRuntime",
					onceKey: "webgl-clustered-light-budget",
				}
			);
		}
		if (activeLights.length <= 0) {
			this._disable(width, height, options);
			return;
		}

		const clusterLists = Array.from({ length: clusterCount }, () => [] as number[]);
		const clusterOverflow = new Uint8Array(clusterCount);
		const lightRanges: Array<ClusteredLightRange | null> = activeLights.map((light) =>
			this._resolveLightRange(
				context,
				light,
				width,
				height,
				tilesX,
				tilesY,
				zSlices,
				near,
				far,
				logScale,
				logBias
			)
		);

		for (let lightIndex = 0; lightIndex < lightRanges.length; lightIndex++) {
			const range = lightRanges[lightIndex];
			if (!range) {
				continue;
			}
			for (let z = range.sliceMin; z <= range.sliceMax; z++) {
				for (let y = range.tileYMin; y <= range.tileYMax; y++) {
					for (let x = range.tileXMin; x <= range.tileXMax; x++) {
						const clusterIndex = x + y * tilesX + z * tilesX * tilesY;
						const list = clusterLists[clusterIndex];
						if (list.length >= maxLightsPerCluster) {
							clusterOverflow[clusterIndex] = 1;
							continue;
						}
						list.push(lightIndex);
					}
				}
			}
		}

		const headerData = new Float32Array(clusterCount * 4);
		const linearIndices: number[] = [];
		for (let i = 0; i < clusterCount; i++) {
			const list = clusterLists[i];
			headerData[i * 4] = linearIndices.length;
			headerData[i * 4 + 1] = list.length;
			headerData[i * 4 + 2] = clusterOverflow[i] ? CLUSTER_FLAG_OVERFLOW : 0;
			headerData[i * 4 + 3] = 0;
			for (const lightIndex of list) {
				linearIndices.push(lightIndex);
			}
		}

		const lightData = this._createLightBuffer(activeLights);
		const indexData = this._createIndexBuffer(linearIndices);

		const headerShape = resolveTextureShape(clusterCount, maxTextureSize);
		const indexShape = resolveTextureShape(indexData.length / 4, maxTextureSize);
		const lightShape = resolveTextureShape(lightData.length / 4, maxTextureSize);
		if (!headerShape || !indexShape || !lightShape) {
			Logger.warn(
				"WebGL clustered lighting exceeded texture capacity; falling back to legacy forward lights",
				{
					scope: "WebGLClusteredLightingRuntime",
					onceKey: "webgl-clustered-texture-size-overflow",
				}
			);
			this._disable(width, height, options);
			return;
		}

		const headerPixels = padTextureData(headerData, headerShape.texelCount);
		const indexPixels = padTextureData(indexData, indexShape.texelCount);
		const lightPixels = padTextureData(lightData, lightShape.texelCount);

		try {
			this._headerTexture = uploadFloatTexture(
				this._gl,
				this._headerTexture,
				headerShape.width,
				headerShape.height,
				headerPixels
			);
			this._indexTexture = uploadFloatTexture(
				this._gl,
				this._indexTexture,
				indexShape.width,
				indexShape.height,
				indexPixels
			);
			this._lightTexture = uploadFloatTexture(
				this._gl,
				this._lightTexture,
				lightShape.width,
				lightShape.height,
				lightPixels
			);
		} catch (error) {
			Logger.warn(
				`WebGL clustered lighting upload failed; falling back to legacy forward lights (${String(error)})`,
				{
					scope: "WebGLClusteredLightingRuntime",
					onceKey: "webgl-clustered-upload-failed",
				}
			);
			this._disable(width, height, options);
			return;
		}

		this._state = {
			enabled: true,
			screenWidth: width,
			screenHeight: height,
			tilesX,
			tilesY,
			zSlices,
			maxLightsPerCluster,
			logScale,
			logBias,
			headerTexture: this._headerTexture,
			headerTexWidth: headerShape.width,
			headerTexHeight: headerShape.height,
			indexTexture: this._indexTexture,
			indexTexWidth: indexShape.width,
			indexTexHeight: indexShape.height,
			lightTexture: this._lightTexture,
			lightTexWidth: lightShape.width,
			lightTexHeight: lightShape.height,
		};
	}

	public destroy(): void {
		if (this._headerTexture) {
			this._gl.deleteTexture(this._headerTexture);
			this._headerTexture = null;
		}
		if (this._indexTexture) {
			this._gl.deleteTexture(this._indexTexture);
			this._indexTexture = null;
		}
		if (this._lightTexture) {
			this._gl.deleteTexture(this._lightTexture);
			this._lightTexture = null;
		}
		this._disable(1, 1, DEFAULT_CLUSTER_OPTIONS);
	}

	private _disable(
		width: number,
		height: number,
		options: ClusteredRuntimeOptions
	): void {
		this._state = {
			enabled: false,
			screenWidth: Math.max(1, width),
			screenHeight: Math.max(1, height),
			tilesX: Math.max(1, Math.ceil(width / Math.max(1, options.tileSizePx))),
			tilesY: Math.max(1, Math.ceil(height / Math.max(1, options.tileSizePx))),
			zSlices: Math.max(1, options.zSlices),
			maxLightsPerCluster: Math.max(1, options.maxLightsPerCluster),
			logScale: 0,
			logBias: 0,
			headerTexture: this._headerTexture,
			headerTexWidth: 1,
			headerTexHeight: 1,
			indexTexture: this._indexTexture,
			indexTexWidth: 1,
			indexTexHeight: 1,
			lightTexture: this._lightTexture,
			lightTexWidth: 1,
			lightTexHeight: 1,
		};
	}

	private _resolveOptions(
		value: FrameContext["features"]["clusteredLightingOptions"]
	): ClusteredRuntimeOptions {
		return {
			tileSizePx: Math.max(
				8,
				Math.floor(finiteOr(value?.tileSizePx, DEFAULT_CLUSTER_OPTIONS.tileSizePx))
			),
			zSlices: Math.max(
				1,
				Math.floor(finiteOr(value?.zSlices, DEFAULT_CLUSTER_OPTIONS.zSlices))
			),
			maxLights: Math.max(
				1,
				Math.floor(finiteOr(value?.maxLights, DEFAULT_CLUSTER_OPTIONS.maxLights))
			),
			maxLightsPerCluster: Math.max(
				1,
				Math.floor(
					finiteOr(
						value?.maxLightsPerCluster,
						DEFAULT_CLUSTER_OPTIONS.maxLightsPerCluster
					)
				)
			),
		};
	}

	private _collectClusteredLights(lights: WebGLLightState): ClusteredLightRecord[] {
		if ((lights.clusteredLights?.length ?? 0) > 0) {
			return lights.clusteredLights.slice();
		}

		const fallback: ClusteredLightRecord[] = [];
		for (const point of lights.pointLights) {
			fallback.push(pointToClusterRecord(point));
		}
		for (let i = 0; i < lights.spotLights.length; i++) {
			fallback.push(
				spotToClusterRecord(lights.spotLights[i], lights.spotShadows[i], i)
			);
		}
		return fallback;
	}

	private _resolveLightRange(
		context: FrameContext,
		light: ClusteredLightRecord,
		screenWidth: number,
		screenHeight: number,
		tilesX: number,
		tilesY: number,
		zSlices: number,
		near: number,
		far: number,
		logScale: number,
		logBias: number
	): ClusteredLightRange | null {
		const viewPosition = Matrix4.transformPoint(context.camera.viewMatrix, {
			x: light.position[0],
			y: light.position[1],
			z: light.position[2],
		});
		const viewDepth = -finiteOr(viewPosition.z, 0);
		const range = Math.max(1e-4, finiteOr(light.range, 0));
		if (viewDepth + range < near || viewDepth - range > far) {
			return null;
		}

		const clipPosition = Matrix4.transformPoint(context.camera.viewProjectionMatrix, {
			x: light.position[0],
			y: light.position[1],
			z: light.position[2],
		});
		const clipW = finiteOr(clipPosition.w, 0);
		if (Math.abs(clipW) < 1e-6) {
			return null;
		}
		const ndcX = clamp(finiteOr(clipPosition.x / clipW, 0), -1.5, 1.5);
		const ndcY = clamp(finiteOr(clipPosition.y / clipW, 0), -1.5, 1.5);
		const aspect = Math.max(
			1e-3,
			finiteOr(context.camera.aspectRatio, screenWidth / Math.max(screenHeight, 1))
		);
		const tanHalfFov = Math.max(
			1e-4,
			Math.tan((finiteOr(context.camera.fov, 60) * Math.PI) / 360)
		);
		const projectedDepth = Math.max(near, viewDepth);
		const radiusNdcY = clamp(range / (projectedDepth * tanHalfFov), 0, 2);
		const radiusNdcX = clamp(radiusNdcY / aspect, 0, 2);

		const tileXMin = clampInt(
			Math.floor((clamp(ndcX - radiusNdcX, -1, 1) * 0.5 + 0.5) * tilesX),
			0,
			tilesX - 1
		);
		const tileXMax = clampInt(
			Math.floor((clamp(ndcX + radiusNdcX, -1, 1) * 0.5 + 0.5) * tilesX),
			0,
			tilesX - 1
		);
		const tileYMin = clampInt(
			Math.floor((clamp(ndcY - radiusNdcY, -1, 1) * 0.5 + 0.5) * tilesY),
			0,
			tilesY - 1
		);
		const tileYMax = clampInt(
			Math.floor((clamp(ndcY + radiusNdcY, -1, 1) * 0.5 + 0.5) * tilesY),
			0,
			tilesY - 1
		);

		const zMin = clamp(Math.max(near, viewDepth - range), near, far);
		const zMax = clamp(Math.min(far, viewDepth + range), near, far);
		if (!(zMax >= zMin)) {
			return null;
		}
		const sliceMin = clampInt(Math.floor(Math.log(zMin) * logScale + logBias), 0, zSlices - 1);
		const sliceMax = clampInt(Math.floor(Math.log(zMax) * logScale + logBias), 0, zSlices - 1);

		return {
			tileXMin: Math.min(tileXMin, tileXMax),
			tileXMax: Math.max(tileXMin, tileXMax),
			tileYMin: Math.min(tileYMin, tileYMax),
			tileYMax: Math.max(tileYMin, tileYMax),
			sliceMin: Math.min(sliceMin, sliceMax),
			sliceMax: Math.max(sliceMin, sliceMax),
		};
	}

	private _createLightBuffer(lights: readonly ClusteredLightRecord[]): Float32Array {
		const data = new Float32Array(Math.max(1, lights.length) * 16);
		for (let i = 0; i < lights.length; i++) {
			const light = lights[i];
			const base = i * 16;
			data[base] = finiteOr(light.position[0], 0);
			data[base + 1] = finiteOr(light.position[1], 0);
			data[base + 2] = finiteOr(light.position[2], 0);
			data[base + 3] = finiteOr(light.range, 0);

			data[base + 4] = finiteOr(light.direction[0], 0);
			data[base + 5] = finiteOr(light.direction[1], 0);
			data[base + 6] = finiteOr(light.direction[2], 0);
			data[base + 7] = finiteOr(light.outerCos, -2);

			data[base + 8] = finiteOr(light.color[0], 0);
			data[base + 9] = finiteOr(light.color[1], 0);
			data[base + 10] = finiteOr(light.color[2], 0);
			data[base + 11] = finiteOr(light.innerCos, -2);

			data[base + 12] = light.type;
			data[base + 13] = light.castsShadow ? 1 : 0;
			data[base + 14] = Math.max(0, light.shadowIndex | 0);
			data[base + 15] = 0;
		}
		return data;
	}

	private _createIndexBuffer(indices: readonly number[]): Float32Array {
		const texelCount = Math.max(1, Math.ceil(indices.length / 4));
		const data = new Float32Array(texelCount * 4);
		data.fill(-1);
		for (let i = 0; i < indices.length; i++) {
			data[i] = indices[i];
		}
		return data;
	}
}

function pointToClusterRecord(point: WebGLPointLight): ClusteredLightRecord {
	return {
		type: 0,
		position: [point.position[0], point.position[1], point.position[2]],
		range: point.range,
		direction: [0, 0, 0],
		outerCos: -2,
		innerCos: -2,
		color: [point.color[0], point.color[1], point.color[2]],
		castsShadow: false,
		shadowIndex: 0,
	};
}

function spotToClusterRecord(
	spot: WebGLSpotLight,
	shadow: WebGLShadowData | undefined,
	shadowIndex: number
): ClusteredLightRecord {
	return {
		type: 1,
		position: [spot.position[0], spot.position[1], spot.position[2]],
		range: spot.range,
		direction: [spot.direction[0], spot.direction[1], spot.direction[2]],
		outerCos: spot.outerCos,
		innerCos: spot.innerCos,
		color: [spot.color[0], spot.color[1], spot.color[2]],
		castsShadow: shadow?.enabled === true,
		shadowIndex: Math.max(0, shadowIndex | 0),
	};
}

function uploadFloatTexture(
	gl: WebGL2RenderingContext,
	texture: WebGLTexture | null,
	width: number,
	height: number,
	pixels: Float32Array
): WebGLTexture {
	const next = texture ?? gl.createTexture();
	if (!next) {
		throw new Error("Failed to create clustered lighting texture.");
	}
	const internalFormat = (gl as WebGL2RenderingContext & { RGBA32F?: number }).RGBA32F ?? gl.RGBA;
	gl.bindTexture(gl.TEXTURE_2D, next);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		internalFormat,
		width,
		height,
		0,
		gl.RGBA,
		gl.FLOAT,
		pixels
	);
	return next;
}

function resolveTextureShape(
	texelCount: number,
	maxTextureSize: number
): { width: number; height: number; texelCount: number } | null {
	const count = Math.max(1, Math.ceil(texelCount));
	const width = Math.min(maxTextureSize, count);
	const height = Math.ceil(count / Math.max(1, width));
	if (height > maxTextureSize) {
		return null;
	}
	return {
		width,
		height,
		texelCount: width * height,
	};
}

function padTextureData(data: Float32Array, texelCapacity: number): Float32Array {
	const required = texelCapacity * 4;
	if (data.length === required) {
		return data;
	}
	const output = new Float32Array(required);
	output.set(data.subarray(0, Math.min(data.length, required)));
	return output;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function clampInt(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value | 0));
}

function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
