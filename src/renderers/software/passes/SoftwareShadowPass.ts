import { clamp } from "../../../maths/Common";
import { Matrix4 } from "../../../maths/Matrix4";
import type { IVector3 } from "../../../maths/types";
import { Vector3 } from "../../../maths/Vector3";
import type { RGB } from "../../../foundation/Color";
import { isShadowCastingLight } from "../../../lights";
import type { ShadowCastingLight } from "../../../lights";
import {
	getPrimaryShadowMap,
	type ShadowMap,
	type ShadowParams,
	type ShadowRenderSet,
} from "../../../lights/ShadowMapping";
import type { IVertex, ProjectedVertex } from "../../../core/types";
import {
	defineTransientKey,
	type FrameContext,
	type TransientStore,
} from "../../../pipeline/types";
import { Projector } from "../Projector";
import type { Rasterizer } from "../Rasterizer";
import type { SoftwarePassLike } from "./types";

export const SOFTWARE_SHADOW_RUNTIME_KEY =
	defineTransientKey<SoftwareShadowRuntimeMap>("software-shadow-runtime");

export interface SoftwareShadowRenderTarget {
	size: number;
	depthBuffer: Float32Array;
	transmissionBuffer: Float32Array;
}

export type SoftwareShadowRuntimeMap = Map<ShadowCastingLight, SoftwareShadowRenderTarget[]>;

interface SoftwareShadowSampleContext {
	worldPoint: IVector3;
	normal?: IVector3 | null;
	shadowMap: ShadowMap;
	runtimeTarget: SoftwareShadowRenderTarget;
	params: ShadowParams;
}

interface ClipVertex {
	x: number;
	y: number;
	z: number;
	w: number;
	u: number;
	v: number;
}

/**
 * Clip-space and frustum constants for software shadow rasterization/sampling.
 */
export class SoftwareShadowConstants {
	static readonly MIN_CLIP_W = 1e-6;
	static readonly MIN_NDC_DEPTH = -1.0;
	static readonly MAX_NDC_DEPTH = 1.0;

	static readonly CLIP_PLANE_MIN_W = 0;
	static readonly CLIP_PLANE_LEFT = 1;
	static readonly CLIP_PLANE_RIGHT = 2;
	static readonly CLIP_PLANE_BOTTOM = 3;
	static readonly CLIP_PLANE_TOP = 4;
	static readonly CLIP_PLANE_NEAR = 5;
	static readonly CLIP_PLANE_FAR = 6;
	static readonly CLIP_PLANE_COUNT = 7;
	static readonly CLIP_EPSILON = 1e-12;
}

export function getSoftwareShadowRuntimeMap(
	transient: TransientStore
): SoftwareShadowRuntimeMap | null {
	return transient.get(SOFTWARE_SHADOW_RUNTIME_KEY) ?? null;
}

export function setSoftwareShadowRuntimeMap(
	transient: TransientStore,
	runtimeMap: SoftwareShadowRuntimeMap
): void {
	transient.set(SOFTWARE_SHADOW_RUNTIME_KEY, runtimeMap);
}

export function ensureSoftwareShadowRenderTarget(
	runtimeMap: SoftwareShadowRuntimeMap,
	light: ShadowCastingLight,
	sliceIndex: number,
	size: number,
): SoftwareShadowRenderTarget {
	let targets = runtimeMap.get(light);
	if (!targets) {
		targets = [];
		runtimeMap.set(light, targets);
	}
	let target = targets[sliceIndex];
	if (!target || target.size !== size) {
		target = {
			size,
			depthBuffer: new Float32Array(size * size),
			transmissionBuffer: new Float32Array(size * size * 3),
		};
		targets[sliceIndex] = target;
	}

	return target;
}

export function clearSoftwareShadowRenderTarget(target: SoftwareShadowRenderTarget): void {
	target.depthBuffer.fill(Infinity);
	target.transmissionBuffer.fill(1.0);
}

export function syncSoftwareShadowRuntimeMap(
	runtimeMap: SoftwareShadowRuntimeMap,
	activeLights: ShadowCastingLight[],
): void {
	for (const [light] of runtimeMap) {
		if (!activeLights.includes(light)) {
			runtimeMap.delete(light);
		}
	}
}

export function trimSoftwareShadowRuntimeTargets(
	runtimeMap: SoftwareShadowRuntimeMap,
	light: ShadowCastingLight,
	sliceCount: number,
): void {
	const targets = runtimeMap.get(light);
	if (!targets) {
		return;
	}
	targets.length = Math.max(0, sliceCount | 0);
}

function isWorldPointInsideShadowMap(shadowMap: ShadowMap, worldPoint: IVector3): boolean {
	if (!shadowMap.viewProjectionMatrix) {
		return false;
	}
	const clip = Matrix4.transformPoint(shadowMap.viewProjectionMatrix, worldPoint);
	if (clip.w <= 1e-6) {
		return false;
	}
	const invW = 1 / clip.w;
	const ndcX = clip.x * invW;
	const ndcY = clip.y * invW;
	const ndcZ = clip.z * invW;
	const uvX = ndcX * 0.5 + 0.5;
	const uvY = 0.5 - ndcY * 0.5;
	return uvX >= 0 && uvX <= 1 && uvY >= 0 && uvY <= 1 && ndcZ >= -1 && ndcZ <= 1;
}

function sampleFromRenderSet(
	renderSet: ShadowRenderSet,
	runtimeTargets: SoftwareShadowRenderTarget[] | undefined,
	worldPoint: IVector3,
	normal?: IVector3 | null,
): RGB {
	if (!runtimeTargets || runtimeTargets.length <= 0) {
		return { r: 1, g: 1, b: 1 };
	}

	const isCSM = renderSet.effectiveStrategyType === "csm" && renderSet.slices.length > 1;
	if (!isCSM) {
		const shadowMap = getPrimaryShadowMap(renderSet);
		const runtimeTarget = runtimeTargets[0];
		if (!shadowMap || !runtimeTarget) {
			return { r: 1, g: 1, b: 1 };
		}
		return sampleSoftwareShadow(shadowMap, runtimeTarget, worldPoint, normal);
	}

	let fallbackShadowMap: ShadowMap | null = null;
	let fallbackRuntimeTarget: SoftwareShadowRenderTarget | null = null;
	for (let index = 0; index < renderSet.slices.length; index++) {
		const slice = renderSet.slices[index];
		const runtimeTarget = runtimeTargets[index];
		if (!runtimeTarget || !slice.shadowMap.viewProjectionMatrix) {
			continue;
		}
		if (!fallbackShadowMap) {
			fallbackShadowMap = slice.shadowMap;
			fallbackRuntimeTarget = runtimeTarget;
		}
		if (isWorldPointInsideShadowMap(slice.shadowMap, worldPoint)) {
			return sampleSoftwareShadow(slice.shadowMap, runtimeTarget, worldPoint, normal);
		}
	}

	if (fallbackShadowMap && fallbackRuntimeTarget) {
		return sampleSoftwareShadow(fallbackShadowMap, fallbackRuntimeTarget, worldPoint, normal);
	}

	return { r: 1, g: 1, b: 1 };
}

export function createSoftwareShadowSampler(
	shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>,
	runtimeMap: SoftwareShadowRuntimeMap | null,
): (light: ShadowCastingLight, worldPoint: IVector3, normal?: IVector3 | null) => RGB {
	return (light: ShadowCastingLight, worldPoint: IVector3, normal?: IVector3 | null): RGB => {
		if (!runtimeMap) return { r: 1, g: 1, b: 1 };

		const shadowRenderSet = shadowMaps.get(light);
		if (!shadowRenderSet) return { r: 1, g: 1, b: 1 };
		const runtimeTargets = runtimeMap.get(light);
		return sampleFromRenderSet(shadowRenderSet, runtimeTargets, worldPoint, normal);
	};
}

function getVogelSample(index: number, numSamples: number, theta: number) {
	const goldenAngle = 2.400049405230919;
	const r = Math.sqrt((index + 0.5) / numSamples);
	const angle = index * goldenAngle + theta;
	return { x: r * Math.cos(angle), y: r * Math.sin(angle) };
}

function calculateShadowFactor(ctx: SoftwareShadowSampleContext): RGB {
	const { worldPoint, normal, shadowMap, runtimeTarget, params } = ctx;
	const { viewProjectionMatrix, latestLightDir, projectionMatrix } = shadowMap;
	const buffer = runtimeTarget.depthBuffer;
	const transmissionBuffer = runtimeTarget.transmissionBuffer;
	const size = runtimeTarget.size;

	if (!viewProjectionMatrix) return { r: 1.0, g: 1.0, b: 1.0 };

	const L = Vector3.normalize({
		x: -latestLightDir.x,
		y: -latestLightDir.y,
		z: -latestLightDir.z,
	});

	const normalBias = params.shadowNormalBias ?? 1.0;
	const normalBiasMin = params.shadowNormalBiasMin ?? 0.05;

	let offsetPoint = worldPoint;
	if (normal) {
		const N = Vector3.normalize(normal);
		const cosTheta = Math.max(0, Vector3.dot(N, L));
		const normalOffset = normalBiasMin + (normalBias - normalBiasMin) * (1.0 - cosTheta);
		offsetPoint = {
			x: worldPoint.x + N.x * normalOffset,
			y: worldPoint.y + N.y * normalOffset,
			z: worldPoint.z + N.z * normalOffset,
		};
	} else {
		const volumeOffset = normalBiasMin;
		offsetPoint = {
			x: worldPoint.x + L.x * volumeOffset,
			y: worldPoint.y + L.y * volumeOffset,
			z: worldPoint.z + L.z * volumeOffset,
		};
	}

	const lightSpacePos = Matrix4.transformPoint(viewProjectionMatrix, offsetPoint);
	const w = lightSpacePos.w;
	if (w <= SoftwareShadowConstants.MIN_CLIP_W) {
		return { r: 1.0, g: 1.0, b: 1.0 };
	}
	const invW = 1 / w;
	const ndcX = lightSpacePos.x * invW;
	const ndcY = lightSpacePos.y * invW;
	const ndcZ = lightSpacePos.z * invW;

	const u = ndcX * 0.5 + 0.5;
	const v = 0.5 - ndcY * 0.5;
	const currentDepth = ndcZ;

	if (
		u < 0 ||
		u > 1 ||
		v < 0 ||
		v > 1 ||
		currentDepth < SoftwareShadowConstants.MIN_NDC_DEPTH ||
		currentDepth > SoftwareShadowConstants.MAX_NDC_DEPTH
	) {
		return { r: 1.0, g: 1.0, b: 1.0 };
	}

	const constantBias = params.shadowBias ?? 0.008;
	const slopeBias = params.shadowSlopeBias ?? 0.03;
	const texelBias = (params.shadowTexelBias ?? 1.0) * (2.0 / size);
	const maxBias = params.shadowMaxBias ?? 0.05;

	const m = projectionMatrix ? projectionMatrix.elements : null;
	const isPerspective = m ? Math.abs(m[3][2] + 1.0) < 1e-6 : false;
	const linearizeDepth = (zNdc: number): number => {
		if (!m) return zNdc;
		if (isPerspective) {
			return m[2][3] / (zNdc + m[2][2]);
		}
		return (m[2][3] - zNdc) / m[2][2];
	};

	const bias = normal
		? Math.min(
				maxBias,
				constantBias +
					slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
					texelBias,
			)
		: Math.min(maxBias, constantBias + texelBias);

	const strength = clamp(params.shadowStrength ?? 1.0);
	const pcfRadiusParams = params.shadowRadius ?? 0;
	const texelSize = 1.0 / size;

	let visibilityR = 0;
	let visibilityG = 0;
	let visibilityB = 0;
	let validSampleCount = 0;

	if (pcfRadiusParams > 0) {
		const theta =
			(worldPoint.x * 12.9898 + worldPoint.y * 78.233 + worldPoint.z * 37.719) %
			(Math.PI * 2);
		const numSearchSamples = Math.floor(params.shadowSearchSamples ?? 16);
		const numSamples = Math.floor(params.shadowSamples ?? 16);
		const maxRadiusUV = pcfRadiusParams * texelSize;

		let numBlockers = 0;
		let avgBlockerDepth = 0;
		for (let i = 0; i < numSearchSamples; i++) {
			const offset = getVogelSample(i, numSearchSamples, theta);
			const su = u + offset.x * maxRadiusUV;
			const sv = v + offset.y * maxRadiusUV;
			if (su >= 0 && su <= 1 && sv >= 0 && sv <= 1) {
				const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
				const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
				const shadowDepth = buffer[ty * size + tx];
				if (currentDepth - bias > shadowDepth) {
					numBlockers++;
					avgBlockerDepth += shadowDepth;
				}
			}
		}

		if (numBlockers === 0) {
			return { r: 1.0, g: 1.0, b: 1.0 };
		}

		avgBlockerDepth /= numBlockers;
		const linCurrent = linearizeDepth(currentDepth);
		const linBlocker = linearizeDepth(avgBlockerDepth);

		let penumbraRatio = 1.0;
		if (linCurrent > linBlocker) {
			const divergence = isPerspective ? linBlocker || 1e-6 : 100.0;
			penumbraRatio = (linCurrent - linBlocker) / divergence;
			penumbraRatio = Math.max(0.0, Math.min(1.0, penumbraRatio));
		} else {
			penumbraRatio = 0;
		}

		const filterRadiusUV = maxRadiusUV * penumbraRatio;
		if (filterRadiusUV < texelSize * 0.1) {
			return calculateShadowFactor({
				...ctx,
				params: {
					...params,
					shadowRadius: 0,
				},
			});
		}

		for (let i = 0; i < numSamples; i++) {
			const offset = getVogelSample(i, numSamples, theta);
			const su = u + offset.x * filterRadiusUV;
			const sv = v + offset.y * filterRadiusUV;
			if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

			const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
			const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
			const idx = ty * size + tx;
			const shadowDepth = buffer[idx];

			validSampleCount++;
			const isOccluded = currentDepth - bias > shadowDepth;
			if (isOccluded) {
				visibilityR += 1.0 - strength;
				visibilityG += 1.0 - strength;
				visibilityB += 1.0 - strength;
				continue;
			}

			const cIdx = idx * 3;
			const transSampleR = transmissionBuffer[cIdx];
			const transSampleG = transmissionBuffer[cIdx + 1];
			const transSampleB = transmissionBuffer[cIdx + 2];
			visibilityR += 1.0 - strength + strength * transSampleR;
			visibilityG += 1.0 - strength + strength * transSampleG;
			visibilityB += 1.0 - strength + strength * transSampleB;
		}
	} else {
		const theta =
			(worldPoint.x * 12.9898 + worldPoint.y * 78.233 + worldPoint.z * 37.719) %
			(Math.PI * 2);
		const pcfRadius = params.shadowPCF ?? 1.5;
		const numSamples = Math.floor(params.shadowSamples ?? 16);
		const radiusUV = pcfRadius * texelSize;

		for (let i = 0; i < numSamples; i++) {
			const offset = getVogelSample(i, numSamples, theta);
			const su = u + offset.x * radiusUV;
			const sv = v + offset.y * radiusUV;
			if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

			const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
			const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
			const idx = ty * size + tx;
			const shadowDepth = buffer[idx];

			validSampleCount++;
			const isOccluded = currentDepth - bias > shadowDepth;
			if (isOccluded) {
				visibilityR += 1.0 - strength;
				visibilityG += 1.0 - strength;
				visibilityB += 1.0 - strength;
				continue;
			}

			const cIdx = idx * 3;
			const transSampleR = transmissionBuffer[cIdx];
			const transSampleG = transmissionBuffer[cIdx + 1];
			const transSampleB = transmissionBuffer[cIdx + 2];
			visibilityR += 1.0 - strength + strength * transSampleR;
			visibilityG += 1.0 - strength + strength * transSampleG;
			visibilityB += 1.0 - strength + strength * transSampleB;
		}
	}

	if (validSampleCount === 0) return { r: 1.0, g: 1.0, b: 1.0 };

	const invCount = 1.0 / validSampleCount;
	return {
		r: clamp(visibilityR * invCount),
		g: clamp(visibilityG * invCount),
		b: clamp(visibilityB * invCount),
	};
}

export function sampleSoftwareShadow(
	shadowMap: ShadowMap,
	runtimeTarget: SoftwareShadowRenderTarget,
	worldPoint: IVector3,
	normal?: IVector3 | null,
): RGB {
	return calculateShadowFactor({
		worldPoint,
		normal,
		shadowMap,
		runtimeTarget,
		params: shadowMap.params,
	});
}

export class SoftwareShadowPass implements SoftwarePassLike {
	private _rasterizer: Rasterizer;
	private _mvpMatrix = Matrix4.identity();
	private _lightDirModel = new Vector3();
	private _shadowLightsScratch: ShadowCastingLight[] = [];
	private _projectedVertsPool: ProjectedVertex[] = [];
	private _projectedVertsView: ProjectedVertex[] = [];
	private _clipInputPool: ClipVertex[] = [];
	private _clipVertsPool: ClipVertex[] = [];
	private _clipPoolCursor = 0;
	private _clipScratchA: ClipVertex[] = [];
	private _clipScratchB: ClipVertex[] = [];
	private _runtimeShadowMaps: SoftwareShadowRuntimeMap = new Map();

	constructor(rasterizer: Rasterizer) {
		this._rasterizer = rasterizer;
		for (let i = 0; i < 4; i++) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}
	}

	public render(context: FrameContext): void {
		const features = context.features;
		if (!features.enableShadows) {
			this._runtimeShadowMaps.clear();
			setSoftwareShadowRuntimeMap(context.transient, this._runtimeShadowMaps);
			return;
		}

		const frame = context.scene;
		const shadowMaps = context.shadowMaps;
		const shadowLights = this._shadowLightsScratch;
		shadowLights.length = 0;
		for (const light of frame.lights) {
			if (isShadowCastingLight(light)) {
				shadowLights.push(light);
			}
		}
		syncSoftwareShadowRuntimeMap(this._runtimeShadowMaps, shadowLights);
		setSoftwareShadowRuntimeMap(context.transient, this._runtimeShadowMaps);

		if (shadowLights.length === 0) {
			this._runtimeShadowMaps.clear();
			return;
		}

		for (const shadowLight of shadowLights) {
			const shadowRenderSet = shadowMaps.get(shadowLight);
			if (!shadowRenderSet || shadowRenderSet.slices.length <= 0) {
				trimSoftwareShadowRuntimeTargets(this._runtimeShadowMaps, shadowLight, 0);
				continue;
			}
			trimSoftwareShadowRuntimeTargets(
				this._runtimeShadowMaps,
				shadowLight,
				shadowRenderSet.slices.length,
			);

			for (let sliceIndex = 0; sliceIndex < shadowRenderSet.slices.length; sliceIndex++) {
				const shadowSlice = shadowRenderSet.slices[sliceIndex];
				const shadowMap = shadowSlice.shadowMap;
				const vpMatrix = shadowMap.viewProjectionMatrix;
				if (!vpMatrix) {
					continue;
				}

				const lightDir = shadowMap.latestLightDir;
				const shadowMapSize = shadowMap.size;
				const shadowRuntime = ensureSoftwareShadowRenderTarget(
					this._runtimeShadowMaps,
					shadowLight,
					sliceIndex,
					shadowMapSize,
				);
				clearSoftwareShadowRenderTarget(shadowRuntime);

				for (const packet of frame.shadowCasterPackets) {
					Matrix4.multiply(vpMatrix, packet.worldMatrix, this._mvpMatrix);
					const inv3x3 = Matrix4.inverse3x3(packet.worldMatrix);
					if (!inv3x3) continue;

					Matrix4.transformNormal(inv3x3, lightDir, this._lightDirModel);

					for (const face of Projector.getPacketFacesWithContext(packet, context)) {
						const dot = Vector3.dot(
							face.normal ?? Vector3.calculateNormal(face.vertices),
							this._lightDirModel,
						);
						if (!packet.material.doubleSided && dot > 0) continue;

						const projected = this._projectFace(face.vertices, shadowMapSize);
						if (!projected) continue;

						this._rasterizer.drawDepthTriangle(
							projected,
							shadowRuntime,
							packet.material,
						);
					}
				}

				for (const packet of frame.shadowTransmitterPackets) {
					Matrix4.multiply(vpMatrix, packet.worldMatrix, this._mvpMatrix);

					for (const face of Projector.getPacketFacesWithContext(packet, context)) {
						const projected = this._projectFace(face.vertices, shadowMapSize);
						if (!projected) continue;

						this._rasterizer.drawTransmissionTriangle(
							projected,
							{
								...face,
								projected,
								center: packet.worldBounds.center,
								depthInfo: { min: 0, max: 0, avg: 0 },
							},
							shadowRuntime,
						);
					}
				}
			}
		}
	}

	private _allocClipVertex(
		x: number,
		y: number,
		z: number,
		w: number,
		uCoord: number = 0,
		vCoord: number = 0,
	): ClipVertex {
		let clipVert = this._clipVertsPool[this._clipPoolCursor];
		if (!clipVert) {
			clipVert = { x: 0, y: 0, z: 0, w: 0, u: 0, v: 0 };
			this._clipVertsPool.push(clipVert);
		}

		clipVert.x = x;
		clipVert.y = y;
		clipVert.z = z;
		clipVert.w = w;
		clipVert.u = uCoord;
		clipVert.v = vCoord;
		this._clipPoolCursor++;
		return clipVert;
	}

	private _clipDistance(vertex: ClipVertex, plane: number): number {
		switch (plane) {
			case SoftwareShadowConstants.CLIP_PLANE_MIN_W:
				return vertex.w - SoftwareShadowConstants.MIN_CLIP_W;
			case SoftwareShadowConstants.CLIP_PLANE_LEFT:
				return vertex.x + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_RIGHT:
				return -vertex.x + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_BOTTOM:
				return vertex.y + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_TOP:
				return -vertex.y + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_NEAR:
				return vertex.z + vertex.w;
			case SoftwareShadowConstants.CLIP_PLANE_FAR:
				return -vertex.z + vertex.w;
			default:
				return -1;
		}
	}

	private _clipAgainstPlane(input: ClipVertex[], output: ClipVertex[], plane: number): void {
		output.length = 0;
		if (input.length === 0) return;

		let previous = input[input.length - 1];
		let previousDistance = this._clipDistance(previous, plane);
		let previousInside = previousDistance >= 0;

		for (let i = 0; i < input.length; i++) {
			const current = input[i];
			const currentDistance = this._clipDistance(current, plane);
			const currentInside = currentDistance >= 0;

			if (currentInside !== previousInside) {
				const denominator = previousDistance - currentDistance;
				const t =
					Math.abs(denominator) > SoftwareShadowConstants.CLIP_EPSILON
						? previousDistance / denominator
						: 0;
				output.push(
					this._allocClipVertex(
						previous.x + (current.x - previous.x) * t,
						previous.y + (current.y - previous.y) * t,
						previous.z + (current.z - previous.z) * t,
						previous.w + (current.w - previous.w) * t,
						previous.u + (current.u - previous.u) * t,
						previous.v + (current.v - previous.v) * t,
					),
				);
			}

			if (currentInside) {
				output.push(
					this._allocClipVertex(
						current.x,
						current.y,
						current.z,
						current.w,
						current.u,
						current.v,
					),
				);
			}

			previous = current;
			previousDistance = currentDistance;
			previousInside = currentInside;
		}
	}

	private _clipToLightFrustum(input: ClipVertex[], count: number): ClipVertex[] {
		this._clipPoolCursor = 0;
		this._clipScratchA.length = 0;
		this._clipScratchB.length = 0;

		for (let i = 0; i < count; i++) {
			const vertex = input[i];
			this._clipScratchA.push(
				this._allocClipVertex(vertex.x, vertex.y, vertex.z, vertex.w, vertex.u, vertex.v),
			);
		}

		let inPolygon = this._clipScratchA;
		let outPolygon = this._clipScratchB;

		for (let plane = 0; plane < SoftwareShadowConstants.CLIP_PLANE_COUNT; plane++) {
			this._clipAgainstPlane(inPolygon, outPolygon, plane);
			if (outPolygon.length < 3) return outPolygon;
			const temp = inPolygon;
			inPolygon = outPolygon;
			outPolygon = temp;
		}

		return inPolygon;
	}

	private _projectFace(vertices: IVertex[], shadowMapSize: number): ProjectedVertex[] | null {
		const count = vertices.length;

		while (this._projectedVertsPool.length < count) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}
		while (this._clipInputPool.length < count) {
			this._clipInputPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				u: 0,
				v: 0,
			});
		}

		let allInside = true;
		let initialOutCodes = -1;

		for (let i = 0; i < count; i++) {
			const vertex = vertices[i];
			const projected = Matrix4.transformPoint(this._mvpMatrix, vertex);
			const clipVertex = this._clipInputPool[i];
			clipVertex.x = projected.x;
			clipVertex.y = projected.y;
			clipVertex.z = projected.z;
			clipVertex.w = projected.w;
			clipVertex.u = vertex.u ?? 0;
			clipVertex.v = vertex.v ?? 0;

			let code = 0;
			if (clipVertex.w < SoftwareShadowConstants.MIN_CLIP_W) code |= 1;
			if (clipVertex.x < -clipVertex.w) code |= 2;
			if (clipVertex.x > clipVertex.w) code |= 4;
			if (clipVertex.y < -clipVertex.w) code |= 8;
			if (clipVertex.y > clipVertex.w) code |= 16;
			if (clipVertex.z < -clipVertex.w) code |= 32;
			if (clipVertex.z > clipVertex.w) code |= 64;

			if (code !== 0) allInside = false;
			if (initialOutCodes === -1) {
				initialOutCodes = code;
			} else {
				initialOutCodes &= code;
			}
		}

		if (initialOutCodes !== 0) return null;

		let clippedVertices: ClipVertex[];
		let clippedCount: number;

		if (allInside) {
			clippedVertices = this._clipInputPool;
			clippedCount = count;
		} else {
			const result = this._clipToLightFrustum(this._clipInputPool, count);
			clippedVertices = result;
			clippedCount = result.length;
			if (clippedCount < 3) return null;
		}

		while (this._projectedVertsPool.length < clippedCount) {
			this._projectedVertsPool.push({
				x: 0,
				y: 0,
				z: 0,
				w: 0,
				world: { x: 0, y: 0, z: 0 },
			});
		}

		const activeVertices = this._projectedVertsPool;
		const projectedView = this._projectedVertsView;
		for (let i = 0; i < clippedCount; i++) {
			const clipVertex = clippedVertices[i];
			const outputVertex = activeVertices[i];
			const invW = 1 / clipVertex.w;
			outputVertex.x = (clipVertex.x * invW * 0.5 + 0.5) * shadowMapSize;
			outputVertex.y = (0.5 - clipVertex.y * invW * 0.5) * shadowMapSize;
			outputVertex.z = clipVertex.z * invW;
			outputVertex.w = invW;
			outputVertex.u = clipVertex.u;
			outputVertex.v = clipVertex.v;
			projectedView[i] = outputVertex;
		}

		projectedView.length = clippedCount;
		return projectedView;
	}
}
