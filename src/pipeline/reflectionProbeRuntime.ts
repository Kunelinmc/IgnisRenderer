import { Texture } from "../core/Texture";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import {
	LightType,
	type ReflectionProbe,
	type ReflectionProbeParallaxMode,
	type ReflectionProbeShape,
	type SceneLight,
} from "../lights";
import {
	directionToEquirectUV as directionToEquirectUVFromEnvironmentMap,
	ensureEnvironmentTextureEquirect,
	getEnvironmentMipLevelCount,
	isTextureReadyForEnvironment as isTextureReadyForEnvironmentShared,
	sampleEnvironmentTextureSpecular,
} from "./environmentMapRuntime";

export const MAX_ACTIVE_REFLECTION_PROBES = 8;
export const REFLECTION_PROBE_WEIGHT_EPSILON = 1e-6;
const REFLECTION_PROBE_RAY_EPSILON = 1e-5;

const _atlasCacheByKey = new Map<string, Texture>();
const _textureIdentityByTexture = new WeakMap<Texture, number>();
let _nextTextureIdentity = 0;

export interface ReflectionProbeSelectionResult {
	firstIndex: number;
	secondIndex: number;
	firstWeight: number;
	secondWeight: number;
}

export interface ReflectionProbeSampleResult {
	direction: IVector3;
	valid: boolean;
}

export interface ReflectionProbeEnvironmentCollection {
	probes: ReflectionProbe[];
	atlas: Texture | null;
}

interface ReflectionProbeAtlasCandidate {
	probe: ReflectionProbe;
	width: number;
	height: number;
	mipCount: number;
	qualityScore: number;
}

export function collectActiveReflectionProbes(
	lights: SceneLight[]
): ReflectionProbe[] {
	const probes: ReflectionProbe[] = [];
	for (const light of lights) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		if (!isTextureReadyForEnvironment(probe.prefilteredMap)) continue;
		probe.getRuntimeCache();
		probes.push(probe);
	}

	probes.sort((left, right) => left.id.localeCompare(right.id));
	return probes;
}

export function collectReflectionProbeEnvironment(
	lights: SceneLight[],
	maxCount = MAX_ACTIVE_REFLECTION_PROBES,
	worldPosition: IVector3 | null = null
): ReflectionProbeEnvironmentCollection {
	const probes = collectActiveReflectionProbes(lights);
	if (probes.length <= 0) {
		return {
			probes,
			atlas: null,
		};
	}
	const atlasCompatibleProbes = limitReflectionProbeEnvironmentProbes(
		selectAtlasCompatibleReflectionProbes(probes),
		maxCount,
		worldPosition
	);
	return {
		probes: atlasCompatibleProbes,
		atlas: buildReflectionProbeAtlasTexture(atlasCompatibleProbes),
	};
}

function limitReflectionProbeEnvironmentProbes(
	probes: ReflectionProbe[],
	maxCount: number,
	worldPosition: IVector3 | null
): ReflectionProbe[] {
	const resolvedMaxCount = Number.isFinite(maxCount) ?
			Math.max(0, Math.floor(maxCount))
		:	MAX_ACTIVE_REFLECTION_PROBES;
	if (resolvedMaxCount <= 0) {
		return [];
	}
	if (probes.length <= resolvedMaxCount) {
		return probes.slice();
	}
	if (!worldPosition) {
		return probes.slice(0, resolvedMaxCount);
	}

	const ranked = probes.map((probe) => {
		const cache = probe.getRuntimeCache();
		const metric = computeProbeMetric(worldPosition, probe);
		const weight = computeProbeRawWeight(
			metric,
			cache.effectiveBlendDistance,
			cache.blendExponent
		);
		return {
			probe,
			metric,
			weight: Number.isFinite(weight) ? weight : 0,
		};
	});

	ranked.sort((left, right) => {
		const leftActive = left.weight > REFLECTION_PROBE_WEIGHT_EPSILON;
		const rightActive = right.weight > REFLECTION_PROBE_WEIGHT_EPSILON;
		if (leftActive !== rightActive) {
			return leftActive ? -1 : 1;
		}
		if (Math.abs(left.weight - right.weight) > REFLECTION_PROBE_WEIGHT_EPSILON) {
			return right.weight - left.weight;
		}
		if (left.metric !== right.metric) {
			return left.metric - right.metric;
		}
		return left.probe.id.localeCompare(right.probe.id);
	});

	const selected = ranked
		.slice(0, resolvedMaxCount)
		.map((entry) => entry.probe);
	selected.sort((left, right) => left.id.localeCompare(right.id));
	return selected;
}

function selectAtlasCompatibleReflectionProbes(
	probes: ReflectionProbe[]
): ReflectionProbe[] {
	if (probes.length <= 1) {
		return probes;
	}

	const candidates = collectReflectionProbeAtlasCandidates(probes);
	if (candidates.length <= 1) {
		return candidates.map((candidate) => candidate.probe);
	}

	const groupsBySignature = new Map<string, ReflectionProbeAtlasCandidate[]>();
	for (const candidate of candidates) {
		const signature = `${candidate.width}x${candidate.height}:${candidate.mipCount}`;
		const group = groupsBySignature.get(signature);
		if (group) {
			group.push(candidate);
			continue;
		}
		groupsBySignature.set(signature, [candidate]);
	}

	let bestGroup: ReflectionProbeAtlasCandidate[] | null = null;
	for (const group of groupsBySignature.values()) {
		if (!bestGroup) {
			bestGroup = group;
			continue;
		}
		const groupScore = computeReflectionProbeAtlasGroupScore(group);
		const bestScore = computeReflectionProbeAtlasGroupScore(bestGroup);
		if (
			group.length > bestGroup.length ||
			(group.length === bestGroup.length && groupScore > bestScore)
		) {
			bestGroup = group;
		}
	}

	return (bestGroup ?? candidates).map((candidate) => candidate.probe);
}

function computeReflectionProbeAtlasGroupScore(
	group: ReflectionProbeAtlasCandidate[]
): number {
	if (group.length <= 0) {
		return 0;
	}
	return group[0].qualityScore;
}

function collectReflectionProbeAtlasCandidates(
	probes: ReflectionProbe[]
): ReflectionProbeAtlasCandidate[] {
	const candidates: ReflectionProbeAtlasCandidate[] = [];
	for (const probe of probes) {
		const sourceMap = probe.prefilteredMap;
		if (!sourceMap) continue;
		const normalizedMap = ensureEnvironmentTextureEquirect(sourceMap);
		if (!isTextureReadyForEnvironmentShared(normalizedMap)) {
			continue;
		}
		candidates.push({
			probe,
			width: normalizedMap.width,
			height: normalizedMap.height,
			mipCount: getEnvironmentMipLevelCount(normalizedMap),
			qualityScore:
				normalizedMap.width *
				normalizedMap.height *
				getEnvironmentMipLevelCount(normalizedMap),
		});
	}
	return candidates;
}

export function refreshReflectionProbeCaches(lights: SceneLight[]): void {
	for (const light of lights) {
		if (light.type !== LightType.ReflectionProbe) continue;
		const probe = light as ReflectionProbe;
		probe.refreshRuntimeCache();
	}
}

export function selectTopTwoReflectionProbes(
	worldPosition: IVector3,
	probes: ReflectionProbe[]
): ReflectionProbeSelectionResult {
	let firstIndex = -1;
	let secondIndex = -1;
	let firstWeight = 0;
	let secondWeight = 0;
	let firstId = "";
	let secondId = "";

	for (let i = 0; i < probes.length; i++) {
		const probe = probes[i];
		const cache = probe.getRuntimeCache();
		const metric = computeProbeMetric(worldPosition, probe);
		let weight = computeProbeRawWeight(
			metric,
			cache.effectiveBlendDistance,
			cache.blendExponent
		);
		if (!Number.isFinite(weight) || weight <= REFLECTION_PROBE_WEIGHT_EPSILON) {
			continue;
		}

		if (firstIndex < 0 || isBetterCandidate(weight, probe.id, firstWeight, firstId)) {
			secondIndex = firstIndex;
			secondWeight = firstWeight;
			secondId = firstId;
			firstIndex = i;
			firstWeight = weight;
			firstId = probe.id;
			continue;
		}

		if (
			secondIndex < 0 ||
			isBetterCandidate(weight, probe.id, secondWeight, secondId)
		) {
			secondIndex = i;
			secondWeight = weight;
			secondId = probe.id;
		}
	}

	if (firstIndex < 0) {
		return {
			firstIndex: -1,
			secondIndex: -1,
			firstWeight: 0,
			secondWeight: 0,
		};
	}

	const sum = firstWeight + Math.max(0, secondWeight);
	if (sum <= REFLECTION_PROBE_WEIGHT_EPSILON) {
		return {
			firstIndex: firstIndex,
			secondIndex: -1,
			firstWeight: 1,
			secondWeight: 0,
		};
	}

	return {
		firstIndex,
		secondIndex: secondIndex >= 0 ? secondIndex : -1,
		firstWeight: firstWeight / sum,
		secondWeight: secondIndex >= 0 ? secondWeight / sum : 0,
	};
}

export function computeProbeMetric(
	worldPosition: IVector3,
	probe: ReflectionProbe
): number {
	const cache = probe.getRuntimeCache();
	const localPosition = Matrix4.transformPoint(cache.worldToProbeMatrix, worldPosition);

	if (probe.shape === "box") {
		return Math.max(
			Math.abs(localPosition.x) * cache.invHalfExtents.x,
			Math.abs(localPosition.y) * cache.invHalfExtents.y,
			Math.abs(localPosition.z) * cache.invHalfExtents.z
		);
	}

	return (
		Math.hypot(localPosition.x, localPosition.y, localPosition.z) * cache.radiusInv
	);
}

export function computeProbeRawWeight(
	metric: number,
	effectiveBlendDistance: number,
	blendExponent: number
): number {
	if (!Number.isFinite(metric)) return 0;
	const safeBlendDistance = Math.max(effectiveBlendDistance, REFLECTION_PROBE_RAY_EPSILON);
	const x = clamp((metric - 1) / safeBlendDistance, 0, 1);
	let weight = 1 - smoothstep(0, 1, x);
	if (blendExponent !== 1) {
		weight = Math.pow(Math.max(0, weight), Math.max(0.01, blendExponent));
	}
	return weight;
}

export function computeProbeDepthOcclusion(
	metric: number,
	effectiveBlendDistance: number
): number {
	if (!Number.isFinite(metric)) return 0;
	const safeBlendDistance = Math.max(
		effectiveBlendDistance,
		REFLECTION_PROBE_RAY_EPSILON
	);
	const normalizedDepth = clamp((1 - metric) / safeBlendDistance, 0, 1);
	return smoothstep(0, 1, normalizedDepth);
}

export function computeParallaxCorrectedDirection(
	worldPosition: IVector3,
	reflectionDirection: IVector3,
	probe: ReflectionProbe
): ReflectionProbeSampleResult {
	const cache = probe.getRuntimeCache();
	const resolvedParallaxMode = resolveParallaxMode(probe.parallaxMode, probe.shape);
	if (resolvedParallaxMode === "off") {
		return {
			direction: normalizeDirection(reflectionDirection),
			valid: false,
		};
	}

	const localOrigin = Matrix4.transformPoint(cache.worldToProbeMatrix, worldPosition);
	const localDirection = normalizeDirection(
		transformDirection3x3(cache.worldToProbe3x3, reflectionDirection)
	);

	let hitLocal: IVector3 | null = null;
	if (resolvedParallaxMode === "box") {
		hitLocal = intersectLocalBox(localOrigin, localDirection, probe);
	} else {
		hitLocal = intersectLocalSphere(localOrigin, localDirection, probe);
	}

	if (!hitLocal) {
		return {
			direction: normalizeDirection(reflectionDirection),
			valid: false,
		};
	}

	const worldHit = Matrix4.transformPoint(cache.probeToWorldMatrix, hitLocal);
	const corrected = {
		x: worldHit.x - cache.captureWorldPosition.x,
		y: worldHit.y - cache.captureWorldPosition.y,
		z: worldHit.z - cache.captureWorldPosition.z,
	};
	return {
		direction: normalizeDirection(corrected),
		valid: true,
	};
}

export function sampleReflectionProbesSpecular(
	worldPosition: IVector3,
	reflectionDirection: IVector3,
	roughness: number,
	probes: ReflectionProbe[],
	fallbackTexture: Texture | null
): { r: number; g: number; b: number } | null {
	const selection = selectTopTwoReflectionProbes(worldPosition, probes);
	if (selection.firstIndex < 0) {
		return fallbackTexture ?
				samplePrefilteredEquirect(
					fallbackTexture,
					reflectionDirection,
					roughness
				)
			:	null;
	}

	const firstProbe = probes[selection.firstIndex];
	const firstDirection = computeParallaxCorrectedDirection(
		worldPosition,
		reflectionDirection,
		firstProbe
	).direction;
	const firstSample = samplePrefilteredEquirect(
		firstProbe.prefilteredMap!,
		firstDirection,
		roughness
	);
	const firstMetric = computeProbeMetric(worldPosition, firstProbe);
	const firstDepthOcclusion = computeProbeDepthOcclusion(
		firstMetric,
		firstProbe.getRuntimeCache().effectiveBlendDistance
	);

	if (selection.secondIndex < 0 || selection.secondWeight <= REFLECTION_PROBE_WEIGHT_EPSILON) {
		return {
			r: firstSample.r * firstDepthOcclusion,
			g: firstSample.g * firstDepthOcclusion,
			b: firstSample.b * firstDepthOcclusion,
		};
	}

	const secondProbe = probes[selection.secondIndex];
	const secondDirection = computeParallaxCorrectedDirection(
		worldPosition,
		reflectionDirection,
		secondProbe
	).direction;
	const secondSample = samplePrefilteredEquirect(
		secondProbe.prefilteredMap!,
		secondDirection,
		roughness
	);
	const secondMetric = computeProbeMetric(worldPosition, secondProbe);
	const secondDepthOcclusion = computeProbeDepthOcclusion(
		secondMetric,
		secondProbe.getRuntimeCache().effectiveBlendDistance
	);

	return {
		r:
			firstSample.r * selection.firstWeight * firstDepthOcclusion +
			secondSample.r * selection.secondWeight * secondDepthOcclusion,
		g:
			firstSample.g * selection.firstWeight * firstDepthOcclusion +
			secondSample.g * selection.secondWeight * secondDepthOcclusion,
		b:
			firstSample.b * selection.firstWeight * firstDepthOcclusion +
			secondSample.b * selection.secondWeight * secondDepthOcclusion,
	};
}

export function samplePrefilteredEquirect(
	texture: Texture,
	direction: IVector3,
	roughness: number
): { r: number; g: number; b: number } {
	return sampleEnvironmentTextureSpecular(texture, direction, roughness);
}

export function buildReflectionProbeAtlasTexture(
	probes: ReflectionProbe[]
): Texture | null {
	if (probes.length <= 0) return null;

	const maps: Texture[] = [];
	const atlasKeyParts: string[] = [];
	for (const probe of probes) {
		const sourceMap = probe.prefilteredMap;
		if (!sourceMap) return null;
		const normalizedMap = ensureEnvironmentTextureEquirect(sourceMap);
		if (!isTextureReadyForEnvironmentShared(normalizedMap)) {
			return null;
		}
		maps.push(normalizedMap);
		const textureIdentity = resolveTextureIdentity(sourceMap);
		atlasKeyParts.push(
			`${probe.id}:r${probe.captureRevision}:t${textureIdentity}:v${sourceMap.version}:${normalizedMap.width}x${normalizedMap.height}:${getEnvironmentMipLevelCount(normalizedMap)}`
		);
	}

	const baseMap = maps[0];
	const baseWidth = baseMap.width;
	const baseHeight = baseMap.height;
	const mipCount = getEnvironmentMipLevelCount(baseMap);
	const atlasKey = atlasKeyParts.join("|");
	const cached = _atlasCacheByKey.get(atlasKey);
	if (cached) return cached;

	for (let i = 1; i < maps.length; i++) {
		const map = maps[i];
		if (
			map.width !== baseWidth ||
			map.height !== baseHeight ||
			getEnvironmentMipLevelCount(map) !== mipCount
		) {
			return null;
		}
	}

	const atlasMipmaps: (Uint8Array | Uint8ClampedArray | Float32Array)[] = [];
	for (let mipLevel = 0; mipLevel < mipCount; mipLevel++) {
		const mipWidth = Math.max(1, baseWidth >> mipLevel);
		const mipHeight = Math.max(1, baseHeight >> mipLevel);
		const atlasWidth = mipWidth * maps.length;
		const pixelCount = atlasWidth * mipHeight * 4;
		const mipData = createCompatibleMipBuffer(baseMap, pixelCount);

		for (let probeIndex = 0; probeIndex < maps.length; probeIndex++) {
			const sourceMap = maps[probeIndex];
			const sourceMip = resolveAtlasSourceMip(sourceMap, mipLevel);
			if (!sourceMip) {
				return null;
			}
			blitMipIntoAtlas(
				sourceMip,
				mipData,
				mipWidth,
				mipHeight,
				atlasWidth,
				probeIndex * mipWidth
			);
		}

		atlasMipmaps.push(mipData);
	}

	const atlas = new Texture(
		atlasMipmaps[0] as Texture["data"],
		baseWidth * maps.length,
		baseHeight,
		baseMap.colorSpace
	);
	atlas.wrapS = "Clamp";
	atlas.wrapT = "Clamp";
	atlas.minFilter = "Linear";
	atlas.magFilter = "Linear";
	atlas.mipmaps = atlasMipmaps;
	atlas.data = atlasMipmaps[0] as Texture["data"];
	if (_atlasCacheByKey.size >= 32) {
		_atlasCacheByKey.clear();
	}
	_atlasCacheByKey.set(atlasKey, atlas);
	return atlas;
}

export function isTextureReadyForEnvironment(
	texture: Texture | null | undefined
): texture is Texture {
	return isTextureReadyForEnvironmentShared(texture);
}

export function directionToEquirectUV(direction: IVector3): { u: number; v: number } {
	return directionToEquirectUVFromEnvironmentMap(direction);
}

function intersectLocalBox(
	localOrigin: IVector3,
	localDirection: IVector3,
	probe: ReflectionProbe
): IVector3 | null {
	const halfX = Math.max(Math.abs(probe.halfExtents.x), REFLECTION_PROBE_RAY_EPSILON);
	const halfY = Math.max(Math.abs(probe.halfExtents.y), REFLECTION_PROBE_RAY_EPSILON);
	const halfZ = Math.max(Math.abs(probe.halfExtents.z), REFLECTION_PROBE_RAY_EPSILON);

	let tMin = -Infinity;
	let tMax = Infinity;

	if (
		!intersectAxis(localOrigin.x, localDirection.x, -halfX, halfX, (min, max) => {
			tMin = Math.max(tMin, min);
			tMax = Math.min(tMax, max);
		})
	) {
		return null;
	}
	if (
		!intersectAxis(localOrigin.y, localDirection.y, -halfY, halfY, (min, max) => {
			tMin = Math.max(tMin, min);
			tMax = Math.min(tMax, max);
		})
	) {
		return null;
	}
	if (
		!intersectAxis(localOrigin.z, localDirection.z, -halfZ, halfZ, (min, max) => {
			tMin = Math.max(tMin, min);
			tMax = Math.min(tMax, max);
		})
	) {
		return null;
	}

	if (tMax < Math.max(tMin, 0)) return null;
	const t = tMin > REFLECTION_PROBE_RAY_EPSILON ? tMin : tMax;
	if (!Number.isFinite(t) || t <= REFLECTION_PROBE_RAY_EPSILON) return null;

	return {
		x: localOrigin.x + localDirection.x * t,
		y: localOrigin.y + localDirection.y * t,
		z: localOrigin.z + localDirection.z * t,
	};
}

function intersectLocalSphere(
	localOrigin: IVector3,
	localDirection: IVector3,
	probe: ReflectionProbe
): IVector3 | null {
	const radius = Math.max(Math.abs(probe.radius), REFLECTION_PROBE_RAY_EPSILON);
	const b =
		localOrigin.x * localDirection.x +
		localOrigin.y * localDirection.y +
		localOrigin.z * localDirection.z;
	const c =
		localOrigin.x * localOrigin.x +
		localOrigin.y * localOrigin.y +
		localOrigin.z * localOrigin.z -
		radius * radius;
	const discriminant = b * b - c;
	if (discriminant < 0) return null;

	const sqrtDisc = Math.sqrt(discriminant);
	const t0 = -b - sqrtDisc;
	const t1 = -b + sqrtDisc;
	let t = Number.POSITIVE_INFINITY;
	if (t0 > REFLECTION_PROBE_RAY_EPSILON) t = t0;
	if (t1 > REFLECTION_PROBE_RAY_EPSILON && t1 < t) t = t1;
	if (!Number.isFinite(t)) return null;

	return {
		x: localOrigin.x + localDirection.x * t,
		y: localOrigin.y + localDirection.y * t,
		z: localOrigin.z + localDirection.z * t,
	};
}

function transformDirection3x3(
	matrix: number[][],
	direction: IVector3
): IVector3 {
	return {
		x:
			matrix[0][0] * direction.x +
			matrix[0][1] * direction.y +
			matrix[0][2] * direction.z,
		y:
			matrix[1][0] * direction.x +
			matrix[1][1] * direction.y +
			matrix[1][2] * direction.z,
		z:
			matrix[2][0] * direction.x +
			matrix[2][1] * direction.y +
			matrix[2][2] * direction.z,
	};
}

function normalizeDirection(direction: IVector3): IVector3 {
	const length = Math.hypot(direction.x, direction.y, direction.z);
	if (length <= REFLECTION_PROBE_RAY_EPSILON) {
		return { x: 0, y: 0, z: 1 };
	}
	const invLength = 1 / length;
	return {
		x: direction.x * invLength,
		y: direction.y * invLength,
		z: direction.z * invLength,
	};
}

function isBetterCandidate(
	weight: number,
	id: string,
	currentWeight: number,
	currentId: string
): boolean {
	if (weight > currentWeight + REFLECTION_PROBE_WEIGHT_EPSILON) return true;
	if (Math.abs(weight - currentWeight) <= REFLECTION_PROBE_WEIGHT_EPSILON) {
		return id.localeCompare(currentId) < 0;
	}
	return false;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = clamp((x - edge0) / Math.max(edge1 - edge0, REFLECTION_PROBE_RAY_EPSILON), 0, 1);
	return t * t * (3 - 2 * t);
}

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}

function resolveParallaxMode(
	mode: ReflectionProbeParallaxMode,
	shape: ReflectionProbeShape
): ReflectionProbeParallaxMode {
	if (mode === "off" || mode === "box" || mode === "sphere") {
		return mode;
	}
	return shape === "box" ? "box" : "off";
}

function intersectAxis(
	origin: number,
	direction: number,
	minValue: number,
	maxValue: number,
	assign: (min: number, max: number) => void
): boolean {
	if (Math.abs(direction) <= REFLECTION_PROBE_RAY_EPSILON) {
		return origin >= minValue && origin <= maxValue;
	}

	const invDirection = 1 / direction;
	let t0 = (minValue - origin) * invDirection;
	let t1 = (maxValue - origin) * invDirection;
	if (t0 > t1) {
		const swap = t0;
		t0 = t1;
		t1 = swap;
	}
	assign(t0, t1);
	return true;
}

function createCompatibleMipBuffer(
	baseMap: Texture,
	pixelCount: number
): Uint8Array | Uint8ClampedArray | Float32Array {
	const mip0 = baseMap.mipmaps[0] ?? baseMap.data;
	if (mip0 instanceof Float32Array) {
		return new Float32Array(pixelCount);
	}
	if (mip0 instanceof Uint8ClampedArray) {
		return new Uint8ClampedArray(pixelCount);
	}
	return new Uint8Array(pixelCount);
}

function resolveAtlasSourceMip(
	texture: Texture,
	mipLevel: number
): Uint8Array | Uint8ClampedArray | Float32Array | null {
	return (
		texture.mipmaps[mipLevel] ??
		(mipLevel === 0 ? texture.data : null) ??
		texture.mipmaps[0] ??
		null
	);
}

function blitMipIntoAtlas(
	source: Uint8Array | Uint8ClampedArray | Float32Array,
	target: Uint8Array | Uint8ClampedArray | Float32Array,
	sourceWidth: number,
	sourceHeight: number,
	targetWidth: number,
	targetOffsetX: number
): void {
	const bytesPerPixel = 4;
	const sourceRowWidth = sourceWidth * bytesPerPixel;
	const targetRowWidth = targetWidth * bytesPerPixel;
	const targetStartPixel = targetOffsetX * bytesPerPixel;
	for (let row = 0; row < sourceHeight; row++) {
		const sourceStart = row * sourceRowWidth;
		const targetStart = row * targetRowWidth + targetStartPixel;
		target.set(source.subarray(sourceStart, sourceStart + sourceRowWidth), targetStart);
	}
}

function resolveTextureIdentity(texture: Texture): number {
	const cached = _textureIdentityByTexture.get(texture);
	if (cached !== undefined) {
		return cached;
	}
	const identity = ++_nextTextureIdentity;
	_textureIdentityByTexture.set(texture, identity);
	return identity;
}
