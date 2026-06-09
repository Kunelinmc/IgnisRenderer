import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import {
	LightType,
	type LightProbe,
	type SceneLight,
} from "../../lights";

export const MAX_ACTIVE_LOCAL_LIGHT_PROBES = 8;
export const LOCAL_LIGHT_PROBE_WEIGHT_EPSILON = 1e-6;
const LOCAL_LIGHT_PROBE_BLEND_EPSILON = 1e-5;

export interface LightProbeSelectionResult {
	firstIndex: number;
	secondIndex: number;
	firstWeight: number;
	secondWeight: number;
	coverage: number;
	priority: number;
}

interface RankedLocalizedProbe {
	probe: LightProbe;
	metric: number;
	weight: number;
	priority: number;
}

export function isLocalizedLightProbe(probe: LightProbe): boolean {
	return probe.shape === "sphere" || probe.shape === "box";
}

export function collectGlobalLightProbes(lights: SceneLight[]): LightProbe[] {
	const probes: LightProbe[] = [];
	for (const light of lights) {
		if (light.type !== LightType.LightProbe) continue;
		const probe = light as LightProbe;
		if (isLocalizedLightProbe(probe)) continue;
		probes.push(probe);
	}
	probes.sort(compareProbeId);
	return probes;
}

export function collectLocalizedLightProbes(lights: SceneLight[]): LightProbe[] {
	const probes: LightProbe[] = [];
	for (const light of lights) {
		if (light.type !== LightType.LightProbe) continue;
		const probe = light as LightProbe;
		if (!isLocalizedLightProbe(probe)) continue;
		probe.getRuntimeCache();
		probes.push(probe);
	}
	probes.sort(compareProbeId);
	return probes;
}

export function collectActiveLocalizedLightProbes(
	lights: SceneLight[],
	maxCount = MAX_ACTIVE_LOCAL_LIGHT_PROBES,
	worldPosition: IVector3 | null = null
): LightProbe[] {
	return limitLocalizedLightProbes(
		collectLocalizedLightProbes(lights),
		maxCount,
		worldPosition
	);
}

export function limitLocalizedLightProbes(
	probes: LightProbe[],
	maxCount = MAX_ACTIVE_LOCAL_LIGHT_PROBES,
	worldPosition: IVector3 | null = null
): LightProbe[] {
	const resolvedMaxCount =
		Number.isFinite(maxCount) ? Math.max(0, Math.floor(maxCount)) : MAX_ACTIVE_LOCAL_LIGHT_PROBES;
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
		const metric = computeLightProbeMetric(worldPosition, probe);
		const weight = computeLightProbeRawWeight(
			metric,
			cache.effectiveBlendDistance
		);
		return {
			probe,
			metric,
			weight: Number.isFinite(weight) ? weight : 0,
			priority: cache.priority,
		};
	});

	ranked.sort(compareRankedLocalizedProbe);
	const selected = ranked
		.slice(0, resolvedMaxCount)
		.map((entry) => entry.probe);
	selected.sort(compareProbeId);
	return selected;
}

export function refreshLightProbeCaches(lights: SceneLight[]): void {
	for (const light of lights) {
		if (light.type !== LightType.LightProbe) continue;
		(light as LightProbe).refreshRuntimeCache();
	}
}

export function selectTopTwoLocalizedLightProbes(
	worldPosition: IVector3,
	probes: LightProbe[]
): LightProbeSelectionResult {
	let bestPriority = Number.NEGATIVE_INFINITY;
	let firstIndex = -1;
	let secondIndex = -1;
	let firstRawWeight = 0;
	let secondRawWeight = 0;
	let firstId = "";
	let secondId = "";

	for (let i = 0; i < probes.length; i++) {
		const probe = probes[i];
		if (!isLocalizedLightProbe(probe)) continue;

		const cache = probe.getRuntimeCache();
		const metric = computeLightProbeMetric(worldPosition, probe);
		const weight = computeLightProbeRawWeight(
			metric,
			cache.effectiveBlendDistance
		);
		if (!Number.isFinite(weight) || weight <= LOCAL_LIGHT_PROBE_WEIGHT_EPSILON) {
			continue;
		}

		const priority = cache.priority;
		if (priority > bestPriority) {
			bestPriority = priority;
			firstIndex = i;
			secondIndex = -1;
			firstRawWeight = weight;
			secondRawWeight = 0;
			firstId = probe.id;
			secondId = "";
			continue;
		}
		if (priority < bestPriority) {
			continue;
		}

		if (
			firstIndex < 0 ||
			isBetterCandidate(weight, probe.id, firstRawWeight, firstId)
		) {
			secondIndex = firstIndex;
			secondRawWeight = firstRawWeight;
			secondId = firstId;
			firstIndex = i;
			firstRawWeight = weight;
			firstId = probe.id;
			continue;
		}

		if (
			secondIndex < 0 ||
			isBetterCandidate(weight, probe.id, secondRawWeight, secondId)
		) {
			secondIndex = i;
			secondRawWeight = weight;
			secondId = probe.id;
		}
	}

	if (firstIndex < 0) {
		return {
			firstIndex: -1,
			secondIndex: -1,
			firstWeight: 0,
			secondWeight: 0,
			coverage: 0,
			priority: 0,
		};
	}

	const rawSum = firstRawWeight + Math.max(0, secondRawWeight);
	const coverage = clamp(rawSum, 0, 1);
	if (rawSum <= LOCAL_LIGHT_PROBE_WEIGHT_EPSILON) {
		return {
			firstIndex,
			secondIndex: -1,
			firstWeight: 1,
			secondWeight: 0,
			coverage,
			priority: bestPriority,
		};
	}

	return {
		firstIndex,
		secondIndex: secondIndex >= 0 ? secondIndex : -1,
		firstWeight: firstRawWeight / rawSum,
		secondWeight: secondIndex >= 0 ? secondRawWeight / rawSum : 0,
		coverage,
		priority: bestPriority,
	};
}

export function computeLightProbeMetric(
	worldPosition: IVector3,
	probe: LightProbe
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

	if (probe.shape === "sphere") {
		return (
			Math.hypot(localPosition.x, localPosition.y, localPosition.z) *
			cache.radiusInv
		);
	}

	return Number.POSITIVE_INFINITY;
}

export function computeLightProbeRawWeight(
	metric: number,
	effectiveBlendDistance: number
): number {
	if (!Number.isFinite(metric)) return 0;
	const safeBlendDistance = Math.max(
		effectiveBlendDistance,
		LOCAL_LIGHT_PROBE_BLEND_EPSILON
	);
	const x = clamp((metric - 1) / safeBlendDistance, 0, 1);
	return 1 - smoothstep(0, 1, x);
}

function compareRankedLocalizedProbe(
	left: RankedLocalizedProbe,
	right: RankedLocalizedProbe
): number {
	const leftActive = left.weight > LOCAL_LIGHT_PROBE_WEIGHT_EPSILON;
	const rightActive = right.weight > LOCAL_LIGHT_PROBE_WEIGHT_EPSILON;
	if (leftActive !== rightActive) {
		return leftActive ? -1 : 1;
	}
	if (left.priority !== right.priority) {
		return right.priority - left.priority;
	}
	if (Math.abs(left.weight - right.weight) > LOCAL_LIGHT_PROBE_WEIGHT_EPSILON) {
		return right.weight - left.weight;
	}
	if (left.metric !== right.metric) {
		return left.metric - right.metric;
	}
	return compareProbeId(left.probe, right.probe);
}

function compareProbeId(left: LightProbe, right: LightProbe): number {
	return left.id.localeCompare(right.id);
}

function isBetterCandidate(
	weight: number,
	id: string,
	bestWeight: number,
	bestId: string
): boolean {
	if (Math.abs(weight - bestWeight) > LOCAL_LIGHT_PROBE_WEIGHT_EPSILON) {
		return weight > bestWeight;
	}
	return id.localeCompare(bestId) < 0;
}

function clamp(value: number, min: number, max: number): number {
	if (value <= min) return min;
	if (value >= max) return max;
	return value;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = clamp((x - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
	return t * t * (3 - 2 * t);
}
