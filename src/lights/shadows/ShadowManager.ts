import type { SceneLight, ShadowCastingLight } from "..";
import { LightType } from "..";
import {
	createShadowRenderSet,
	ensureShadowRenderSetMatchesConfig,
	type ShadowConfig,
	type ShadowRenderSet,
} from "../ShadowMapping";
import type { IVector3 } from "../../maths/types";
import { CSMShadowMap } from "./CSMShadowMap";
import { ShadowFrameState } from "./ShadowFrameState";
import { SingleShadowMap } from "./SingleShadowMap";
import { VSMShadowMap, type VSMShadowMapOptions } from "./VSMShadowMap";
import type { ShadowMapBase } from "./ShadowMapBase";
import type {
	IShadowBackendCapabilities,
	ShadowBindingRecord,
	ShadowCSMOptions,
	ShadowMapBaseOptions,
} from "./types";

interface ShadowBudgetCandidate {
	light: ShadowCastingLight;
	shadowMap: ShadowMapBase;
	score: number;
	size: number;
	cascadeCount: number;
	filterMode: "pcf" | "vsm";
	config: ShadowConfig;
	cost: number;
}

export interface ShadowFrameBuildOptions {
	lights: SceneLight[];
	backendCapabilities?: IShadowBackendCapabilities;
	cameraPosition?: IVector3 | null;
	enableShadows?: boolean;
}

export class ShadowManager {
	private _mapsById = new Map<string, ShadowMapBase>();
	private _shadowByLight = new Map<ShadowCastingLight, ShadowMapBase>();
	private _lightsByShadowId = new Map<string, Set<ShadowCastingLight>>();
	private _shadowRenderSets = new Map<ShadowCastingLight, ShadowRenderSet>();
	private _version = 0;
	private _lastFrameState = new ShadowFrameState(0, [], new Map());

	public createSingle(options: ShadowMapBaseOptions = {}): SingleShadowMap {
		const shadowMap = new SingleShadowMap(options);
		this._mapsById.set(shadowMap.id, shadowMap);
		this._version++;
		return shadowMap;
	}

	public createVSM(options: VSMShadowMapOptions = {}): VSMShadowMap {
		const shadowMap = new VSMShadowMap(options);
		this._mapsById.set(shadowMap.id, shadowMap);
		this._version++;
		return shadowMap;
	}

	public createCSM(options: ShadowCSMOptions = {}): CSMShadowMap {
		const shadowMap = new CSMShadowMap(options);
		this._mapsById.set(shadowMap.id, shadowMap);
		this._version++;
		return shadowMap;
	}

	public bind(light: ShadowCastingLight, shadowMap: ShadowMapBase): void {
		const existing = this._shadowByLight.get(light);
		if (existing?.id === shadowMap.id) {
			return;
		}
		if (existing) {
			this._detachBinding(light, existing);
		}

		this._mapsById.set(shadowMap.id, shadowMap);
		this._shadowByLight.set(light, shadowMap);
		let lights = this._lightsByShadowId.get(shadowMap.id);
		if (!lights) {
			lights = new Set();
			this._lightsByShadowId.set(shadowMap.id, lights);
		}
		lights.add(light);
		this._version++;
	}

	public rebind(light: ShadowCastingLight, shadowMap: ShadowMapBase): void {
		this.bind(light, shadowMap);
	}

	public unbindLight(light: ShadowCastingLight): void {
		const existing = this._shadowByLight.get(light);
		if (!existing) {
			return;
		}
		this._detachBinding(light, existing);
		this._shadowRenderSets.delete(light);
		this._version++;
	}

	public destroy(shadowMap: ShadowMapBase): void {
		const lights = this._lightsByShadowId.get(shadowMap.id);
		if (lights) {
			for (const light of lights) {
				this._shadowByLight.delete(light);
				this._shadowRenderSets.delete(light);
			}
		}
		this._lightsByShadowId.delete(shadowMap.id);
		this._mapsById.delete(shadowMap.id);
		this._version++;
	}

	public clear(): void {
		this._mapsById.clear();
		this._shadowByLight.clear();
		this._lightsByShadowId.clear();
		this._shadowRenderSets.clear();
		this._version++;
	}

	public get version(): number {
		return this._version;
	}

	public getBoundShadowMap(light: ShadowCastingLight): ShadowMapBase | undefined {
		return this._shadowByLight.get(light);
	}

	public getLegacyShadowConfig(light: ShadowCastingLight): ShadowConfig | undefined {
		const shadowMap = this._shadowByLight.get(light);
		if (!shadowMap || shadowMap.enabled === false) {
			return undefined;
		}
		return shadowMap.toLegacyShadowConfig(light.type);
	}

	public getLastFrameState(): ShadowFrameState {
		return this._lastFrameState;
	}

	public buildFrameState(options: ShadowFrameBuildOptions): ShadowFrameState {
		const activeLights = new Set<ShadowCastingLight>();
		for (const light of options.lights) {
			if (isShadowBindableLightType(light)) {
				activeLights.add(light);
			}
		}
		for (const light of this._shadowByLight.keys()) {
			if (activeLights.has(light)) {
				continue;
			}
			this.unbindLight(light);
		}

		const enabled = options.enableShadows !== false;
		if (!enabled || options.lights.length <= 0) {
			this._shadowRenderSets.clear();
			this._lastFrameState = new ShadowFrameState(
				this._version,
				[],
				this._shadowRenderSets
			);
			return this._lastFrameState;
		}

		const candidates = this._collectCandidates(options);
		const selectedCandidates = this._applyDynamicBudget(
			candidates,
			options.backendCapabilities
		);
		const selectedLights = new Set<ShadowCastingLight>(
			selectedCandidates.map((candidate) => candidate.light)
		);
		for (const [light] of this._shadowRenderSets) {
			if (!selectedLights.has(light)) {
				this._shadowRenderSets.delete(light);
			}
		}

		const records: ShadowBindingRecord[] = [];
		for (const candidate of selectedCandidates) {
			const existing = this._shadowRenderSets.get(candidate.light);
			const nextRenderSet =
				!existing ?
					createShadowRenderSet(candidate.config)
				:	ensureShadowRenderSetMatchesConfig(existing, candidate.config);
			this._shadowRenderSets.set(candidate.light, nextRenderSet);

			records.push({
				light: candidate.light,
				shadowMapId: candidate.shadowMap.id,
				shadowMapKind: candidate.shadowMap.kind,
				filterMode: candidate.filterMode,
				priority: candidate.shadowMap.priority,
				renderSet: nextRenderSet,
				cost: candidate.cost,
				score: candidate.score,
			});
		}

		this._lastFrameState = new ShadowFrameState(
			this._version,
			records,
			this._shadowRenderSets
		);
		return this._lastFrameState;
	}

	private _collectCandidates(options: ShadowFrameBuildOptions): ShadowBudgetCandidate[] {
		const candidates: ShadowBudgetCandidate[] = [];
		const cameraPosition = options.cameraPosition ?? null;
		for (const light of options.lights) {
			if (!isShadowBindableLightType(light)) {
				continue;
			}
			const shadowMap = this._shadowByLight.get(light);
			if (!shadowMap || shadowMap.enabled === false) {
				continue;
			}

			const cascadeCount = this._resolveCascadeCount(shadowMap, light);
			const config = shadowMap.toLegacyShadowConfig(light.type, {
				size: shadowMap.size,
				cascadeCount,
			});
			const score = this._resolveCandidateScore(light, shadowMap, cameraPosition);
			const cost = shadowMap.estimateCost(light.type, shadowMap.size, cascadeCount);
			candidates.push({
				light,
				shadowMap,
				score,
				size: shadowMap.size,
				cascadeCount,
				filterMode: shadowMap.filterMode,
				config,
				cost,
			});
		}
		candidates.sort((left, right) => right.score - left.score);
		return candidates;
	}

	private _applyDynamicBudget(
		candidates: ShadowBudgetCandidate[],
		capabilities?: IShadowBackendCapabilities
	): ShadowBudgetCandidate[] {
		const budget =
			typeof capabilities?.maxDynamicShadowCost === "number" &&
			Number.isFinite(capabilities.maxDynamicShadowCost) &&
			capabilities.maxDynamicShadowCost > 0 ?
				capabilities.maxDynamicShadowCost
			:	Number.POSITIVE_INFINITY;

		let consumed = 0;
		const selected: ShadowBudgetCandidate[] = [];
		for (const candidate of candidates) {
			if (consumed + candidate.cost <= budget) {
				consumed += candidate.cost;
				selected.push(candidate);
				continue;
			}

			const degraded = this._degradeCandidateToFitBudget(
				candidate,
				Math.max(0, budget - consumed)
			);
			if (!degraded) {
				continue;
			}
			consumed += degraded.cost;
			selected.push(degraded);
		}

		return selected;
	}

	private _degradeCandidateToFitBudget(
		candidate: ShadowBudgetCandidate,
		remainingBudget: number
	): ShadowBudgetCandidate | null {
		if (!(remainingBudget > 0)) {
			return null;
		}

		let best: ShadowBudgetCandidate | null = null;
		const minSize = 128;
		let size = candidate.size;
		let cascadeCount = candidate.cascadeCount;
		while (size >= minSize) {
			while (cascadeCount >= 1) {
				const cost = candidate.shadowMap.estimateCost(
					candidate.light.type,
					size,
					cascadeCount
				);
				if (cost <= remainingBudget) {
					const config = candidate.shadowMap.toLegacyShadowConfig(
						candidate.light.type,
						{
							size,
							cascadeCount,
						}
					);
					best = {
						...candidate,
						size,
						cascadeCount,
						cost,
						config,
					};
					break;
				}
				cascadeCount--;
			}
			if (best) {
				break;
			}
			size = Math.floor(size * 0.5);
			cascadeCount = Math.max(1, candidate.cascadeCount);
		}

		return best;
	}

	private _resolveCascadeCount(
		shadowMap: ShadowMapBase,
		light: ShadowCastingLight
	): number {
		if (shadowMap.kind !== "csm") {
			return 1;
		}
		return (shadowMap as CSMShadowMap).getCascadeCountForLightType(light.type);
	}

	private _resolveCandidateScore(
		light: ShadowCastingLight,
		shadowMap: ShadowMapBase,
		cameraPosition: IVector3 | null
	): number {
		let score = shadowMap.priority * 1000 + light.intensity * 10;
		if (cameraPosition && isFiniteVector3(cameraPosition)) {
			const lightPosition = resolveLightWorldPosition(light);
			if (lightPosition) {
				const dx = lightPosition.x - cameraPosition.x;
				const dy = lightPosition.y - cameraPosition.y;
				const dz = lightPosition.z - cameraPosition.z;
				const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
				score += 1 / Math.max(1, distance);
			}
		}
		return score;
	}

	private _detachBinding(
		light: ShadowCastingLight,
		shadowMap: ShadowMapBase
	): void {
		this._shadowByLight.delete(light);
		const lights = this._lightsByShadowId.get(shadowMap.id);
		if (lights) {
			lights.delete(light);
			if (lights.size <= 0) {
				this._lightsByShadowId.delete(shadowMap.id);
			}
		}
	}
}

function resolveLightWorldPosition(light: ShadowCastingLight): IVector3 | null {
	switch (light.type) {
		case LightType.Point:
		case LightType.Spot:
		case LightType.RectArea:
			return light.getWorldPosition();
		default:
			return null;
	}
}

function isFiniteVector3(value: IVector3): boolean {
	return (
		Number.isFinite(value.x) &&
		Number.isFinite(value.y) &&
		Number.isFinite(value.z)
	);
}

function isShadowBindableLightType(
	light: SceneLight
): light is ShadowCastingLight {
	return (
		light.type === LightType.Directional ||
		light.type === LightType.Point ||
		light.type === LightType.Spot ||
		light.type === LightType.RectArea
	);
}
