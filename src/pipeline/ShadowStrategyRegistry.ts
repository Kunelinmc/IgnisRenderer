import { LightType, type ShadowCastingLight } from "../lights";
import { CascadedShadowMap } from "../lights/shadows/CascadedShadowMap";
import { SingleShadowMap } from "../lights/shadows/SingleShadowMap";
import type { ShadowConfig, ShadowStrategyType } from "../lights/shadows/ShadowMapping";
import type {
	SceneBounds,
	IShadowStrategyProvider,
	ShadowSliceDescriptor,
	ShadowStrategyBuildContext,
	ShadowStrategyCamera,
} from "../lights/shadows/types";

export type { ShadowStrategyCamera, ShadowSliceDescriptor, SceneBounds };

interface IShadowStrategyBuilder {
	buildSlices(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[];
}

export interface ShadowBackendCapabilities {
	backendKey: string;
	supportsSingleMap: boolean;
	supportsDirectionalCSM: boolean;
	supportsSpotCSM?: boolean;
	supportsPointCSM?: boolean;
	maxCsmDirectionalLights: number;
	maxDynamicShadowCost?: number;
	supportsPagedShadows?: boolean;
	maxPagedShadowPages?: number;
	pagedShadowPageSizeRange?: [number, number];
}

export class ShadowStrategyRegistry {
	private static readonly _defaultRegistry = new ShadowStrategyRegistry()
		.register("single-map", SingleShadowMap)
		.register("csm", CascadedShadowMap);

	private readonly _providers = new Map<ShadowStrategyType, IShadowStrategyProvider>();

	public register(provider: IShadowStrategyProvider): this;
	public register(
		type: ShadowStrategyType,
		strategy: IShadowStrategyBuilder,
		supports?: (light: ShadowCastingLight) => boolean
	): this;
	public register(
		providerOrType: IShadowStrategyProvider | ShadowStrategyType,
		strategy?: IShadowStrategyBuilder,
		supports: (light: ShadowCastingLight) => boolean = () => true
	): this {
		if (typeof providerOrType !== "string") {
			this._providers.set(providerOrType.type, providerOrType);
			return this;
		}

		if (!strategy) {
			throw new Error(
				`Shadow strategy ${providerOrType} requires a builder with buildSlices().`
			);
		}

		this._providers.set(providerOrType, {
			type: providerOrType,
			supports,
			build: (context) => strategy.buildSlices(context),
		});
		return this;
	}

	public get(type: ShadowStrategyType): IShadowStrategyProvider | null {
		return this._providers.get(type) ?? null;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		const provider = this.get(context.config.strategy);
		if (!provider || !provider.supports(context.light)) {
			return [];
		}
		return provider.build(context);
	}

	public static getDefault(): ShadowStrategyRegistry {
		return ShadowStrategyRegistry._defaultRegistry;
	}

	public static buildWithDefault(
		context: ShadowStrategyBuildContext
	): ShadowSliceDescriptor[] {
		return ShadowStrategyRegistry._defaultRegistry.build(context);
	}
}

export function getDefaultShadowStrategyRegistry(): ShadowStrategyRegistry {
	return ShadowStrategyRegistry.getDefault();
}

function resolveShadowPriority(light: ShadowCastingLight): number {
	const config = light.scene?.shadows.getLegacyShadowConfig(light);
	if (!config) {
		return 0;
	}
	const priority = (config as { priority?: unknown }).priority;
	if (typeof priority !== "number" || !Number.isFinite(priority)) {
		return 0;
	}
	return priority;
}

export function selectCSMDirectionalLights(
	lights: ShadowCastingLight[],
	maxCount: number,
	resolveConfig: (light: ShadowCastingLight) => ShadowConfig | undefined = (light) =>
		light.scene?.shadows.getLegacyShadowConfig(light)
): Set<ShadowCastingLight> {
	const requested = lights.filter(
		(light) =>
			light.type === LightType.Directional &&
			resolveConfig(light)?.strategy === "csm"
	);
	if (requested.length <= 0 || maxCount <= 0) {
		return new Set();
	}

	requested.sort((left, right) => {
		const priorityDelta = resolveShadowPriority(right) - resolveShadowPriority(left);
		if (priorityDelta !== 0) {
			return priorityDelta;
		}
		const intensityDelta = right.intensity - left.intensity;
		if (intensityDelta !== 0) {
			return intensityDelta;
		}
		return left.id.localeCompare(right.id);
	});

	return new Set(requested.slice(0, maxCount));
}
