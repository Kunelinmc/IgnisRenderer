import { LightType, type ShadowCastingLight } from "../lights";
import { CSMShadowMap } from "../lights/shadows/CSMShadowMap";
import { SingleShadowMap } from "../lights/shadows/SingleShadowMap";
import type { ShadowConfig, ShadowStrategyType } from "../lights/shadows/ShadowMapping";
import type {
	IShadowStrategyProvider,
	ShadowSliceDescriptor,
	ShadowStrategyBuildContext,
} from "../lights/shadows/ShadowStrategyTypes";

export type {
	IShadowStrategyProvider,
	SceneBounds,
	ShadowSliceDescriptor,
	ShadowStrategyBuildContext,
	ShadowStrategyCamera,
} from "../lights/shadows/ShadowStrategyTypes";

export interface ShadowBackendCapabilities {
	backendKey: string;
	supportsSingleMap: boolean;
	supportsDirectionalCSM: boolean;
	supportsSpotCSM?: boolean;
	supportsPointCSM?: boolean;
	maxCsmDirectionalLights: number;
	maxDynamicShadowCost?: number;
}

class SingleMapShadowStrategyProvider implements IShadowStrategyProvider {
	public readonly type: ShadowStrategyType = "single-map";

	public supports(_light: ShadowCastingLight): boolean {
		return true;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		return SingleShadowMap.buildSlices(context);
	}
}

class CSMShadowStrategyProvider implements IShadowStrategyProvider {
	public readonly type: ShadowStrategyType = "csm";

	public supports(_light: ShadowCastingLight): boolean {
		return true;
	}

	public build(context: ShadowStrategyBuildContext): ShadowSliceDescriptor[] {
		return CSMShadowMap.buildSlices(context);
	}
}

export class ShadowStrategyRegistry {
	private readonly _providers = new Map<ShadowStrategyType, IShadowStrategyProvider>();

	public register(provider: IShadowStrategyProvider): this {
		this._providers.set(provider.type, provider);
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
}

const _defaultRegistry = new ShadowStrategyRegistry()
	.register(new SingleMapShadowStrategyProvider())
	.register(new CSMShadowStrategyProvider());

export function getDefaultShadowStrategyRegistry(): ShadowStrategyRegistry {
	return _defaultRegistry;
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
