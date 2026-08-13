import type { WebGPUFrameConfiguration } from "./WebGPUFrameConfiguration";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_FRAME_FEATURE_STATES,
	type WebGPUFrameConfigurationDemand,
	type WebGPUFrameTargetClass,
} from "./WebGPUFrameMessages";
import type { WebGPUFrameMessageHandler } from "./WebGPUFrameMessage";
import {
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "../constants";

/** @internal Aggregates analyzed frame work into one effective configuration. */
export class WebGPUFrameConfigurationModule {
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "reduce-demands",
		moduleId: "frame-configuration",
		phase: "configuration",
		inputs: [
			{ descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE },
			{ descriptor: WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, required: false },
		],
		outputs: [WEBGPU_FRAME_CONFIGURATION_MESSAGE],
		run: (messages, publisher) => {
			publisher.publish(
				WEBGPU_FRAME_CONFIGURATION_MESSAGE,
				this._reduce(
					messages.get(WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE),
					messages.getAll(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE),
				),
			);
		},
	}];

	private _reduce(
		request: import("./WebGPUFrameMessages").WebGPUFrameConfigurationRequest,
		demands: readonly WebGPUFrameConfigurationDemand[],
	): WebGPUFrameConfiguration {
		const { capabilities, options } = request;
		const diagnostics = demands.flatMap((demand) => demand.diagnostics ?? []);
		const resources = mergeResourceDemands(demands);
		const featureStates = mergeFeatureStates(demands);
		const mrtSupported =
			capabilities.maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
			capabilities.maxColorAttachmentBytesPerSample >=
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
		if (!mrtSupported) {
			if (capabilities.maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT) {
				diagnostics.push({
					code: "webgpu-mrt-disabled-attachments",
					message:
						`WebGPU device maxColorAttachments is ${capabilities.maxColorAttachments}, ` +
						`requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`,
				});
			}
			if (
				capabilities.maxColorAttachmentBytesPerSample <
				WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
			) {
				diagnostics.push({
					code: "webgpu-mrt-disabled-bytes",
					message:
						"WebGPU device maxColorAttachmentBytesPerSample is " +
						`${capabilities.maxColorAttachmentBytesPerSample}, requires ` +
						`${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`,
				});
			}
		}

		let targetClass: WebGPUFrameTargetClass =
			options.samplePlan.sampleCount > 1 ? "color" : "single";
		if (options.forceForwardMrt === true) targetClass = "mrt";
		for (const demand of demands) {
			if (demand.targetClass && targetRank(demand.targetClass) > targetRank(targetClass)) {
				targetClass = demand.targetClass;
			}
		}
		const deferredActive = getFeatureState(
			featureStates,
			WEBGPU_FRAME_FEATURE_STATES.deferredActive,
			false,
		);
		const deferredSupported = getFeatureState(
			featureStates,
			WEBGPU_FRAME_FEATURE_STATES.deferredSupported,
			false,
		);
		const oitActive = getFeatureState(
			featureStates,
			WEBGPU_FRAME_FEATURE_STATES.oitActive,
			false,
		);
		if (deferredActive) targetClass = "gbuffer";
		if (!mrtSupported && !deferredActive) {
			targetClass = "single";
		}
		const sceneTargetMode = targetClass;
		const deferredGBufferLayout = getFeatureState(
			featureStates,
			WEBGPU_FRAME_FEATURE_STATES.deferredGBufferLayout,
			"extended" as const,
		);
		const needsPostProcessTargets = hasResource(
			resources,
			WEBGPU_FRAME_LOGICAL_RESOURCES.postProcessTargets,
		);
		const needsTransmissionTargets = hasResource(
			resources,
			WEBGPU_FRAME_LOGICAL_RESOURCES.transmissionTargets,
		);
		const needsPlanarReflectionMask = hasResource(
			resources,
			WEBGPU_FRAME_LOGICAL_RESOURCES.planarReflectionMask,
		);
		const needsHiZTarget = hasResource(
			resources,
			WEBGPU_FRAME_LOGICAL_RESOURCES.hiZTarget,
		);
		return {
			mrtSupported,
			deferredSupported,
			deferredActive,
			oitActive,
			transparencyMode: oitActive ? "oit" : "legacy",
			sceneTargetMode,
			deferredGBufferLayout,
			targetRequirements: sceneTargetMode === "single" ? null : {
				sceneTargetMode,
				needsPostProcessTargets,
				needsOITTargets: resources.has(WEBGPU_FRAME_LOGICAL_RESOURCES.oitTargets),
				needsTransmissionTargets,
				needsPlanarReflectionMask,
				needsHiZTarget,
				deferredGBufferLayout,
			},
			needsHiZBuild: hasBooleanDemand(demands, "needsHiZBuild"),
			needsOcclusionTest: hasBooleanDemand(demands, "needsOcclusionTest"),
			enableEarlyZPrepass: options.enableEarlyZPrepass,
			samplePlan: options.samplePlan,
			diagnostics,
		};
	}
}

function mergeFeatureStates(
	demands: readonly WebGPUFrameConfigurationDemand[],
): ReadonlyMap<
	string,
	import("./WebGPUFrameMessages").WebGPUFrameFeatureStateValue
> {
	const states = new Map<
		string,
		import("./WebGPUFrameMessages").WebGPUFrameFeatureStateValue
	>();
	for (const demand of demands) {
		for (const [key, value] of Object.entries(demand.featureStates ?? {})) {
			const prior = states.get(key);
			if (prior !== undefined && prior !== value) {
				throw new Error(
					`WebGPU frame configuration has conflicting feature state "${key}".`,
				);
			}
			states.set(key, value);
		}
	}
	return states;
}

function mergeResourceDemands(
	demands: readonly WebGPUFrameConfigurationDemand[],
): ReadonlySet<import("./WebGPUFrameMessages").WebGPUFrameLogicalResourceId> {
	const resources = new Set<
		import("./WebGPUFrameMessages").WebGPUFrameLogicalResourceId
	>();
	const exclusive = new Map<string, string>();
	for (const demand of demands) {
		for (const resource of demand.resources ?? []) {
			resources.add(resource.id);
			if (!resource.exclusiveGroup) continue;
			const prior = exclusive.get(resource.exclusiveGroup);
			if (prior && prior !== resource.id) {
				throw new Error(
					`WebGPU frame configuration has conflicting exclusive resource ` +
					`demands in group "${resource.exclusiveGroup}".`,
				);
			}
			exclusive.set(resource.exclusiveGroup, resource.id);
		}
	}
	return resources;
}

function targetRank(target: WebGPUFrameTargetClass): number {
	return { single: 0, color: 1, mrt: 2, gbuffer: 3 }[target];
}

function hasResource<TValue>(
	values: ReadonlySet<TValue>,
	value: TValue,
): boolean {
	return values.has(value);
}

function hasBooleanDemand(
	demands: readonly WebGPUFrameConfigurationDemand[],
	key: keyof WebGPUFrameConfigurationDemand,
): boolean {
	return demands.some((demand) => demand[key] === true);
}

function getFeatureState<TValue>(
	states: ReadonlyMap<string, import("./WebGPUFrameMessages").WebGPUFrameFeatureStateValue>,
	key: string,
	fallback: TValue,
): TValue {
	return (states.get(key) ?? fallback) as TValue;
}
