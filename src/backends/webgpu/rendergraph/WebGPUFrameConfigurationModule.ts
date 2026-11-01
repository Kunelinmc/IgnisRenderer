import { TextureFormat } from "../../../core/TextureFormat";
import type { FrameContext } from "../../../pipeline/types";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import type {
	WebGPUFrameConfiguration,
	WebGPUFrameSamplePlan,
} from "./WebGPUFrameConfiguration";
import type { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_FRAME_FEATURE_STATES,
	type WebGPUFrameConfigurationDemand,
	type WebGPUFrameTargetClass,
} from "./WebGPUFrameMessages";
import type {
	WebGPUFrameMessageHandler,
	WebGPUFrameMessageSnapshot,
} from "./WebGPUFrameMessage";
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
		for (const demand of demands) {
			if (demand.targetClass && targetRank(demand.targetClass) > targetRank(targetClass)) {
				targetClass = demand.targetClass;
			}
		}
		const oitActive = getFeatureState(
			featureStates,
			WEBGPU_FRAME_FEATURE_STATES.oitActive,
			false,
		);
		if (!mrtSupported) {
			targetClass = "single";
		}
		const sceneTargetMode = targetClass;
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
			oitActive,
			transparencyMode: oitActive ? "oit" : "legacy",
			sceneTargetMode,
			targetRequirements: sceneTargetMode === "single" ? null : {
				sceneTargetMode,
				needsPostProcessTargets,
				needsOITTargets: resources.has(WEBGPU_FRAME_LOGICAL_RESOURCES.oitTargets),
				needsTransmissionTargets,
				needsPlanarReflectionMask,
				needsHiZTarget,
			},
			needsHiZBuild: hasBooleanDemand(demands, "needsHiZBuild"),
			needsOcclusionTest: hasBooleanDemand(demands, "needsOcclusionTest"),
			enableEarlyZPrepass: options.enableEarlyZPrepass,
			samplePlan: options.samplePlan,
			diagnostics,
		};
	}
}

/**
 * Resolves target demands and effective sampling without allocating frame targets.
 *
 * @internal WebGPU frame configuration shared by execution and warmup;
 * applications use `Renderer.renderFrame()` and `Renderer.warmup()`.
 * @param input Retained feature analysis, device policy, and sample-count resolver.
 * @returns The converged configuration and its sealed configuration messages.
 * @sideEffects May populate device sample-count capability caches through probes.
 */
export async function resolveWebGPUFrameConfiguration(input: {
	readonly modules: Pick<WebGPUFrameGraphModuleRegistry, "dispatchMessages">;
	readonly analysisMessages: WebGPUFrameMessageSnapshot;
	readonly context: FrameContext;
	readonly device: GPUDevice | null;
	readonly sampleCountResolver: Pick<WebGPUSampleCountResolver, "resolveDomainSampleCount">;
	readonly requestedSampleCount: number;
	readonly enableEarlyZPrepass: boolean;
	readonly supportsInFrameTextureCopy: boolean;
}): Promise<{
	readonly configuration: WebGPUFrameConfiguration;
	readonly messages: WebGPUFrameMessageSnapshot;
}> {
	const capabilities = {
		maxColorAttachments: input.device?.limits?.maxColorAttachments ?? 8,
		maxColorAttachmentBytesPerSample:
			input.device?.limits?.maxColorAttachmentBytesPerSample ?? 32,
	};
	let samplePlan: WebGPUFrameSamplePlan = {
		requestedSampleCount: input.requestedSampleCount,
		sampleCount: input.requestedSampleCount,
		selectionSignature: "main-scene:pending",
		runtimeFallbackActive: false,
	};
	const visited = new Set<string>();
	for (let attempts = 0; attempts < 6; attempts++) {
		const messages = await input.modules.dispatchMessages("configuration", {
			prior: input.analysisMessages,
			seeds: [{
				descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
				value: {
					context: input.context,
					capabilities,
					options: {
						enableEarlyZPrepass: input.enableEarlyZPrepass,
						samplePlan,
						supportsInFrameTextureCopy: input.supportsInFrameTextureCopy,
					},
				},
			}],
		});
		const configuration = messages.get(WEBGPU_FRAME_CONFIGURATION_MESSAGE);
		const formats = getSampleCountProbeFormats(configuration);
		let nextPlan: WebGPUFrameSamplePlan;
		if (formats.length === 0) {
			nextPlan = {
				requestedSampleCount: input.requestedSampleCount,
				sampleCount: 1,
				selectionSignature: "main-scene:single",
				runtimeFallbackActive: false,
			};
		} else {
			const selection = input.sampleCountResolver.resolveDomainSampleCount(
				"main-scene", input.requestedSampleCount, formats,
			);
			nextPlan = {
				requestedSampleCount: selection.requestedSampleCount,
				sampleCount: Math.min(samplePlan.sampleCount, selection.sampleCount),
				selectionSignature: selection.signature,
				runtimeFallbackActive: selection.runtimeFallbackActive,
			};
		}
		if (nextPlan.sampleCount === samplePlan.sampleCount &&
			nextPlan.selectionSignature === samplePlan.selectionSignature) {
			return { configuration, messages };
		}
		const stateKey = `${nextPlan.sampleCount}|${nextPlan.selectionSignature}`;
		if (visited.has(stateKey)) {
			throw new Error("WebGPU scene sample-count planning did not converge.");
		}
		visited.add(stateKey);
		samplePlan = nextPlan;
	}
	throw new Error("WebGPU scene sample-count planning exceeded its retry limit.");
}

function getSampleCountProbeFormats(
	configuration: WebGPUFrameConfiguration,
): GPUTextureFormat[] {
	const requirements = configuration.targetRequirements;
	if (!requirements) return [];
	const formats = new Set<GPUTextureFormat>([
		TextureFormat.RGBA16Float as GPUTextureFormat,
		TextureFormat.Depth32Float as GPUTextureFormat,
	]);
	if (requirements.sceneTargetMode === "mrt") {
		formats.add(TextureFormat.RGBA8Unorm as GPUTextureFormat);
	}
	if (requirements.needsPlanarReflectionMask) {
		formats.add(TextureFormat.R8Unorm as GPUTextureFormat);
	}
	return Array.from(formats);
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
	return { single: 0, color: 1, mrt: 2 }[target];
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
