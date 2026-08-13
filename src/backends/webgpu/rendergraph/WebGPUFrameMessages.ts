import type { PlannedPostProcessPass } from "../../../postprocess";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { FrameContext, FramePass } from "../../../pipeline/types";

import type {
	WebGPUFrameCapabilitySnapshot,
	WebGPUFrameConfiguration,
	WebGPUFrameConfigurationOptions,
} from "./WebGPUFrameConfiguration";
import { defineWebGPUFrameMessage } from "./WebGPUFrameMessage";
import type { WebGPUFrameGraphContribution } from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameResourceAllocationSnapshot,
	WebGPUFrameGraphResourceId,
} from "./types";

export const WEBGPU_FRAME_CONTEXT_MESSAGE = defineWebGPUFrameMessage<FrameContext>({
	id: "webgpu:frame-context",
	ownerId: "frame-core",
	phase: "analysis",
	seeded: true,
});

export const WEBGPU_FRAME_PACKETS_MESSAGE =
	defineWebGPUFrameMessage<PreparedFramePacketSet>({
		id: "webgpu:frame-packets",
		ownerId: "frame-core",
		phase: "analysis",
		seeded: true,
	});

export const WEBGPU_POST_PROCESS_PASSES_MESSAGE =
	defineWebGPUFrameMessage<readonly PlannedPostProcessPass[]>({
		id: "webgpu:post-process-passes",
		ownerId: "frame-core",
		phase: "analysis",
		seeded: true,
	});

export interface WebGPUFrameConfigurationRequest {
	readonly context: FrameContext;
	readonly capabilities: WebGPUFrameCapabilitySnapshot;
	readonly options: WebGPUFrameConfigurationOptions;
}

export const WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFrameConfigurationRequest>({
		id: "webgpu:configuration-request",
		ownerId: "frame-core",
		phase: "configuration",
		seeded: true,
	});

export type WebGPUFrameTargetClass = "single" | "color" | "mrt" | "gbuffer";

export const WEBGPU_FRAME_LOGICAL_RESOURCES = {
	postProcessTargets: "frame:post-process-targets",
	oitTargets: "frame:oit-targets",
	transmissionTargets: "frame:transmission-targets",
	planarReflectionMask: "frame:planar-reflection-mask",
	hiZTarget: "frame:hiz-target",
} as const;

export type WebGPUFrameLogicalResourceId =
	(typeof WEBGPU_FRAME_LOGICAL_RESOURCES)[keyof typeof WEBGPU_FRAME_LOGICAL_RESOURCES];

export interface WebGPUFrameLogicalResourceDemand {
	readonly id: WebGPUFrameLogicalResourceId;
	readonly exclusiveGroup?: string;
}

export const WEBGPU_FRAME_FEATURE_STATES = {
	deferredSupported: "deferred:supported",
	deferredActive: "deferred:active",
	deferredGBufferLayout: "deferred:gbuffer-layout",
	oitActive: "transparency:oit-active",
	transparencyMode: "transparency:mode",
} as const;

export type WebGPUFrameFeatureStateValue = boolean | string | number;

export interface WebGPUFrameConfigurationDemand {
	readonly source: string;
	readonly targetClass?: WebGPUFrameTargetClass;
	readonly resources?: readonly WebGPUFrameLogicalResourceDemand[];
	readonly featureStates?: Readonly<Record<string, WebGPUFrameFeatureStateValue>>;
	readonly needsHiZBuild?: boolean;
	readonly needsOcclusionTest?: boolean;
	readonly needsPlanarReflectionComposite?: boolean;
	readonly hasOITMeshContributors?: boolean;
	readonly hasTransmissionPackets?: boolean;
	readonly hasAlphaBillboardParticles?: boolean;
	readonly hasAdditiveBillboardParticles?: boolean;
	readonly diagnostics?: readonly {
		readonly code: string;
		readonly message: string;
	}[];
}

export const WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFrameConfigurationDemand>({
		id: "webgpu:configuration-demand",
		ownerId: "frame-core",
		phase: "configuration",
		cardinality: "many",
	});

export const WEBGPU_FRAME_CONFIGURATION_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFrameConfiguration>({
		id: "webgpu:configuration",
		ownerId: "frame-core",
		phase: "configuration",
	});

export interface WebGPUFramePlanningRequest {
	readonly pass: FramePass;
	readonly context: FrameContext;
	readonly state: WebGPUFrameResourceAllocationSnapshot;
	readonly finalization?: boolean;
	readonly finalColorResource?: WebGPUFrameGraphResourceId;
}

export const WEBGPU_FRAME_PLANNING_REQUEST_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFramePlanningRequest>({
		id: "webgpu:planning-request",
		ownerId: "frame-core",
		phase: "planning",
		seeded: true,
	});

export interface WebGPUFrameGraphFragmentMessage {
	readonly moduleId: string;
	readonly contribution: WebGPUFrameGraphContribution;
}

export const WEBGPU_FRAME_GRAPH_FRAGMENT_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFrameGraphFragmentMessage>({
		id: "webgpu:graph-fragment",
		ownerId: "frame-core",
		phase: "planning",
		cardinality: "many",
	});

export interface WebGPUFrameFinalOutputMessage {
	readonly resource: WebGPUFrameGraphResourceId;
	readonly colorDomain: import("../../../postprocess/PostProcessPass").PostProcessColorDomain;
}

export const WEBGPU_FRAME_FINAL_OUTPUT_MESSAGE =
	defineWebGPUFrameMessage<WebGPUFrameFinalOutputMessage>({
		id: "webgpu:final-output",
		ownerId: "frame-core",
		phase: "planning",
		cardinality: "many",
	});
