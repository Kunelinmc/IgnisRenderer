import type { FrameContext, FramePass } from "../../../pipeline/types";

import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphResourceId,
	WebGPUFrameGraphResourceRef,
} from "./types";

export function createWebGPUFrameGraphNode(
	pass: FramePass,
	kind: WebGPUFrameGraphNode["kind"],
	label: string,
	resources: Pick<
		WebGPUFrameGraphNode,
		"creates" | "reads" | "writes" | "destroys"
	> = {},
): WebGPUFrameGraphNode {
	return {
		id: `${pass.stage}:${kind}`,
		stage: pass.stage,
		kind,
		label,
		...resources,
	};
}

export function readWebGPUFrameGraphResource(
	id: WebGPUFrameGraphResourceId,
	usage: WebGPUFrameGraphResourceRef["usage"],
	optional = false,
): WebGPUFrameGraphResourceRef {
	return { id, usage, optional };
}

export function writeWebGPUFrameGraphResource(
	id: WebGPUFrameGraphResourceId,
	usage: WebGPUFrameGraphResourceRef["usage"],
	optional = false,
): WebGPUFrameGraphResourceRef {
	return { id, usage, optional };
}

export function hasWebGPUPagedShadowWork(context: FrameContext): boolean {
	return context.shadowPlan.hasPagedWork;
}

export function createWebGPUPagedShadowLightingReads(
	context: FrameContext,
): WebGPUFrameGraphResourceRef[] {
	if (!hasWebGPUPagedShadowWork(context)) return [];
	return [
		readWebGPUFrameGraphResource(
			"paged-shadow:page-table-texture",
			"texture-binding",
			true,
		),
		readWebGPUFrameGraphResource(
			"paged-shadow:physical-depth",
			"texture-binding",
			true,
		),
	];
}

export function createWebGPUForwardGraphResources(
	state: WebGPUFrameGraphPlannerState,
	loadExistingColor: boolean,
	context: FrameContext,
): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
	const useCanvas = state.sceneTargetMode === "single" || !state.hasFrameTargets;
	const sceneColor = useCanvas
		? WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor
		: WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
	const depth = useCanvas
		? WEBGPU_FRAME_GRAPH_RESOURCES.canvasDepth
		: WEBGPU_FRAME_GRAPH_RESOURCES.frameDepth;
	const reads: WebGPUFrameGraphResourceRef[] = [
		readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
		readWebGPUFrameGraphResource(
			"shadow-transmittance-atlas",
			"texture-binding",
			true,
		),
		...createWebGPUPagedShadowLightingReads(context),
	];
	if (loadExistingColor) {
		reads.push(readWebGPUFrameGraphResource(sceneColor, "render-attachment", true));
		reads.push(readWebGPUFrameGraphResource(depth, "depth-attachment", true));
	}
	return {
		reads,
		writes: [
			writeWebGPUFrameGraphResource(sceneColor, "render-attachment"),
			writeWebGPUFrameGraphResource(depth, "depth-attachment"),
		],
	};
}
