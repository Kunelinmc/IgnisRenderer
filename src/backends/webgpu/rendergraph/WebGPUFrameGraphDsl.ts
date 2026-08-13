import type { FramePass } from "../../../pipeline/types";

import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphResourceId,
	WebGPUFrameGraphResourceRef,
} from "./types";

/** @internal Creates one backend-private logical frame-graph node. */
export function createWebGPUFrameGraphNode(
	pass: FramePass,
	kind: WebGPUFrameGraphNode["kind"],
	label: string,
	resources: Pick<
		WebGPUFrameGraphNode,
		"creates" | "reads" | "writes" | "destroys"
	> = {},
	localId: string = kind,
): WebGPUFrameGraphNode {
	return {
		id: `${pass.stage}:${kind}`,
		localId,
		stage: pass.stage,
		kind,
		label,
		...resources,
	};
}

/** @internal Declares one logical resource read. */
export function readWebGPUFrameGraphResource(
	id: WebGPUFrameGraphResourceId,
	usage: WebGPUFrameGraphResourceRef["usage"],
	optional = false,
): WebGPUFrameGraphResourceRef {
	return { id, usage, optional };
}

/** @internal Declares one logical resource write. */
export function writeWebGPUFrameGraphResource(
	id: WebGPUFrameGraphResourceId,
	usage: WebGPUFrameGraphResourceRef["usage"],
	optional = false,
): WebGPUFrameGraphResourceRef {
	return { id, usage, optional };
}
