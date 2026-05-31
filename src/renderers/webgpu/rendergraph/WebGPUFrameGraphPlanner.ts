import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphStagePlan,
} from "./types";

/**
 * Builds WebGPU-internal frame graph nodes for renderer-level backend stages.
 */
export class WebGPUFrameGraphPlanner {
	public planStage(
		pass: FramePass,
		_context: FrameContext,
		state: WebGPUFrameGraphPlannerState
	): WebGPUFrameGraphStagePlan {
		return {
			pass,
			nodes: this._createNodes(pass, state),
		};
	}

	private _createNodes(
		pass: FramePass,
		state: WebGPUFrameGraphPlannerState
	): WebGPUFrameGraphNode[] {
		switch (pass.stage) {
			case "shadow":
				return [this._node(pass, "shadow", "WebGPUShadow")];
			case "reflection":
				return [
					this._node(
						pass,
						"planar-reflection-capture",
						"WebGPUPlanarReflectionCapture"
					),
				];
			case "main-opaque":
				return [
					this._node(
						pass,
						"opaque-scene",
						state.deferredActive ?
							"WebGPUOpaqueDeferred"
						:	`WebGPUOpaque${state.sceneTargetMode}`
					),
				];
			case "main-transparent":
				return [
					state.oitActive ?
						this._node(pass, "oit-transparent", "WebGPUOITTransparent")
					:	this._node(pass, "transparent-scene", "WebGPUTransparent"),
				];
			case "particles":
				return [
					state.oitActive ?
						this._node(pass, "oit-particles", "WebGPUOITParticles")
					:	this._node(pass, "particles", "WebGPUParticles"),
				];
			default:
				return [];
		}
	}

	private _node(
		pass: FramePass,
		kind: WebGPUFrameGraphNode["kind"],
		label: string
	): WebGPUFrameGraphNode {
		return {
			id: `${pass.stage}:${kind}`,
			stage: pass.stage,
			kind,
			label,
		};
	}
}
