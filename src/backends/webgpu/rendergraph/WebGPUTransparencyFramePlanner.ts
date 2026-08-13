import type { FrameContext, FramePass } from "../../../pipeline/types";

import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphDsl";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import type {
	WebGPUFrameGraphNode,
	WebGPUFrameResourceAllocationSnapshot,
	WebGPUFrameGraphResourceId,
} from "./types";

interface WebGPUTransparencyPlanningState {
	readonly hasOITMeshContributors?: boolean;
	readonly hasTransmissionPackets?: boolean;
	readonly hasAlphaBillboardParticles?: boolean;
	readonly hasAdditiveBillboardParticles?: boolean;
}

/** @internal Owns graph planning for WebGPU transparency and particle stages. */
export class WebGPUTransparencyFramePlanner {
	public plan(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	): WebGPUFrameGraphNode[] {
		if (pass.stage === "main-transparent") {
			return this._createTransparentNodes(pass, context, state);
		}
		return pass.stage === "particles"
			? this._createParticleNodes(pass, context, state)
			: [];
	}

	private _createTransparentNodes(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	): WebGPUFrameGraphNode[] {
		const hasMeshContributors = state.hasOITMeshContributors !== false;
		const hasTransmissionPackets = state.hasTransmissionPackets !== false;
		if (!state.oitActive) {
			const nodes: WebGPUFrameGraphNode[] = [];
			if (hasMeshContributors) {
				nodes.push(createWebGPUFrameGraphNode(
					pass,
					"transparent-forward",
					"WebGPUTransparentForward",
					createWebGPUForwardGraphResources(state, true, context),
				));
			}
			if (hasTransmissionPackets) {
				nodes.push(this._createTransmissionNode(pass, context, state));
			}
			return nodes;
		}
		if (!hasMeshContributors) return [];
		const nodes = [
			this._createOITPrepareNode(pass),
			this._createOITClearNode(pass),
			this._createOITMeshAccumulateNode(pass, context),
		];
		if (state.hasAlphaBillboardParticles !== true) {
			nodes.push(this._createOITResolveNode(pass));
			if (hasTransmissionPackets) {
				nodes.push(this._createTransmissionNode(pass, context, state));
			}
		}
		return nodes;
	}

	private _createParticleNodes(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	): WebGPUFrameGraphNode[] {
		const hasAlphaParticles = state.hasAlphaBillboardParticles !== false;
		const hasAdditiveParticles = state.hasAdditiveBillboardParticles !== false;
		if (!state.oitActive) {
			const nodes: WebGPUFrameGraphNode[] = [];
			if (hasAlphaParticles) {
				nodes.push(createWebGPUFrameGraphNode(
					pass,
					"particle-alpha-forward",
					"WebGPUParticlesAlpha",
					createWebGPUForwardGraphResources(state, true, context),
				));
			}
			if (hasAdditiveParticles) {
				nodes.push(createWebGPUFrameGraphNode(
					pass,
					"particle-additive",
					"WebGPUParticlesAdditive",
					createWebGPUForwardGraphResources(state, true, context),
				));
			}
			return nodes;
		}
		const nodes: WebGPUFrameGraphNode[] = [];
		if (hasAlphaParticles) {
			if (state.hasOITMeshContributors !== true) {
				nodes.push(this._createOITPrepareNode(pass), this._createOITClearNode(pass));
			}
			nodes.push(this._createOITParticleAccumulateNode(pass, context));
			nodes.push(this._createOITResolveNode(pass));
			if (state.hasTransmissionPackets === true) {
				nodes.push(this._createTransmissionNode(pass, context, state));
			}
		}
		if (hasAdditiveParticles) {
			nodes.push(createWebGPUFrameGraphNode(
				pass,
				"particle-additive",
				"WebGPUParticlesAdditive",
				createWebGPUForwardGraphResources(state, true, context),
			));
		}
		return nodes;
	}

	private _createOITPrepareNode(pass: FramePass): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(pass, "oit-prepare", "WebGPUOITPrepare", {
			reads: [readWebGPUFrameGraphResource(
				"frame:scene-color-main",
				"copy-src",
			)],
			writes: [writeWebGPUFrameGraphResource(
				"oit:scene-color-copy",
				"copy-dst",
			)],
		});
	}

	private _createOITClearNode(pass: FramePass): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(pass, "oit-clear", "WebGPUOITClear", {
			writes: [
				writeWebGPUFrameGraphResource("oit:accum", "render-attachment"),
				writeWebGPUFrameGraphResource("oit:reveal", "render-attachment"),
			],
		});
	}

	private _createOITMeshAccumulateNode(
		pass: FramePass,
		context: FrameContext,
	): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(
			pass,
			"oit-mesh-accumulate",
			"WebGPUOITMeshAccumulate",
			this._createOITAccumulateResources(context),
		);
	}

	private _createOITParticleAccumulateNode(
		pass: FramePass,
		context: FrameContext,
	): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(
			pass,
			"oit-particle-accumulate",
			"WebGPUOITParticleAccumulate",
			this._createOITAccumulateResources(context),
		);
	}

	private _createOITAccumulateResources(context: FrameContext) {
		return {
			reads: [
				readWebGPUFrameGraphResource("frame:depth", "depth-attachment"),
				readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
				readWebGPUFrameGraphResource(
					"shadow-transmittance-atlas",
					"texture-binding",
					true,
				),
				...createWebGPUPagedShadowLightingReads(context),
			],
			writes: [
				writeWebGPUFrameGraphResource("oit:accum", "render-attachment"),
				writeWebGPUFrameGraphResource("oit:reveal", "render-attachment"),
			],
		};
	}

	private _createOITResolveNode(pass: FramePass): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(pass, "oit-resolve", "WebGPUOITResolve", {
			reads: [
				readWebGPUFrameGraphResource("oit:scene-color-copy", "texture-binding"),
				readWebGPUFrameGraphResource("oit:accum", "texture-binding"),
				readWebGPUFrameGraphResource("oit:reveal", "texture-binding"),
			],
			writes: [writeWebGPUFrameGraphResource(
				"frame:scene-color-main",
				"render-attachment",
			)],
		});
	}

	private _createTransmissionNode(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	): WebGPUFrameGraphNode {
		return createWebGPUFrameGraphNode(
			pass,
			"transmission",
			"WebGPUTransmission",
			this._withTransmissionCaptureResources(
				createWebGPUForwardGraphResources(state, true, context),
				state,
			),
		);
	}

	private _withTransmissionCaptureResources(
		resources: Pick<WebGPUFrameGraphNode, "reads" | "writes">,
		state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		if (!state.needsTransmissionTargets) return resources;
		const frameAttachmentIds = new Set<WebGPUFrameGraphResourceId>([
			WEBGPU_FRAME_GRAPH_RESOURCES.frameColor,
			WEBGPU_FRAME_GRAPH_RESOURCES.frameDepth,
			WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor,
			WEBGPU_FRAME_GRAPH_RESOURCES.canvasDepth,
		]);
		return {
			reads: [
				...(resources.reads ?? []).filter(
					(resource) => !frameAttachmentIds.has(resource.id),
				),
				readWebGPUFrameGraphResource(
					"frame:scene-color-main",
					"copy-src",
					true,
				),
				readWebGPUFrameGraphResource("frame:depth", "copy-src", true),
			],
			writes: [
				writeWebGPUFrameGraphResource(
					"transmission:scene-color-copy",
					"copy-dst",
				),
				writeWebGPUFrameGraphResource(
					"transmission:lighting",
					"render-attachment",
				),
				...[
					"transmission:surface0",
					"transmission:surface1",
					"transmission:surface2",
				].map((id) => writeWebGPUFrameGraphResource(id, "render-attachment")),
				writeWebGPUFrameGraphResource("transmission:depth", "copy-dst"),
				writeWebGPUFrameGraphResource(
					"transmission:depth",
					"depth-attachment",
				),
			],
		};
	}
}

function createWebGPUPagedShadowLightingReads(
	_context: FrameContext,
) {
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

function createWebGPUForwardGraphResources(
	state: WebGPUFrameResourceAllocationSnapshot & WebGPUTransparencyPlanningState,
	loadExistingColor: boolean,
	_context: FrameContext,
): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
	const useCanvas = state.sceneTargetMode === "single" || !state.hasFrameTargets;
	const color = useCanvas
		? WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor
		: WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
	const depth = useCanvas
		? WEBGPU_FRAME_GRAPH_RESOURCES.canvasDepth
		: WEBGPU_FRAME_GRAPH_RESOURCES.frameDepth;
	const reads = [
		readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
		readWebGPUFrameGraphResource(
			"shadow-transmittance-atlas",
			"texture-binding",
			true,
		),
		...createWebGPUPagedShadowLightingReads(_context),
	];
	if (loadExistingColor) {
		reads.push(readWebGPUFrameGraphResource(color, "render-attachment", true));
		reads.push(readWebGPUFrameGraphResource(depth, "depth-attachment", true));
	}
	return {
		reads,
		writes: [
			writeWebGPUFrameGraphResource(color, "render-attachment"),
			writeWebGPUFrameGraphResource(depth, "depth-attachment"),
		],
	};
}
