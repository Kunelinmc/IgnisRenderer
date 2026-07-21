import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphResourceRef,
	WebGPUFrameGraphResourceId,
	WebGPUFrameGraphStagePlan,
} from "./types";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";

const DEFERRED_GBUFFER_RENDER_RESOURCE_IDS = [
	"gbuffer:albedo-alpha",
	"gbuffer:normal-rough-metal",
	"gbuffer:emissive-occlusion",
	"gbuffer:motion-depth",
	"gbuffer:specular",
	"gbuffer:coat-sheen",
	"gbuffer:sheen-reflectance",
] as const;

const DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS = [
	"gbuffer:material-ext0",
	"gbuffer:material-ext1",
	"gbuffer:material-ext2",
	"gbuffer:material-ext3",
] as const;

const DEFERRED_GBUFFER_RESOURCE_IDS = [
	...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS,
	...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS,
] as const;

/**
 * Builds WebGPU-internal frame graph nodes for renderer-level backend stages.
 */
export class WebGPUFrameGraphPlanner {
	private readonly _stagePlanners: Map<
		FramePass["stage"],
		(
			pass: FramePass,
			context: FrameContext,
			state: WebGPUFrameGraphPlannerState,
		) => WebGPUFrameGraphNode[]
	>;

	constructor() {
		this._stagePlanners = this._createStagePlanners();
	}

	public planStage(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphStagePlan {
		const planner = this._stagePlanners.get(pass.stage);
		return {
			pass,
			nodes: planner ? planner(pass, context, state) : [],
		};
	}

	public planFinalization(
		pass: FramePass,
		state: WebGPUFrameGraphPlannerState,
		sourceResource: WebGPUFrameGraphResourceId = WEBGPU_FRAME_GRAPH_RESOURCES.frameColor,
	): WebGPUFrameGraphStagePlan {
		const r = WEBGPU_FRAME_GRAPH_RESOURCES;
		return {
			pass,
			nodes:
				state.hasFrameTargets === true
					? [this._node(pass, "presentation", "WebGPUPresentation", {
							reads: [this._read(sourceResource, "texture-binding")],
							writes: [this._write(r.canvasColor, "present")],
						})]
					: [],
		};
	}

	private _createStagePlanners(): Map<
		FramePass["stage"],
		(
			pass: FramePass,
			context: FrameContext,
			state: WebGPUFrameGraphPlannerState,
		) => WebGPUFrameGraphNode[]
	> {
		return new Map([
			["shadow", (pass, context) => this._createShadowNodes(pass, context)],
			[
				"reflection",
				(pass) => [
					this._node(pass, "planar-reflection-capture", "WebGPUPlanarReflectionCapture", {
						reads: [this._read("shadow-atlas", "texture-binding", true)],
						writes: [this._write("planar-reflection:capture", "render-attachment")],
					}),
				],
			],
			[
				"main-opaque",
				(pass, context, state) => {
					if (state.deferredActive && state.sceneTargetMode === "gbuffer") {
						const nodes: WebGPUFrameGraphNode[] = [
							this._node(
								pass,
								"opaque-scene",
								"WebGPUGBuffer",
								this._createOpaqueResources(state, context),
							),
						];
						if ((context.scene?.decalPackets.length ?? 0) > 0) {
							nodes.push(
								this._node(
									pass,
									"deferred-decal",
									"WebGPUDeferredDecal",
									this._createDeferredDecalResources(),
								),
							);
						}
						nodes.push(
							this._node(
								pass,
								"deferred-lighting",
								"WebGPUDeferredLighting",
								this._createDeferredLightingResources(context),
							),
						);
						return this._appendPagedShadowFeedbackNode(
							pass,
							context,
							state,
							this._appendHiZNodes(
								pass,
								state,
								this._appendReflectionComposite(pass, state, nodes),
							),
						);
					}
					return this._appendPagedShadowFeedbackNode(
						pass,
						context,
						state,
						this._appendHiZNodes(pass, state, this._appendReflectionComposite(pass, state, [
							this._node(
								pass,
								"opaque-scene",
								`WebGPUOpaque${state.sceneTargetMode}`,
								this._createOpaqueResources(state, context),
							),
						])),
					);
				},
			],
			[
				"main-transparent",
				(pass, context, state) => this._createTransparentNodes(pass, context, state),
			],
			[
				"particles",
				(pass, context, state) => this._createParticleNodes(pass, context, state),
			],
		]);
	}

	private _createTransparentNodes(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphNode[] {
		const hasMeshContributors = state.hasOITMeshContributors !== false;
		const hasTransmissionPackets = state.hasTransmissionPackets !== false;
		if (!state.oitActive) {
			const nodes: WebGPUFrameGraphNode[] = [];
			if (hasMeshContributors) {
				nodes.push(this._node(
					pass,
					"transparent-forward",
					"WebGPUTransparentForward",
					this._createForwardResources(state, true, context),
				));
			}
			if (hasTransmissionPackets) {
				nodes.push(this._node(
					pass,
					"transmission",
					"WebGPUTransmission",
					this._withTransmissionCaptureResources(
						this._createForwardResources(state, true, context),
						state,
					),
				));
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
				nodes.push(this._node(
					pass,
					"transmission",
					"WebGPUTransmission",
					this._withTransmissionCaptureResources(
						this._createForwardResources(state, true, context),
						state,
					),
				));
			}
		}
		return nodes;
	}

	private _createParticleNodes(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphNode[] {
		const hasAlphaParticles = state.hasAlphaBillboardParticles !== false;
		const hasAdditiveParticles = state.hasAdditiveBillboardParticles !== false;
		if (!state.oitActive) {
			const nodes: WebGPUFrameGraphNode[] = [];
			if (hasAlphaParticles) {
				nodes.push(this._node(
					pass,
					"particle-alpha-forward",
					"WebGPUParticlesAlpha",
					this._createForwardResources(state, true, context),
				));
			}
			if (hasAdditiveParticles) {
				nodes.push(this._node(
					pass,
					"particle-additive",
					"WebGPUParticlesAdditive",
					this._createForwardResources(state, true, context),
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
				nodes.push(this._node(
					pass,
					"transmission",
					"WebGPUTransmission",
					this._withTransmissionCaptureResources(
						this._createForwardResources(state, true, context),
						state,
					),
				));
			}
		}
		if (hasAdditiveParticles) {
			nodes.push(this._node(
				pass,
				"particle-additive",
				"WebGPUParticlesAdditive",
				this._createForwardResources(state, true, context),
			));
		}
		return nodes;
	}

	private _createOITPrepareNode(pass: FramePass): WebGPUFrameGraphNode {
		return this._node(pass, "oit-prepare", "WebGPUOITPrepare", {
			reads: [this._read("frame:scene-color-main", "copy-src")],
			writes: [this._write("oit:scene-color-copy", "copy-dst")],
		});
	}

	private _createOITClearNode(pass: FramePass): WebGPUFrameGraphNode {
		return this._node(pass, "oit-clear", "WebGPUOITClear", {
			writes: [
				this._write("oit:accum", "render-attachment"),
				this._write("oit:reveal", "render-attachment"),
			],
		});
	}

	private _createOITMeshAccumulateNode(
		pass: FramePass,
		context: FrameContext,
	): WebGPUFrameGraphNode {
		return this._node(pass, "oit-mesh-accumulate", "WebGPUOITMeshAccumulate", {
			reads: [
				this._read("frame:depth", "depth-attachment"),
				...this._createPagedShadowLightingReads(context),
			],
			writes: [
				this._write("oit:accum", "render-attachment"),
				this._write("oit:reveal", "render-attachment"),
			],
		});
	}

	private _createOITParticleAccumulateNode(
		pass: FramePass,
		context: FrameContext,
	): WebGPUFrameGraphNode {
		return this._node(pass, "oit-particle-accumulate", "WebGPUOITParticleAccumulate", {
			reads: [
				this._read("frame:depth", "depth-attachment"),
				...this._createPagedShadowLightingReads(context),
			],
			writes: [
				this._write("oit:accum", "render-attachment"),
				this._write("oit:reveal", "render-attachment"),
			],
		});
	}

	private _createOITResolveNode(pass: FramePass): WebGPUFrameGraphNode {
		return this._node(pass, "oit-resolve", "WebGPUOITResolve", {
			reads: [
				this._read("oit:scene-color-copy", "texture-binding"),
				this._read("oit:accum", "texture-binding"),
				this._read("oit:reveal", "texture-binding"),
			],
			writes: [this._write("frame:scene-color-main", "render-attachment")],
		});
	}

	private _appendReflectionComposite(
		pass: FramePass,
		state: WebGPUFrameGraphPlannerState,
		nodes: WebGPUFrameGraphNode[],
	): WebGPUFrameGraphNode[] {
		if (!state.needsPlanarReflectionComposite) return nodes;
		const r = WEBGPU_FRAME_GRAPH_RESOURCES;
		nodes.push(this._node(pass, "planar-reflection-composite", "WebGPUPlanarReflectionComposite", {
			reads: [
				this._read(r.planarReflectionCapture, "texture-binding"),
				this._read(r.frameColor, "render-attachment", true),
			],
			writes: [
				this._write(r.frameColor, "render-attachment", true),
				this._write(r.planarReflectionMask, "render-attachment", true),
			],
		}));
		return nodes;
	}

	private _node(
		pass: FramePass,
		kind: WebGPUFrameGraphNode["kind"],
		label: string,
		resources: Pick<WebGPUFrameGraphNode, "creates" | "reads" | "writes" | "destroys"> = {},
	): WebGPUFrameGraphNode {
		return {
			id: `${pass.stage}:${kind}`,
			stage: pass.stage,
			kind,
			label,
			...resources,
		};
	}

	private _createShadowNodes(pass: FramePass, context: FrameContext): WebGPUFrameGraphNode[] {
		const nodes = [
			this._node(pass, "shadow", "WebGPUShadow", {
				writes: [this._write("shadow-atlas", "render-attachment")],
			}),
		];
		if (!this._hasPagedShadowWork(context)) {
			return nodes;
		}
		nodes.push(
			this._node(pass, "paged-shadow-page-mark", "WebGPUPagedShadowPageMark", {
				reads: [this._read("paged-shadow:feedback-flags", "storage-binding", true)],
				writes: [
					this._write("paged-shadow:page-request-flags", "storage-binding"),
					this._write("paged-shadow:page-requests", "storage-binding"),
					this._write("paged-shadow:counters", "storage-binding"),
				],
			}),
			this._node(pass, "paged-shadow-page-allocate", "WebGPUPagedShadowPageAllocate", {
				reads: [
					this._read("paged-shadow:page-requests", "storage-binding"),
					this._read("paged-shadow:page-request-flags", "storage-binding"),
				],
				writes: [
					this._write("paged-shadow:page-table", "storage-binding"),
					this._write("paged-shadow:page-metadata", "storage-binding"),
					this._write("paged-shadow:residency-state", "storage-binding"),
					this._write("paged-shadow:free-list", "storage-binding"),
					this._write("paged-shadow:counters", "storage-binding"),
					this._write("paged-shadow:dirty-physical-pages", "storage-binding"),
				],
			}),
			this._node(pass, "paged-shadow-page-table-copy", "WebGPUPagedShadowPageTableCopy", {
				reads: [this._read("paged-shadow:page-table", "storage-binding")],
				writes: [this._write("paged-shadow:page-table-texture", "storage-binding")],
			}),
			this._node(pass, "paged-shadow-depth", "WebGPUPagedShadowDepth", {
				reads: [
					this._read("paged-shadow:page-table", "storage-binding"),
					this._read("paged-shadow:page-metadata", "storage-binding"),
					this._read("paged-shadow:dirty-physical-pages", "storage-binding"),
				],
				writes: [
					this._write("paged-shadow:draw-instances", "storage-binding"),
					this._write("paged-shadow:draw-indirect-args", "storage-binding"),
					this._write("paged-shadow:clear-draw-indirect-args", "storage-binding"),
					this._write("paged-shadow:physical-depth", "render-attachment"),
				],
			}),
		);
		return nodes;
	}

	private _hasPagedShadowWork(context: FrameContext): boolean {
		if (context.backendProfile?.shadow?.supportsPagedShadowRendering !== true) {
			return false;
		}
		const shadowMaps = context.shadowMaps;
		if (!shadowMaps || typeof shadowMaps.values !== "function") {
			return false;
		}
		for (const renderSet of shadowMaps.values()) {
			if (renderSet.storageMode === "paged") {
				return true;
			}
		}
		return false;
	}

	private _createOpaqueResources(
		state: WebGPUFrameGraphPlannerState,
		context: FrameContext,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		if (state.sceneTargetMode === "single") {
			return this._createForwardResources(state, false, context);
		}
		if (state.sceneTargetMode === "color") {
			return {
				reads: [
					this._read("shadow-atlas", "texture-binding", true),
					...this._createPagedShadowLightingReads(context),
				],
				writes: [
					this._write("frame:scene-color-main", "render-attachment"),
					this._write("frame:depth", "depth-attachment"),
				],
			};
		}
		const writes = [
			this._write("frame:scene-color-main", "render-attachment"),
			this._write("gbuffer:albedo-alpha", "render-attachment"),
			this._write("gbuffer:normal-rough-metal", "render-attachment"),
			this._write("gbuffer:emissive-occlusion", "render-attachment"),
			this._write("gbuffer:motion-depth", "render-attachment"),
			this._write("frame:depth", "depth-attachment"),
		];
		if (state.sceneTargetMode === "gbuffer") {
			writes.push(
				this._write("gbuffer:specular", "render-attachment"),
				this._write("gbuffer:coat-sheen", "render-attachment"),
				this._write("gbuffer:sheen-reflectance", "render-attachment"),
				this._write("gbuffer:material-ext0", "storage-binding"),
				this._write("gbuffer:material-ext1", "storage-binding"),
				this._write("gbuffer:material-ext2", "storage-binding"),
				this._write("gbuffer:material-ext3", "storage-binding"),
			);
		}
		const reads = [
			this._read("shadow-atlas", "texture-binding", true),
			...this._createPagedShadowLightingReads(context),
			this._read("planar-reflection:capture", "texture-binding", true),
		];
		if (state.needsPlanarReflectionMask) {
			writes.push(this._write("planar-reflection:mask", "render-attachment"));
		}
		return { reads, writes };
	}

	private _createDeferredDecalResources(): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		return {
			reads: DEFERRED_GBUFFER_RESOURCE_IDS.map((id) => this._read(id, "copy-src")),
			writes: [
				...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS.map((id) =>
					this._write(id, "render-attachment"),
				),
				...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS.map((id) =>
					this._write(id, "storage-binding"),
				),
			],
		};
	}

	private _createDeferredLightingResources(
		context: FrameContext,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		return {
			reads: [
				...DEFERRED_GBUFFER_RESOURCE_IDS.map((id) => this._read(id, "texture-binding")),
				this._read("shadow-atlas", "texture-binding", true),
				...this._createPagedShadowLightingReads(context),
			],
			writes: [this._write("frame:scene-color-main", "render-attachment")],
		};
	}

	private _appendHiZNodes(
		pass: FramePass,
		state: WebGPUFrameGraphPlannerState,
		nodes: WebGPUFrameGraphNode[],
	): WebGPUFrameGraphNode[] {
		if (!state.needsHiZBuild || state.sceneTargetMode === "single") {
			return nodes;
		}
		nodes.push(
			this._node(pass, "hiz-build", "WebGPUHiZBuild", {
				reads: [this._read("gbuffer:motion-depth", "texture-binding")],
				writes: [this._write("frame:hiz", "storage-binding")],
			}),
		);
		if (!state.needsOcclusionTest) return nodes;
		nodes.push(
			this._node(pass, "occlusion-test", "WebGPUOcclusionTest", {
				reads: [this._read("frame:hiz", "texture-binding")],
				writes: [this._write("occlusion:results", "storage-binding")],
			}),
		);
		return nodes;
	}

	private _appendPagedShadowFeedbackNode(
		pass: FramePass,
		context: FrameContext,
		state: WebGPUFrameGraphPlannerState,
		nodes: WebGPUFrameGraphNode[],
	): WebGPUFrameGraphNode[] {
		if (!this._hasPagedShadowWork(context) || state.sceneTargetMode === "single") {
			return nodes;
		}
		let hasScreenFeedback = false;
		for (const renderSet of context.shadowMaps.values()) {
			if (
				renderSet.storageMode === "paged" &&
				renderSet.layout?.paged?.feedbackMode === "screen-feedback"
			) {
				hasScreenFeedback = true;
				break;
			}
		}
		if (!hasScreenFeedback) {
			return nodes;
		}
		nodes.push(
			this._node(pass, "paged-shadow-feedback", "WebGPUPagedShadowFeedback", {
				reads: [
					this._read("frame:depth", "texture-binding", true),
					this._read("paged-shadow:page-table", "storage-binding", true),
				],
				writes: [this._write("paged-shadow:next-feedback-flags", "storage-binding")],
			}),
		);
		return nodes;
	}

	private _withTransmissionCaptureResources(
		resources: Pick<WebGPUFrameGraphNode, "reads" | "writes">,
		state: WebGPUFrameGraphPlannerState,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		if (!state.needsTransmissionTargets) {
			return resources;
		}
		return {
			reads: [...(resources.reads ?? []), ...this._createTransmissionReads(state)],
			writes: [...(resources.writes ?? []), ...this._createTransmissionWrites(state)],
		};
	}

	private _createTransmissionReads(
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphResourceRef[] {
		if (!state.needsTransmissionTargets) {
			return [];
		}
		return [
			this._read("frame:scene-color-main", "copy-src", true),
			this._read("frame:depth", "copy-src", true),
		];
	}

	private _createTransmissionDepthReads(
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphResourceRef[] {
		if (!state.needsTransmissionTargets) {
			return [];
		}
		return [this._read("frame:depth", "copy-src", true)];
	}

	private _createTransmissionWrites(
		state: WebGPUFrameGraphPlannerState,
	): WebGPUFrameGraphResourceRef[] {
		if (!state.needsTransmissionTargets) {
			return [];
		}
		return [
			this._write("transmission:scene-color-copy", "copy-dst"),
			this._write("transmission:lighting", "render-attachment"),
			this._write("transmission:surface0", "render-attachment"),
			this._write("transmission:surface1", "render-attachment"),
			this._write("transmission:surface2", "render-attachment"),
			this._write("transmission:depth", "copy-dst"),
			this._write("transmission:depth", "depth-attachment"),
		];
	}

	private _createForwardResources(
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
			this._read("shadow-atlas", "texture-binding", true),
			...this._createPagedShadowLightingReads(context),
		];
		if (loadExistingColor) {
			reads.push(this._read(sceneColor, "texture-binding", true));
			reads.push(this._read(depth, "depth-attachment", true));
		}
		return {
			reads,
			writes: [
				this._write(sceneColor, "render-attachment"),
				this._write(depth, "depth-attachment"),
			],
		};
	}

	private _createPagedShadowLightingReads(context: FrameContext): WebGPUFrameGraphResourceRef[] {
		if (!this._hasPagedShadowWork(context)) {
			return [];
		}
		return [
			this._read("paged-shadow:page-table-texture", "texture-binding", true),
			this._read("paged-shadow:physical-depth", "texture-binding", true),
		];
	}

	private _read(
		id: WebGPUFrameGraphResourceId,
		usage: WebGPUFrameGraphResourceRef["usage"],
		optional = false,
	): WebGPUFrameGraphResourceRef {
		return { id, usage, optional };
	}

	private _write(
		id: WebGPUFrameGraphResourceId,
		usage: WebGPUFrameGraphResourceRef["usage"],
		optional = false,
	): WebGPUFrameGraphResourceRef {
		return { id, usage, optional };
	}
}
