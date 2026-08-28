import type { WebGPUShadowRenderProvider } from "../WebGPUResourceContracts";
import type { WebGPUPagedShadowFrameRequest } from "../WebGPUPagedShadowTechnique";

import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	createWebGPUFrameGraphNode,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphDsl";
import type {
	WebGPURecordingFrameSession as WebGPUFrameSession,
} from "./WebGPUFrameSession";

/** @internal Owns shadow graph-node execution for one WebGPU runtime. */
export class WebGPUShadowFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "shadow";
	public readonly executors = {
		shadow: async (_node: unknown, session: WebGPUFrameSession) => {
			const packets = this._requirePackets(session);
			await this._renderer.renderShadows(
				session.context,
				packets,
				session.commands.encoder ?? undefined,
			);
		},
		"paged-shadow-page-mark": async (_node: unknown, session: WebGPUFrameSession) => {
			const request = this._createRequest(session);
			this._renderer.preparePagedShadowFrame(request);
			await this._renderer.recordPagedShadowPageMarkPass(request);
		},
		"paged-shadow-page-allocate": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._renderer.recordPagedShadowPageAllocationPass(this._createRequest(session));
		},
		"paged-shadow-page-table-copy": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._renderer.recordPagedShadowPageTableCopyPass(this._createRequest(session));
		},
		"paged-shadow-depth": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._renderer.recordPagedShadowDepthPass(this._createRequest(session));
		},
		"paged-shadow-feedback": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._renderer.recordPagedShadowFeedbackPass(this._createRequest(session));
		},
	};
	constructor(
		private readonly _renderer: WebGPUShadowRenderProvider,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		const nodes =
			input.pass.stage === "shadow"
				? this._createShadowNodes(input)
				: this._createFeedbackNodes(input);
		if (nodes.length === 0) return [];
		return [{
			lane: input.pass.stage === "shadow" ? "geometry" : "visibility",
			after: input.pass.stage === "shadow" ? undefined : ["visibility"],
			nodes,
		}];
	}

	public destroy(): void {}

	private _createRequest(session: WebGPUFrameSession): WebGPUPagedShadowFrameRequest {
		const packets = this._requirePackets(session);
		const targets = session.targets.frameTargets;
		const pagedFrame = this._renderer.resolvePagedShadowFrame(session.context);
		if (!pagedFrame) {
			throw new Error("WebGPU paged shadow node has no selected experiment light.");
		}
		return {
			context: session.context,
			encoder: session.commands.encoder,
			pagedFrame,
			shadowCasterPackets: packets.shadowCasters.slice(),
			shadowTransmitterPackets: packets.shadowTransmitters.slice(),
			feedbackDepthTexture: targets.depth,
			feedbackMotionDepthTexture: targets.gMotionDepth ?? null,
		};
	}

	private _requirePackets(session: WebGPUFrameSession) {
		if (!session.framePackets) {
			throw new Error("WebGPU frame session has no prepared frame packets.");
		}
		return session.framePackets;
	}

	private _createShadowNodes(input: WebGPUFrameModulePlanningInput) {
		if (input.pass.stage !== "shadow") return [];
		const nodes = [];
		if (input.context.shadowPlan.hasRasterWork) {
			nodes.push(createWebGPUFrameGraphNode(input.pass, "shadow", "WebGPUShadow", {
				writes: [
					writeWebGPUFrameGraphResource("shadow-atlas", "render-attachment"),
					writeWebGPUFrameGraphResource(
						"shadow-transmittance-atlas",
						"render-attachment",
					),
				],
			}));
		}
		if (!this._renderer.resolvePagedShadowFrame(input.context)) return nodes;
		nodes.push(
			createWebGPUFrameGraphNode(
				input.pass,
				"paged-shadow-page-mark",
				"WebGPUPagedShadowPageMark",
				{
					reads: [
						readWebGPUFrameGraphResource(
							"paged-shadow:feedback-flags",
							"storage-binding",
							true,
						),
					],
					writes: [
						writeWebGPUFrameGraphResource(
							"paged-shadow:page-request-flags",
							"storage-binding",
						),
						writeWebGPUFrameGraphResource(
							"paged-shadow:page-requests",
							"storage-binding",
						),
						writeWebGPUFrameGraphResource("paged-shadow:counters", "storage-binding"),
					],
				},
			),
			createWebGPUFrameGraphNode(
				input.pass,
				"paged-shadow-page-allocate",
				"WebGPUPagedShadowPageAllocate",
				{
					reads: [
						readWebGPUFrameGraphResource(
							"paged-shadow:page-requests",
							"storage-binding",
						),
						readWebGPUFrameGraphResource(
							"paged-shadow:page-request-flags",
							"storage-binding",
						),
					],
					writes: [
						"page-table",
						"page-metadata",
						"residency-state",
						"free-list",
						"counters",
						"dirty-physical-pages",
					].map((id) =>
						writeWebGPUFrameGraphResource(`paged-shadow:${id}`, "storage-binding"),
					),
				},
			),
			createWebGPUFrameGraphNode(
				input.pass,
				"paged-shadow-page-table-copy",
				"WebGPUPagedShadowPageTableCopy",
				{
					reads: [
						readWebGPUFrameGraphResource("paged-shadow:page-table", "storage-binding"),
					],
					writes: [
						writeWebGPUFrameGraphResource(
							"paged-shadow:page-table-texture",
							"storage-binding",
						),
					],
				},
			),
			createWebGPUFrameGraphNode(input.pass, "paged-shadow-depth", "WebGPUPagedShadowDepth", {
				reads: ["page-table", "page-metadata", "dirty-physical-pages"].map((id) =>
					readWebGPUFrameGraphResource(`paged-shadow:${id}`, "storage-binding"),
				),
				writes: [
					writeWebGPUFrameGraphResource("paged-shadow:draw-instances", "storage-binding"),
					writeWebGPUFrameGraphResource(
						"paged-shadow:draw-indirect-args",
						"storage-binding",
					),
					writeWebGPUFrameGraphResource(
						"paged-shadow:clear-draw-indirect-args",
						"storage-binding",
					),
					writeWebGPUFrameGraphResource(
						"paged-shadow:physical-depth",
						"render-attachment",
					),
				],
			}),
		);
		return nodes;
	}

	private _createFeedbackNodes(input: WebGPUFrameModulePlanningInput) {
		const pagedFrame = this._renderer.resolvePagedShadowFrame(input.context);
		if (
			input.pass.stage !== "main-opaque" ||
			!pagedFrame ||
			input.state.sceneTargetMode === "single"
		)
			return [];
		if (pagedFrame.settings.feedbackMode !== "screen-feedback") return [];
		return [
			createWebGPUFrameGraphNode(
				input.pass,
				"paged-shadow-feedback",
				"WebGPUPagedShadowFeedback",
				{
					reads: [
						readWebGPUFrameGraphResource("frame:depth", "texture-binding", true),
						readWebGPUFrameGraphResource(
							"paged-shadow:page-table",
							"storage-binding",
							true,
						),
					],
					writes: [
						writeWebGPUFrameGraphResource(
							"paged-shadow:next-feedback-flags",
							"storage-binding",
						),
					],
				},
			),
		];
	}
}
