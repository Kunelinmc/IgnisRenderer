import type { WebGPUShadowRenderProvider } from "../WebGPUResourceContracts";
import type { WebGPUPagedShadowFrameRequest } from "../WebGPUPagedShadowRuntime";

import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	createWebGPUFrameGraphNode,
	hasWebGPUPagedShadowWork,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphPlanningUtils";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";

/** @internal Owns shadow graph-node execution for one WebGPU runtime. */
export class WebGPUShadowFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "shadow";
	public readonly executors = {
		shadow: async (_node: unknown, session: WebGPUFrameSession) => {
			const packets = this._requirePackets(session);
			await this._renderer.renderShadows(
				session.context,
				packets,
				session.encoder ?? undefined,
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
		private readonly _recording: WebGPUFrameGraphRecordingContext,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		const nodes =
			input.pass.stage === "shadow"
				? this._createShadowNodes(input)
				: this._createFeedbackNodes(input);
		if (nodes.length === 0) return [];
		return [{ order: input.pass.stage === "shadow" ? 100 : 500, nodes }];
	}

	public destroy(): void {}

	private _createRequest(session: WebGPUFrameSession): WebGPUPagedShadowFrameRequest {
		const packets = this._requirePackets(session);
		const targets = this._recording.getFrameTargets();
		return {
			context: session.context,
			encoder: session.encoder,
			renderSets: session.context.shadowMaps,
			shadowCasterPackets: packets.shadowCasters.slice(),
			shadowTransmitterPackets: packets.shadowTransmitters.slice(),
			feedbackDepthTexture: targets?.depth ?? null,
			feedbackMotionDepthTexture: targets?.gMotionDepth ?? null,
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
		const nodes = [
			createWebGPUFrameGraphNode(input.pass, "shadow", "WebGPUShadow", {
				writes: [writeWebGPUFrameGraphResource("shadow-atlas", "render-attachment")],
			}),
		];
		if (!hasWebGPUPagedShadowWork(input.context)) return nodes;
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
		if (
			input.pass.stage !== "main-opaque" ||
			!hasWebGPUPagedShadowWork(input.context) ||
			input.state.sceneTargetMode === "single"
		)
			return [];
		const hasScreenFeedback = Array.from(input.context.shadowMaps.values()).some(
			(renderSet) =>
				renderSet.storageMode === "paged" &&
				renderSet.layout?.paged?.feedbackMode === "screen-feedback",
		);
		if (!hasScreenFeedback) return [];
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
