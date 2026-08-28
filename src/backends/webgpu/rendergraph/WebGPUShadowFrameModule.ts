import type { WebGPUShadowRenderProvider } from "../WebGPUResourceContracts";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	createWebGPUFrameGraphNode,
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
	};

	constructor(
		private readonly _renderer: WebGPUShadowRenderProvider,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage !== "shadow") {
			return [];
		}
		const nodes = this._createShadowNodes(input);
		if (nodes.length === 0) return [];
		return [{
			lane: "geometry",
			nodes,
		}];
	}

	public destroy(): void {}

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
		return nodes;
	}
}
