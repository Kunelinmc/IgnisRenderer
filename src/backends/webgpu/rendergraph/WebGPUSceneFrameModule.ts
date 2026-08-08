import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameGraphModule,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	createWebGPUForwardGraphResources,
	createWebGPUFrameGraphNode,
	createWebGPUPagedShadowLightingReads,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphPlanningUtils";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import type { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameGraphNode } from "./types";

/** @internal Owns main-scene graph execution and pass-local lifecycle. */
export class WebGPUSceneFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "scene";
	public readonly executors = {
		"frame-setup": async () => {},
		"opaque-external": async () => {},
		"opaque-scene": async (_node: unknown, session: WebGPUFrameSession) => {
			const packets = session.framePackets;
			if (!packets) {
				throw new Error("WebGPU frame session has no prepared frame packets.");
			}
			session.deferredOpaqueFrameState = await this.recorder.recordOpaque(
				session.context,
				packets,
				session.configuration?.deferredActive === true,
			);
		},
	};
	constructor(
		public readonly recorder: WebGPUScenePassRecorder,
		private readonly _depthDirtyClearPass: WebGPUDepthDirtyClearPass,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage === "webgpu-setup") {
			return [
				{
					order: 100,
					nodes: [
						{
							...createWebGPUFrameGraphNode(
								input.pass,
								"frame-setup",
								"WebGPUFrameSetup",
							),
							domain: "cpu",
							retention: "always",
						},
					],
				},
			];
		}
		if (input.pass.stage === "particle-sim") {
			return [
				{
					order: 100,
					nodes: [
						{
							...createWebGPUFrameGraphNode(
								input.pass,
								"opaque-external",
								"WebGPUOpaque:particle-sim",
							),
							domain: "cpu",
							retention: "always",
							opaque: true,
						},
					],
				},
			];
		}
		if (input.pass.stage !== "main-opaque") return [];
		const nodes = [
			createWebGPUFrameGraphNode(
				input.pass,
				"opaque-scene",
				input.state.deferredActive && input.state.sceneTargetMode === "gbuffer"
					? "WebGPUGBuffer"
					: `WebGPUOpaque${input.state.sceneTargetMode}`,
				this._createOpaqueResources(input),
			),
		];
		return nodes.length > 0 ? [{ order: 100, nodes }] : [];
	}

	public onShaderRuntimeChanged(): void {
		this._depthDirtyClearPass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._depthDirtyClearPass.destroy();
	}

	private _createOpaqueResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const { state, context } = input;
		if (state.sceneTargetMode === "single") {
			return createWebGPUForwardGraphResources(state, false, context);
		}
		if (state.sceneTargetMode === "color") {
			return {
				reads: [
					readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
					readWebGPUFrameGraphResource(
						"shadow-transmittance-atlas",
						"texture-binding",
						true,
					),
					...createWebGPUPagedShadowLightingReads(context),
				],
				writes: [
					writeWebGPUFrameGraphResource("frame:scene-color-main", "render-attachment"),
					writeWebGPUFrameGraphResource("frame:depth", "depth-attachment"),
				],
			};
		}
		const writes = [
			writeWebGPUFrameGraphResource("frame:scene-color-main", "render-attachment"),
			writeWebGPUFrameGraphResource("gbuffer:albedo-alpha", "render-attachment"),
			writeWebGPUFrameGraphResource("gbuffer:normal-rough-metal", "render-attachment"),
			writeWebGPUFrameGraphResource("gbuffer:emissive-occlusion", "render-attachment"),
			writeWebGPUFrameGraphResource("gbuffer:motion-depth", "render-attachment"),
			writeWebGPUFrameGraphResource("frame:depth", "depth-attachment"),
		];
		if (state.sceneTargetMode === "gbuffer" && state.deferredGBufferLayout !== "base") {
			writes.push(
				writeWebGPUFrameGraphResource("gbuffer:specular", "render-attachment"),
				writeWebGPUFrameGraphResource("gbuffer:coat-sheen", "render-attachment"),
				writeWebGPUFrameGraphResource("gbuffer:sheen-reflectance", "render-attachment"),
				writeWebGPUFrameGraphResource("gbuffer:material-ext0", "storage-binding"),
				writeWebGPUFrameGraphResource("gbuffer:material-ext3", "storage-binding"),
			);
		}
		const reads = [
			readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
			readWebGPUFrameGraphResource(
				"shadow-transmittance-atlas",
				"texture-binding",
				true,
			),
			...createWebGPUPagedShadowLightingReads(context),
			readWebGPUFrameGraphResource("planar-reflection:capture", "texture-binding", true),
		];
		if (state.needsPlanarReflectionMask) {
			writes.push(
				writeWebGPUFrameGraphResource("planar-reflection:mask", "render-attachment"),
			);
		}
		return { reads, writes };
	}
}
