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
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import type {
	WebGPURecordingFrameSession as WebGPUFrameSession,
} from "./WebGPUFrameSession";
import type { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import type { WebGPUColorDirtyClearPass } from "./WebGPUColorDirtyClearPass";
import type { WebGPUDeferredOpaqueStatePort } from "./WebGPUDeferredFrameModule";
import type { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameGraphNode } from "./types";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";

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
			this._deferredOpaqueState.publish(await this.recorder.recordOpaque(
				session.context,
				packets,
				session.configuration?.deferredActive === true,
			));
		},
	};
	constructor(
		public readonly recorder: WebGPUScenePassRecorder,
		private readonly _depthDirtyClearPass: WebGPUDepthDirtyClearPass,
		private readonly _colorDirtyClearPass: WebGPUColorDirtyClearPass,
		private readonly _deferredOpaqueState: WebGPUDeferredOpaqueStatePort,
	) {}

	public activateFrame(context: WebGPUFrameExecutionContext): void {
		this.recorder.bindFrame(context);
	}

	public closeFrame(): void {
		this.recorder.closeFrame();
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (input.pass.stage === "webgpu-setup") {
			return [
				{
					lane: "setup",
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
					lane: "setup",
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
		return nodes.length > 0 ? [{ lane: "geometry", nodes }] : [];
	}

	public onShaderRuntimeChanged(): void {
		this._depthDirtyClearPass.onShaderRuntimeChanged();
		this._colorDirtyClearPass.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._depthDirtyClearPass.destroy();
		this._colorDirtyClearPass.destroy();
	}

	private _createOpaqueResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const { state } = input;
		if (state.sceneTargetMode === "single") {
			return this._createForwardResources(input, false);
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
					...this._createShadowReads(),
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
			...this._createShadowReads(),
			readWebGPUFrameGraphResource("planar-reflection:capture", "texture-binding", true),
		];
		if (state.needsPlanarReflectionMask) {
			writes.push(
				writeWebGPUFrameGraphResource("planar-reflection:mask", "render-attachment"),
			);
		}
		return { reads, writes };
	}

	private _createForwardResources(
		input: WebGPUFrameModulePlanningInput,
		loadExistingColor: boolean,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const useCanvas = input.state.sceneTargetMode === "single" ||
			!input.state.hasFrameTargets;
		const color = useCanvas
			? WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor
			: WEBGPU_FRAME_GRAPH_RESOURCES.frameColor;
		const depth = useCanvas
			? WEBGPU_FRAME_GRAPH_RESOURCES.canvasDepth
			: WEBGPU_FRAME_GRAPH_RESOURCES.frameDepth;
		const reads = this._createShadowReads();
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

	private _createShadowReads() {
		return [
			readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
			readWebGPUFrameGraphResource(
				"shadow-transmittance-atlas",
				"texture-binding",
				true,
			),
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
}
