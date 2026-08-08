import type { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import type { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import type {
	WebGPUFrameGraphModule,
	WebGPUFrameGraphContribution,
	WebGPUFrameModuleAnalysisInput,
	WebGPUFrameModuleConfigurationInput,
	WebGPUFrameModulePlanningInput,
	WebGPUFrameModuleStateStore,
} from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";
import { WEBGPU_DEFERRED_FEATURE_ANALYSIS } from "./WebGPUFrameModuleStateKeys";
import { analyzeWebGPUDeferredFeatures } from "./WebGPUFrameFeatureAnalyzer";
import {
	createWebGPUFrameGraphNode,
	createWebGPUPagedShadowLightingReads,
	readWebGPUFrameGraphResource,
	writeWebGPUFrameGraphResource,
} from "./WebGPUFrameGraphPlanningUtils";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import type { WebGPUFrameGraphNode } from "./types";

interface WebGPUReflectionCompositePort {
	composite(session: WebGPUFrameSession): Promise<void>;
}

const DEFERRED_GBUFFER_RENDER_RESOURCE_IDS = [
	"gbuffer:albedo-alpha",
	"gbuffer:normal-rough-metal",
	"gbuffer:emissive-occlusion",
	"gbuffer:motion-depth",
	"gbuffer:specular",
	"gbuffer:coat-sheen",
	"gbuffer:sheen-reflectance",
] as const;

const DEFERRED_GBUFFER_BASE_RESOURCE_IDS =
	DEFERRED_GBUFFER_RENDER_RESOURCE_IDS.slice(0, 4);

const DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS = [
	"gbuffer:material-ext0",
	"gbuffer:material-ext3",
] as const;

const DEFERRED_GBUFFER_RESOURCE_IDS = [
	...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS,
	...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS,
] as const;

/** @internal Owns deferred decal, lighting, and fallback graph execution. */
export class WebGPUDeferredFrameModule implements WebGPUFrameGraphModule {
	public readonly id = "deferred";
	public readonly executors = {
		"deferred-decal": async (_node: unknown, session: WebGPUFrameSession) => {
			if (session.deferredOpaqueFrameState?.lightingEnabled) {
				await this._decalPass.recordDecalPass(session.context);
			}
		},
		"deferred-lighting": async (_node: unknown, session: WebGPUFrameSession) => {
			await this._recordLighting(session);
		},
	};
	public constructor(
		private readonly _lightingPass: WebGPUDeferredLightingPass,
		private readonly _decalPass: WebGPUDeferredDecalPass,
		private readonly _sceneRecorder: WebGPUScenePassRecorder,
		private readonly _reflection: WebGPUReflectionCompositePort,
	) {}

	public analyze(
		input: WebGPUFrameModuleAnalysisInput,
		state: WebGPUFrameModuleStateStore,
	): void {
		state.set(
			WEBGPU_DEFERRED_FEATURE_ANALYSIS,
			analyzeWebGPUDeferredFeatures(input.context, input.framePackets),
		);
	}

	public contributeConfiguration(
		input: WebGPUFrameModuleConfigurationInput,
	): WebGPUFrameModuleConfigurationContribution {
		const analysis = input.state.require(WEBGPU_DEFERRED_FEATURE_ANALYSIS);
		return (builder) => builder.setDeferred(analysis);
	}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		if (
			input.pass.stage !== "main-opaque" ||
			!input.state.deferredActive ||
			input.state.sceneTargetMode !== "gbuffer"
		) return [];
		const nodes: WebGPUFrameGraphNode[] = [];
		if ((input.context.scene?.decalPackets.length ?? 0) > 0) {
			nodes.push(createWebGPUFrameGraphNode(
				input.pass,
				"deferred-decal",
				"WebGPUDeferredDecal",
				this._createDecalResources(input),
			));
		}
		nodes.push(createWebGPUFrameGraphNode(
			input.pass,
			"deferred-lighting",
			"WebGPUDeferredLighting",
			this._createLightingResources(input),
		));
		return nodes.length > 0 ? [{ order: 200, nodes }] : [];
	}

	public invalidateFrameResources(): void {
		this._destroyBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyBindings();
	}

	public destroy(): void {
		this._destroyBindings();
	}

	private async _recordLighting(session: WebGPUFrameSession): Promise<void> {
		const state = session.deferredOpaqueFrameState;
		if (!state) return;
		try {
			if (state.lightingEnabled) {
				await this._lightingPass.recordLightingPass(
					session.context,
					state.clearSceneColor,
				);
			}
			if (state.fallbackPackets.length > 0) {
				await this._sceneRecorder.recordMainPass(
					session.context,
					state.fallbackPackets,
					false,
					false,
				);
			}
			await this._reflection.composite(session);
		} finally {
			session.deferredOpaqueFrameState = null;
		}
	}

	private _destroyBindings(): void {
		this._lightingPass.destroyBindings();
		this._decalPass.destroyBindings();
	}

	private _createDecalResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const resourceIds = input.state.deferredGBufferLayout === "base"
			? DEFERRED_GBUFFER_BASE_RESOURCE_IDS
			: DEFERRED_GBUFFER_RESOURCE_IDS;
		return {
			reads: resourceIds.map((id) =>
				readWebGPUFrameGraphResource(id, "copy-src")),
			writes: [
				...DEFERRED_GBUFFER_RENDER_RESOURCE_IDS.map((id) =>
					writeWebGPUFrameGraphResource(id, "render-attachment")),
				...DEFERRED_GBUFFER_STORAGE_RESOURCE_IDS.map((id) =>
					writeWebGPUFrameGraphResource(id, "storage-binding")),
			],
		};
	}

	private _createLightingResources(
		input: WebGPUFrameModulePlanningInput,
	): Pick<WebGPUFrameGraphNode, "reads" | "writes"> {
		const resourceIds = input.state.deferredGBufferLayout === "base"
			? DEFERRED_GBUFFER_BASE_RESOURCE_IDS
			: DEFERRED_GBUFFER_RESOURCE_IDS;
		return {
			reads: [
				...resourceIds.map((id) =>
					readWebGPUFrameGraphResource(id, "texture-binding")),
				readWebGPUFrameGraphResource("shadow-atlas", "texture-binding", true),
				readWebGPUFrameGraphResource(
					"shadow-transmittance-atlas",
					"texture-binding",
					true,
				),
				...createWebGPUPagedShadowLightingReads(input.context),
			],
			writes: [
				writeWebGPUFrameGraphResource(
					"frame:scene-color-main",
					"render-attachment",
				),
			],
		};
	}
}
