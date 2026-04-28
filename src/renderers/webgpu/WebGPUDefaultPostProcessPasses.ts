import {
	INTERACTION_TRANSIENT_STATE_KEY,
	isFogPostProcessEnabled,
} from "../../pipeline/types";
import type { IBindingGroup, IRenderTexture } from "../types";
import type { WebGPULightingState } from "./types";
import type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
} from "./WebGPUPostProcessRuntime";
import type { WebGPUPostProcessPassPlugin } from "./WebGPUPostProcessGraph";

export interface WebGPUDefaultPostProcessPassFactoryDeps {
	executeRuntimePass(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult>;
	getFrameBinding(): IBindingGroup;
	getLightingState(): WebGPULightingState | null;
	presentToCanvas(source: IRenderTexture, applyGamma: boolean): Promise<void>;
	getTAAHistoryValid(): boolean;
	getSSRHistoryValid(): boolean;
	getVolumetricHistoryValid(): boolean;
	getMotionHistoryValid(): boolean;
	setTAAHistoryUpdated(updated: boolean): void;
	setSSRHistoryUpdated(updated: boolean): void;
	setVolumetricHistoryUpdated(updated: boolean): void;
}

/**
 * Build the default WebGPU post-process pass graph.
 */
export function createWebGPUDefaultPostProcessPasses(
	deps: WebGPUDefaultPostProcessPassFactoryDeps
): WebGPUPostProcessPassPlugin[] {
	return [
		{
			id: "ssao",
			kind: "compute",
			dependsOn: [],
			precompileHints: ["postprocess:ssao"],
			isEnabled: (features) => features.enableSSAO,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "ssao",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "ssgi",
			kind: "compute",
			dependsOn: ["ssao"],
			precompileHints: ["postprocess:ssgi"],
			isEnabled: (features) => features.enableSSGI,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "ssgi",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "taa",
			kind: "compute",
			dependsOn: ["ssgi", "ssao"],
			precompileHints: ["postprocess:taa"],
			isEnabled: (features) => features.enableTAA,
			execute: async (ctx) => {
				const result = await deps.executeRuntimePass({
					passId: "taa",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
					historyValid: deps.getTAAHistoryValid() && deps.getMotionHistoryValid(),
				});
				deps.setTAAHistoryUpdated(result.historyUpdated === true);
			},
		},
		{
			id: "ssr",
			kind: "compute",
			dependsOn: ["taa"],
			precompileHints: ["postprocess:ssr", "postprocess:hiz"],
			isEnabled: (features) => features.enableSSR,
			execute: async (ctx) => {
				const result = await deps.executeRuntimePass({
					passId: "ssr",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
					historyValid: deps.getSSRHistoryValid() && deps.getMotionHistoryValid(),
					frameBinding: deps.getFrameBinding(),
				});
				deps.setSSRHistoryUpdated(result.historyUpdated === true);
			},
		},
		{
			id: "volumetric",
			kind: "compute",
			dependsOn: ["ssr"],
			precompileHints: ["postprocess:volumetric", "postprocess:hiz"],
			isEnabled: (features) => features.enableVolumetric,
			execute: async (ctx) => {
				const result = await deps.executeRuntimePass({
					passId: "volumetric",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
					historyValid:
						deps.getVolumetricHistoryValid() && deps.getMotionHistoryValid(),
					frameBinding: deps.getFrameBinding(),
					lightingState: deps.getLightingState(),
				});
				deps.setVolumetricHistoryUpdated(result.historyUpdated === true);
			},
		},
		{
			id: "fog",
			kind: "compute",
			dependsOn: ["volumetric"],
			precompileHints: ["postprocess:fog"],
			isEnabled: (features) => isFogPostProcessEnabled(features),
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "fog",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "motion-blur",
			kind: "compute",
			dependsOn: ["fog"],
			precompileHints: ["postprocess:motion-blur"],
			isEnabled: (features) => features.enableMotionBlur,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "motion-blur",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "dof",
			kind: "compute",
			dependsOn: ["motion-blur"],
			precompileHints: ["postprocess:dof"],
			isEnabled: (features) => features.enableDOF,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "dof",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "bloom",
			kind: "compute",
			dependsOn: ["dof"],
			precompileHints: ["postprocess:bloom"],
			isEnabled: (features) => features.enableBloom,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "bloom",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "fxaa",
			kind: "compute",
			dependsOn: ["bloom"],
			precompileHints: ["postprocess:fxaa"],
			isEnabled: (features) => features.enableFXAA,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "fxaa",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "interaction-outline",
			kind: "compute",
			dependsOn: ["fxaa"],
			precompileHints: ["postprocess:interaction-outline"],
			isEnabled: () => true,
			execute: async (ctx) => {
				const interaction = ctx.frameContext.transient.get(
					INTERACTION_TRANSIENT_STATE_KEY
				);
				if ((interaction?.selectedEntityIds?.length ?? 0) === 0) {
					return;
				}
				await deps.executeRuntimePass({
					passId: "interaction-outline",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
					state: interaction,
				});
			},
		},
		{
			id: "tonemap",
			kind: "compute",
			dependsOn: ["interaction-outline"],
			precompileHints: ["postprocess:tonemap"],
			isEnabled: (features) => features.enableGamma,
			execute: async (ctx) => {
				await deps.executeRuntimePass({
					passId: "tonemap",
					encoder: ctx.encoder,
					targets: ctx.targets,
					frameContext: ctx.frameContext,
				});
			},
		},
		{
			id: "gamma",
			kind: "render",
			dependsOn: ["tonemap"],
			precompileHints: [],
			isEnabled: (features) => features.enableGamma,
			execute: async (ctx) => {
				await deps.presentToCanvas(ctx.targets.sceneColor, true);
			},
		},
	];
}
