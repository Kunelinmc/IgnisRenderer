import type { FrameContext, InteractionTransientState } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import type { IBindingGroup } from "../../types";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessGraph";
import type { WebGPULightingState } from "../types";

export const WEBGPU_POST_PROCESS_PASS_IDS = [
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"fxaa",
	"interaction-outline",
] as const;

export type WebGPUPostProcessPassId =
	(typeof WEBGPU_POST_PROCESS_PASS_IDS)[number];

interface WebGPUPostProcessExecuteBaseRequest {
	passId: WebGPUPostProcessPassId;
	encoder: ICommandEncoder;
	targets: WebGPUFrameTargets;
	frameContext: FrameContext;
}

export interface WebGPUPostProcessSSAOExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "ssao";
}

export interface WebGPUPostProcessSSGIExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "ssgi";
}

export interface WebGPUPostProcessTAAExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "taa";
	historyValid: boolean;
}

export interface WebGPUPostProcessSSRExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "ssr";
	historyValid: boolean;
	frameBinding: IBindingGroup;
}

export interface WebGPUPostProcessVolumetricExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "volumetric";
	historyValid: boolean;
	frameBinding: IBindingGroup;
	lightingState: WebGPULightingState | null;
}

export interface WebGPUPostProcessFogExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "fog";
}

export interface WebGPUPostProcessMotionBlurExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "motion-blur";
}

export interface WebGPUPostProcessDOFExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "dof";
}

export interface WebGPUPostProcessBloomExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "bloom";
}

export interface WebGPUPostProcessFXAAExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "fxaa";
}

export interface WebGPUPostProcessInteractionOutlineExecuteRequest
	extends WebGPUPostProcessExecuteBaseRequest {
	passId: "interaction-outline";
	state?: InteractionTransientState | null;
}

export type WebGPUPostProcessExecuteRequest =
	| WebGPUPostProcessSSAOExecuteRequest
	| WebGPUPostProcessSSGIExecuteRequest
	| WebGPUPostProcessTAAExecuteRequest
	| WebGPUPostProcessSSRExecuteRequest
	| WebGPUPostProcessVolumetricExecuteRequest
	| WebGPUPostProcessFogExecuteRequest
	| WebGPUPostProcessMotionBlurExecuteRequest
	| WebGPUPostProcessDOFExecuteRequest
	| WebGPUPostProcessBloomExecuteRequest
	| WebGPUPostProcessFXAAExecuteRequest
	| WebGPUPostProcessInteractionOutlineExecuteRequest;

export interface WebGPUPostProcessExecuteResult {
	ran: boolean;
	historyUpdated?: boolean;
}

export interface WebGPUPostProcessPassDelegate {
	readonly passIds: readonly WebGPUPostProcessPassId[];
	invalidateBindings(): void;
	onShaderRuntimeChanged(): void;
	warmupHint(hint: string): Promise<boolean>;
	execute(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult | null>;
}
