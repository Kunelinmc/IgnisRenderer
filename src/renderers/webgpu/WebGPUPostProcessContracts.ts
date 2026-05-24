import type { IRenderTexture } from "../types";
import type { WebGPUPostProcessPassId } from "./postprocess/types";

export type WebGPUBuiltinPostProcessPassId =
	| WebGPUPostProcessPassId
	| "gamma";

const WEBGPU_BUILTIN_POST_PROCESS_PASS_ID_SET = new Set<string>([
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"tonemap",
	"gamma",
]);

/**
 * Returns whether `id` is reserved by a built-in WebGPU post-process pass.
 *
 * @param id Candidate runtime pass id.
 * @returns `true` when the id cannot be used by custom post-process passes.
 */
export function isWebGPUBuiltinPostProcessPassId(id: string): boolean {
	return WEBGPU_BUILTIN_POST_PROCESS_PASS_ID_SET.has(id);
}

export interface WebGPUFrameTargets {
	sceneColor: IRenderTexture;
	sceneColorMain: IRenderTexture;
	postPing: IRenderTexture;
	postPong: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	gSpecular?: IRenderTexture | null;
	gCoatSheen?: IRenderTexture | null;
	gSheenReflectance?: IRenderTexture | null;
	gMaterialExt0?: IRenderTexture | null;
	gMaterialExt1?: IRenderTexture | null;
	gMaterialExt2?: IRenderTexture | null;
	gMaterialExt3?: IRenderTexture | null;
	depth: IRenderTexture;
	oitAccum: IRenderTexture;
	oitReveal: IRenderTexture;
	oitSceneColorCopy: IRenderTexture;
	planarReflectionMask: IRenderTexture;
	aoRaw: IRenderTexture;
	aoBlur: IRenderTexture;
	ssrRaw: IRenderTexture;
	hiZ: IRenderTexture;
	historyRead: IRenderTexture;
	historyWrite: IRenderTexture;
	ssrHistoryRead: IRenderTexture;
	ssrHistoryWrite: IRenderTexture;
	volumetricHistoryRead: IRenderTexture;
	volumetricHistoryWrite: IRenderTexture;
	volumetricReservoirHistoryRead: IRenderTexture;
	volumetricReservoirHistoryWrite: IRenderTexture;
	motionHistoryRead: IRenderTexture;
	motionHistoryWrite: IRenderTexture;
}
