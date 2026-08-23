import type {
	WebGPUDrawPassDescriptor,
	WebGPUSceneTargetMode,
} from "./WebGPUScenePassDescriptors";

export interface WebGPUPlanarReflectionPassDescriptor
	extends WebGPUDrawPassDescriptor {
	readonly drawMode: "planar-reflection-composite";
}

/** @internal Resolves the fixed planar-reflection composite pipeline contract. */
export function resolveWebGPUPlanarReflectionPassDescriptor(
	sceneTargetMode: WebGPUSceneTargetMode,
): WebGPUPlanarReflectionPassDescriptor {
	return {
		sceneTargetMode,
		transparentMode: "default",
		drawMode: "planar-reflection-composite",
		pipelineLayoutKind: "planar-reflection",
		fragmentTargetKind: "planar-reflection",
		shaderEntryMode: "planar-reflection-composite",
		depthStateMode: "planar-reflection",
		frontFace: "ccw",
		sampleCountMode: "mrt-msaa",
		depthFormatMode: "depth32float",
		pipelineKeyPart:
			`${sceneTargetMode}|default|planar-reflection-composite|` +
			"layout:planar-reflection|targets:planar-reflection",
	};
}
