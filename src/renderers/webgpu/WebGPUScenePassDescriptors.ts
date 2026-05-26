export type WebGPUSceneTargetMode = "single" | "mrt" | "gbuffer";
export type WebGPUTransparentPipelineMode =
	| "default"
	| "transmission"
	| "oit";
export type WebGPUScenePipelineDrawMode =
	| "default"
	| "early-z-color"
	| "early-z-prepass"
	| "reflection-capture"
	| "planar-reflection-composite";

export type WebGPUScenePipelineLayoutKind =
	| "scene"
	| "scene-gbuffer"
	| "scene-depth-prepass"
	| "planar-reflection";
export type WebGPUSceneFragmentTargetKind =
	| "single"
	| "mrt"
	| "gbuffer"
	| "oit"
	| "depth-only"
	| "planar-reflection";
export type WebGPUSceneShaderEntryMode =
	| "single"
	| "mrt"
	| "gbuffer"
	| "oit"
	| "early-z-prepass"
	| "planar-reflection-composite";
export type WebGPUSceneDepthStateMode =
	| "default"
	| "early-z-color"
	| "early-z-prepass"
	| "planar-reflection";
export type WebGPUSampleCountMode = "single-sample" | "mrt-msaa";
export type WebGPUDepthFormatMode = "canvas" | "depth32float";

export interface WebGPUScenePassDescriptor {
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly transparentMode: WebGPUTransparentPipelineMode;
	readonly drawMode: WebGPUScenePipelineDrawMode;
	readonly pipelineLayoutKind: WebGPUScenePipelineLayoutKind;
	readonly fragmentTargetKind: WebGPUSceneFragmentTargetKind;
	readonly shaderEntryMode: WebGPUSceneShaderEntryMode;
	readonly depthStateMode: WebGPUSceneDepthStateMode;
	readonly frontFace: "ccw" | "cw";
	readonly sampleCountMode: WebGPUSampleCountMode;
	readonly depthFormatMode: WebGPUDepthFormatMode;
	readonly pipelineKeyPart: string;
}

export function resolveWebGPUScenePassDescriptor(
	sceneTargetMode: WebGPUSceneTargetMode,
	transparentMode: WebGPUTransparentPipelineMode = "default",
	drawMode: WebGPUScenePipelineDrawMode = "default"
): WebGPUScenePassDescriptor {
	const sampleCountMode =
		sceneTargetMode === "mrt" ? "mrt-msaa" : "single-sample";
	const depthFormatMode =
		sceneTargetMode === "single" ? "canvas" : "depth32float";

	if (drawMode === "early-z-prepass") {
		return {
			sceneTargetMode,
			transparentMode,
			drawMode,
			pipelineLayoutKind: "scene-depth-prepass",
			fragmentTargetKind: "depth-only",
			shaderEntryMode: "early-z-prepass",
			depthStateMode: "early-z-prepass",
			frontFace: "ccw",
			sampleCountMode,
			depthFormatMode,
			pipelineKeyPart:
				`${sceneTargetMode}|${transparentMode}|${drawMode}|` +
				`layout:scene-depth-prepass|targets:depth-only`,
		};
	}

	if (drawMode === "planar-reflection-composite") {
		return {
			sceneTargetMode,
			transparentMode,
			drawMode,
			pipelineLayoutKind: "planar-reflection",
			fragmentTargetKind: "planar-reflection",
			shaderEntryMode: "planar-reflection-composite",
			depthStateMode: "planar-reflection",
			frontFace: "ccw",
			sampleCountMode,
			depthFormatMode,
			pipelineKeyPart:
				`${sceneTargetMode}|${transparentMode}|${drawMode}|` +
				`layout:planar-reflection|targets:planar-reflection`,
		};
	}

	const fragmentTargetKind =
		sceneTargetMode === "gbuffer" ? "gbuffer"
		: sceneTargetMode === "single" ? "single"
		: transparentMode === "oit" ? "oit"
		: "mrt";
	const shaderEntryMode =
		sceneTargetMode === "gbuffer" ? "gbuffer"
		: sceneTargetMode === "single" ? "single"
		: transparentMode === "oit" ? "oit"
		: "mrt";
	const pipelineLayoutKind =
		sceneTargetMode === "gbuffer" ? "scene-gbuffer" : "scene";

	return {
		sceneTargetMode,
		transparentMode,
		drawMode,
		pipelineLayoutKind,
		fragmentTargetKind,
		shaderEntryMode,
		depthStateMode: drawMode === "early-z-color" ? "early-z-color" : "default",
		frontFace: drawMode === "reflection-capture" ? "cw" : "ccw",
		sampleCountMode,
		depthFormatMode,
		pipelineKeyPart:
			`${sceneTargetMode}|${transparentMode}|${drawMode}|` +
			`layout:${pipelineLayoutKind}|targets:${fragmentTargetKind}`,
	};
}
