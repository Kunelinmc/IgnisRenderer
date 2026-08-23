export type WebGPUSceneTargetMode = "single" | "color" | "mrt" | "gbuffer";
export type WebGPUTransparentPipelineMode =
	| "default"
	| "transmission"
	| "transmission-capture"
	| "oit";
export type WebGPUScenePipelineDrawMode =
	| "default"
	| "early-z-color"
	| "early-z-prepass"
	| "reflection-capture";
export type WebGPUDrawPipelineMode =
	| WebGPUScenePipelineDrawMode
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
	| "gbuffer-base"
	| "oit"
	| "transmission-capture"
	| "depth-only"
	| "planar-reflection";
export type WebGPUSceneShaderEntryMode =
	| "single"
	| "mrt"
	| "gbuffer"
	| "gbuffer-base"
	| "oit"
	| "transmission-capture"
	| "early-z-prepass"
	| "planar-reflection-composite";
export type WebGPUSceneDepthStateMode =
	| "default"
	| "early-z-color"
	| "early-z-prepass"
	| "planar-reflection";
export type WebGPUSampleCountMode = "single-sample" | "mrt-msaa";
export type WebGPUDepthFormatMode = "canvas" | "depth32float";

export interface WebGPUDrawPassDescriptor {
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly transparentMode: WebGPUTransparentPipelineMode;
	readonly drawMode: WebGPUDrawPipelineMode;
	readonly pipelineLayoutKind: WebGPUScenePipelineLayoutKind;
	readonly fragmentTargetKind: WebGPUSceneFragmentTargetKind;
	readonly shaderEntryMode: WebGPUSceneShaderEntryMode;
	readonly depthStateMode: WebGPUSceneDepthStateMode;
	readonly frontFace: "ccw" | "cw";
	readonly sampleCountMode: WebGPUSampleCountMode;
	readonly depthFormatMode: WebGPUDepthFormatMode;
	readonly pipelineKeyPart: string;
}

export interface WebGPUScenePassDescriptor extends WebGPUDrawPassDescriptor {
	readonly drawMode: WebGPUScenePipelineDrawMode;
}

export function resolveWebGPUScenePassDescriptor(
	sceneTargetMode: WebGPUSceneTargetMode,
	transparentMode: WebGPUTransparentPipelineMode = "default",
	drawMode: WebGPUScenePipelineDrawMode = "default",
	deferredGBufferLayout: "base" | "extended" = "extended"
): WebGPUScenePassDescriptor {
	const sampleCountMode =
		sceneTargetMode === "mrt" || sceneTargetMode === "color" ?
			"mrt-msaa"
		:	"single-sample";
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

	const fragmentTargetKind =
		transparentMode === "transmission-capture" ? "transmission-capture" :
		sceneTargetMode === "gbuffer" ?
			deferredGBufferLayout === "base" ? "gbuffer-base" : "gbuffer"
		: sceneTargetMode === "single" ? "single"
		: sceneTargetMode === "color" && transparentMode !== "oit" ? "single"
		: transparentMode === "oit" ? "oit"
		: "mrt";
	const shaderEntryMode =
		transparentMode === "transmission-capture" ? "transmission-capture" :
		sceneTargetMode === "gbuffer" ?
			deferredGBufferLayout === "base" ? "gbuffer-base" : "gbuffer"
		: sceneTargetMode === "single" ? "single"
		: sceneTargetMode === "color" && transparentMode !== "oit" ? "single"
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
			`layout:${pipelineLayoutKind}|targets:${fragmentTargetKind}|` +
			`gbuffer:${deferredGBufferLayout}`,
	};
}
