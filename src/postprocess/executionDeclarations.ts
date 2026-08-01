import type {
	PostProcessExecutionDeclaration,
	PostProcessExecutionResourceUse,
	PostProcessSharedResourceDeclaration,
} from "./types";

export const POST_PROCESS_CPU_READ = Object.freeze({
	access: "read",
	usage: "cpu-read",
} as const satisfies PostProcessExecutionResourceUse);

export const POST_PROCESS_CPU_WRITE = Object.freeze({
	access: "write",
	usage: "cpu-write",
} as const satisfies PostProcessExecutionResourceUse);

export const POST_PROCESS_SAMPLED_READ = Object.freeze({
	access: "read",
	usage: "sampled",
} as const satisfies PostProcessExecutionResourceUse);

export const POST_PROCESS_STORAGE_WRITE = Object.freeze({
	access: "write",
	usage: "storage",
} as const satisfies PostProcessExecutionResourceUse);

export const POST_PROCESS_COLOR_ATTACHMENT_WRITE = Object.freeze({
	access: "write",
	usage: "color-attachment",
} as const satisfies PostProcessExecutionResourceUse);

export const SOFTWARE_IN_PLACE_EXECUTION = Object.freeze({
	color: { access: "read-write", output: "preserve" },
} as const satisfies PostProcessExecutionDeclaration);

export const WEBGPU_VERSIONED_EXECUTION = Object.freeze({
	color: { access: "read", output: "new-version" },
} as const satisfies PostProcessExecutionDeclaration);

export const WEBGL_VERSIONED_EXECUTION = Object.freeze({
	color: { access: "read", output: "new-version" },
} as const satisfies PostProcessExecutionDeclaration);

export const POST_PROCESS_SHARED_RESOURCE_IDS = Object.freeze({
	frameHiZ: "backend:frame-hiz",
	planarReflectionMask: "backend:planar-reflection-mask",
	transmissionSceneColor: "backend:transmission-scene-color",
	transmissionLighting: "backend:transmission-lighting",
	transmissionSurface1: "backend:transmission-surface-1",
	transmissionSurface2: "backend:transmission-surface-2",
} as const);

export type BuiltInPostProcessSharedResourceId =
	(typeof POST_PROCESS_SHARED_RESOURCE_IDS)[keyof typeof POST_PROCESS_SHARED_RESOURCE_IDS];

export const WEBGPU_HIZ_SHARED_RESOURCE = Object.freeze({
	id: POST_PROCESS_SHARED_RESOURCE_IDS.frameHiZ,
	access: "read",
	usage: "sampled",
} as const satisfies PostProcessSharedResourceDeclaration);
