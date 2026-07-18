import type { RenderBackendType } from "../renderers/IRenderBackend";
import type { PostProcessGraphMetadata } from "./types";

/** @internal Resolves strict graph profiles for engine-owned post-process passes. */
export function resolveBuiltinPostProcessGraphMetadata(
	backend: RenderBackendType
): PostProcessGraphMetadata {
	const color = backend === "software" ?
		{ access: "read-write" as const, output: "preserve" as const } :
		{ access: "read" as const, output: "new-version" as const };
	return Object.freeze({ color, outputValidation: "strict" });
}

/** @internal Strict WebGPU graph profile for passes that consume frame Hi-Z. */
export const WEBGPU_HIZ_POST_PROCESS_GRAPH_METADATA = Object.freeze({
	color: Object.freeze({ access: "read", output: "new-version" }),
	backendShared: Object.freeze([
		Object.freeze({
			id: "backend:frame-hiz",
			access: "read",
			usage: "sampled",
		}),
	]),
	outputValidation: "strict",
} as const satisfies PostProcessGraphMetadata);
