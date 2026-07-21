import type { RenderBackendType } from "../backends/IRenderBackend";
import type {
	LogicalGBufferSemantic,
	PostProcessExecutionDeclaration,
	PostProcessHistoryDescriptor,
	PostProcessSharedResourceDeclaration,
	PostProcessTransientDescriptor,
} from "./types";

export interface PostProcessExecutionDeclarationOptions {
	readonly gBuffer?: readonly LogicalGBufferSemantic[];
	readonly histories?: readonly PostProcessHistoryDescriptor[];
	readonly transients?: readonly PostProcessTransientDescriptor[];
	readonly shared?: readonly PostProcessSharedResourceDeclaration[];
	readonly color?: PostProcessExecutionDeclaration["color"];
}

/** @internal Creates the complete declaration used by engine-owned passes. */
export function createPostProcessExecutionDeclaration(
	backend: RenderBackendType,
	options: PostProcessExecutionDeclarationOptions = {}
): PostProcessExecutionDeclaration {
	const writeUsage = backend === "webgl" ? "color-attachment" :
		backend === "software" ? "cpu-write" : "storage";
	const readUsage = backend === "software" ? "cpu-read" : "sampled";
	return {
		color: options.color ?? (backend === "software" ? {
			access: "read-write",
			output: "preserve",
		} : {
			access: "read",
			output: "new-version",
		}),
		gBuffer: options.gBuffer?.map((semantic) => ({
			semantic,
			access: "read",
			usage: readUsage,
		})),
		histories: options.histories?.map((descriptor) => ({
			descriptor,
			read: [{ access: "read", usage: readUsage }],
			write: [{ access: "write", usage: writeUsage }],
		})),
		transients: options.transients?.map((descriptor) => ({
			descriptor,
			uses: [{ access: "write", usage: writeUsage }],
		})),
		shared: options.shared,
	};
}

export const WEBGPU_HIZ_SHARED_RESOURCE = Object.freeze({
	id: "backend:frame-hiz",
	access: "read",
	usage: "sampled",
} as const satisfies PostProcessSharedResourceDeclaration);
