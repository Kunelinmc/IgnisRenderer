import type {
	PostProcessBaseResourceDescriptor,
	PostProcessResourceDescriptor,
	PostProcessResourceMipMode,
	PostProcessScaledResourceDescriptor,
} from "./types";

const DEFAULT_POST_PROCESS_RESOURCE_FORMAT = "rgba16float";
const DEFAULT_POST_PROCESS_RESOURCE_USAGE = [
	"sampled",
	"storage",
	"render-target",
] as const;
const DEFAULT_POST_PROCESS_RESOURCE_MIP_MODE: PostProcessResourceMipMode = "single";

export interface PostProcessResourceDescriptorResolveOptions {
	readonly includeMipMode?: boolean;
}

export interface PostProcessResourceDescriptorKeyOptions {
	readonly includeMipMode?: boolean;
	readonly sortUsage?: boolean;
}

interface PostProcessResourceDescriptorWithMip
	extends PostProcessBaseResourceDescriptor {
	readonly mipMode?: PostProcessResourceMipMode;
}

/**
 * Resolves a scale-based logical descriptor into a concrete resource descriptor.
 *
 * @internal Owned by post-process graph resource managers.
 * @param descriptor Logical scale-based descriptor.
 * @param width Full-size frame width.
 * @param height Full-size frame height.
 * @param options Optional resource resolution flags.
 * @returns Concrete backend resource descriptor.
 * @sideEffects None.
 */
export function resolvePostProcessResourceDescriptor(
	descriptor: PostProcessScaledResourceDescriptor & {
		readonly mipMode?: PostProcessResourceMipMode;
	},
	width: number,
	height: number,
	options: PostProcessResourceDescriptorResolveOptions = {}
): PostProcessResourceDescriptor {
	const resourceDescriptor: PostProcessResourceDescriptor = {
		id: descriptor.id,
		width: resolvePostProcessResourceDimension(width, descriptor.widthScale),
		height: resolvePostProcessResourceDimension(height, descriptor.heightScale),
		format: descriptor.format ?? DEFAULT_POST_PROCESS_RESOURCE_FORMAT,
		usage: descriptor.usage ?? DEFAULT_POST_PROCESS_RESOURCE_USAGE,
	};
	if (options.includeMipMode) {
		return {
			...resourceDescriptor,
			mipMode: descriptor.mipMode ?? DEFAULT_POST_PROCESS_RESOURCE_MIP_MODE,
		};
	}
	return resourceDescriptor;
}

/**
 * Creates a descriptor key for graph-level conflict detection.
 *
 * @internal Owned by `PostProcessGraphCompiler`.
 * @param descriptor Logical descriptor declared by a pass.
 * @param options Optional key composition flags.
 * @returns Stable descriptor compatibility key.
 * @sideEffects None.
 */
export function createPostProcessScaledResourceDescriptorKey(
	descriptor: PostProcessScaledResourceDescriptor & {
		readonly mipMode?: PostProcessResourceMipMode;
	},
	options: PostProcessResourceDescriptorKeyOptions = {}
): string {
	return [
		descriptor.widthScale ?? 1,
		descriptor.heightScale ?? 1,
		descriptor.format ?? DEFAULT_POST_PROCESS_RESOURCE_FORMAT,
		...(options.includeMipMode ?
			[descriptor.mipMode ?? DEFAULT_POST_PROCESS_RESOURCE_MIP_MODE]
		:	[]),
		resolvePostProcessResourceUsageKey(descriptor, options.sortUsage ?? true),
	].join("|");
}

/**
 * Creates a descriptor key for backend resource allocation reuse.
 *
 * @internal Owned by post-process resource managers.
 * @param backend Backend that owns the concrete resource.
 * @param descriptor Concrete backend resource descriptor.
 * @param options Optional key composition flags.
 * @returns Stable allocation compatibility key.
 * @sideEffects None.
 */
export function createPostProcessResourceAllocationKey(
	backend: string,
	descriptor: PostProcessResourceDescriptor,
	options: PostProcessResourceDescriptorKeyOptions = {}
): string {
	return [
		backend,
		descriptor.width,
		descriptor.height,
		descriptor.format,
		...(options.includeMipMode ?
			[descriptor.mipMode ?? DEFAULT_POST_PROCESS_RESOURCE_MIP_MODE]
		:	[]),
		resolvePostProcessResourceUsageKey(descriptor, options.sortUsage ?? false),
	].join("|");
}

function resolvePostProcessResourceDimension(
	size: number,
	scale: number | undefined
): number {
	return Math.max(1, Math.floor(size * (scale ?? 1)));
}

function resolvePostProcessResourceUsageKey(
	descriptor: PostProcessResourceDescriptorWithMip,
	sortUsage: boolean
): string {
	const usage = descriptor.usage ?? DEFAULT_POST_PROCESS_RESOURCE_USAGE;
	return (sortUsage ? [...usage].sort() : usage).join(",");
}
