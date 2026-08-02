import type {
	RenderGraphDiagnostic,
	RenderGraphNodeId,
	RenderGraphNormalizedSubresourceRange,
	RenderGraphResourceDescriptor,
	RenderGraphSubresourceRange,
	RenderGraphTextureAspect,
} from "./types";

const ALL_TEXTURE_ASPECTS: readonly RenderGraphTextureAspect[] = Object.freeze([
	"color",
	"depth",
	"stencil",
]);

/** @internal Normalizes and validates one logical resource range. */
export function normalizeRenderGraphSubresource(
	descriptor: RenderGraphResourceDescriptor,
	range: RenderGraphSubresourceRange | undefined,
	context: {
		readonly nodeId?: RenderGraphNodeId;
		readonly stage?: string;
	},
): {
	readonly range: RenderGraphNormalizedSubresourceRange | undefined;
	readonly diagnostic: RenderGraphDiagnostic | null;
} {
	if (descriptor.kind === "external") {
		if (!range) return { range: undefined, diagnostic: null };
		return invalidRange(descriptor, context, "External resources cannot select a subresource range.");
	}
	if (descriptor.kind === "buffer") {
		if (range && range.kind !== "buffer") {
			return invalidRange(descriptor, context, "A buffer resource requires a buffer byte range.");
		}
		const bufferRange = range?.kind === "buffer" ? range : undefined;
		const totalSize = descriptor.size;
		const offset = bufferRange?.offset ?? 0;
		const size = bufferRange?.size ??
			(totalSize === undefined ? Number.MAX_SAFE_INTEGER : totalSize - offset);
		if (
			!Number.isSafeInteger(offset) || offset < 0 ||
			!Number.isSafeInteger(size) || size <= 0 ||
			(totalSize !== undefined && offset + size > totalSize)
		) {
			return invalidRange(descriptor, context, "Buffer range is invalid or out of bounds.");
		}
		return {
			range: Object.freeze({ kind: "buffer", offset, size }),
			diagnostic: null,
		};
	}
	if (range && range.kind !== "texture") {
		return invalidRange(descriptor, context, "A texture resource requires a texture subresource range.");
	}
	const textureRange = range?.kind === "texture" ? range : undefined;
	const mipLevelCount = descriptor.mipLevelCount ?? inferLegacyMipCount(descriptor);
	const layerCount = descriptor.depthOrArrayLayers ?? 1;
	const mipStart = textureRange?.mipStart ?? 0;
	const normalizedMipCount = textureRange?.mipCount ?? Math.max(1, mipLevelCount - mipStart);
	const layerStart = textureRange?.layerStart ?? 0;
	const normalizedLayerCount = textureRange?.layerCount ?? Math.max(1, layerCount - layerStart);
	const aspects = normalizeAspects(textureRange?.aspects, descriptor.format);
	if (
		!Number.isSafeInteger(mipStart) || mipStart < 0 ||
		!Number.isSafeInteger(normalizedMipCount) || normalizedMipCount <= 0 ||
		mipStart + normalizedMipCount > mipLevelCount ||
		!Number.isSafeInteger(layerStart) || layerStart < 0 ||
		!Number.isSafeInteger(normalizedLayerCount) || normalizedLayerCount <= 0 ||
		layerStart + normalizedLayerCount > layerCount ||
		aspects.length === 0
	) {
		return invalidRange(descriptor, context, "Texture range is invalid or out of bounds.");
	}
	return {
		range: Object.freeze({
			kind: "texture",
			mipStart,
			mipCount: normalizedMipCount,
			layerStart,
			layerCount: normalizedLayerCount,
			aspects: Object.freeze(aspects),
		}),
		diagnostic: null,
	};
}

/** @internal Returns whether two normalized ranges overlap. */
export function renderGraphSubresourcesOverlap(
	left: RenderGraphNormalizedSubresourceRange | undefined,
	right: RenderGraphNormalizedSubresourceRange | undefined,
): boolean {
	if (!left || !right) return true;
	if (left.kind !== right.kind) return false;
	if (left.kind === "buffer" && right.kind === "buffer") {
		return left.offset < right.offset + right.size && right.offset < left.offset + left.size;
	}
	if (left.kind !== "texture" || right.kind !== "texture") return false;
	const mipOverlap =
		left.mipStart < right.mipStart + right.mipCount &&
		right.mipStart < left.mipStart + left.mipCount;
	const layerOverlap =
		left.layerStart < right.layerStart + right.layerCount &&
		right.layerStart < left.layerStart + left.layerCount;
	const aspectOverlap = left.aspects.some((aspect) => right.aspects.includes(aspect));
	return mipOverlap && layerOverlap && aspectOverlap;
}

/** @internal Returns a deterministic allocation compatibility key. */
export function createRenderGraphCompatibilityKey(
	descriptor: RenderGraphResourceDescriptor,
): string {
	if (descriptor.kind === "external") return `external:${descriptor.id}`;
	if (descriptor.kind === "buffer") {
		return [
			"buffer",
			descriptor.size ?? "unknown",
			...(descriptor.allowedUsages ?? []).slice().sort(),
		].join(":");
	}
	return [
		"texture",
		descriptor.format ?? "unknown",
		descriptor.width ?? "unknown",
		descriptor.height ?? "unknown",
		descriptor.depthOrArrayLayers ?? 1,
		descriptor.dimension ?? "2d",
		descriptor.sampleCount ?? 1,
		descriptor.mipLevelCount ?? inferLegacyMipCount(descriptor),
		...(descriptor.allowedUsages ?? []).slice().sort(),
	].join(":");
}

function inferLegacyMipCount(descriptor: {
	readonly width?: number;
	readonly height?: number;
	readonly mipMode?: "single" | "full-chain";
}): number {
	if (descriptor.mipMode !== "full-chain") return 1;
	const maximumDimension = Math.max(1, descriptor.width ?? 1, descriptor.height ?? 1);
	return Math.floor(Math.log2(maximumDimension)) + 1;
}

function normalizeAspects(
	aspects: readonly RenderGraphTextureAspect[] | undefined,
	format: string | undefined,
): RenderGraphTextureAspect[] {
	if (aspects) return Array.from(new Set(aspects));
	if (!format) return [...ALL_TEXTURE_ASPECTS];
	const lower = format.toLowerCase();
	const resolved: RenderGraphTextureAspect[] = [];
	if (lower.includes("depth")) resolved.push("depth");
	if (lower.includes("stencil")) resolved.push("stencil");
	if (resolved.length === 0) resolved.push("color");
	return resolved;
}

function invalidRange(
	descriptor: RenderGraphResourceDescriptor,
	context: { readonly nodeId?: RenderGraphNodeId; readonly stage?: string },
	message: string,
): {
	readonly range: undefined;
	readonly diagnostic: RenderGraphDiagnostic;
} {
	return {
		range: undefined,
		diagnostic: {
			phase: "compile",
			enforcement: "enforced",
			severity: "error",
			code: "invalid-subresource-range",
			stage: context.stage,
			nodeId: context.nodeId,
			resourceId: descriptor.id,
			message: `${message} Resource "${descriptor.id}".`,
		},
	};
}
