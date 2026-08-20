import type { RenderBackendType } from "../backends/IRenderBackend";
import type { FrameContext } from "../pipeline/types";
import type {
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
} from "./types";

const SYNTHETIC_GBUFFER_SEMANTICS: readonly LogicalGBufferSemantic[] = [
	"color",
	"depth",
	"normal",
	"motion",
	"world-position",
	"albedo",
	"roughness",
	"metallic",
	"specular",
	"transmission",
	"emissive",
	"occlusion",
];

export interface SyntheticLogicalGBufferBridgeOptions {
	readonly backend: RenderBackendType;
	readonly normalSpace: LogicalGBufferBridge["normalSpace"];
	readonly depthEncoding: LogicalGBufferBridge["depthEncoding"];
	readonly motionEncoding?: LogicalGBufferBridge["motionEncoding"];
}

/**
 * Creates allocation-free logical G-buffer metadata for declaration planning.
 *
 * @internal Owned by backend post-process executors. Runtime frame execution
 * must use a physical bridge instead.
 */
export function createSyntheticLogicalGBufferBridge(
	context: FrameContext,
	options: SyntheticLogicalGBufferBridgeOptions,
): LogicalGBufferBridge {
	const width = Math.max(1, context.attachments?.width ?? 1);
	const height = Math.max(1, context.attachments?.height ?? 1);
	const channels: LogicalGBufferBridge["channels"] = {};
	for (const semantic of SYNTHETIC_GBUFFER_SEMANTICS) {
		channels[semantic] = {
			semantic,
			handle: { backend: options.backend, resource: null },
			width,
			height,
		};
	}
	return {
		width,
		height,
		normalSpace: options.normalSpace,
		depthEncoding: options.depthEncoding,
		...(options.motionEncoding ? { motionEncoding: options.motionEncoding } : {}),
		channels,
		worldPosition: {
			source: "derived",
			available: true,
		},
	};
}
