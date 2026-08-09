import type { FrameContext } from "../../pipeline/types";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	LogicalGBufferSemantic,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import { createPostProcessResourceAccessor } from "../../postprocess/PostProcessResourceAccessor";
import type { SoftwareFrameView } from "./SoftwareFrameView";
import type { PostProcessColorDomain } from "../../postprocess/PostProcessPass";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";

/**
 * Executes logical post-process passes on the software backend.
 */
export class SoftwarePostProcessExecutor implements IPostProcessExecutor {
	public readonly backend = "software";
	private _activeFrame: SoftwareFrameView | null = null;
	private _outputColorDomain: PostProcessColorDomain = "scene-linear-hdr";

	constructor(
		private readonly _getSceneColor?: () => Float32Array,
		private readonly _getDisplayOutputState?: () => DisplayOutputState,
	) {}

	public get outputColorDomain(): PostProcessColorDomain {
		return this._outputColorDomain;
	}

	public bindSoftwareFrame(
		frame: SoftwareFrameView,
		initialColorDomain: PostProcessColorDomain = "scene-linear-hdr",
	): void {
		this._activeFrame = frame;
		this._outputColorDomain = initialColorDomain;
	}

	public unbindSoftwareFrame(): void {
		this._activeFrame = null;
	}

	/**
	 * Allocates a CPU-backed post-process resource.
	 *
	 * @param desc Resource descriptor from the backend post-process resource pool.
	 * @returns Software resource handle wrapping a `Float32Array`.
	 * @sideEffects Allocates typed-array storage.
	 */
	public createResource(desc: PostProcessResourceDescriptor): PostProcessResourceHandle {
		const floatsPerPixel = 4;
		return {
			id: desc.id,
			backend: this.backend,
			width: desc.width,
			height: desc.height,
			format: desc.format,
			resource: new Float32Array(desc.width * desc.height * floatsPerPixel),
		};
	}

	/**
	 * Releases a CPU-backed post-process resource.
	 *
	 * @param _handle Resource handle to release.
	 * @returns Nothing.
	 * @sideEffects None; JavaScript owns typed-array memory reclamation.
	 */
	public destroyResource(_handle: PostProcessResourceHandle): void {}

	/**
	 * Creates a logical G-buffer bridge for the current software frame.
	 *
	 * @param context Current renderer frame context.
	 * @returns Logical bridge wrapping CPU frame attachments.
	 * @sideEffects None.
	 */
	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return createSoftwareGBufferBridge(context, this._resolveSceneColor(context));
	}

	/**
	 * Provides CPU helper objects for pass-owned software implementations.
	 *
	 * @param request Pass-owned implementation context request.
	 * @returns Context object expected by the selected software implementation.
	 * @sideEffects None.
	 */
	public createPassExecutionContext(request: PostProcessPassExecutionContextRequest): unknown {
		const getGBuffer = (semantic: LogicalGBufferSemantic) => {
			const handle = request.gBuffer.channels[semantic]?.handle;
			return handle?.backend === "software" && "data" in handle ? handle.data : null;
		};
		const frame = this._activeFrame;
		if (!frame) {
			throw new Error("Software post-process executor requires an active frame view.");
		}
		const context: Record<string, unknown> = {
			attachments: request.frameContext.attachments,
			canvasContext: null,
			displayOutput: this._getDisplayOutputState?.(),
			dirtyRects: frame.clipRegions.map((region) => ({
				minX: region.minX,
				minY: region.minY,
				maxX: region.maxXExclusive - 1,
				maxY: region.maxYExclusive - 1,
			})),
			resources: createPostProcessResourceAccessor<ArrayBufferView>({
				passId: request.passId,
				declaration: request.declaration,
				colorInput: frame.attachments.color,
				colorOutput:
					request.declaration.color.access === "read-write"
						? frame.attachments.color
						: null,
				getGBuffer: (semantic) => getGBuffer(semantic),
				getHistory: (id) => {
					const slot = request.histories[id];
					return slot
						? {
								read: slot.read.resource as ArrayBufferView,
								write: slot.write.resource as ArrayBufferView,
								valid: slot.valid,
							}
						: null;
				},
				getTransient: (id) =>
					(request.transients[id]?.handle.resource as ArrayBufferView | null) ?? null,
				getShared: () => null,
			}),
		};
		return Object.freeze(context);
	}

	public completePass(request: PostProcessPassRequest, result: PostProcessPassResult): void {
		if (!result.ran) return;
		const contract = request.pass.colorContract;
		if (contract?.input === this._outputColorDomain) {
			this._outputColorDomain = contract.output;
		}
	}

	private _resolveSceneColor(context: FrameContext): Float32Array {
		if (this._activeFrame) return this._activeFrame.attachments.color;
		if (this._getSceneColor) return this._getSceneColor();
		return new Float32Array(context.attachments.width * context.attachments.height * 4);
	}
}

/**
 * Creates a semantic G-buffer bridge for software frame attachments.
 *
 * @param context Current frame context.
 * @returns Logical bridge wrapping CPU color, depth, and normal buffers.
 * @sideEffects None.
 */
export function createSoftwareGBufferBridge(
	context: FrameContext,
	sceneColor?: Float32Array,
): LogicalGBufferBridge {
	const attachments = context.attachments;
	const width = Math.max(1, attachments.width);
	const height = Math.max(1, attachments.height);
	return {
		width,
		height,
		normalSpace: "view",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			color: {
				semantic: "color",
				handle: {
					backend: "software",
					data: sceneColor ?? null,
					stride: 4,
				},
				width,
				height,
				format: "rgba32float",
			},
			depth: attachments.depthBuffer ?
				{
					semantic: "depth",
					handle: {
						backend: "software",
						data: attachments.depthBuffer,
						stride: 1,
					},
					width,
					height,
					format: "float32",
					encoding: "linear-view-z",
				}
			:	undefined,
			normal: attachments.normalBuffer ?
				{
					semantic: "normal",
					handle: {
						backend: "software",
						data: attachments.normalBuffer,
						stride: 3,
					},
					width,
					height,
					format: "float32x3",
					encoding: "view-normal",
				}
			:	undefined,
			motion: attachments.motionBuffer ?
				{
					semantic: "motion",
					handle: {
						backend: "software",
						data: attachments.motionBuffer,
						stride: 4,
					},
					width,
					height,
					format: "float32x4",
					encoding: "motion-depth.xy",
				}
			:	undefined,
		},
		worldPosition: {
			source: "derived",
			available: !!attachments.depthBuffer,
		},
	};
}
