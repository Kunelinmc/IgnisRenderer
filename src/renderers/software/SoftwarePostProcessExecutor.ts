import type { FrameContext } from "../../pipeline/types";
import type {
	IPostProcessExecutor,
	LogicalGBufferBridge,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type { SoftwareVolumetricLightingContext } from "../../postprocess/passes/VolumetricLightingPass";
import type { SoftwareBuiltinPostProcessContext } from "../../postprocess/passes/BuiltinScreenPasses";

export interface SoftwarePostProcessExecutorHost {
	getCanvasContext(): CanvasRenderingContext2D | null;
}

/**
 * Executes logical post-process passes on the software backend.
 */
export class SoftwarePostProcessExecutor implements IPostProcessExecutor {
	public readonly backend = "software";
	private _host: SoftwarePostProcessExecutorHost;

	constructor(host: SoftwarePostProcessExecutorHost) {
		this._host = host;
	}

	/**
	 * Allocates a CPU-backed post-process resource.
	 *
	 * @param desc Resource descriptor from the backend post-process resource pool.
	 * @returns Software resource handle wrapping a `Float32Array`.
	 * @sideEffects Allocates typed-array storage.
	 */
	public createResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
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
		return createSoftwareGBufferBridge(context);
	}

	/**
	 * Executes one logical software post-process pass.
	 *
	 * @param passId Logical pass id.
	 * @param request Current pass request.
	 * @returns Execution result for pipeline history tracking.
	 * @sideEffects Mutates the current software color buffer or canvas output.
	 */
	public executePass(
		passId: string,
		request: PostProcessPassRequest
	): PostProcessPassResult {
		void passId;
		void request;
		return { ran: false };
	}

	/**
	 * Provides CPU helper objects for pass-owned software implementations.
	 *
	 * @param request Pass-owned implementation context request.
	 * @returns Context object expected by the selected software implementation.
	 * @sideEffects None.
	 */
	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		switch (request.passId) {
			case "ssao":
				return {
					attachments: request.frameContext.attachments,
				};
			case "taa":
				return {
					attachments: request.frameContext.attachments,
				};
			case "volumetric": {
				const context: SoftwareVolumetricLightingContext = {
					canvasContext: this._host.getCanvasContext(),
				};
				return context;
			}
			case "fxaa":
				return {
					attachments: request.frameContext.attachments,
					canvasContext: this._host.getCanvasContext(),
				};
			case "tonemap":
			case "color-filter":
			case "interaction-outline":
			case "gamma": {
				const context: SoftwareBuiltinPostProcessContext = {
					canvasContext: this._host.getCanvasContext(),
				};
				return context;
			}
			default:
				return undefined;
		}
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
	context: FrameContext
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
					data: attachments.pixels ?? null,
					stride: 4,
				},
				width,
				height,
				format: "rgba8unorm",
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
