import type { DrawPacket, FrameContext } from "../../../pipeline/types";

export interface SoftwarePassLike<
	TRenderArgs extends unknown[] = [FrameContext],
	TRenderResult = void | Promise<void>,
> {
	render(...args: TRenderArgs): TRenderResult;
	destroy?(): void;
}

/**
 * @internal Software-only surface composite pass executed after a material pass.
 */
export interface SoftwareSurfaceCompositePass {
	composite(context: FrameContext, packets: DrawPacket[]): void | Promise<void>;
}
