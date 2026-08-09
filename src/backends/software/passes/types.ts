import type { DrawPacket } from "../../../pipeline/types";
import type { SoftwarePassContext } from "../SoftwareFrameServices";

export interface SoftwarePassLike<
	TRenderArgs extends unknown[] = [SoftwarePassContext],
	TRenderResult = void | Promise<void>,
> {
	render(...args: TRenderArgs): TRenderResult;
	destroy?(): void;
}

/**
 * @internal Software-only surface composite pass executed after a material pass.
 */
export interface SoftwareSurfaceCompositePass {
	composite(context: SoftwarePassContext, packets: DrawPacket[]): void | Promise<void>;
}
