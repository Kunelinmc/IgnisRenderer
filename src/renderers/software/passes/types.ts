import type { FrameContext } from "../../../pipeline/types";

export interface SoftwarePassLike<
	TRenderArgs extends unknown[] = [FrameContext],
	TRenderResult = void | Promise<void>,
> {
	render(...args: TRenderArgs): TRenderResult;
	destroy?(): void;
}
