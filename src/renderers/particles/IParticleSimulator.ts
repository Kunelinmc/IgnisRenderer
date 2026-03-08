import type { FrameContext } from "../../pipeline/types";

export interface IParticleSimulator {
	beginFrame(context: FrameContext): void;
	simulate(context: FrameContext, deltaTimeMs: number): void;
	emitRenderBatches(context: FrameContext): void;
	endFrame(): void;
}
