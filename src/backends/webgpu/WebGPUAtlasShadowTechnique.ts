import type { FrameContext } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { WebGPULightingState } from "./types";
import type { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type { WebGPUShadowCasterRenderer } from "./WebGPUShadowCasterRenderer";

/**
 * Owns atlas-specific allocation and frame execution.
 *
 * @internal `WebGPUShadowRuntime` owns this technique and the shared caster
 * renderer it uses.
 */
export class WebGPUAtlasShadowTechnique {
	constructor(
		private readonly _allocator: WebGPUShadowAtlasAllocator,
		private readonly _casterRenderer: WebGPUShadowCasterRenderer,
	) {}

	public get allocator(): WebGPUShadowAtlasAllocator {
		return this._allocator;
	}

	public prepare(lightingState: WebGPULightingState, tileSize: number): void {
		this._allocator.prepare(lightingState, tileSize);
	}

	public render(context: FrameContext, encoder?: ICommandEncoder | null): Promise<void> {
		return this._casterRenderer.renderAtlas(context, this._allocator, encoder);
	}

	public destroy(): void {
		this._allocator.destroy();
	}
}
