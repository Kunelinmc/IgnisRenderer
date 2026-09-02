import type { FrameContext } from "../../pipeline/types";
import type { PreparedFramePacketSet } from "../../pipeline/FramePackets";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { WebGPULightingState } from "./types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import { WebGPUAtlasShadowTechnique } from "./WebGPUAtlasShadowTechnique";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowCasterRenderer } from "./WebGPUShadowCasterRenderer";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { IRenderTexture } from "../types";

/**
 * Sole WebGPU owner for atlas shadow technique.
 *
 * @internal `WebGPUFrameServiceOwner` delegates shadow resource lifecycle and
 * frame execution to this runtime.
 */
export class WebGPUShadowRuntime {
	private readonly _casterRenderer: WebGPUShadowCasterRenderer;
	private readonly _atlasTechnique: WebGPUAtlasShadowTechnique;

	constructor(
		backend: WebGPUDeviceResourceHost,
		geometryRegistry: WebGPUGeometryRegistry,
		animationPayloads: WebGPUAnimationPayloadPool,
	) {
		this._casterRenderer = new WebGPUShadowCasterRenderer(
			backend,
			geometryRegistry,
			animationPayloads
		);
		this._atlasTechnique = new WebGPUAtlasShadowTechnique(
			new WebGPUShadowAtlasAllocator(backend),
			this._casterRenderer,
		);
	}

	public warmup(): Promise<void> {
		return this._casterRenderer.warmup();
	}

	public renderShadows(
		context: FrameContext,
		framePackets: PreparedFramePacketSet,
		encoder?: ICommandEncoder | null,
	): Promise<void> {
		return this._atlasTechnique.render(
			context,
			framePackets.shadowCasterSubmissions,
			framePackets.shadowTransmitterSubmissions,
			encoder,
		);
	}

	public prepareAtlas(lightingState: WebGPULightingState, tileSize: number): void {
		this._atlasTechnique.prepare(lightingState, tileSize);
	}

	public get atlas(): IRenderTexture | null {
		return this._atlasTechnique.allocator.atlas;
	}

	public get transmittanceAtlas(): IRenderTexture | null {
		return this._atlasTechnique.allocator.transmittanceAtlas;
	}

	public ensureAtlasForTileSize(tileSize: number): IRenderTexture {
		return this._atlasTechnique.allocator.ensureAtlasForTileSize(tileSize);
	}

	public onShaderRuntimeChanged(): void {
		this._casterRenderer.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._atlasTechnique.destroy();
		this._casterRenderer.destroy();
	}
}
