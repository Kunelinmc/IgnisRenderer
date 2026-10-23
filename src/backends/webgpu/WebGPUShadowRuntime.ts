import type { FrameContext } from "../../pipeline/types";
import type { PreparedFramePacketSet } from "../../pipeline/FramePackets";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { WebGPULightingState } from "./types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowCasterRenderer } from "./WebGPUShadowCasterRenderer";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { IRenderTexture } from "../types";

/**
 * Coordinates WebGPU shadow atlas resources and caster execution.
 *
 * @internal `WebGPUFrameServiceOwner` delegates shadow resource lifecycle and
 * frame execution to this runtime.
 */
export class WebGPUShadowRuntime {
	private readonly _casterRenderer: WebGPUShadowCasterRenderer;
	private readonly _atlasAllocator: WebGPUShadowAtlasAllocator;

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
		this._atlasAllocator = new WebGPUShadowAtlasAllocator(backend);
	}

	public warmup(): Promise<void> {
		return this._casterRenderer.warmup();
	}

	public renderShadows(
		context: FrameContext,
		framePackets: PreparedFramePacketSet,
		encoder?: ICommandEncoder | null,
	): Promise<void> {
		return this._casterRenderer.renderAtlas(
			context,
			framePackets.shadowCasterSubmissions,
			framePackets.shadowTransmitterSubmissions,
			this._atlasAllocator,
			encoder,
		);
	}

	public prepareAtlas(lightingState: WebGPULightingState, tileSize: number): void {
		this._atlasAllocator.prepare(lightingState, tileSize);
	}

	public get atlas(): IRenderTexture | null {
		return this._atlasAllocator.atlas;
	}

	public get transmittanceAtlas(): IRenderTexture | null {
		return this._atlasAllocator.transmittanceAtlas;
	}

	public ensureAtlasForTileSize(tileSize: number): IRenderTexture {
		return this._atlasAllocator.ensureAtlasForTileSize(tileSize);
	}

	public onShaderRuntimeChanged(): void {
		this._casterRenderer.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._atlasAllocator.destroy();
		this._casterRenderer.destroy();
	}
}
