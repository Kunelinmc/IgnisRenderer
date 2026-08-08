import type { FrameContext } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { WebGPULightingState } from "./types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import {
	WebGPUPagedShadowTechnique,
	type WebGPUPagedShadowFrameRequest,
	type WebGPUPagedShadowSamplingResources,
} from "./WebGPUPagedShadowTechnique";
import type { WebGPUResourceManager } from "./WebGPUResourceManager";
import { WebGPUAtlasShadowTechnique } from "./WebGPUAtlasShadowTechnique";
import { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import { WebGPUShadowCasterRenderer } from "./WebGPUShadowCasterRenderer";
import type { IRenderTexture } from "../types";

/**
 * Sole WebGPU owner for atlas and paged shadow techniques.
 *
 * @internal `WebGPUFrameServiceOwner` delegates shadow resource lifecycle and
 * frame execution to this runtime.
 */
export class WebGPUShadowRuntime {
	private readonly _casterRenderer: WebGPUShadowCasterRenderer;
	private readonly _atlasTechnique: WebGPUAtlasShadowTechnique;
	private readonly _pagedTechnique: WebGPUPagedShadowTechnique;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		geometryRegistry: WebGPUGeometryRegistry,
	) {
		this._casterRenderer = new WebGPUShadowCasterRenderer(backend, geometryRegistry);
		this._atlasTechnique = new WebGPUAtlasShadowTechnique(
			new WebGPUShadowAtlasAllocator(backend),
			this._casterRenderer,
		);
		this._pagedTechnique = new WebGPUPagedShadowTechnique(
			backend,
			resourceManager,
			this._casterRenderer,
		);
	}

	public warmup(): Promise<void> {
		return this._casterRenderer.warmup();
	}

	public renderAtlas(context: FrameContext, encoder?: ICommandEncoder | null): Promise<void> {
		return this._atlasTechnique.render(context, encoder);
	}

	public prepareAtlas(lightingState: WebGPULightingState, tileSize: number): void {
		this._atlasTechnique.prepare(lightingState, tileSize);
	}

	public preparePaged(request: WebGPUPagedShadowFrameRequest): void {
		this._pagedTechnique.prepareFrame(request);
	}

	public recordPageMark(request: WebGPUPagedShadowFrameRequest): void | Promise<void> {
		return this._pagedTechnique.recordPageMarkPass(request);
	}

	public recordPageAllocation(request: WebGPUPagedShadowFrameRequest): void | Promise<void> {
		return this._pagedTechnique.recordPageAllocationPass(request);
	}

	public recordPageTableCopy(request: WebGPUPagedShadowFrameRequest): void | Promise<void> {
		return this._pagedTechnique.recordPageTableCopyPass(request);
	}

	public recordPagedDepth(request: WebGPUPagedShadowFrameRequest): Promise<void> {
		return this._pagedTechnique.recordDepthPass(request);
	}

	public recordFeedback(request: WebGPUPagedShadowFrameRequest): void | Promise<void> {
		return this._pagedTechnique.recordFeedbackPass(request);
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

	public getSamplingResources(): WebGPUPagedShadowSamplingResources {
		return this._pagedTechnique.getSamplingResources();
	}

	public onShaderRuntimeChanged(): void {
		this._casterRenderer.onShaderRuntimeChanged();
	}

	public destroy(): void {
		this._pagedTechnique.destroy();
		this._atlasTechnique.destroy();
		this._casterRenderer.destroy();
	}
}
