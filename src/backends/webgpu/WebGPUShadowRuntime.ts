import type { FrameContext } from "../../pipeline/types";
import type { PreparedFramePacketSet } from "../../pipeline/FramePackets";
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
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { IRenderTexture } from "../types";
import {
	DEFAULT_WEBGPU_PAGED_SHADOW_EXPERIMENT,
	WebGPUPagedShadowExperiment,
	type WebGPUPagedShadowExperimentConfig,
	type WebGPUPagedShadowFrameState,
} from "./WebGPUPagedShadowExperiment";

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
	private readonly _pagedExperiment: WebGPUPagedShadowExperiment;

	constructor(
		backend: WebGPUDeviceResourceHost,
		resourceManager: WebGPUResourceManager,
		geometryRegistry: WebGPUGeometryRegistry,
		animationPayloads: WebGPUAnimationPayloadPool,
		pagedExperimentConfig: Readonly<WebGPUPagedShadowExperimentConfig> =
			DEFAULT_WEBGPU_PAGED_SHADOW_EXPERIMENT,
	) {
		this._pagedExperiment = new WebGPUPagedShadowExperiment(pagedExperimentConfig);
		this._casterRenderer = new WebGPUShadowCasterRenderer(
			backend,
			geometryRegistry,
			animationPayloads
		);
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

	public renderShadows(
		context: FrameContext,
		framePackets: PreparedFramePacketSet,
		encoder?: ICommandEncoder | null,
	): Promise<void> {
		return this._atlasTechnique.render(
			{
				...context,
				scene: {
					...context.scene,
					shadowCasterPackets: framePackets.shadowCasters.slice(),
					shadowTransmitterPackets: framePackets.shadowTransmitters.slice(),
				},
			},
			encoder,
		);
	}

	public prepareAtlas(lightingState: WebGPULightingState, tileSize: number): void {
		this._atlasTechnique.prepare(lightingState, tileSize);
	}

	public preparePagedShadowFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._pagedTechnique.prepareFrame(request);
	}

	public resolvePagedShadowFrame(context: FrameContext): WebGPUPagedShadowFrameState | null {
		return this._pagedExperiment.resolve(context);
	}

	public recordPagedShadowPageMarkPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._pagedTechnique.recordPageMarkPass(request);
	}

	public recordPagedShadowPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._pagedTechnique.recordPageAllocationPass(request);
	}

	public recordPagedShadowPageTableCopyPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
		return this._pagedTechnique.recordPageTableCopyPass(request);
	}

	public recordPagedShadowDepthPass(
		request: WebGPUPagedShadowFrameRequest,
	): Promise<void> {
		return this._pagedTechnique.recordDepthPass(request);
	}

	public recordPagedShadowFeedbackPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void> {
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
