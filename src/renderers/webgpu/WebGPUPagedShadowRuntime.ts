import type { ShadowCastingLight } from "../../lights";
import type { ShadowRenderSet } from "../../lights/shadows/ShadowMapping";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IRenderBuffer,
	IRenderTexture,
} from "../types";

export interface WebGPUPagedShadowFrameRequest {
	context: FrameContext;
	encoder: ICommandEncoder | null;
	renderSets: ReadonlyMap<ShadowCastingLight, ShadowRenderSet>;
	shadowCasterPackets: readonly DrawPacket[];
	shadowTransmitterPackets: readonly DrawPacket[];
}

export interface WebGPUPagedShadowResources {
	pageTable: IRenderBuffer | IRenderTexture;
	physicalDepthAtlas: IRenderTexture;
	physicalTransmittanceAtlas: IRenderTexture | null;
	pageMetadataBuffer: IRenderBuffer;
}

/**
 * WebGPU paged shadow runtime skeleton.
 *
 * V1 records no GPU work and allocates no page-table resources; it only gives
 * the frame graph a stable integration point for future virtual shadow passes.
 */
export class WebGPUPagedShadowRuntime {
	private _lastRequest: WebGPUPagedShadowFrameRequest | null = null;

	/**
	 * @internal WebGPU frame graph preparation hook.
	 *
	 * @param request Current frame context and paged shadow render sets.
	 * @remarks V1 stores the request for diagnostics and allocates no resources.
	 */
	public prepareFrame(request: WebGPUPagedShadowFrameRequest): void {
		this._lastRequest = request;
	}

	/**
	 * @internal WebGPU frame graph page-mark pass hook.
	 *
	 * @param request Current frame context and shadow caster packets.
	 * @returns Nothing in v1; future versions may return asynchronous GPU work.
	 */
	public recordPageMarkPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		this._lastRequest = request;
	}

	/**
	 * @internal WebGPU frame graph page-allocation pass hook.
	 *
	 * @param request Current frame context and paged shadow render sets.
	 * @returns Nothing in v1; future versions may allocate page table resources.
	 */
	public recordPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		this._lastRequest = request;
	}

	/**
	 * @internal WebGPU frame graph depth pass hook.
	 *
	 * @param request Current frame context and shadow draw packets.
	 * @returns Nothing in v1; future versions may render dirty physical pages.
	 */
	public recordDepthPass(
		request: WebGPUPagedShadowFrameRequest
	): void | Promise<void> {
		this._lastRequest = request;
	}

	/**
	 * @internal WebGPU resource query hook for shadow samplers and debug tools.
	 *
	 * @returns `null` in v1 because no page table or physical atlas is allocated.
	 */
	public getResources(): WebGPUPagedShadowResources | null {
		return null;
	}

	/**
	 * @internal WebGPU runtime lifecycle hook.
	 *
	 * @remarks Clears cached frame state. No GPU resources are owned in v1.
	 */
	public destroy(): void {
		this._lastRequest = null;
	}
}
