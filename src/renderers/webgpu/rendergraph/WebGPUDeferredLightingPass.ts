import type { FrameContext } from "../../../pipeline/types";
import {
	type IBindingGroup,
	type IRenderTexture,
} from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { GBufferSlot } from "../constants";
import type { WebGPUDeferredResourceProvider } from "../WebGPUResourceContracts";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";

export interface WebGPUDeferredLightingPassCallbacks {
	readonly recordingContext: WebGPUFrameGraphRecordingContext;
}

/**
 * Owns deferred G-buffer bindings and full-screen lighting resolve.
 */
export class WebGPUDeferredLightingPass {
	private readonly _host: WebGPUFrameHost;
	private readonly _resources: WebGPUDeferredResourceProvider;
	private readonly _recordingContext: WebGPUFrameGraphRecordingContext;
	private _gbufferWriteBinding: IBindingGroup | null = null;
	private _gbufferWriteBindingSources: IRenderTexture[] = [];
	private _gbufferReadBinding: IBindingGroup | null = null;
	private _gbufferReadBindingSources: IRenderTexture[] = [];

	public constructor(
		host: WebGPUFrameHost,
		resources: WebGPUDeferredResourceProvider,
		callbacks: WebGPUDeferredLightingPassCallbacks
	) {
		this._host = host;
		this._resources = resources;
		this._recordingContext = callbacks.recordingContext;
	}

	public destroyBindings(): void {
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = null;
		this._gbufferWriteBindingSources = [];
		this._destroyBindingGroup(this._gbufferReadBinding);
		this._gbufferReadBinding = null;
		this._gbufferReadBindingSources = [];
	}

	public getGBufferWriteBinding(): IBindingGroup {
		const targets = this._recordingContext.getFrameTargets();
		if (
			!targets?.gMaterialExt0 ||
			!targets.gMaterialExt1 ||
			!targets.gMaterialExt2 ||
			!targets.gMaterialExt3
		) {
			throw new Error("WebGPU deferred G-buffer storage targets are unavailable.");
		}
		const sources = [
			targets.gMaterialExt0,
			targets.gMaterialExt1,
			targets.gMaterialExt2,
			targets.gMaterialExt3,
		];
		if (
			this._gbufferWriteBinding &&
			this._gbufferWriteBindingSources.length === sources.length &&
			this._gbufferWriteBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._gbufferWriteBinding;
		}
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = this._host.createBindingGroup({
			layout: this._resources.getGBufferWriteLayout(),
			entries: [
				{ binding: 0, resource: sources[0] },
				{ binding: 1, resource: sources[1] },
				{ binding: 2, resource: sources[2] },
				{ binding: 3, resource: sources[3] },
			],
			label: "WebGPUGBufferWriteBinding",
		});
		this._gbufferWriteBindingSources = sources;
		return this._gbufferWriteBinding;
	}

	public getGBufferReadBinding(): IBindingGroup {
		const targets = this._recordingContext.getFrameTargets();
		if (
			!targets?.gSpecular ||
			!targets.gCoatSheen ||
			!targets.gSheenReflectance ||
			!targets.gMaterialExt0 ||
			!targets.gMaterialExt1 ||
			!targets.gMaterialExt2 ||
			!targets.gMaterialExt3
		) {
			throw new Error("WebGPU deferred G-buffer read targets are unavailable.");
		}
		const sources: IRenderTexture[] = [];
		sources[GBufferSlot.AlbedoAlpha] = targets.gAlbedoAlpha;
		sources[GBufferSlot.NormalRoughMetal] = targets.gNormalRoughMetal;
		sources[GBufferSlot.EmissiveOcclusion] = targets.gEmissiveOcclusion;
		sources[GBufferSlot.MotionDepth] = targets.gMotionDepth;
		sources[GBufferSlot.Specular] = targets.gSpecular;
		sources[GBufferSlot.CoatSheen] = targets.gCoatSheen;
		sources[GBufferSlot.SheenReflectance] = targets.gSheenReflectance;
		sources.push(
			targets.gMaterialExt0,
			targets.gMaterialExt1,
			targets.gMaterialExt2,
			targets.gMaterialExt3
		);
		if (
			this._gbufferReadBinding &&
			this._gbufferReadBindingSources.length === sources.length &&
			this._gbufferReadBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._gbufferReadBinding;
		}
		this._destroyBindingGroup(this._gbufferReadBinding);
		this._gbufferReadBinding = this._host.createBindingGroup({
			layout: this._resources.getGBufferReadLayout(),
			entries: sources.map((resource, binding) => ({
				binding,
				resource,
			})),
			label: "WebGPUGBufferReadBinding",
		});
		this._gbufferReadBindingSources = sources as IRenderTexture[];
		return this._gbufferReadBinding;
	}

	public async recordLightingPass(
		context: FrameContext,
		clearSceneColor: boolean
	): Promise<void> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets) {
			return;
		}
		const pipeline = await this._resources.getDeferredLightingPipeline();
		const gbufferReadBinding = this.getGBufferReadBinding();
		encoder.beginRenderPass({
			label: "WebGPUDeferredLighting",
			colorAttachments: [
				{
					view: targets.sceneColorMain,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearSceneColor ? "clear" : "load",
					storeOp: "store",
				},
			],
		});
		encoder.setPipeline(pipeline);
		const frameResources = this._recordingContext.requireFrameResources();
		encoder.setBindingGroup(0, frameResources.frameBinding);
		encoder.setBindingGroup(1, this._resources.getDeferredUnusedBinding());
		encoder.setBindingGroup(2, frameResources.clusteredSceneBinding);
		encoder.setBindingGroup(3, gbufferReadBinding);
		const dirtyRects = this._recordingContext.resolveDirtyRects(
			context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height
		);
		for (const rect of dirtyRects) {
			encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			encoder.draw(3);
		}
		encoder.endRenderPass();
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}
}
