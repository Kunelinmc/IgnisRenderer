import type { FrameContext } from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	type IBindingGroup,
	type IRenderTexture,
} from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "../WebGPURenderResources";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessContracts";

export interface WebGPUDeferredLightingPassCallbacks {
	getEncoder(): ICommandEncoder | null;
	getFrameTargets(): WebGPUFrameTargets | null;
	requireFrameResources(): WebGPUPreparedFrameResources;
	resolveDirtyRects(
		context: FrameContext,
		width: number,
		height: number
	): Array<{ x: number; y: number; width: number; height: number }>;
}

/**
 * Owns deferred G-buffer bindings and full-screen lighting resolve.
 */
export class WebGPUDeferredLightingPass {
	private readonly _backend: WebGPUBackend;
	private readonly _resources: WebGPURenderResources;
	private readonly _callbacks: WebGPUDeferredLightingPassCallbacks;
	private _gbufferWriteBinding: IBindingGroup | null = null;
	private _gbufferWriteBindingSources: IRenderTexture[] = [];
	private _gbufferReadBinding: IBindingGroup | null = null;
	private _gbufferReadBindingSources: IRenderTexture[] = [];

	public constructor(
		backend: WebGPUBackend,
		resources: WebGPURenderResources,
		callbacks: WebGPUDeferredLightingPassCallbacks
	) {
		this._backend = backend;
		this._resources = resources;
		this._callbacks = callbacks;
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
		const targets = this._callbacks.getFrameTargets();
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
		this._gbufferWriteBinding = this._backend.createBindingGroup({
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
		const targets = this._callbacks.getFrameTargets();
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
		const sources = [
			targets.gAlbedoAlpha,
			targets.gNormalRoughMetal,
			targets.gEmissiveOcclusion,
			targets.gMotionDepth,
			targets.gSpecular,
			targets.gCoatSheen,
			targets.gSheenReflectance,
			targets.gMaterialExt0,
			targets.gMaterialExt1,
			targets.gMaterialExt2,
			targets.gMaterialExt3,
		];
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
		this._gbufferReadBinding = this._backend.createBindingGroup({
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
		const encoder = this._callbacks.getEncoder();
		const targets = this._callbacks.getFrameTargets();
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
		const frameResources = this._callbacks.requireFrameResources();
		encoder.setBindingGroup(0, frameResources.frameBinding);
		encoder.setBindingGroup(1, this._resources.getDeferredUnusedBinding());
		encoder.setBindingGroup(2, frameResources.clusteredSceneBinding);
		encoder.setBindingGroup(3, gbufferReadBinding);
		const dirtyRects = this._callbacks.resolveDirtyRects(
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
