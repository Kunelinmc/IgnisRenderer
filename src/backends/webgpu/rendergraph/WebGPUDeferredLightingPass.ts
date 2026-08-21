import type { FrameContext } from "../../../pipeline/types";
import {
	type IBindingGroup,
	type IRenderTexture,
} from "../../types";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { GBufferSlot } from "../constants";
import type { WebGPUDeferredResourceProvider } from "../WebGPUResourceContracts";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";

/**
 * Owns deferred G-buffer bindings and full-screen lighting resolve.
 */
export class WebGPUDeferredLightingPass {
	private readonly _host: WebGPUFrameHost;
	private readonly _resources: WebGPUDeferredResourceProvider;
	private _frame: Pick<
		WebGPUFrameExecutionContext,
		"commands" | "targets" | "resources" | "dirtyRects"
	> | null = null;
	private _gbufferWriteBinding: IBindingGroup | null = null;
	private _gbufferWriteBindingSources: IRenderTexture[] = [];
	private _gbufferReadBinding: IBindingGroup | null = null;
	private _gbufferReadBindingSources: IRenderTexture[] = [];

	constructor(
		host: WebGPUFrameHost,
		resources: WebGPUDeferredResourceProvider,
	) {
		this._host = host;
		this._resources = resources;
	}

	public bindFrame(frame: WebGPUFrameExecutionContext): void {
		this._frame = frame;
	}

	public closeFrame(): void {
		this._frame = null;
	}

	public destroyBindings(): void {
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = null;
		this._gbufferWriteBindingSources = [];
		this._destroyBindingGroup(this._gbufferReadBinding);
		this._gbufferReadBinding = null;
		this._gbufferReadBindingSources = [];
	}

	/** Ensures all frame-global deferred bindings and resolve pipeline exist. */
	public async preflight(): Promise<void> {
		this.getGBufferWriteBinding();
		this.getGBufferReadBinding();
		await this._resources.getDeferredLightingPipeline();
	}

	public getGBufferWriteBinding(): IBindingGroup {
		const targets = this._requireFrame().targets.frameTargets;
		const placeholders =
			targets.gMaterialExt0 && targets.gMaterialExt3
				? null
				: this._resources.getDeferredPlaceholderTextures();
		const sources = [
			targets.gMaterialExt0 ?? placeholders!.rgba16Float,
			targets.gMaterialExt3 ?? placeholders!.rgba16Uint,
		];
		if (
			this._gbufferWriteBinding &&
			this._gbufferWriteBindingSources.length === sources.length &&
			this._gbufferWriteBindingSources.every((source, index) => source === sources[index])
		) {
			return this._gbufferWriteBinding;
		}
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = this._host.createBindingGroup({
			layout: this._resources.getGBufferWriteLayout(),
			entries: [
				{ binding: 0, resource: sources[0] },
				{ binding: 1, resource: sources[1] },
			],
			label: "WebGPUGBufferWriteBinding",
		});
		this._gbufferWriteBindingSources = sources;
		return this._gbufferWriteBinding;
	}

	public getGBufferReadBinding(): IBindingGroup {
		const targets = this._requireFrame().targets.frameTargets;
		const placeholders =
			targets.gSpecular &&
			targets.gCoatSheen &&
			targets.gSheenReflectance &&
			targets.gMaterialExt0 &&
			targets.gMaterialExt3
				? null
				: this._resources.getDeferredPlaceholderTextures();
		const sources: IRenderTexture[] = [];
		sources[GBufferSlot.AlbedoAlpha] = targets.gAlbedoAlpha;
		sources[GBufferSlot.NormalRoughMetal] = targets.gNormalRoughMetal;
		sources[GBufferSlot.EmissiveOcclusion] = targets.gEmissiveOcclusion;
		sources[GBufferSlot.MotionDepth] = targets.gMotionDepth;
		sources[GBufferSlot.Specular] = targets.gSpecular ?? placeholders!.rgba16Float;
		sources[GBufferSlot.CoatSheen] = targets.gCoatSheen ?? placeholders!.rgba16Float;
		sources[GBufferSlot.SheenReflectance] =
			targets.gSheenReflectance ?? placeholders!.rgba8Unorm;
		sources.push(
			targets.gMaterialExt0 ?? placeholders!.rgba16Float,
			targets.gMaterialExt3 ?? placeholders!.rgba16Uint,
		);
		if (
			this._gbufferReadBinding &&
			this._gbufferReadBindingSources.length === sources.length &&
			this._gbufferReadBindingSources.every((source, index) => source === sources[index])
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
		clearSceneColor: boolean,
	): Promise<void> {
		const frame = this._requireFrame();
		const encoder = frame.commands.encoder;
		const targets = frame.targets.frameTargets;
		if (!encoder) {
			return;
		}
		const pipeline = await this._resources.getDeferredLightingPipeline();
		const gbufferReadBinding = this.getGBufferReadBinding();
		encoder.beginRenderPass({
			label: "WebGPUDeferredLighting",
			colorAttachments: [
				{
					view: targets.sceneColorMain,
					clearValue: {
						r: 0,
						g: 0,
						b: 0,
						a: context.presentationAlphaMode === "premultiplied" ? 0 : 1,
					},
					loadOp: clearSceneColor ? "clear" : "load",
					storeOp: "store",
				},
			],
		});
		encoder.setPipeline(pipeline);
		const frameResources = frame.resources;
		encoder.setBindingGroup(0, frameResources.frameBinding);
		encoder.setBindingGroup(1, this._resources.getDeferredUnusedBinding());
		encoder.setBindingGroup(2, frameResources.clusteredSceneBinding);
		encoder.setBindingGroup(3, gbufferReadBinding);
		const dirtyRects = frame.dirtyRects.resolveDirtyRects(
			context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height,
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

	private _requireFrame() {
		if (!this._frame) {
			throw new Error("WebGPU deferred lighting frame is not active.");
		}
		return this._frame;
	}
}
