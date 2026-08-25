import type { ICommandEncoder } from "../ICommandEncoder";
import { TextureFormat } from "../../core/TextureFormat";
import { createRenderViewTransient } from "../../pipeline/RenderViewTransient";
import {
	prepareFramePackets,
	type PreparedFramePacketSet,
} from "../../pipeline/FramePackets";
import type { FrameContext } from "../../pipeline/types";
import type {
	CustomRenderTargetExecutionTarget,
	PreparedRenderTargetJob,
} from "../../rendering/CustomRenderTargets";
import { RenderTargetRegistrySnapshot } from "../../rendering/CustomRenderTargets";
import { TextureUsage, type IRenderTexture } from "../types";
import { submitWebGPUDraws } from "./WebGPUDrawSubmission";
import type {
	WebGPUFrameResourceProvider,
	WebGPUParticleBillboardRenderer,
	WebGPUParticleBillboardRendererProvider,
	WebGPUPreparedFrameResources,
	WebGPUFrameResourceScope,
	WebGPUSceneResourceProvider,
} from "./WebGPUResourceContracts";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";

/** Records a prepared renderer view into an arbitrary single-color target. */
export class WebGPURenderTargetViewExecutor {
	private readonly _particleRenderer: WebGPUParticleBillboardRenderer;
	private readonly _scratchDepth = new Map<string, IRenderTexture>();
	private readonly _pendingScopes: WebGPUFrameResourceScope[] = [];

	public constructor(
		private readonly _host: WebGPUFrameHost,
		private readonly _resources: WebGPUFrameResourceProvider &
			WebGPUSceneResourceProvider & WebGPUParticleBillboardRendererProvider,
	) {
		this._particleRenderer = _resources.getParticleBillboardRenderer();
	}

	public async execute(
		encoder: ICommandEncoder,
		baseContext: FrameContext,
		job: PreparedRenderTargetJob,
		target: CustomRenderTargetExecutionTarget,
	): Promise<void> {
		if (job.descriptor.kind !== "scene-view" || !job.scene) return;
		const context = this._createContext(baseContext, job, target);
		const packets = prepareFramePackets(context, "render-target-view");
		const scope = this._resources.createFrameScope();
		let retainedForSubmit = false;
		try {
			const frameResources = scope.prepare(context, {
				sceneTargetMode: "color",
				framePackets: packets,
				temporalStateMode: "disabled",
			});
			await this._resources.buildClusteredLighting(encoder, frameResources);
			const depth = target.depth?.texture ?? this._getScratchDepth(target);
			const drewEnvironment = await this._recordEnvironment(
				encoder,
				context,
				target.color[0].texture,
				depth,
				frameResources,
			);
			await this._recordScene(
				encoder,
				context,
				target.color[0].texture,
				depth,
				drewEnvironment,
				packets,
				frameResources,
			);
			if (job.descriptor.content?.particles !== false) {
				await this._particleRenderer.renderParticles(
					encoder,
					context,
					{
						label: "WebGPURenderTargetViewParticles",
						sampleCount: 1,
						colorAttachments: [{
							view: target.color[0].texture,
							loadOp: "load",
							storeOp: "store",
						}],
						depth,
					},
					frameResources,
					"color",
					{ pipelineMode: "legacy" },
				);
			}
			this._pendingScopes.push(scope);
			retainedForSubmit = true;
		} finally {
			if (!retainedForSubmit) scope.destroy();
		}
	}

	/** Releases frame scopes only after their recorded commands were submitted. */
	public releaseSubmittedScopes(): void {
		this._destroyPendingScopes();
	}

	/** Discards retained scopes when the owning frame transaction aborts. */
	public discardPendingScopes(): void {
		this._destroyPendingScopes();
	}

	public destroy(): void {
		this._destroyPendingScopes();
		for (const depth of this._scratchDepth.values()) depth.destroy();
		this._scratchDepth.clear();
	}

	public releaseTarget(targetId: string): void {
		for (const [key, depth] of this._scratchDepth) {
			if (!key.startsWith(`${targetId}:`)) continue;
			depth.destroy();
			this._scratchDepth.delete(key);
		}
	}

	private _createContext(
		base: FrameContext,
		job: PreparedRenderTargetJob,
		target: CustomRenderTargetExecutionTarget,
	): FrameContext {
		const scene = job.scene!;
		const includeShadows = job.descriptor.kind === "scene-view" &&
			job.descriptor.content?.shadows !== "disabled";
		return {
			...base,
			presentationAlphaMode: "opaque",
			viewCamera: scene.camera,
			attachments: { width: target.width, height: target.height },
			features: {
				...base.features,
				enableReflection: false,
				enableShadows: includeShadows && base.features.enableShadows,
				warnings: base.features.warnings.slice(),
			},
			postProcess: base.postProcess.withPassDisabled("ssr"),
			renderTargets: new RenderTargetRegistrySnapshot(),
			renderTargetJobs: undefined,
			shadowPlan: scene.shadowPlan,
			scene,
			sceneState: scene,
			view: scene,
			incremental: {
				enabled: false,
				forceFullFrame: true,
				dirtyRects: [{ x: 0, y: 0, width: target.width, height: target.height }],
				dirtyTileSize: Math.max(target.width, target.height),
				dirtyTileColumns: 1,
				dirtyTileRows: 1,
				dirtyTiles: [0],
				dirtyAreaRatio: 1,
				firstPass: null,
				postProcessStartPass: null,
				reasonMask: 0,
				temporalHistoryReset: true,
			},
			transient: createRenderViewTransient(base.transient, scene.camera),
		};
	}

	private async _recordEnvironment(
		encoder: ICommandEncoder,
		context: FrameContext,
		color: IRenderTexture,
		depth: IRenderTexture,
		frameResources: WebGPUPreparedFrameResources,
	): Promise<boolean> {
		if (
			!context.features.enableEnvironment ||
			!context.scene.environment.backgroundEnabled
		) return false;
		const environment = await this._resources.getEnvironmentResources(
			frameResources,
			"color",
			{ sampleCount: 1 },
		);
		if (!environment) return false;
		encoder.beginRenderPass({
			label: "WebGPURenderTargetViewEnvironment",
			colorAttachments: [{
				view: color,
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
				loadOp: "clear",
				storeOp: "store",
			}],
			depthStencilAttachment: {
				view: depth,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});
		encoder.setPipeline(environment.pipeline);
		encoder.setBindingGroup(0, environment.frameBinding);
		encoder.draw(3);
		encoder.endRenderPass();
		return true;
	}

	private async _recordScene(
		encoder: ICommandEncoder,
		context: FrameContext,
		color: IRenderTexture,
		depth: IRenderTexture,
		drewEnvironment: boolean,
		packets: PreparedFramePacketSet,
		frameResources: WebGPUPreparedFrameResources,
	): Promise<void> {
		encoder.beginRenderPass({
			label: "WebGPURenderTargetViewScene",
			colorAttachments: [{
				view: color,
				clearValue: { r: 0, g: 0, b: 0, a: 1 },
				loadOp: drewEnvironment ? "load" : "clear",
				storeOp: "store",
			}],
			depthStencilAttachment: {
				view: depth,
				depthClearValue: 1,
				depthLoadOp: drewEnvironment ? "load" : "clear",
				depthStoreOp: "store",
			},
		});
		await submitWebGPUDraws({
			encoder,
			resources: this._resources,
			frameResources,
			packets: packets.all.slice(),
			resolveDrawOptions: () => ({
				sceneTargetMode: "color",
				sampleCount: 1,
			}),
		});
		encoder.endRenderPass();
	}

	private _getScratchDepth(target: CustomRenderTargetExecutionTarget): IRenderTexture {
		const key = `${target.id}:${target.width}x${target.height}`;
		let depth = this._scratchDepth.get(key);
		if (!depth) {
			depth = this._host.createTexture({
				width: target.width,
				height: target.height,
				format: TextureFormat.Depth32Float,
				usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
				label: `WebGPURenderTargetViewDepth_${target.id}`,
			});
			this._scratchDepth.set(key, depth);
		}
		return depth;
	}

	private _destroyPendingScopes(): void {
		for (const scope of this._pendingScopes.splice(0)) scope.destroy();
	}
}
