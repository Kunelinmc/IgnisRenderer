import type { DrawPacket } from "../../pipeline/types";
import type { WarmupPhaseCounters } from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import type { WebGPUDrawResourceAssembler } from "./WebGPUDrawResourceAssembler";
import type {
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
} from "./WebGPUResourceContracts";
import type { WebGPUScenePipelineResources } from "./WebGPUScenePipelineResources";
import type { WebGPUSceneTargetMode } from "./WebGPUScenePassDescriptors";
import type { WebGPUFeatureWarmupContributor } from "./WebGPUFeatureWarmup";

export interface WebGPUSceneWarmupRequest {
	readonly phase?: string;
	readonly packets: readonly DrawPacket[];
	readonly frameResources: WebGPUPreparedFrameResources;
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly sampleCount: number;
	readonly enableEarlyZPrepass: boolean;
	readonly deferredGBufferLayout?: "base" | "extended";
	yieldIfNeeded(): Promise<void>;
}

/** @internal Scene draw preparation paired with scene-owned pipelines. */
export class WebGPUSceneDrawResources
	implements WebGPUFeatureWarmupContributor<WebGPUSceneWarmupRequest> {
	public constructor(
		private readonly _assembler: WebGPUDrawResourceAssembler,
		private readonly _pipelines: WebGPUScenePipelineResources,
	) {}

	public getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
	): Promise<WebGPUDrawResources[] | null> {
		return this._assembler.getDrawResources(
			packet,
			frameResources,
			options,
			this._pipelines,
		);
	}

	public async warmup(
		request: WebGPUSceneWarmupRequest,
	): Promise<WarmupPhaseCounters> {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors = [];
		for (const packet of request.packets) {
			const compileVariant = async (
				options: WebGPUDrawResourceOptions,
			): Promise<boolean> => {
				total++;
				try {
					const draws = await this.getDrawResources(
						packet,
						request.frameResources,
						options,
					);
					if (draws && draws.length > 0) {
						compiled++;
						return true;
					}
					skipped++;
					return false;
				} catch (error) {
					failed++;
					errors.push(toShaderCompileError(
						error,
						"webgpu",
						`WebGPUSceneWarmup:${packet.submission.id}:${options.drawMode ?? "default"}`,
					));
					return false;
				} finally {
					await request.yieldIfNeeded();
				}
			};
			await compileVariant({
				sceneTargetMode: request.sceneTargetMode,
				deferredGBufferLayout: request.deferredGBufferLayout,
				sampleCount: request.sampleCount,
			});
			if (request.enableEarlyZPrepass) {
				const earlyZCompiled = await compileVariant({
					sceneTargetMode: request.sceneTargetMode,
					drawMode: "early-z-prepass",
					deferredGBufferLayout: request.deferredGBufferLayout,
					sampleCount: request.sampleCount,
				});
				if (earlyZCompiled) await compileVariant({
					sceneTargetMode: request.sceneTargetMode,
					drawMode: "early-z-color",
					deferredGBufferLayout: request.deferredGBufferLayout,
					sampleCount: request.sampleCount,
				});
			}
		}
		return {
			phase: request.phase ?? "webgpu-scene",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
	}

	public onShaderRuntimeChanged(): void {
		this._pipelines.invalidateShaderRuntimeCaches();
	}

	public destroy(): void {
		this._pipelines.destroy();
	}
}
