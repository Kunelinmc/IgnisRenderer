import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { Logger } from "../../foundation/Logger";
import type { DrawPacket } from "../../pipeline/types";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import type { WebGPUMaterialBindingCache } from "./WebGPUMaterialBindingCache";
import {
	readWebGPUShaderRuntimeView,
	type WebGPUMaterialPipelinePurpose,
	type WebGPUMaterialPipelineResolver,
} from "./WebGPUMaterialPipelineResolver";
import type { WebGPUMaterialSnapshotCache } from "./WebGPUMaterialSnapshotCache";
import type {
	WebGPUDrawPipelineProvider,
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
} from "./WebGPUResourceContracts";
import {
	resolveWebGPUScenePassDescriptor,
	type WebGPUScenePassDescriptor,
} from "./WebGPUScenePassDescriptors";
import type { WebGPUStaticMeshBatcher } from "./WebGPUStaticMeshBatcher";

/** @internal Shared draw preparation independent of feature pipeline ownership. */
export class WebGPUDrawResourceAssembler {
	public constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _geometry: WebGPUGeometryRegistry,
		private readonly _animation: WebGPUAnimationPayloadPool,
		private readonly _snapshots: WebGPUMaterialSnapshotCache,
		private readonly _bindings: WebGPUMaterialBindingCache,
		private readonly _batcher: WebGPUStaticMeshBatcher,
		private readonly _materialPipelines: WebGPUMaterialPipelineResolver,
	) {}

	public async getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
		pipelines: WebGPUDrawPipelineProvider,
	): Promise<WebGPUDrawResources[] | null> {
		const transparentPipelineMode = options.transparentPipelineMode ?? "default";
		const sceneTargetMode = options.sceneTargetMode ?? frameResources.sceneTargetMode;
		const drawMode = options.drawMode ?? "default";
		const descriptor = resolveWebGPUScenePassDescriptor(
			sceneTargetMode,
			transparentPipelineMode,
			drawMode,
			options.deferredGBufferLayout,
		);
		const geometry = this._geometry.getGeometry(packet.primitive);
		const animationPayload = this._animation.getScenePayload(
			packet,
			geometry,
			frameResources.jointMatrixMap,
			frameResources.morphWeightMap,
		);
		const results: WebGPUDrawResources[] = [];
		const solidSnapshot = await this._snapshots.resolve(packet.material, false);
		for (const warning of solidSnapshot.data.warnings) {
			Logger.warn(`[${warning.key}] ${warning.message}`, {
				scope: "WebGPUDrawResourceAssembler",
				onceKey: warning.key,
			});
		}
		const solidState = this._resolveMaterialState(
			packet,
			solidSnapshot.data,
			false,
			descriptor,
		);
		if (!solidState) return null;
		const solidPipeline = await pipelines.resolvePipeline({
			materialState: solidState,
			pass: descriptor,
			topology: geometry.topology,
			geometryLayout: geometry,
			sampleCount: options.sampleCount,
		});
		if (!solidPipeline) return null;
		const staticDraw = this._batcher.getDrawState(
			packet,
			solidPipeline,
			geometry,
			solidSnapshot,
			drawMode,
		);
		const solidModelBinding = staticDraw?.modelBinding ?? this._bindings.getBinding(
			packet,
			solidPipeline,
			solidSnapshot.data,
			solidSnapshot.textures,
			solidSnapshot.samplers,
			animationPayload,
			geometry.morphPositionBuffer,
			geometry.morphNormalBuffer,
		);
		results.push({
			pipeline: solidPipeline,
			frameBinding: frameResources.frameBinding,
			modelBinding: solidModelBinding,
			clusteredBinding: frameResources.clusteredSceneBinding,
			vertexBindings: geometry.vertexBindings,
			indexBuffer: geometry.indexBuffer,
			indexFormat: geometry.indexFormat,
			indexCount: geometry.indexCount,
			staticBatchKey: staticDraw?.batchKey,
			firstInstance: staticDraw?.firstInstance,
			resolvedInputs: {
				materialData: solidSnapshot.data,
				textures: solidSnapshot.textures,
				samplers: solidSnapshot.samplers,
				geometry,
			},
		});

		if (
			drawMode === "early-z-prepass" ||
			!packet.material.wireframe ||
			geometry.topology !== DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
		) return results;

		const wireGeometry = this._geometry.getWireframeGeometry(packet.primitive);
		const wireSnapshot = await this._snapshots.resolve(packet.material, true);
		const wireState = this._resolveMaterialState(
			packet,
			wireSnapshot.data,
			true,
			descriptor,
		);
		if (!wireState) return results;
		const wirePipeline = await pipelines.resolvePipeline({
			materialState: wireState,
			pass: descriptor,
			topology: geometry.topology,
			geometryLayout: wireGeometry,
			sampleCount: options.sampleCount,
		});
		if (!wirePipeline) return results;
		const wireModelBinding = this._bindings.getBinding(
			packet,
			wirePipeline,
			wireSnapshot.data,
			wireSnapshot.textures,
			wireSnapshot.samplers,
			animationPayload,
			wireGeometry.morphPositionBuffer,
			wireGeometry.morphNormalBuffer,
		);
		results.push({
			pipeline: wirePipeline,
			frameBinding: frameResources.frameBinding,
			modelBinding: wireModelBinding,
			clusteredBinding: frameResources.clusteredSceneBinding,
			vertexBindings: wireGeometry.vertexBindings,
			indexBuffer: wireGeometry.wireframeIndexBuffer!,
			indexFormat: wireGeometry.wireframeIndexFormat,
			indexCount: wireGeometry.wireframeIndexCount,
			resolvedInputs: {
				materialData: wireSnapshot.data,
				textures: wireSnapshot.textures,
				samplers: wireSnapshot.samplers,
				geometry: wireGeometry,
			},
		});
		return results;
	}

	private _resolveMaterialState(
		packet: DrawPacket,
		materialData: WebGPUDrawResources["resolvedInputs"]["materialData"],
		wireframe: boolean,
		descriptor: WebGPUScenePassDescriptor,
	) {
		const purpose: WebGPUMaterialPipelinePurpose =
			descriptor.drawMode === "early-z-prepass" ? "early-z" : "scene";
		try {
			return this._materialPipelines.resolve(
				packet.material,
				materialData,
				wireframe,
				resolveShaderTargetMode(descriptor.sceneTargetMode),
				purpose,
				readWebGPUShaderRuntimeView(this._backend),
			);
		} catch (error) {
			if (purpose !== "early-z") throw error;
			const shaderId = "shaderId" in packet.material
				? String((packet.material as { shaderId: number }).shaderId)
				: "unknown";
			const key = `webgpu-earlyz-shader-material-skip-${shaderId}`;
			Logger.warn(
				`[${key}] ShaderMaterial ${packet.material.name} early-z pre-pass is skipped: ${String(error)}`,
				{ scope: "WebGPUDrawResourceAssembler", onceKey: key },
			);
			return null;
		}
	}
}

function resolveShaderTargetMode(
	mode: WebGPUScenePassDescriptor["sceneTargetMode"],
): "single" | "mrt" | "deferred" {
	if (mode === "gbuffer") return "deferred";
	return mode === "color" ? "single" : mode;
}
