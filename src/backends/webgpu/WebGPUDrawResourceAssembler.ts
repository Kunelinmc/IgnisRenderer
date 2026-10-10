import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { Logger } from "../../foundation/Logger";
import type { Material } from "../../materials/Material";
import type { DrawPacket } from "../../pipeline/types";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryHandle, WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import type { WebGPUMaterialBindingCache } from "./WebGPUMaterialBindingCache";
import {
	readWebGPUShaderRuntimeView,
	type WebGPUMaterialPipelinePurpose,
	type WebGPUMaterialPipelineResolver,
	type WebGPUMaterialPipelineState,
	type WebGPUShaderRuntimeView,
} from "./WebGPUMaterialPipelineResolver";
import type {
	WebGPUMaterialSnapshotCache,
	WebGPUResolvedMaterialSnapshot,
} from "./WebGPUMaterialSnapshotCache";
import type {
	WebGPUDrawPipelineProvider,
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUDrawResourceResult,
	WebGPUPreparedFrameResources,
} from "./WebGPUResourceContracts";
import {
	resolveWebGPUScenePassDescriptor,
	type WebGPUDrawPassDescriptor,
} from "./WebGPUScenePassDescriptors";
import { resolveWebGPUPlanarReflectionPassDescriptor } from "./WebGPUPlanarReflectionPassDescriptor";
import type { WebGPUStaticMeshBatcher } from "./WebGPUStaticMeshBatcher";

interface SharedMaterialPreparation {
	validatedEpoch: number;
	readonly revision: number;
	readonly inputs: Promise<{
		snapshot: WebGPUResolvedMaterialSnapshot;
		state: WebGPUMaterialPipelineState | null;
	}>;
	readonly geometries: WeakMap<WebGPUGeometryHandle, SharedDrawPreparation>;
}

interface SharedDrawPreparation {
	readonly epoch: number;
	ready: WebGPUDrawResources | null;
	pending: Promise<WebGPUDrawResources | null>;
}

interface PreparationScope {
	readonly providerRevision: number;
	readonly bindingRevision: number;
	readonly runtime: WebGPUShaderRuntimeView;
	readonly variants: Map<string, WeakMap<Material, SharedMaterialPreparation>>;
}

/** @internal Shared draw preparation independent of feature pipeline ownership. */
export class WebGPUDrawResourceAssembler {
	private _epoch = 0;
	private _frameScopes = new WeakMap<WebGPUPreparedFrameResources, object>();
	private readonly _releasedScopes = new WeakSet<object>();
	private _scopes = new WeakMap<
		object,
		Map<WebGPUDrawPipelineProvider, PreparationScope>
	>();
	public constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _geometry: WebGPUGeometryRegistry,
		private readonly _animation: WebGPUAnimationPayloadPool,
		private readonly _snapshots: WebGPUMaterialSnapshotCache,
		private readonly _bindings: WebGPUMaterialBindingCache,
		private readonly _batcher: WebGPUStaticMeshBatcher,
		private readonly _materialPipelines: WebGPUMaterialPipelineResolver,
	) {}

	/** @internal WebGPU scene preparation clears templates; public rendering uses Renderer. */
	public clear(): void {
		this._epoch++;
		this._scopes = new WeakMap();
	}

	/** @internal WebGPU frame orchestration retains ready templates; use Renderer.renderFrame(). */
	public beginFrame(): void {
		this._epoch++;
	}

	/** @internal WebGPU frame services register scope ownership; use Renderer.renderFrame(). */
	public bindFrameScope(frame: WebGPUPreparedFrameResources, owner: object): void {
		this._frameScopes.set(frame, owner);
	}

	/** @internal WebGPU scope teardown invalidates preparation; Renderer owns public rendering. */
	public releaseFrameScope(owner: object): void {
		this._releasedScopes.add(owner);
		this._scopes.delete(owner);
	}

	/** @internal Reuses ready static scene preparation; other draws use the legacy path. */
	public getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
		pipelines: WebGPUDrawPipelineProvider,
	): WebGPUDrawResourceResult {
		const scopeOwner = this._frameScopes.get(frameResources) ?? frameResources;
		if (this._releasedScopes.has(scopeOwner)) return null;
		const drawMode = options.drawMode ?? "default";
		const geometry = this._geometry.getGeometry(packet.submission.geometry);
		const firstInstance = this._batcher.getPreparedInstanceIndex(packet, geometry, drawMode);
		const providerRevision = pipelines.preparationRevision;
		if (firstInstance === undefined || providerRevision === undefined ||
			drawMode === "planar-reflection-composite" || drawMode === "reflection-capture") {
			return this._resolveDrawResources(packet, frameResources, options, pipelines);
		}
		const runtime = readWebGPUShaderRuntimeView(this._backend);
		let providers = this._scopes.get(scopeOwner);
		if (!providers) {
			providers = new Map();
			this._scopes.set(scopeOwner, providers);
		}
		let scope = providers.get(pipelines);
		if (!scope || scope.providerRevision !== providerRevision ||
			scope.bindingRevision !== this._batcher.bindingRevision ||
			!sameRuntime(scope.runtime, runtime)) {
			scope = {
				providerRevision, bindingRevision: this._batcher.bindingRevision, runtime,
				variants: new Map(),
			};
			providers.set(pipelines, scope);
		}
		const sceneTargetMode = options.sceneTargetMode ?? frameResources.sceneTargetMode;
		const transparentMode = options.transparentPipelineMode ?? "default";
		const key = `${sceneTargetMode}|${transparentMode}|${drawMode}|` +
			`${options.sampleCount}|${options.deferredGBufferLayout ?? "extended"}`;
		let materials = scope.variants.get(key);
		if (!materials) {
			materials = new WeakMap();
			scope.variants.set(key, materials);
		}
		const material = packet.submission.material.effective;
		let shared = materials.get(material);
		const materialRevision = shared?.validatedEpoch === this._epoch
			? material._getRevisionInternal() : this._snapshots.refreshMaterialRevision(material);
		if (shared) shared.validatedEpoch = this._epoch;
		if (!shared || shared.revision !== materialRevision) {
			const descriptor = resolveWebGPUScenePassDescriptor(
				sceneTargetMode, transparentMode, drawMode, options.deferredGBufferLayout,
			);
			const snapshot = this._snapshots.resolve(material, false);
			shared = {
				validatedEpoch: this._epoch,
				revision: material._getRevisionInternal(),
				inputs: snapshot.then((resolved) => {
					for (const warning of resolved.data.warnings) {
						Logger.warn(`[${warning.key}] ${warning.message}`, {
							scope: "WebGPUDrawResourceAssembler", onceKey: warning.key,
						});
					}
					return {
						snapshot: resolved,
						state: this._resolveMaterialState(packet, resolved.data, false, descriptor),
					};
				}),
				geometries: new WeakMap(),
			};
			materials.set(material, shared);
			const entry = shared;
			void shared.inputs.catch(() => {
				if (materials.get(material) === entry) materials.delete(material);
			});
		}
		let group = shared.geometries.get(geometry);
		if (group?.ready && !this._batcher.touchMaterialBinding(
			group.ready.resolvedInputs.materialData, group.ready.modelBinding,
		)) {
			shared.geometries.delete(geometry);
			group = undefined;
		}
		if (group && !group.ready && group.epoch !== this._epoch) {
			shared.geometries.delete(geometry);
			group = undefined;
		}
		if (group?.ready) return instantiateStaticDraw(group.ready, frameResources, firstInstance);
		const epoch = this._epoch;
		const arenaRevision = this._batcher.arenaRevision;
		const geometryVersion = packet.submission.geometry.version;
		const entry = shared;
		const currentScope = scope;
		const isCurrent = () => this._epoch === epoch &&
			!this._releasedScopes.has(scopeOwner) &&
			this._batcher.arenaRevision === arenaRevision &&
			this._batcher.bindingRevision === currentScope.bindingRevision &&
			pipelines.preparationRevision === currentScope.providerRevision &&
			material._getRevisionInternal() === entry.revision &&
			packet.submission.geometry.version === geometryVersion &&
			this._geometry.isCurrentGeometry(packet.submission.geometry, geometry) &&
			sameRuntime(runtime, readWebGPUShaderRuntimeView(this._backend));
		if (!group) {
			const descriptor = resolveWebGPUScenePassDescriptor(
				sceneTargetMode, transparentMode, drawMode, options.deferredGBufferLayout,
			);
			const pending = this._prepareStaticDraw(
				packet, frameResources, options, pipelines, geometry, descriptor, shared,
				isCurrent,
			).then((ready) => {
				if (!isCurrent()) ready = null;
				currentGroup.ready = ready;
				if (!ready && entry.geometries.get(geometry) === currentGroup) {
					entry.geometries.delete(geometry);
				}
				return ready;
			}, (error) => {
				if (entry.geometries.get(geometry) === currentGroup) {
					entry.geometries.delete(geometry);
				}
				throw error;
			});
			group = { epoch, ready: null, pending };
			const currentGroup = group;
			shared.geometries.set(geometry, group);
		}
		return group.pending.then((ready) => ready && isCurrent()
			? instantiateStaticDraw(ready, frameResources, firstInstance) : null);
	}

	private async _prepareStaticDraw(
		packet: DrawPacket,
		frame: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
		pipelines: WebGPUDrawPipelineProvider,
		geometry: WebGPUGeometryHandle,
		descriptor: WebGPUDrawPassDescriptor,
		shared: SharedMaterialPreparation,
		isCurrent: () => boolean,
	): Promise<WebGPUDrawResources | null> {
		const { snapshot, state } = await shared.inputs;
		if (!state || !isCurrent()) return null;
		const pipeline = await pipelines.resolvePipeline({
			materialState: state, pass: descriptor, topology: geometry.topology,
			geometryLayout: geometry, sampleCount: options.sampleCount,
		});
		if (!pipeline || !isCurrent()) return null;
		const draw = this._batcher.getDrawState(
			packet, pipeline, geometry, snapshot, descriptor.drawMode,
		);
		if (!draw) return null;
		return {
			pipeline, frameBinding: frame.frameBinding, modelBinding: draw.modelBinding,
			clusteredBinding: frame.clusteredSceneBinding, vertexBindings: geometry.vertexBindings,
			indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
			indexCount: geometry.indexCount, staticBatchKey: draw.batchKey,
			resolvedInputs: {
				materialData: snapshot.data, textures: snapshot.textures,
				samplers: snapshot.samplers, geometry,
			},
		};
	}

	private async _resolveDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
		pipelines: WebGPUDrawPipelineProvider,
	): Promise<WebGPUDrawResources[] | null> {
		const transparentPipelineMode = options.transparentPipelineMode ?? "default";
		const sceneTargetMode = options.sceneTargetMode ?? frameResources.sceneTargetMode;
		const drawMode = options.drawMode ?? "default";
		const descriptor = drawMode === "planar-reflection-composite"
			? resolveWebGPUPlanarReflectionPassDescriptor(sceneTargetMode)
			: resolveWebGPUScenePassDescriptor(
				sceneTargetMode,
				transparentPipelineMode,
				drawMode,
				options.deferredGBufferLayout,
			);
		const geometry = this._geometry.getGeometry(packet.submission.geometry);
		const animationPayload = this._animation.getScenePayload(
			packet,
			geometry,
			frameResources.jointMatrixMap,
			frameResources.morphWeightMap,
		);
		if (!animationPayload) return null;
		const results: WebGPUDrawResources[] = [];
		const solidSnapshot = await this._snapshots.resolve(packet.submission.material.effective, false);
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
			!packet.submission.material.effective.wireframe ||
			geometry.topology !== DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
		) return results;

		const wireGeometry = this._geometry.getWireframeGeometry(
			packet.submission.geometry,
		);
		const wireSnapshot = await this._snapshots.resolve(packet.submission.material.effective, true);
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
		descriptor: WebGPUDrawPassDescriptor,
	) {
		const purpose: WebGPUMaterialPipelinePurpose =
			descriptor.drawMode === "early-z-prepass" ? "early-z" : "scene";
		try {
			return this._materialPipelines.resolve(
				packet.submission.material.effective,
				materialData,
				wireframe,
				resolveShaderTargetMode(descriptor.sceneTargetMode),
				purpose,
				readWebGPUShaderRuntimeView(this._backend),
			);
		} catch (error) {
			if (purpose !== "early-z") throw error;
			const shaderId = "shaderId" in packet.submission.material.effective
				? String((packet.submission.material.effective as { shaderId: number }).shaderId)
				: "unknown";
			const key = `webgpu-earlyz-shader-material-skip-${shaderId}`;
			Logger.warn(
				`[${key}] ShaderMaterial ${packet.submission.material.effective.name} early-z pre-pass is skipped: ${String(error)}`,
				{ scope: "WebGPUDrawResourceAssembler", onceKey: key },
			);
			return null;
		}
	}
}

function sameRuntime(a: WebGPUShaderRuntimeView, b: WebGPUShaderRuntimeView): boolean {
	return a.revision === b.revision && a.mode === b.mode &&
		a.directiveCacheTag === b.directiveCacheTag;
}

function instantiateStaticDraw(
	ready: WebGPUDrawResources,
	frame: WebGPUPreparedFrameResources,
	firstInstance: number,
): WebGPUDrawResources[] {
	return [{
		...ready, firstInstance, frameBinding: frame.frameBinding,
		clusteredBinding: frame.clusteredSceneBinding,
	}];
}

function resolveShaderTargetMode(
	mode: WebGPUDrawPassDescriptor["sceneTargetMode"],
): "single" | "mrt" | "deferred" {
	if (mode === "gbuffer") return "deferred";
	return mode === "color" ? "single" : mode;
}
