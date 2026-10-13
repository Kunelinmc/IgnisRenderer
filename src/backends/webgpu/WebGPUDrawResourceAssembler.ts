import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { Logger } from "../../foundation/Logger";
import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { DrawPacket } from "../../pipeline/types";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryHandle, WebGPUGeometryRegistry } from "./WebGPUGeometryRegistry";
import type { WebGPUDepthMaterialResources } from "./WebGPUDepthMaterialResources";
import type { WebGPUMaterialUniformData } from "./types";
import {
	readWebGPUShaderRuntimeView,
	type WebGPUMaterialPipelinePurpose,
	type WebGPUMaterialPipelineResolver,
	type WebGPUMaterialPipelineState,
	type WebGPUShaderRuntimeView,
} from "./WebGPUMaterialPipelineResolver";
import type { WebGPUDrawResourceCache } from "./WebGPUDrawResourceCache";
import type {
	WebGPUMaterialResource,
	WebGPUDrawPipelineProvider,
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUDrawResourceResult,
	WebGPUPreparedFrameResources,
	WebGPUResolvedDrawInputs,
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
		snapshot: WebGPUMaterialResource;
		state: WebGPUMaterialPipelineState | null;
	}>;
	readonly geometries: WeakMap<WebGPUGeometryHandle, SharedDrawPreparation>;
}

interface SharedDrawPreparation {
	readonly epoch: number;
	ready: PreparedSceneDraw | null;
	pending: Promise<PreparedSceneDraw | null>;
}

type PreparedSceneDraw = WebGPUDrawResources & { resolvedInputs: WebGPUResolvedDrawInputs };

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
		private readonly _resources: WebGPUDrawResourceCache,
		private readonly _batcher: WebGPUStaticMeshBatcher,
		private readonly _materialPipelines: WebGPUMaterialPipelineResolver,
		private readonly _depthMaterials: WebGPUDepthMaterialResources,
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
		if (drawMode === "early-z-prepass" &&
			!(packet.submission.material.effective instanceof ShaderMaterial)) {
			return this._resolveBuiltinDepthDraw(packet, frameResources, options, pipelines);
		}
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
			? material._getRevisionInternal() : this._resources.refreshMaterialRevision(material);
		if (shared) shared.validatedEpoch = this._epoch;
		if (!shared || shared.revision !== materialRevision) {
			const descriptor = resolveWebGPUScenePassDescriptor(
				sceneTargetMode, transparentMode, drawMode, options.deferredGBufferLayout,
			);
			const snapshot = this._resources.resolveMaterial(material, false);
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
			group.ready.resolvedInputs.materialData, group.ready.bindings[1].group,
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
	): Promise<PreparedSceneDraw | null> {
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
			pipeline,
			bindings: [
				{ slot: 0, group: frame.frameBinding },
				{ slot: 1, group: draw.modelBinding },
				{ slot: 2, group: frame.clusteredSceneBinding },
			],
			vertexBindings: geometry.vertexBindings,
			indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
			indexCount: geometry.indexCount, staticBatchKey: draw.batchKey,
			resolvedInputs: {
				materialData: snapshot.data, textures: snapshot.textures,
				samplers: snapshot.samplers, geometry,
			},
		};
	}

	private async _resolveBuiltinDepthDraw(
		packet: DrawPacket,
		frame: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
		pipelines: WebGPUDrawPipelineProvider,
	): Promise<WebGPUDrawResources[] | null> {
		const material = packet.submission.material.effective;
		const runtime = readWebGPUShaderRuntimeView(this._backend);
		const state = this._materialPipelines.resolveBuiltinDepth(material, runtime);
		if (state.transparent || !state.depthWrite) return null;
		const epoch = this._epoch;
		const scopeOwner = this._frameScopes.get(frame) ?? frame;
		const revision = pipelines.preparationRevision;
		const geometry = this._geometry.getGeometry(packet.submission.geometry);
		const geometryVersion = packet.submission.geometry.version;
		const pass = resolveWebGPUScenePassDescriptor(
			options.sceneTargetMode ?? frame.sceneTargetMode,
			options.transparentPipelineMode ?? "default", "early-z-prepass",
			options.deferredGBufferLayout,
		);
		const isCurrent = () => epoch === this._epoch &&
			!this._releasedScopes.has(scopeOwner) &&
			pipelines.preparationRevision === revision &&
			packet.submission.geometry.version === geometryVersion &&
			this._geometry.isCurrentGeometry(packet.submission.geometry, geometry) &&
			sameRuntime(runtime, readWebGPUShaderRuntimeView(this._backend));
		const pipeline = await pipelines.resolvePipeline({
			materialState: state, pass, topology: geometry.topology,
			geometryLayout: geometry, sampleCount: options.sampleCount,
		});
		if (!pipeline || !isCurrent()) return null;
		const depthMaterial = await this._depthMaterials.resolve(material);
		if (!isCurrent() || !this._depthMaterials.isCurrent(material, depthMaterial)) return null;
		const currentState = this._materialPipelines.resolveBuiltinDepth(material, runtime);
		if (currentState.cullMode !== state.cullMode || currentState.alphaMode !== state.alphaMode ||
			currentState.transparent || !currentState.depthWrite) return null;
		const animation = this._animation.getScenePayload(
			packet, geometry, frame.jointMatrixMap, frame.morphWeightMap,
		);
		if (!animation) return null;
		const staticDraw = this._batcher.getDepthDrawState(
			packet, pipeline, geometry, depthMaterial.binding,
		);
		const modelBinding = this._resources.getDepthBinding(
			packet, animation, geometry.morphPositionBuffer, staticDraw ?? undefined,
		);
		return [{
			pipeline,
			bindings: [
				{ slot: 0, group: frame.depthFrameBinding },
				{ slot: 1, group: modelBinding },
				{ slot: 2, group: depthMaterial.binding },
			],
			vertexBindings: geometry.vertexBindings,
			indexBuffer: geometry.indexBuffer, indexFormat: geometry.indexFormat,
			indexCount: geometry.indexCount, firstInstance: staticDraw?.firstInstance,
			staticBatchKey: staticDraw?.batchKey,
		}];
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
		const solidSnapshot = await this._resources.resolveMaterial(
			packet.submission.material.effective, false,
		);
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
		const solidModelBinding = staticDraw?.modelBinding ?? this._resources.getMaterialBinding(
			packet,
			solidSnapshot,
			animationPayload,
			geometry.morphPositionBuffer,
			geometry.morphNormalBuffer,
		);
		results.push({
			pipeline: solidPipeline,
			bindings: [
				{ slot: 0, group: frameResources.frameBinding },
				{ slot: 1, group: solidModelBinding },
				{ slot: 2, group: frameResources.clusteredSceneBinding },
			],
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
		const wireSnapshot = await this._resources.resolveMaterial(
			packet.submission.material.effective, true,
		);
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
		const wireModelBinding = this._resources.getMaterialBinding(
			packet,
			wireSnapshot,
			animationPayload,
			wireGeometry.morphPositionBuffer,
			wireGeometry.morphNormalBuffer,
		);
		results.push({
			pipeline: wirePipeline,
			bindings: [
				{ slot: 0, group: frameResources.frameBinding },
				{ slot: 1, group: wireModelBinding },
				{ slot: 2, group: frameResources.clusteredSceneBinding },
			],
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
		materialData: WebGPUMaterialUniformData,
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
		...ready, firstInstance,
		bindings: [
			{ slot: 0, group: frame.frameBinding },
			ready.bindings[1],
			{ slot: 2, group: frame.clusteredSceneBinding },
		],
	}];
}

function resolveShaderTargetMode(
	mode: WebGPUDrawPassDescriptor["sceneTargetMode"],
): "single" | "mrt" | "deferred" {
	if (mode === "gbuffer") return "deferred";
	return mode === "color" ? "single" : mode;
}
