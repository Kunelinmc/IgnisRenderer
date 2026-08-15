import { WebGPUFrameOrchestrator } from "../../src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts";
import { createWebGPUFrameRuntimeComposition } from "../../src/backends/webgpu/rendergraph/WebGPUFrameRuntimeComposition.ts";
import { WebGPUPostProcessExecutor } from "../../src/backends/webgpu/WebGPUPostProcessExecutor.ts";
import { Logger } from "../../src/foundation/Logger.ts";
import { Camera } from "../../src/cameras/Camera.ts";
import { Material } from "../../src/materials/Material.ts";
import { PBRMaterial } from "../../src/materials/PBRMaterial.ts";
import { Matrix4 } from "../../src/maths/Matrix4.ts";
import { BackendPostProcessRuntime } from "../../src/postprocess/BackendPostProcessRuntime.ts";
import { FramePacketContributorRegistry } from "../../src/pipeline/FramePacketContributorRegistry.ts";
import { WebGPUParticleMeshPacketContributor } from "../../src/backends/webgpu/WebGPUParticleMeshPacketContributor.ts";
import { PARTICLE_MESH_TRANSIENT_BATCHES_KEY } from "../../src/pipeline/types.ts";
import { EMPTY_SHADOW_FRAME_PLAN } from "../../src/lights/shadows/ShadowFramePlan.ts";

import { FakeWebGPUBackend as FakeBackend } from "./fakes.mjs";
import { createResolvedPostProcess } from "./postprocess.mjs";

class WebGPUFrameExecutor extends WebGPUFrameOrchestrator {
	constructor(host, resources, msaa, options, particleRenderer = resources) {
		const framePackets = new FramePacketContributorRegistry();
		framePackets.register(new WebGPUParticleMeshPacketContributor());
		const sampleCounts = msaa ?? createMSAAContext(1);
		const frameServices = new Proxy(resources, {
			get(target, property, receiver) {
				if (property === "getParticleBillboardRenderer") {
					return () => particleRenderer;
				}
				return Reflect.get(target, property, receiver);
			},
		});
		const frameRuntime = createWebGPUFrameRuntimeComposition({
			host,
			frameServices,
			framePackets,
			sampleCountResolver: sampleCounts,
			warnOnce: (code, message, cause) =>
				Logger.warn(`[${code}] ${message}${cause ? ` ${String(cause)}` : ""}`),
		});
		const diagnostics = createFrameDiagnosticsCollector(frameRuntime);
		super(
			host,
			resources.createFrameScope(),
			framePackets,
			sampleCounts,
			sampleCounts.sampleCount,
			frameRuntime.modules,
			{
				enableEarlyZPrepass: options?.enableEarlyZPrepass ?? host.enableEarlyZPrepass,
				enableDeferredLighting:
					options?.enableDeferredLighting ?? host.enableDeferredLighting,
				frameGraphValidationMode:
					options?.frameGraphValidationMode ?? host.frameGraphValidationMode,
				diagnosticsObserver: diagnostics,
			},
		);
		this.frameRuntime = frameRuntime;
		this.diagnostics = diagnostics;
	}

	abortFrame(error) {
		this.abortRecording(error);
		this.abortFrameState(error);
	}
}

function createFrameDiagnosticsCollector(frameRuntime) {
	const state = {
		active: false,
		targetManager: null,
		compiledGraph: null,
		compiledStages: [],
		lastPlannedNodeIds: [],
		lastExecutedNodeIds: [],
		commit: null,
		transitions: [],
	};
	return {
		state,
		onSessionTransition({ previous, next }) {
			state.transitions.push([previous, next]);
			if (next === "preparing" || next === "skipped") {
				state.compiledGraph = null;
				state.compiledStages = [];
				state.lastPlannedNodeIds = [];
				state.lastExecutedNodeIds = [];
				state.commit = null;
			}
			state.active = next !== null;
		},
		onTargetsConfigured(targets) {
			state.targetManager = targets;
		},
		onGraphCompiled(graph, stages) {
			if (graph) {
				state.compiledGraph = graph.graph;
				state.compiledStages = [...stages];
			} else {
				state.compiledStages.push(...stages);
			}
			state.lastPlannedNodeIds = state.compiledStages.flatMap(
				(stage) => stage.nodes.map((node) => node.id),
			);
		},
		onNodeExecuted(nodeId) {
			state.lastExecutedNodeIds.push(nodeId);
		},
		onCommitSettled(commit) {
			state.commit = commit;
		},
		getDebugState() {
			const target = state.targetManager ?? {
				width: 0,
				height: 0,
				sampleCount: 1,
				texturePoolOwnerCount: 0,
				sceneTargetMode: "single",
				deferredGBufferLayout: "extended",
				needsOITTargets: false,
				frameTargets: null,
				msaaTargets: null,
			};
			return {
				...state,
				frameTargets: target.frameTargets,
				msaaTargets: target.msaaTargets,
				texturePoolOwnerCount: target.texturePoolOwnerCount,
				sceneTargetMode: target.sceneTargetMode,
				oitActive: state.active && target.needsOITTargets,
				targetManager: target,
				postProcess: frameRuntime.postProcess.getDebugState(),
			};
		},
	};
}

function createPreparedFrameResources(options = {}) {
	return {
		scopeKey: options.scopeKey ?? "main",
		sceneTargetMode: options.sceneTargetMode ?? "mrt",
		frameBinding: { id: "frame-binding" },
		decalFrameBinding: { id: "decal-frame-binding" },
		environmentBinding: { id: "environment-binding" },
		clusteredSceneBinding: { id: "clustered-binding" },
		lightingState: {},
		featureData: { get: () => undefined },
		featureState: {},
		environmentState: {},
		jointMatrixMap: null,
		morphWeightMap: null,
	};
}

function createFrameScopeAdapter(resources, state = {}) {
	return {
		prepare: (context, options) => {
			state.lastPrepared = resources.prepareFrame(context, options);
			return state.lastPrepared;
		},
		updateParticleShadowVolumes(context) {
			state.particleShadowVolumeUpdates ??= [];
			state.particleShadowVolumeUpdates.push(context);
		},
		destroy() {},
		_state: state,
	};
}

function createResourcesStub() {
	const state = {};
	return {
		createFrameScope() { return createFrameScopeAdapter(this, state); },
		sceneFrameLayout: {},
		prepareFrame(_context, options = {}) {
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {},
		_state: state,
	};
}

function createModeTrackingResourcesStub() {
	const state = {
		mode: "single",
		modeTransitions: [],
		environmentModeAtRequest: null,
		drawModeAtRequest: null,
		drawPipelineModeAtRequest: null,
	};
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		sceneFrameLayout: {},
		prepareFrame(_context, options = {}) {
			state.mode = options.sceneTargetMode ?? state.mode;
			state.modeTransitions.push(state.mode);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources(_frameResources, sceneTargetMode) {
			state.environmentModeAtRequest = sceneTargetMode ?? state.mode;
			return {
				pipeline: {},
				frameBinding: {},
			};
		},
		async getDrawResources(_packet, _frameResources, options = {}) {
			state.drawModeAtRequest = options.sceneTargetMode ?? state.mode;
			state.drawPipelineModeAtRequest = options.drawMode ?? "default";
			return null;
		},
		async renderParticles() {},
		_state: state,
	};
}

function createFrameContext(width, height) {
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		viewCamera: {},
		attachments: { width, height },
		features: {
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			enableOIT: false,
			enableClusteredLighting: false,
			warnings: [],
			clusteredLightingOptions: {},
		},
		postProcess: createResolvedPostProcess({
			ssao: { enabled: true },
			taa: { enabled: true },
		}, "webgpu"),
		shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
		scene: {
			shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: {},
		transient: new Map(),
	};
}

function createOITBackend({ sampleCount = 1 } = {}) {
	const backend = new FakeBackend();
	backend.msaaContext = createMSAAContext(sampleCount);
	return backend;
}

function createMSAAContext(initialSampleCount = 1) {
	let sampleCount = initialSampleCount;
	return {
		createFrameScope() { return createFrameScopeAdapter(this); },
		get sampleCount() {
			return sampleCount;
		},
		resolveSupportedSampleCount(requested) {
			return Math.max(1, Math.floor(requested));
		},
		resolveDomainSampleCount(domain, requested, formats) {
			return {
				domain,
				requestedSampleCount: Math.max(1, Math.floor(requested)),
				sampleCount,
				signature: `${domain}|${formats.slice().sort().join(",")}`,
				runtimeFallbackActive: sampleCount === 1 && initialSampleCount > 1,
			};
		},
		fallbackToSingleSample() {
			if (sampleCount === 1) {
				return false;
			}
			sampleCount = 1;
			return true;
		},
	};
}

function findEncoderCallIndex(backend, predicate) {
	const encoder = backend.commandEncoders[0];
	if (!encoder) {
		return -1;
	}
	return encoder.calls.findIndex(predicate);
}

function getFrameGraphDebugState(executor) {
	return executor.diagnostics.getDebugState();
}

function getFrameTargets(executor) {
	return getFrameGraphDebugState(executor).frameTargets;
}

function getMSAATargets(executor) {
	return getFrameGraphDebugState(executor).msaaTargets;
}

function createOITSequencingResourcesStub() {
	const state = {
		events: [],
		drawOptions: [],
	};
	const drawResource = {
		pipeline: {},
		frameBinding: {},
		modelBinding: {},
		clusteredBinding: {},
		vertexBuffer: {},
		indexBuffer: {},
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() { return createFrameScopeAdapter(this); },
		prepareFrame(_context, options = {}) {
			state.events.push(`prepare:${options.sceneTargetMode ?? "default"}`);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {
			state.events.push("clustered:build");
		},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.drawOptions.push({
				packetId: packet.id,
				sceneTargetMode: options.sceneTargetMode ?? null,
				transparentPipelineMode: options.transparentPipelineMode ?? "default",
				drawMode: options.drawMode ?? "default",
			});
			state.events.push(
				`draw:${packet.id}:${options.transparentPipelineMode ?? "default"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles(
			encoder,
			_context,
			targets,
			_frameResources,
			_mode,
			options = {}
		) {
			const blendModes = options.includeBlendModes ?? [];
			state.events.push(
				`particles:${targets.label}:${options.pipelineMode ?? "legacy"}:${blendModes.join(",")}`
			);
			encoder.beginRenderPass({
				label: targets.label,
				colorAttachments: targets.colorAttachments,
				depthStencilAttachment: {
					view: targets.depth,
					depthLoadOp: "load",
					depthStoreOp: "store",
				},
			});
			encoder.endRenderPass();
			return options.pipelineMode === "oit" ? 1 : 1;
		},
		_state: state,
	};
}

function createDeferredLightingResourcesStub() {
	const state = {
		deferredUnusedBinding: { id: "deferred-unused-binding" },
		events: [],
	};
	const drawResource = {
		pipeline: { id: "gbuffer-pipeline" },
		frameBinding: { id: "frame-binding" },
		modelBinding: { id: "model-binding" },
		clusteredBinding: { id: "clustered-binding" },
		vertexBuffer: { id: "vertex-buffer" },
		indexBuffer: { id: "index-buffer" },
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() { return createFrameScopeAdapter(this); },
		prepareFrame(_context, options = {}) {
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting() {},
		renderShadows() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "none"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
		getGBufferWriteLayout() {
			return { id: "gbuffer-write-layout" };
		},
		getGBufferReadLayout() {
			return { id: "gbuffer-read-layout" };
		},
		async getDeferredLightingPipeline() {
			return { id: "deferred-lighting-pipeline" };
		},
		getDeferredUnusedBinding() {
			return state.deferredUnusedBinding;
		},
		_state: state,
	};
}

function createPlanarReflectionResourcesStub() {
	const state = {
		events: [],
		drawOptions: [],
		environmentOptions: [],
		prepareContexts: [],
		throwOnClusteredBuild: false,
	};
	const drawResource = {
		pipeline: { id: "draw-pipeline" },
		frameBinding: { id: "frame-binding" },
		modelBinding: { id: "model-binding" },
		clusteredBinding: { id: "clustered-binding" },
		vertexBuffer: { id: "vertex-buffer" },
		indexBuffer: { id: "index-buffer" },
		indexCount: 3,
	};
	return {
		sceneFrameLayout: {},
		createFrameScope() {
			return {
				prepare: (context, options) => {
					state.lastPrepared = this.prepareFrame(context, options);
					state.mainPrepared ??= state.lastPrepared;
					return state.lastPrepared;
				},
				updateParticleShadowVolumes() {},
				destroy() { state.events.push("scope:destroy"); },
			};
		},
		prepareFrame(context, options = {}) {
			state.prepareContexts.push(context);
			state.events.push(
				`prepare:reflection:${context.features.enableReflection}:ssr:${context.postProcess.isEnabled("ssr")}:opaque:${context.scene.opaquePackets.map((packet) => packet.id).join(",")}`
			);
			return createPreparedFrameResources(options);
		},
		async buildClusteredLighting(_encoder, frameResources) {
			state.events.push("clustered:build");
			if (frameResources) {
				state.events.push(`clustered-scope:${frameResources.scopeKey}`);
			}
			if (state.throwOnClusteredBuild) {
				throw new Error("simulated planar capture failure");
			}
		},
		renderShadows() {},
		async getEnvironmentResources(_frameResources, sceneTargetMode, options = {}) {
			state.environmentOptions.push({
				sceneTargetMode: sceneTargetMode ?? null,
				sampleCount: options.sampleCount ?? null,
			});
			return null;
		},
		async getDrawResources(packet, _frameResources, options = {}) {
			state.drawOptions.push({
				packetId: packet.id,
				sceneTargetMode: options.sceneTargetMode ?? null,
				drawMode: options.drawMode ?? "default",
				sampleCount: options.sampleCount ?? null,
			});
			state.events.push(
				`draw:${packet.id}:${options.sceneTargetMode ?? "default"}:${options.drawMode ?? "default"}`
			);
			return [drawResource];
		},
		async renderParticles() {},
		getPlanarReflectionLayout() {
			return { id: "planar-reflection-layout" };
		},
		releaseScope(scopeKey) {
			state.events.push(`release:${scopeKey}`);
		},
		_state: state,
	};
}

const ISOLATED_GLOBAL_KEYS = ["GPUBufferUsage", "GPUMapMode", "GPUShaderStage", "GPUTextureUsage"];

export function initializeIsolatedWebGPUTestState() {
	const previousGlobals = new Map(
		ISOLATED_GLOBAL_KEYS.map((key) => [
			key,
			Object.prototype.hasOwnProperty.call(globalThis, key)
				? globalThis[key]
				: undefined,
		]),
	);
	const previousLoggerLevel = Logger.getLevel();

	if (globalThis.GPUBufferUsage === undefined) {
		globalThis.GPUBufferUsage = {
			COPY_SRC: 1,
			COPY_DST: 2,
			MAP_READ: 4,
			MAP_WRITE: 8,
			RENDER_ATTACHMENT: 16,
		};
	}
	if (globalThis.GPUMapMode === undefined) {
		globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
	}
	if (globalThis.GPUShaderStage === undefined) {
		globalThis.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
	}
	if (globalThis.GPUTextureUsage === undefined) {
		globalThis.GPUTextureUsage = {
			COPY_SRC: 1,
			COPY_DST: 2,
			TEXTURE_BINDING: 4,
			STORAGE_BINDING: 8,
			RENDER_ATTACHMENT: 16,
		};
	}
	Logger.configure({ level: "silent", resetOnceKeys: true });

	return () => {
		for (const [key, value] of previousGlobals) {
			if (value === undefined) {
				delete globalThis[key];
			} else {
				globalThis[key] = value;
			}
		}
		Logger.reset();
		if (previousLoggerLevel !== "info") {
			Logger.setLevel(previousLoggerLevel);
		}
	};
}

function createPlanarPacket(id, material, y) {
	const worldMatrix = Matrix4.identity();
	return {
		id,
		meshInstance: { id: `${id}-mesh`, worldMatrix, mesh: { primitives: [] } },
		mesh: { primitives: [], boundingSphere: { center: { x: 0, y, z: 0 }, radius: 1 } },
		primitive: {
			id: `${id}-primitive`,
			material,
			geometry: {},
			boundingSphere: { center: { x: 0, y, z: 0 }, radius: 1 },
		},
		material,
		geometry: {},
		worldMatrix,
		normalMatrix: worldMatrix,
		worldBounds: { center: { x: 0, y, z: 0 }, radius: 1 },
		sortDepth: 1,
		pipelineKey: id,
		passFlags: 0,
	};
}

export {
	BackendPostProcessRuntime,
	Camera,
	FakeBackend,
	Logger,
	Material,
	Matrix4,
	PBRMaterial,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	WebGPUFrameExecutor,
	WebGPUPostProcessExecutor,
	createDeferredLightingResourcesStub,
	createFrameContext,
	createFrameScopeAdapter,
	createMSAAContext,
	createModeTrackingResourcesStub,
	createOITBackend,
	createOITSequencingResourcesStub,
	createPlanarPacket,
	createPlanarReflectionResourcesStub,
	createPreparedFrameResources,
	createResolvedPostProcess,
	createResourcesStub,
	findEncoderCallIndex,
	getFrameGraphDebugState,
	getFrameTargets,
	getMSAATargets,
};
