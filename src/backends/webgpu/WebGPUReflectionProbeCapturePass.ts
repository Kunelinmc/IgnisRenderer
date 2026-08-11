import { Camera } from "../../cameras/Camera";
import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3 } from "../../maths/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	FrameContext,
	PreparedScene,
	ResolvedFeatureState,
} from "../../pipeline/types";
import type {
	ParticleRenderBatch,
	ParticleRenderItem,
} from "../../particles/ParticleRenderBatch";
import type {
	FramePacketProvider,
	PreparedFramePacketSet,
} from "../../pipeline/FramePacketContributorRegistry";
import {
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
	createTransientStore,
} from "../../pipeline/types";
import { PreparedSceneBuilder } from "../../pipeline/PreparedSceneBuilder";
import type { ResolvedPostProcessState } from "../../postprocess";
import type { IncrementalFrameContext } from "../../pipeline/incremental";
import type { ProbeCaptureFaceRequest } from "../../lights/runtime/ProbeCaptureRuntime";
import { LightType } from "../../lights";
import { ComputeRuntime } from "./ComputeRuntime";
import type {
	WebGPUPreparedFrameResources,
	WebGPUFrameResourceProvider,
	WebGPUParticleBillboardRenderer,
	WebGPUSceneResourceProvider,
} from "./WebGPUResourceContracts";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";
import { TextureFormat, TextureUsage, type IRenderTexture } from "../types";
import { submitWebGPUDraws } from "./WebGPUDrawSubmission";
import { WEBGPU_MRT_COLOR_FORMATS } from "./constants";
import {
	CustomRenderPassRegistrySnapshot,
	RenderTargetRegistrySnapshot,
} from "../../rendering/CustomRenderTargets";

const CAPTURE_CAMERA_NEAR = 0.1;
const CUBE_FACE_DIRECTIONS: IVector3[] = [
	{ x: 1, y: 0, z: 0 },
	{ x: -1, y: 0, z: 0 },
	{ x: 0, y: 1, z: 0 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: 0, z: 1 },
	{ x: 0, y: 0, z: -1 },
];
const CUBE_FACE_UP_VECTORS: IVector3[] = [
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: 0, z: 1 },
	{ x: 0, y: 0, z: -1 },
	{ x: 0, y: -1, z: 0 },
	{ x: 0, y: -1, z: 0 },
];

export class WebGPUReflectionProbeCapturePass {
	private _backend: WebGPUFrameHost;
	private _captureResources: WebGPUFrameResourceProvider &
		WebGPUSceneResourceProvider;
	private _particleRenderer: WebGPUParticleBillboardRenderer;
	private readonly _framePacketProvider: FramePacketProvider;
	private _readbackRuntime: ComputeRuntime | null = null;
	private _destroyed = false;

	constructor(
		backend: WebGPUFrameHost,
		captureResources: WebGPUFrameResourceProvider & WebGPUSceneResourceProvider,
		framePacketProvider: FramePacketProvider,
		particleRenderer: WebGPUParticleBillboardRenderer,
	) {
		this._backend = backend;
		this._captureResources = captureResources;
		this._framePacketProvider = framePacketProvider;
		this._particleRenderer = particleRenderer;
	}

	public async captureFace(
		request: ProbeCaptureFaceRequest
	): Promise<Float32Array | null> {
		const faceSize = Math.max(1, Math.floor(request.faceSize));
		if (faceSize <= 0) {
			return null;
		}

		const resolvedFaceIndex = clampFaceIndex(request.faceIndex);
		const captureCamera = createCubeFaceCamera(
			request.captureWorldPosition,
			request.captureFar,
			resolvedFaceIndex
		);
		const captureScene = buildCapturePreparedScene(
			request.frameContext,
			captureCamera,
			request.includeMeshes,
			request.includeEnvironment,
			request.includeTransparent,
			request.includeParticles,
			request.targetKind,
			request.targetId
		);
		const captureFeatures = createCaptureFeatures(
			request.frameContext.features,
			request.includeShadows
		);
		const capturePostProcess = createCapturePostProcess(
			request.frameContext.postProcess
		);
		const captureTransient = createTransientStore();
		if (request.includeParticles) {
			populateParticleBatchesForCapture(
				request.frameContext,
				captureTransient,
				captureCamera
			);
		} else {
			captureTransient.set(PARTICLE_TRANSIENT_BATCHES_KEY, []);
		}

		const captureContext: FrameContext = {
			backendProfile: request.frameContext.backendProfile,
			viewCamera: captureCamera,
			attachments: {
				width: faceSize,
				height: faceSize,
			},
			features: captureFeatures,
			postProcess: capturePostProcess,
			renderTargets: new RenderTargetRegistrySnapshot(),
			customRenderPasses: new CustomRenderPassRegistrySnapshot(),
			shadowPlan: captureScene.shadowPlan,
			scene: captureScene,
			shCoeffs: request.frameContext.shCoeffs,
			shAmbientCoeffs: request.frameContext.shAmbientCoeffs,
			worldMatrix: request.frameContext.worldMatrix,
			incremental: createFullFrameIncrementalContext(faceSize),
			transient: captureTransient,
		};
		const framePackets = this._framePacketProvider.prepare(
			captureContext,
			"probe-capture",
		);
		const captureTargets = createCaptureRenderTargets(
			this._backend,
			faceSize,
			resolvedFaceIndex
		);
		const scope = this._captureResources.createFrameScope();
		let frameResources: WebGPUPreparedFrameResources | null = null;

		try {
			frameResources = scope.prepare(captureContext, {
				sceneTargetMode: "mrt",
				framePackets,
				temporalStateMode: "disabled",
			});
			const encoder = this._backend.createCommandEncoder();
			await this._captureResources.buildClusteredLighting(encoder, frameResources);
			await this._recordSceneCapture(
				encoder,
				captureContext,
				captureTargets,
				request.includeEnvironment,
				framePackets,
				frameResources
			);
			if (request.includeParticles) {
				await this._particleRenderer.renderParticles(
					encoder,
					captureContext,
					{
						label: "WebGPUReflectionProbeCaptureParticles",
						sampleCount: 1,
						colorAttachments: [
							{
								view: captureTargets.sceneColor,
								loadOp: "load",
								storeOp: "store",
							},
						],
						depth: captureTargets.depth,
					},
					frameResources,
					"mrt",
					{
						pipelineMode: "legacy",
					}
				);
			}

			this._backend.submit([encoder.finish()]);
			const readback = await this._getReadbackRuntime().readTexture({
				texture: captureTargets.sceneColor,
				width: faceSize,
				height: faceSize,
				format: TextureFormat.RGBA16Float,
			});
			return flipFaceRowsVertically(
				readback.toRGBAFloat32(),
				faceSize
			);
		} finally {
			destroyCaptureRenderTargets(captureTargets);
			scope.destroy();
		}
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this._readbackRuntime?.destroy();
		this._readbackRuntime = null;
	}

	private _getReadbackRuntime(): ComputeRuntime {
		if (this._destroyed) {
			throw new Error(
				"WebGPU reflection probe capture pass is destroyed; cannot read back captures."
			);
		}
		if (!this._readbackRuntime) {
			this._readbackRuntime = new ComputeRuntime(this._backend.computeFacade);
		}
		return this._readbackRuntime;
	}

	private async _recordSceneCapture(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: CaptureRenderTargets,
		includeEnvironment: boolean,
		framePackets: PreparedFramePacketSet,
		frameResources: WebGPUPreparedFrameResources
	): Promise<void> {
		const drewEnvironment = includeEnvironment ?
				await this._recordEnvironmentCapturePass(
					encoder,
					targets,
					frameResources
				)
			:	false;

		encoder.beginRenderPass({
			label: "WebGPUReflectionProbeCaptureMain",
			colorAttachments: [
				{
					view: targets.sceneColor,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: drewEnvironment ? "load" : "clear",
					storeOp: "store",
				},
				{
					view: targets.gAlbedoAlpha,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gNormalRoughMetal,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gEmissiveOcclusion,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gMotionDepth,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.depth,
				depthClearValue: 1,
				depthLoadOp: drewEnvironment ? "load" : "clear",
				depthStoreOp: "store",
			},
		});

		const packets = framePackets.all.slice();
		await submitWebGPUDraws({
			encoder,
			resources: this._captureResources,
			frameResources,
			packets,
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				sampleCount: 1,
			}),
		});
		encoder.endRenderPass();
	}

	private async _recordEnvironmentCapturePass(
		encoder: ICommandEncoder,
		targets: CaptureRenderTargets,
		frameResources: WebGPUPreparedFrameResources
	): Promise<boolean> {
		const environmentResources =
			await this._captureResources.getEnvironmentResources(frameResources, "mrt", {
				sampleCount: 1,
			});
		if (!environmentResources) {
			return false;
		}

		encoder.beginRenderPass({
			label: "WebGPUReflectionProbeCaptureEnvironment",
			colorAttachments: [
				{
					view: targets.sceneColor,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.depth,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});
		encoder.setPipeline(environmentResources.pipeline);
		encoder.setBindingGroup(0, environmentResources.frameBinding);
		encoder.draw(3);
		encoder.endRenderPass();
		return true;
	}
}

interface CaptureRenderTargets {
	sceneColor: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	depth: IRenderTexture;
}

function createCaptureRenderTargets(
	backend: WebGPUFrameHost,
	faceSize: number,
	faceIndex: number
): CaptureRenderTargets {
	return {
		sceneColor: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: WEBGPU_MRT_COLOR_FORMATS[0],
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.CopySrc |
				TextureUsage.TextureBinding,
			label: `WebGPUReflectionProbeCaptureSceneColor_face${faceIndex}`,
		}),
		gAlbedoAlpha: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: WEBGPU_MRT_COLOR_FORMATS[1],
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureAlbedo_face${faceIndex}`,
		}),
		gNormalRoughMetal: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: WEBGPU_MRT_COLOR_FORMATS[2],
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureNormal_face${faceIndex}`,
		}),
		gEmissiveOcclusion: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: WEBGPU_MRT_COLOR_FORMATS[3],
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureEmissive_face${faceIndex}`,
		}),
		gMotionDepth: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: WEBGPU_MRT_COLOR_FORMATS[4],
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureMotion_face${faceIndex}`,
		}),
		depth: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: TextureFormat.Depth32Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureDepth_face${faceIndex}`,
		}),
	};
}

function destroyCaptureRenderTargets(targets: CaptureRenderTargets): void {
	targets.sceneColor.destroy();
	targets.gAlbedoAlpha.destroy();
	targets.gNormalRoughMetal.destroy();
	targets.gEmissiveOcclusion.destroy();
	targets.gMotionDepth.destroy();
	targets.depth.destroy();
}

function clampFaceIndex(faceIndex: number): number {
	if (!Number.isFinite(faceIndex)) return 0;
	return Math.max(0, Math.min(5, Math.floor(faceIndex)));
}

function createCubeFaceCamera(
	captureWorldPosition: IVector3,
	captureFarInput: number,
	faceIndex: number
): Camera {
	const direction = CUBE_FACE_DIRECTIONS[faceIndex] ?? CUBE_FACE_DIRECTIONS[0];
	const up = CUBE_FACE_UP_VECTORS[faceIndex] ?? CUBE_FACE_UP_VECTORS[0];
	const captureFar = Math.max(1, captureFarInput);
	const camera = new Camera({
		fov: 90,
		aspectRatio: 1,
		near: CAPTURE_CAMERA_NEAR,
		far: captureFar,
	});
	camera.position.set(
		captureWorldPosition.x,
		captureWorldPosition.y,
		captureWorldPosition.z
	);
	camera.updateWorldMatrix();

	const target = {
		x: captureWorldPosition.x + direction.x,
		y: captureWorldPosition.y + direction.y,
		z: captureWorldPosition.z + direction.z,
	};
	camera.viewMatrix = Matrix4.lookAt(captureWorldPosition, target, up);
	camera.projectionMatrix = Matrix4.perspective(
		camera.fov,
		camera.aspectRatio,
		camera.near,
		camera.far
	);
	camera.viewProjectionMatrix = Matrix4.multiply(
		camera.projectionMatrix,
		camera.viewMatrix
	);
	camera.frustum.setFromMatrix(camera.viewProjectionMatrix);
	return camera;
}

function buildCapturePreparedScene(
	frameContext: FrameContext,
	captureCamera: Camera,
	includeMeshes: boolean,
	includeEnvironment: boolean,
	includeTransparent: boolean,
	includeParticles: boolean,
	targetKind: ProbeCaptureFaceRequest["targetKind"],
	targetId: string
): PreparedScene {
	const baseScene = frameContext.scene;
	const rebuiltScene =
		includeMeshes ?
			PreparedSceneBuilder.rebuildForCamera(baseScene, captureCamera, {
				visibilityScene: frameContext.viewCamera.scene ?? null,
			})
		:	baseScene;
	const lights = filterCapturePreparedSceneLights(
		baseScene.lights,
		targetKind,
		targetId
	);

	return {
		...rebuiltScene,
		lights,
		particleSystems: includeParticles ? baseScene.particleSystems : [],
		camera: captureCamera,
		environment: {
			...baseScene.environment,
			backgroundEnabled:
				includeEnvironment && baseScene.environment.backgroundEnabled,
			backgroundTexture:
				includeEnvironment ? baseScene.environment.backgroundTexture : null,
		},
		opaquePackets: includeMeshes ? rebuiltScene.opaquePackets : [],
		transparentPackets:
			includeMeshes && includeTransparent ? rebuiltScene.transparentPackets : [],
		shadowCasterPackets: baseScene.shadowCasterPackets,
		shadowTransmitterPackets: baseScene.shadowTransmitterPackets,
		reflectivePackets: [],
		decalPackets: [],
		occlusion: null,
		spatialIndex: null,
	};
}

function filterCapturePreparedSceneLights(
	lights: PreparedScene["lights"],
	targetKind: ProbeCaptureFaceRequest["targetKind"],
	targetId: string
): PreparedScene["lights"] {
	if (targetKind !== "grid") {
		return lights;
	}
	return lights.filter(
		(light) =>
			light.type !== LightType.IrradianceProbeGrid ||
			light.id !== targetId
	);
}

function populateParticleBatchesForCapture(
	frameContext: FrameContext,
	captureTransient: ReturnType<typeof createTransientStore>,
	captureCamera: Camera
): void {
	const sourceBatches =
		frameContext.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
	const rebasedBatches: ParticleRenderBatch[] = [];
	for (const batch of sourceBatches) {
		const particles: ParticleRenderItem[] = [];
		for (const particle of batch.particles) {
			const cameraSpace = Matrix4.transformPoint(
				captureCamera.viewMatrix,
				particle.position
			);
			const depth = -cameraSpace.z;
			if (depth <= 0) continue;
			particles.push({
				position: {
					x: particle.position.x,
					y: particle.position.y,
					z: particle.position.z,
				},
				size: particle.size,
				color: {
					r: particle.color.r,
					g: particle.color.g,
					b: particle.color.b,
					a: particle.color.a,
				},
				rotation: particle.rotation,
				depth,
				uvRect: {
					u0: particle.uvRect.u0,
					v0: particle.uvRect.v0,
					u1: particle.uvRect.u1,
					v1: particle.uvRect.v1,
				},
			});
		}
		particles.sort((left, right) => right.depth - left.depth);
		if (particles.length <= 0) continue;
		rebasedBatches.push({
			systemId: batch.systemId,
			blendMode: batch.blendMode,
			texture: batch.texture,
			receiveShadows: batch.receiveShadows,
			castShadows: batch.castShadows,
			shadowDensity: batch.shadowDensity,
			shadowSoftness: batch.shadowSoftness,
			particles,
		});
	}
	captureTransient.set(PARTICLE_TRANSIENT_BATCHES_KEY, rebasedBatches);

	captureTransient.set(
		PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
		frameContext.transient.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY) ?? [],
	);
}

function createCaptureFeatures(
	features: ResolvedFeatureState,
	includeShadows: boolean
): ResolvedFeatureState {
	return {
		...features,
		enableReflection: false,
		enableShadows: includeShadows && features.enableShadows,
		warnings: features.warnings.slice(),
	};
}

function createCapturePostProcess(
	postProcess: ResolvedPostProcessState
): ResolvedPostProcessState {
	return postProcess.withPassDisabled("ssr");
}

function createFullFrameIncrementalContext(
	faceSize: number
): IncrementalFrameContext {
	return {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: faceSize, height: faceSize }],
		dirtyTileSize: faceSize,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		postProcessStartPass: null,
		reasonMask: 0,
		temporalHistoryReset: true,
	};
}

function flipFaceRowsVertically(data: Float32Array, faceSize: number): Float32Array {
	const rowStride = faceSize * 4;
	const flipped = new Float32Array(data.length);
	for (let y = 0; y < faceSize; y++) {
		const srcOffset = y * rowStride;
		const dstOffset = (faceSize - 1 - y) * rowStride;
		flipped.set(data.subarray(srcOffset, srcOffset + rowStride), dstOffset);
	}
	return flipped;
}
