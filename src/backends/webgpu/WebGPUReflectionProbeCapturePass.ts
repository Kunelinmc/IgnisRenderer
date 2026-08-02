import { Camera } from "../../cameras/Camera";
import { AlphaMode } from "../../materials/Material";
import { isMaterialTransparentPass } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, Matrix3Arr } from "../../maths/types";
import type { MeshInstance } from "../../meshes";
import type { IPrimitive } from "../../core/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	DrawPacket,
	FrameContext,
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
	ParticleRenderBatch,
	ParticleRenderItem,
	PreparedScene,
	ResolvedFeatureState,
} from "../../pipeline/types";
import {
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
	createTransientStore,
} from "../../pipeline/types";
import type { ResolvedPostProcessState } from "../../postprocess";
import type { IncrementalFrameContext } from "../../pipeline/incremental";
import type { ProbeCaptureFaceRequest } from "../../lights/runtime/ProbeCaptureRuntime";
import { LightType } from "../../lights";
import { ComputeRuntime } from "./ComputeRuntime";
import type {
	WebGPUPreparedFrameResources,
	WebGPUFrameResourceProvider,
	WebGPUParticleRenderProvider,
	WebGPUSceneResourceProvider,
} from "./WebGPUResourceContracts";
import type { WebGPUFrameHost } from "./rendergraph/WebGPUFrameHost";
import { TextureFormat, TextureUsage, type IRenderTexture } from "../types";
import { submitWebGPUDraws } from "./WebGPUDrawSubmission";
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
	private _resources: WebGPUFrameResourceProvider &
		WebGPUSceneResourceProvider;
	private _particleResources: WebGPUParticleRenderProvider;
	private _readbackRuntime: ComputeRuntime | null = null;
	private _destroyed = false;

	constructor(
		backend: WebGPUFrameHost,
		resources: WebGPUFrameResourceProvider & WebGPUSceneResourceProvider,
		particleResources: WebGPUParticleRenderProvider,
	) {
		this._backend = backend;
		this._resources = resources;
		this._particleResources = particleResources;
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
			shadowMaps: captureScene.shadowMaps,
			scene: captureScene,
			shCoeffs: request.frameContext.shCoeffs,
			shAmbientCoeffs: request.frameContext.shAmbientCoeffs,
			worldMatrix: request.frameContext.worldMatrix,
			incremental: createFullFrameIncrementalContext(faceSize),
			transient: captureTransient,
		};
		const captureTargets = createCaptureRenderTargets(
			this._backend,
			faceSize,
			resolvedFaceIndex
		);
		const scope = this._resources.createFrameScope();
		let frameResources: WebGPUPreparedFrameResources | null = null;

		try {
			frameResources = scope.prepare(captureContext, {
				sceneTargetMode: "mrt",
				temporalStateMode: "disabled",
			});
			const encoder = this._backend.createCommandEncoder();
			await this._resources.buildClusteredLighting(encoder, frameResources);
			await this._recordSceneCapture(
				encoder,
				captureContext,
				captureTargets,
				request.includeEnvironment,
				frameResources
			);
			if (request.includeParticles) {
				await this._particleResources.renderParticles(
					encoder,
					captureContext,
					{
						label: "WebGPUReflectionProbeCaptureParticles",
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
						sampleCountOverride: 1,
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

		const packets = [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
			...this._buildParticleMeshDrawPackets(context),
		];
		await submitWebGPUDraws({
			encoder,
			resources: this._resources,
			frameResources,
			packets,
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				sampleCountOverride: 1,
			}),
		});
		encoder.endRenderPass();
	}

	private _buildParticleMeshDrawPackets(context: FrameContext): DrawPacket[] {
		return this._particleResources.buildParticleMeshDrawPackets(context);
	}

	private async _recordEnvironmentCapturePass(
		encoder: ICommandEncoder,
		targets: CaptureRenderTargets,
		frameResources: WebGPUPreparedFrameResources
	): Promise<boolean> {
		const environmentResources =
			await this._resources.getEnvironmentResources(frameResources, "mrt", {
				sampleCountOverride: 1,
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
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.CopySrc |
				TextureUsage.TextureBinding,
			label: `WebGPUReflectionProbeCaptureSceneColor_face${faceIndex}`,
		}),
		gAlbedoAlpha: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureAlbedo_face${faceIndex}`,
		}),
		gNormalRoughMetal: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureNormal_face${faceIndex}`,
		}),
		gEmissiveOcclusion: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment,
			label: `WebGPUReflectionProbeCaptureEmissive_face${faceIndex}`,
		}),
		gMotionDepth: backend.createTexture({
			width: faceSize,
			height: faceSize,
			format: TextureFormat.RGBA16Float,
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
	const opaquePackets: DrawPacket[] = [];
	const transparentPackets: DrawPacket[] = [];
	const lights = filterCapturePreparedSceneLights(
		baseScene.lights,
		targetKind,
		targetId
	);

	if (includeMeshes) {
		const meshInstances = baseScene.meshInstances.filter(
			(meshInstance) => meshInstance.visible !== false
		);
		const bypassFrustumMeshInstances: MeshInstance[] = [];
		const frustumCulledMeshInstances: MeshInstance[] = [];

		for (const meshInstance of meshInstances) {
			if (isAnimationDrivenMeshInstance(meshInstance)) {
				bypassFrustumMeshInstances.push(meshInstance);
			} else {
				frustumCulledMeshInstances.push(meshInstance);
			}
		}

		const sceneRef = frameContext.viewCamera.scene;
		const visibleSet = new Set<MeshInstance>();
		if (sceneRef) {
			const visible = sceneRef.queryMeshInstancesInFrustum(
				captureCamera,
				frustumCulledMeshInstances
			);
			for (const meshInstance of visible) {
				visibleSet.add(meshInstance);
			}
		} else {
			for (const meshInstance of frustumCulledMeshInstances) {
				visibleSet.add(meshInstance);
			}
		}
		for (const meshInstance of bypassFrustumMeshInstances) {
			visibleSet.add(meshInstance);
		}

		for (const meshInstance of meshInstances) {
			if (!visibleSet.has(meshInstance)) continue;
			for (const packet of buildMeshPackets(meshInstance, captureCamera)) {
				const transparent =
					(packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
				if (transparent) {
					if (includeTransparent) {
						transparentPackets.push(packet);
					}
				} else {
					opaquePackets.push(packet);
				}
			}
		}
	}

	opaquePackets.sort(compareOpaquePackets);
	transparentPackets.sort(compareTransparentPackets);

	return {
		sceneBounds: baseScene.sceneBounds,
		lights,
		particleSystems: includeParticles ? baseScene.particleSystems : [],
		hasActiveAnimations: baseScene.hasActiveAnimations,
		camera: captureCamera,
		environment: {
			...baseScene.environment,
			backgroundEnabled:
				includeEnvironment && baseScene.environment.backgroundEnabled,
			backgroundTexture:
				includeEnvironment ? baseScene.environment.backgroundTexture : null,
		},
		meshInstances: baseScene.meshInstances,
		shadowMaps: baseScene.shadowMaps,
		opaquePackets,
		transparentPackets,
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

function buildMeshPackets(
	meshInstance: MeshInstance,
	camera: Camera
): DrawPacket[] {
	const worldMatrix = meshInstance.worldMatrix;
	const normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
	const cameraSpaceCenter = Matrix4.transformPoint(
		camera.viewMatrix,
		Matrix4.transformPoint(worldMatrix, meshInstance.mesh.boundingSphere.center)
	);
	const meshDepth = -cameraSpaceCenter.z;
	const worldScale = getMaxScaleFromMatrix(worldMatrix) || 1;

	return meshInstance.mesh.primitives
		.filter((primitive) => primitive.visible !== false)
		.map((primitive) =>
			createDrawPacket(
				meshInstance,
				primitive,
				worldMatrix,
				normalMatrix,
				worldScale,
				meshDepth
			)
		);
}

function createDrawPacket(
	meshInstance: MeshInstance,
	primitive: IPrimitive,
	worldMatrix: Matrix4,
	normalMatrix: Matrix3Arr,
	worldScale: number,
	meshDepth: number
): DrawPacket {
	const material = primitive.material;
	const isTransparent = isMaterialTransparentPass(material);
	let passFlags = 0;
	if (isTransparent) {
		passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
	}

	const worldCenter = Matrix4.transformPoint(
		worldMatrix,
		primitive.boundingSphere.center
	);
	return {
		id: `${meshInstance.id}:${primitive.id}`,
		meshInstance,
		mesh: meshInstance.mesh,
		primitive,
		material,
		geometry: primitive.geometry,
		worldMatrix,
		normalMatrix,
		worldBounds: {
			center: {
				x: worldCenter.x,
				y: worldCenter.y,
				z: worldCenter.z,
			},
			radius: primitive.boundingSphere.radius * worldScale,
		},
		sortDepth: meshDepth,
		pipelineKey: [
			material.type,
			material.shading,
			material.alphaMode ?? AlphaMode.Opaque,
			material.doubleSided ? "double" : "single",
			material.depthWrite ? "depth-write" : "depth-read",
		].join(":"),
		passFlags,
	};
}

function compareOpaquePackets(left: DrawPacket, right: DrawPacket): number {
	const keyCompare = left.pipelineKey.localeCompare(right.pipelineKey);
	if (keyCompare !== 0) return keyCompare;

	if (left.material !== right.material) {
		return left.material.name.localeCompare(right.material.name);
	}

	if (left.geometry !== right.geometry) {
		return left.id.localeCompare(right.id);
	}

	return left.sortDepth - right.sortDepth;
}

function compareTransparentPackets(
	left: DrawPacket,
	right: DrawPacket
): number {
	if (left.sortDepth !== right.sortDepth) {
		return right.sortDepth - left.sortDepth;
	}
	return left.id.localeCompare(right.id);
}

function getMaxScaleFromMatrix(matrix: Matrix4): number {
	const elements = matrix.elements;
	const sx = Math.hypot(elements[0][0], elements[1][0], elements[2][0]);
	const sy = Math.hypot(elements[0][1], elements[1][1], elements[2][1]);
	const sz = Math.hypot(elements[0][2], elements[1][2], elements[2][2]);
	return Math.max(sx, sy, sz);
}

function isAnimationDrivenMeshInstance(meshInstance: MeshInstance): boolean {
	if (meshInstance.skeleton) return true;
	for (const primitive of meshInstance.mesh.primitives) {
		if ((primitive.geometry.morphTargets?.length ?? 0) > 0) {
			return true;
		}
	}
	return false;
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

	const sourceMeshBatches =
		frameContext.transient.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY) ?? [];
	const rebasedMeshBatches: ParticleMeshRenderBatch[] = [];
	for (const batch of sourceMeshBatches) {
		const particles: ParticleMeshRenderItem[] = [];
		for (const particle of batch.particles) {
			const cameraSpace = Matrix4.transformPoint(
				captureCamera.viewMatrix,
				particle.position
			);
			const depth = -cameraSpace.z;
			if (depth <= 0) continue;
			particles.push({
				templateIndex: particle.templateIndex,
				position: {
					x: particle.position.x,
					y: particle.position.y,
					z: particle.position.z,
				},
				previousPosition: {
					x: particle.previousPosition.x,
					y: particle.previousPosition.y,
					z: particle.previousPosition.z,
				},
				size: particle.size,
				color: {
					r: particle.color.r,
					g: particle.color.g,
					b: particle.color.b,
					a: particle.color.a,
				},
				rotation: particle.rotation,
				previousRotation: particle.previousRotation,
				depth,
			});
		}
		particles.sort((left, right) => right.depth - left.depth);
		if (particles.length <= 0) continue;
		rebasedMeshBatches.push({
			...batch,
			particles,
		});
	}
	captureTransient.set(
		PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
		rebasedMeshBatches
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
