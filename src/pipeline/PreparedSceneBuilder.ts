import type { Camera } from "../cameras/Camera";
import { AlphaMode } from "../materials/Material";
import { ShaderMaterial } from "../materials/ShaderMaterial";
import { isMaterialTransparentPass } from "../materials/transparency";
import { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr } from "../maths/types";
import type { Scene } from "../core/Scene";
import { EMPTY_SHADOW_FRAME_PLAN } from "./shadows/ShadowFramePlan";
import type { IPrimitive } from "../core/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type { Decal } from "../decals";
import { resolveDecalReceiverLayerMask } from "../decals/evaluation";
import { MeshInstance } from "../meshes";
import {
	buildOcclusionCandidate,
	type NormalizedOcclusionCullingOptions,
	type OcclusionCandidate,
	type OcclusionVisibilityProvider,
} from "./OcclusionCulling";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_RECEIVER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	type DecalPacket,
	type DrawPacket,
	type PreparedScene,
} from "./types";

export interface PreparedSceneBuildOptions {
	viewportWidth?: number;
	viewportHeight?: number;
	occlusionCullingOptions?: NormalizedOcclusionCullingOptions;
	occlusionVisibilityProvider?: OcclusionVisibilityProvider | null;
}

export interface PreparedSceneBuildSource {
	readonly scene: Scene;
	readonly camera: Camera;
	readonly hasActiveAnimations: boolean;
}

export class PreparedSceneBuilder {
	/**
	 * Rebuilds camera-visible draw packet lists from an existing prepared scene.
	 *
	 * @internal Pipeline and backend passes use this when they need a secondary
	 * camera view, such as mirrored planar reflection capture, without rebuilding
	 * scene-wide lighting, environment, and shadow metadata from `Scene`.
	 * Application code should normally let `Renderer` prepare scenes.
	 * @param source - Prepared scene whose mesh instances and shared scene state
	 * are reused as the rebuild source.
	 * @param camera - Camera used for frustum culling and packet depth sorting.
	 * @param options - Optional occlusion inputs for the rebuilt camera view.
	 * @returns A prepared scene sharing source scene metadata but with packets
	 * rebuilt for `camera`.
	 * @sideEffects None. The source prepared scene is not mutated.
	 */
	public static rebuildForCamera(
		source: PreparedScene,
		camera: Camera,
		options: PreparedSceneBuildOptions = {}
	): PreparedScene {
		const renderableMeshInstances = source.meshInstances.filter(
			(meshInstance) => meshInstance.visible !== false
		);
		const cameraVisibleMeshInstances = this._resolveVisibleMeshInstances(
			renderableMeshInstances,
			camera,
			null
		);
		return this._buildFromMeshInstances({
			meshInstances: source.meshInstances,
			renderableMeshInstances,
			cameraVisibleMeshInstances,
			camera,
			sceneBounds: source.sceneBounds,
			environment: source.environment,
			lights: source.lights,
			particleSystems: source.particleSystems,
			hasActiveAnimations: source.hasActiveAnimations,
			shadowPlan: source.shadowPlan,
			decals: source.decalPackets.map((packet) => packet.decal),
			options,
		});
	}

	public static build(
		source: PreparedSceneBuildSource,
		options: PreparedSceneBuildOptions = {}
	): PreparedScene {
		const meshInstances = source.scene.getMeshInstances();
		const renderableMeshInstances = meshInstances.filter(
			(meshInstance) => meshInstance.visible !== false
		);
		const cameraVisibleMeshInstances = this._resolveVisibleMeshInstances(
			renderableMeshInstances,
			source.camera,
			source.scene
		);
		const environment = source.scene.environment;
		return this._buildFromMeshInstances({
			meshInstances,
			renderableMeshInstances,
			cameraVisibleMeshInstances,
			camera: source.camera,
			sceneBounds: source.scene.getBounds(),
			environment: {
				backgroundEnabled: environment.backgroundEnabled,
				lightingEnabled: environment.lightingEnabled,
				backgroundTexture: environment.backgroundTexture,
				iblTexture: environment.iblTexture,
				backgroundStrength: environment.backgroundStrength,
				diffuseStrength: environment.diffuseStrength,
				specularStrength: environment.specularStrength,
				backgroundTintLinear: environment.backgroundTintLinear,
				backgroundExposure: environment.backgroundExposure,
			},
			lights: source.scene.ecs.findLights(),
			particleSystems: source.scene.ecs.findParticleSystems(),
			hasActiveAnimations: source.hasActiveAnimations,
			shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
			decals: source.scene.getDecals(),
			options,
		});
	}

	private static _resolveVisibleMeshInstances(
		renderableMeshInstances: MeshInstance[],
		camera: Camera,
		scene: Scene | null
	): Set<MeshInstance> {
		const bypassFrustumMeshInstances: MeshInstance[] = [];
		const frustumCulledMeshInstances: MeshInstance[] = [];

		for (const meshInstance of renderableMeshInstances) {
			if (isAnimationDrivenMeshInstance(meshInstance)) {
				bypassFrustumMeshInstances.push(meshInstance);
			} else {
				frustumCulledMeshInstances.push(meshInstance);
			}
		}

		const frustumVisibleMeshInstances =
			scene ?
				scene.queryMeshInstancesInFrustum(camera, frustumCulledMeshInstances)
			:	this._filterMeshInstancesInFrustum(camera, frustumCulledMeshInstances);
		const cameraVisibleMeshInstances = new Set<MeshInstance>(
			frustumVisibleMeshInstances
		);
		for (const meshInstance of bypassFrustumMeshInstances) {
			cameraVisibleMeshInstances.add(meshInstance);
		}
		return cameraVisibleMeshInstances;
	}

	private static _filterMeshInstancesInFrustum(
		camera: Camera,
		meshInstances: MeshInstance[]
	): MeshInstance[] {
		const result: MeshInstance[] = [];
		const bounds = {
			center: { x: 0, y: 0, z: 0 },
			radius: 0,
		};
		for (const meshInstance of meshInstances) {
			meshInstance.getWorldBoundingSphere(bounds);
			if (camera.frustum.intersectsSphere(bounds.center, bounds.radius)) {
				result.push(meshInstance);
			}
		}
		return result;
	}

	private static _buildFromMeshInstances(input: {
		meshInstances: MeshInstance[];
		renderableMeshInstances: MeshInstance[];
		cameraVisibleMeshInstances: Set<MeshInstance>;
		camera: Camera;
		sceneBounds: PreparedScene["sceneBounds"];
		environment: PreparedScene["environment"];
		lights: PreparedScene["lights"];
		particleSystems: PreparedScene["particleSystems"];
		hasActiveAnimations: boolean;
		shadowPlan: PreparedScene["shadowPlan"];
		decals: Decal[];
		options: PreparedSceneBuildOptions;
	}): PreparedScene {
		const opaquePackets: DrawPacket[] = [];
		const transparentPackets: DrawPacket[] = [];
		const shadowCasterPackets: DrawPacket[] = [];
		const shadowTransmitterPackets: DrawPacket[] = [];
		const reflectivePackets: DrawPacket[] = [];

		for (const meshInstance of input.renderableMeshInstances) {
			const visibleInCamera =
				input.cameraVisibleMeshInstances.has(meshInstance);
			const packets = this._buildMeshPackets(meshInstance, input.camera);
			for (const packet of packets) {
				if (packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
					if (visibleInCamera) {
						transparentPackets.push(packet);
					}
				} else if (visibleInCamera) {
					opaquePackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) {
					shadowCasterPackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) {
					shadowTransmitterPackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_REFLECTIVE) {
					if (visibleInCamera) {
						reflectivePackets.push(packet);
					}
				}
			}
		}

		const occlusion = this._resolveOcclusionState(
			opaquePackets,
			input.camera,
			input.options
		);
		if (occlusion) {
			if (occlusion.culledPacketIds.length > 0) {
				const culled = new Set(occlusion.culledPacketIds);
				for (let index = opaquePackets.length - 1; index >= 0; index--) {
					if (culled.has(opaquePackets[index].id)) {
						opaquePackets.splice(index, 1);
					}
				}
			}
		}

		opaquePackets.sort(compareOpaquePackets);
		transparentPackets.sort(compareTransparentPackets);
		const decalPackets = this._buildDecalPackets(
			input.decals,
			opaquePackets
		);

		return {
			sceneBounds: input.sceneBounds,
			lights: input.lights,
			particleSystems: input.particleSystems,
			hasActiveAnimations: input.hasActiveAnimations,
			camera: input.camera,
			environment: input.environment,
			meshInstances: input.meshInstances,
			shadowPlan: input.shadowPlan,
			opaquePackets,
			transparentPackets,
			shadowCasterPackets,
			shadowTransmitterPackets,
			reflectivePackets,
			decalPackets,
			occlusion,
			spatialIndex: null,
		};
	}

	private static _resolveOcclusionState(
		opaquePackets: DrawPacket[],
		camera: Camera,
		options: PreparedSceneBuildOptions
	): PreparedScene["occlusion"] {
		const provider = options.occlusionVisibilityProvider ?? null;
		const cullingOptions = options.occlusionCullingOptions;
		const width = Math.max(0, Math.floor(options.viewportWidth ?? 0));
		const height = Math.max(0, Math.floor(options.viewportHeight ?? 0));
		if (!provider || !cullingOptions || width <= 0 || height <= 0) {
			return null;
		}
		const candidates: OcclusionCandidate[] = [];
		const culledPacketIds: string[] = [];
		let eligibleCandidateCount = 0;
		let visibleCandidateCount = 0;
		for (const packet of opaquePackets) {
			const candidate = buildOcclusionCandidate(
				packet,
				camera,
				width,
				height,
				cullingOptions
			);
			if (!candidate) {
				continue;
			}
			candidates.push(candidate);
			if (!candidate.eligible) {
				visibleCandidateCount++;
				continue;
			}
			eligibleCandidateCount++;
			if (provider.isPacketVisible(candidate)) {
				visibleCandidateCount++;
			} else {
				culledPacketIds.push(candidate.packetId);
			}
		}
		return {
			enabled: true,
			sourceFrameIndex: provider.sourceFrameIndex,
			candidates,
			culledPacketIds,
			visibleCandidateCount,
			eligibleCandidateCount,
		};
	}

	private static _buildMeshPackets(
		meshInstance: MeshInstance,
		camera: Camera
	): DrawPacket[] {
		const worldMatrix = meshInstance.worldMatrix;
		const normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
		const cameraSpaceCenter = Matrix4.transformPoint(
			camera.viewMatrix,
			Matrix4.transformPoint(
				worldMatrix,
				meshInstance.mesh.boundingSphere.center
			)
		);
		const meshDepth = -cameraSpaceCenter.z;
		const worldScale = getMaxScaleFromMatrix(worldMatrix) || 1;

		return meshInstance.mesh.primitives
			.filter((primitive) => primitive.visible !== false)
			.map((primitive) =>
				this._createPacket(
					meshInstance,
					primitive,
					worldMatrix,
					normalMatrix,
					worldScale,
					meshDepth
				)
			);
	}

	private static _createPacket(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		worldMatrix: Matrix4,
		normalMatrix: Matrix3Arr,
		worldScale: number,
		meshDepth: number
	): DrawPacket {
		const material = primitive.material;
		const isTransparent = isMaterialTransparentPass(material);
		const isReflective =
			material.reflectivity > 0 && material.mirrorPlane !== null;
		const supportsShadowCasting =
			(primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
			DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;

		let passFlags = 0;
		if (isTransparent) {
			passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
			if (primitive.castShadows && supportsShadowCasting) {
				passFlags |= DRAW_PACKET_FLAG_SHADOW_TRANSMITTER;
			}
		} else if (primitive.castShadows && supportsShadowCasting) {
			passFlags |= DRAW_PACKET_FLAG_SHADOW_CASTER;
		}

		if (isReflective) {
			passFlags |= DRAW_PACKET_FLAG_REFLECTIVE;
		}
		if (primitive.receiveShadows !== false) {
			passFlags |= DRAW_PACKET_FLAG_SHADOW_RECEIVER;
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

	private static _buildDecalPackets(
		decals: Decal[],
		opaquePackets: DrawPacket[]
	): DecalPacket[] {
		const packets: DecalPacket[] = [];
		let sceneOrder = 0;
		for (const decal of decals) {
			const packet = this._createDecalPacket(
				decal,
				opaquePackets,
				sceneOrder
			);
			sceneOrder++;
			if (packet) {
				packets.push(packet);
			}
		}
		packets.sort(compareDecalPackets);
		return packets;
	}

	private static _createDecalPacket(
		decal: Decal,
		opaquePackets: DrawPacket[],
		sceneOrder: number
	): DecalPacket | null {
		if (decal.visible === false || decal.opacity <= 0) {
			return null;
		}
		const material = decal.material;
		if (!material || material instanceof ShaderMaterial) {
			return null;
		}
		const receiverLayerMask = resolveDecalReceiverLayerMask(
			decal.receiverLayerMask
		);
		if (receiverLayerMask === 0) {
			return null;
		}
		const hasReceiver = opaquePackets.some(
			(packet) => (packet.meshInstance.renderLayers & receiverLayerMask) !== 0
		);
		if (!hasReceiver) {
			return null;
		}
		const inverseWorldMatrix = Matrix4.inverse(decal.worldMatrix);
		if (!inverseWorldMatrix) {
			return null;
		}
		const normalMatrix = Matrix4.normalMatrix(decal.worldMatrix) as Matrix3Arr;
		const worldCenter = Matrix4.transformPoint(decal.worldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const elements = decal.worldMatrix.elements;
		const halfDiagonal = Math.hypot(
			Math.hypot(elements[0][0], elements[1][0], elements[2][0]),
			Math.hypot(elements[0][1], elements[1][1], elements[2][1]),
			Math.hypot(elements[0][2], elements[1][2], elements[2][2])
		) * 0.5;
		if (halfDiagonal <= 0) {
			return null;
		}
		return {
			id: decal.id,
			decal,
			material,
			worldMatrix: decal.worldMatrix,
			inverseWorldMatrix,
			normalMatrix,
			worldBounds: {
				center: {
					x: worldCenter.x,
					y: worldCenter.y,
					z: worldCenter.z,
				},
				radius: halfDiagonal,
			},
			receiverLayerMask,
			priority: decal.priority,
			opacity: decal.opacity,
			edgeFade: decal.edgeFade,
			channelBlendModes: { ...decal.channelBlendModes },
			sceneOrder,
		};
	}
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

function compareDecalPackets(left: DecalPacket, right: DecalPacket): number {
	if (left.priority !== right.priority) {
		return left.priority - right.priority;
	}
	if (left.sceneOrder !== right.sceneOrder) {
		return left.sceneOrder - right.sceneOrder;
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
