import type { Camera } from "../cameras/Camera";
import { AlphaMode } from "../materials/Material";
import type { Material } from "../materials/Material";
import { ShaderMaterial } from "../materials/ShaderMaterial";
import { isMaterialTransparentPass } from "../materials/transparency";
import { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr } from "../maths/types";
import type { Scene } from "../core/Scene";
import { EMPTY_SHADOW_FRAME_PLAN } from "../lights/shadows/ShadowFramePlan";
import type { IPrimitive } from "../core/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type { Decal } from "../decals";
import type {
	PrimitiveDeformationMap,
	PrimitiveDeformationState,
} from "../simulation/animation/types";
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
	/** @internal Scene whose spatial index may accelerate secondary-view culling. */
	visibilityScene?: Scene | null;
	/** @internal View-local packet cache owned by frame coordination. */
	packetCache?: PreparedScenePacketCache | null;
}

interface PreparedPacketCacheSignature {
	readonly mesh: DrawPacket["mesh"];
	readonly material: DrawPacket["material"];
	readonly geometry: DrawPacket["geometry"];
	readonly transformRevision: number;
	readonly boundsVersion: number;
	readonly meshBoundsVersion: number;
	readonly geometryVersion: number;
	readonly deformationRevision: number;
	readonly primitiveVisible: boolean;
	readonly castShadows: boolean;
	readonly receiveShadows: boolean;
	readonly renderLayers: number;
	readonly materialRevision: number;
}

interface PreparedPacketCacheEntry {
	readonly key: string;
	readonly owner: Map<string, PreparedPacketCacheEntry>;
	readonly packet: DrawPacket;
	signature: PreparedPacketCacheSignature;
	lastUsedFrame: number;
}

/** @internal Bounded view-local cache for prepared draw packets. */
export class PreparedScenePacketCache {
	private _views = new WeakMap<object, Map<string, PreparedPacketCacheEntry>>();
	private readonly _lru = new Set<PreparedPacketCacheEntry>();
	private _materialRevisions = new WeakMap<Material, number>();
	private _frame = 0;
	private _frameHits = 0;
	private _frameRebuilds = 0;

	public constructor(private readonly _maxEntries = 65_536) {}

	public beginFrame(): void {
		this._frame++;
		this._frameHits = 0;
		this._frameRebuilds = 0;
		this._materialRevisions = new WeakMap<Material, number>();
	}

	public getReusablePacket(
		view: Camera,
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformationRevision: number,
	): DrawPacket | null {
		const packets = this._views.get(view);
		if (!packets) return null;
		const key = `${meshInstance.id}:${primitive.id}`;
		const entry = packets.get(key);
		if (!entry) return null;
		if (!this._isSignatureCurrent(
			entry.signature,
			meshInstance,
			primitive,
			deformationRevision,
		)) return null;
		this._frameHits++;
		updatePacketSortDepth(entry.packet, view);
		this._touchEntry(entry);
		return entry.packet;
	}

	public storePacket(
		view: Camera,
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformationRevision: number,
		packet: DrawPacket,
	): DrawPacket {
		let packets = this._views.get(view);
		if (!packets) {
			packets = new Map();
			this._views.set(view, packets);
		}
		const key = `${meshInstance.id}:${primitive.id}`;
		const signature = this._createSignature(
			meshInstance,
			primitive,
			deformationRevision,
		);
		let entry = packets.get(key);
		this._frameRebuilds++;
		if (entry) {
			Object.assign(entry.packet, packet);
			entry.signature = signature;
			this._touchEntry(entry);
			return entry.packet;
		}
		entry = {
			key,
			owner: packets,
			packet,
			signature,
			lastUsedFrame: this._frame,
		};
		packets.set(key, entry);
		this._lru.add(entry);
		return packet;
	}

	public endFrame(): void {
		while (this._lru.size > this._maxEntries) {
			const oldest = this._lru.values().next().value as
				| PreparedPacketCacheEntry
				| undefined;
			if (!oldest || oldest.lastUsedFrame === this._frame) break;
			this._lru.delete(oldest);
			oldest.owner.delete(oldest.key);
		}
	}

	public clear(): void {
		this._views = new WeakMap();
		this._lru.clear();
		this._materialRevisions = new WeakMap();
		this._frame = 0;
	}

	public getDebugStats(): {
		readonly entries: number;
		readonly frameHits: number;
		readonly frameRebuilds: number;
	} {
		return {
			entries: this._lru.size,
			frameHits: this._frameHits,
			frameRebuilds: this._frameRebuilds,
		};
	}

	private _createSignature(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformationRevision: number,
	): PreparedPacketCacheSignature {
		return {
			mesh: meshInstance.mesh,
			material: primitive.material,
			geometry: primitive.geometry,
			transformRevision: meshInstance.worldTransformRevision,
			boundsVersion: meshInstance.worldBoundsVersion,
			meshBoundsVersion: meshInstance.mesh.boundsVersion,
			geometryVersion: primitive.geometryVersion ?? 0,
			deformationRevision,
			primitiveVisible: primitive.visible !== false,
			castShadows: primitive.castShadows === true,
			receiveShadows: primitive.receiveShadows !== false,
			renderLayers: meshInstance.renderLayers,
			materialRevision: this._getMaterialRevision(primitive.material),
		};
	}

	private _isSignatureCurrent(
		signature: PreparedPacketCacheSignature,
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformationRevision: number,
	): boolean {
		return (
			signature.mesh === meshInstance.mesh &&
			signature.material === primitive.material &&
			signature.geometry === primitive.geometry &&
			signature.transformRevision === meshInstance.worldTransformRevision &&
			signature.boundsVersion === meshInstance.worldBoundsVersion &&
			signature.meshBoundsVersion === meshInstance.mesh.boundsVersion &&
			signature.geometryVersion === (primitive.geometryVersion ?? 0) &&
			signature.deformationRevision === deformationRevision &&
			signature.primitiveVisible === (primitive.visible !== false) &&
			signature.castShadows === (primitive.castShadows === true) &&
			signature.receiveShadows === (primitive.receiveShadows !== false) &&
			signature.renderLayers === meshInstance.renderLayers &&
			signature.materialRevision === this._getMaterialRevision(primitive.material)
		);
	}

	private _getMaterialRevision(material: Material): number {
		let revision = this._materialRevisions.get(material);
		if (revision !== undefined) return revision;
		revision = material.revision;
		this._materialRevisions.set(material, revision);
		return revision;
	}

	private _touchEntry(entry: PreparedPacketCacheEntry): void {
		if (entry.lastUsedFrame === this._frame) return;
		this._lru.delete(entry);
		this._lru.add(entry);
		entry.lastUsedFrame = this._frame;
	}
}

export interface PreparedSceneBuildSource {
	readonly scene: Scene;
	readonly camera: Camera;
	readonly hasActiveAnimations: boolean;
	readonly deformationStates?: PrimitiveDeformationMap | null;
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
	 * @param options - Optional visibility acceleration and occlusion inputs for
	 * the rebuilt camera view.
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
			options.visibilityScene ?? null
		);
		const opaquePackets: DrawPacket[] = [];
		const transparentPackets: DrawPacket[] = [];
		const reflectivePackets: DrawPacket[] = [];
		for (const meshInstance of renderableMeshInstances) {
			if (!cameraVisibleMeshInstances.has(meshInstance)) {
				continue;
			}
			this._forEachMeshPacket(
				meshInstance,
				camera,
				source.deformationStates ?? null,
				options.packetCache ?? null,
				(packet) => this._appendViewPacket(
					packet,
					opaquePackets,
					transparentPackets,
					reflectivePackets,
				),
			);
		}
		const viewState = this._finalizeViewPackets(
			opaquePackets,
			transparentPackets,
			camera,
			source.decalPackets.map((packet) => packet.decal),
			options
		);
		return {
			...source,
			camera,
			opaquePackets,
			transparentPackets,
			reflectivePackets,
			decalPackets: viewState.decalPackets,
			occlusion: viewState.occlusion,
			spatialIndex: null,
		};
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
			deformationStates: source.deformationStates ?? null,
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
		deformationStates: PrimitiveDeformationMap | null;
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
			this._forEachMeshPacket(
				meshInstance,
				input.camera,
				input.deformationStates,
				input.options.packetCache ?? null,
				(packet) => {
				if (visibleInCamera) {
					this._appendViewPacket(
						packet,
						opaquePackets,
						transparentPackets,
						reflectivePackets
					);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) {
					shadowCasterPackets.push(packet);
				}

				if (packet.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) {
					shadowTransmitterPackets.push(packet);
				}
				},
			);
		}

		const viewState = this._finalizeViewPackets(
			opaquePackets,
			transparentPackets,
			input.camera,
			input.decals,
			input.options
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
			decalPackets: viewState.decalPackets,
			occlusion: viewState.occlusion,
			spatialIndex: null,
			deformationStates: input.deformationStates,
		};
	}

	private static _appendViewPacket(
		packet: DrawPacket,
		opaquePackets: DrawPacket[],
		transparentPackets: DrawPacket[],
		reflectivePackets: DrawPacket[]
	): void {
		if (packet.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
			transparentPackets.push(packet);
		} else {
			opaquePackets.push(packet);
		}
		if (packet.passFlags & DRAW_PACKET_FLAG_REFLECTIVE) {
			reflectivePackets.push(packet);
		}
	}

	private static _finalizeViewPackets(
		opaquePackets: DrawPacket[],
		transparentPackets: DrawPacket[],
		camera: Camera,
		decals: Decal[],
		options: PreparedSceneBuildOptions
	): Pick<PreparedScene, "decalPackets" | "occlusion"> {
		const occlusion = this._resolveOcclusionState(
			opaquePackets,
			camera,
			options
		);
		if (occlusion?.culledPacketIds.length) {
			const culled = new Set(occlusion.culledPacketIds);
			for (let index = opaquePackets.length - 1; index >= 0; index--) {
				if (culled.has(opaquePackets[index].id)) {
					opaquePackets.splice(index, 1);
				}
			}
		}

		opaquePackets.sort(compareOpaquePackets);
		transparentPackets.sort(compareTransparentPackets);
		return {
			decalPackets: this._buildDecalPackets(decals, opaquePackets),
			occlusion,
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

	private static _forEachMeshPacket(
		meshInstance: MeshInstance,
		camera: Camera,
		deformationStates: PrimitiveDeformationMap | null,
		packetCache: PreparedScenePacketCache | null,
		visitor: (packet: DrawPacket) => void,
	): void {
		const worldMatrix = meshInstance.worldMatrix;
		let normalMatrix: Matrix3Arr | null = null;
		let worldScale = 1;
		let transformStatePrepared = false;

		for (const primitive of meshInstance.mesh.primitives) {
			if (primitive.visible === false) continue;
			const deformation =
				deformationStates?.get(`${meshInstance.id}:${primitive.id}`) ?? null;
			const cached = packetCache?.getReusablePacket(
				camera,
				meshInstance,
				primitive,
				deformation?.revision ?? 0,
			);
			if (cached) {
				visitor(cached);
				continue;
			}
			if (!transformStatePrepared) {
				transformStatePrepared = true;
				normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
				worldScale = getMaxScaleFromMatrix(worldMatrix) || 1;
			}
			const created = this._createPacket(
					meshInstance,
					primitive,
					worldMatrix,
					normalMatrix!,
					worldScale,
					camera,
					deformation,
			);
			visitor(packetCache
				? packetCache.storePacket(
					camera,
					meshInstance,
					primitive,
					deformation?.revision ?? 0,
					created,
				)
				: created);
		}
	}

	private static _createPacket(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		worldMatrix: Matrix4,
		normalMatrix: Matrix3Arr,
		worldScale: number,
		camera: Camera,
		deformation: PrimitiveDeformationState | null
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

		const localBounds = deformation?.localBounds ?? primitive.boundingSphere;
		const worldCenter = Matrix4.transformPoint(
			worldMatrix,
			localBounds.center
		);
		const cameraSpaceCenter = Matrix4.transformPoint(
			camera.viewMatrix,
			worldCenter
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
				radius: localBounds.radius * worldScale,
			},
			deformationRevision: deformation?.revision ?? 0,
			sortDepth: -cameraSpaceCenter.z,
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

function updatePacketSortDepth(packet: DrawPacket, camera: Camera): void {
	const center = packet.worldBounds.center;
	const view = camera.viewMatrix.elements;
	packet.sortDepth = -(
		view[2][0] * center.x +
		view[2][1] * center.y +
		view[2][2] * center.z +
		view[2][3]
	);
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
