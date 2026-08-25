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
	type DrawGeometryBinding,
	type DrawInstanceBinding,
	type DrawPacket,
	type DrawSourceRef,
	type DrawSubmission,
	type PreparedScene,
} from "./types";

const geometryBindings = new WeakMap<IPrimitive, DrawGeometryBinding>();
const NO_DEFORMATION = Object.freeze({
	mode: "none",
	revision: 0,
	jointPayloadKey: null,
	morphPayloadKey: null,
} as const);

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
	readonly mesh: MeshInstance["mesh"];
	readonly primitive: IPrimitive;
	readonly material: Material;
	readonly geometry: IPrimitive["geometry"];
	readonly topology: IPrimitive["topology"];
	readonly transformRevision: number;
	readonly boundsVersion: number;
	readonly meshBoundsVersion: number;
	readonly geometryVersion: number;
	readonly deformationRevision: number;
	readonly deformationMode: PrimitiveDeformationState["mode"] | "none";
	readonly jointPayloadKey: string | null;
	readonly morphPayloadKey: string | null;
	readonly primitiveVisible: boolean;
	readonly castShadows: boolean;
	readonly receiveShadows: boolean;
	readonly renderLayers: number;
	readonly materialRevision: number;
}

interface MutableDrawPacket {
	submission: DrawSubmission;
	sortDepth: number;
}

interface PreparedViewPacketCacheEntry {
	readonly key: string;
	readonly owner: Map<string, PreparedViewPacketCacheEntry>;
	readonly submissionEntry: PreparedSubmissionCacheEntry;
	packet: MutableDrawPacket;
}

interface PreparedSubmissionCacheEntry {
	readonly key: string;
	submission: DrawSubmission;
	signature: PreparedPacketCacheSignature;
	readonly viewEntries: Set<PreparedViewPacketCacheEntry>;
	lastUsedFrame: number;
}

/** @internal Bounded submission cache with isolated view-local packets. */
export class PreparedScenePacketCache {
	private _views = new WeakMap<object, Map<string, PreparedViewPacketCacheEntry>>();
	private readonly _submissions = new Map<string, PreparedSubmissionCacheEntry>();
	private readonly _lru = new Set<PreparedSubmissionCacheEntry>();
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

	public getReusableSubmission(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformation: PrimitiveDeformationState | null,
	): DrawSubmission | null {
		const key = `${meshInstance.id}:${primitive.id}`;
		const entry = this._submissions.get(key);
		if (!entry) return null;
		if (!this._isSignatureCurrent(
			entry.signature,
			meshInstance,
			primitive,
			deformation,
		)) return null;
		this._frameHits++;
		this._touchEntry(entry);
		return entry.submission;
	}

	public storeSubmission(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		deformation: PrimitiveDeformationState | null,
		submission: DrawSubmission,
	): DrawSubmission {
		const key = `${meshInstance.id}:${primitive.id}`;
		const signature = this._createSignature(
			meshInstance,
			primitive,
			deformation,
		);
		let entry = this._submissions.get(key);
		this._frameRebuilds++;
		if (entry) {
			entry.submission = submission;
			entry.signature = signature;
			this._touchEntry(entry);
			return submission;
		}
		entry = {
			key,
			submission,
			signature,
			viewEntries: new Set(),
			lastUsedFrame: this._frame,
		};
		this._submissions.set(key, entry);
		this._lru.add(entry);
		return submission;
	}

	public getViewPacket(
		view: Camera,
		submission: DrawSubmission,
		sortDepth: number,
	): DrawPacket {
		let packets = this._views.get(view);
		if (!packets) {
			packets = new Map();
			this._views.set(view, packets);
		}
		const submissionEntry = this._submissions.get(submission.id);
		if (!submissionEntry) return { submission, sortDepth };
		let entry = packets.get(submission.id);
		if (!entry) {
			entry = {
				key: submission.id,
				owner: packets,
				submissionEntry,
				packet: { submission, sortDepth },
			};
			packets.set(submission.id, entry);
			submissionEntry.viewEntries.add(entry);
			return entry.packet;
		}
		if (entry.packet.submission !== submission) {
			entry.packet = { submission, sortDepth };
		} else {
			entry.packet.sortDepth = sortDepth;
		}
		return entry.packet;
	}

	public endFrame(): void {
		while (this._lru.size > this._maxEntries) {
			const oldest = this._lru.values().next().value as
				| PreparedSubmissionCacheEntry
				| undefined;
			if (!oldest || oldest.lastUsedFrame === this._frame) break;
			this._lru.delete(oldest);
			this._submissions.delete(oldest.key);
			for (const viewEntry of oldest.viewEntries) {
				viewEntry.owner.delete(viewEntry.key);
			}
		}
	}

	public clear(): void {
		this._views = new WeakMap();
		this._submissions.clear();
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
		deformation: PrimitiveDeformationState | null,
	): PreparedPacketCacheSignature {
		return {
			mesh: meshInstance.mesh,
			primitive,
			material: primitive.material,
			geometry: primitive.geometry,
			topology: primitive.topology,
			transformRevision: meshInstance.worldTransformRevision,
			boundsVersion: meshInstance.worldBoundsVersion,
			meshBoundsVersion: meshInstance.mesh.boundsVersion,
			geometryVersion: primitive.geometryVersion ?? 0,
			deformationRevision: deformation?.revision ?? 0,
			deformationMode: deformation?.mode ?? "none",
			jointPayloadKey: deformation?.jointPayloadKey ?? null,
			morphPayloadKey: deformation?.morphPayloadKey ?? null,
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
		deformation: PrimitiveDeformationState | null,
	): boolean {
		return (
			signature.mesh === meshInstance.mesh &&
			signature.primitive === primitive &&
			signature.material === primitive.material &&
			signature.geometry === primitive.geometry &&
			signature.topology === primitive.topology &&
			signature.transformRevision === meshInstance.worldTransformRevision &&
			signature.boundsVersion === meshInstance.worldBoundsVersion &&
			signature.meshBoundsVersion === meshInstance.mesh.boundsVersion &&
			signature.geometryVersion === (primitive.geometryVersion ?? 0) &&
			signature.deformationRevision === (deformation?.revision ?? 0) &&
			signature.deformationMode === (deformation?.mode ?? "none") &&
			signature.jointPayloadKey === (deformation?.jointPayloadKey ?? null) &&
			signature.morphPayloadKey === (deformation?.morphPayloadKey ?? null) &&
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

	private _touchEntry(entry: PreparedSubmissionCacheEntry): void {
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
	 * Builds camera-visible draw packet lists over an existing prepared scene.
	 *
	 * @internal Pipeline and backend passes use this when they need a secondary
	 * camera view, such as mirrored planar reflection capture, without rebuilding
	 * scene-wide lighting, environment, and shadow metadata from `Scene`.
	 * Application code should normally let `Renderer` prepare scenes.
	 * @param source - Prepared scene whose mesh instances and shared scene state
	 * are reused as the view source.
	 * @param camera - Camera used for frustum culling and packet depth sorting.
	 * @param options - Optional visibility acceleration and occlusion inputs for
	 * the camera-local view.
	 * @returns A prepared scene sharing source scene metadata but with packets
	 * built for `camera`.
	 * @sideEffects None. The source prepared scene is not mutated.
	 */
	public static buildView(
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
		const visibleIds = new Set(
			Array.from(cameraVisibleMeshInstances, (meshInstance) => meshInstance.id),
		);
		const submissions = source.submissions?.length ? source.submissions : Array.from(new Map(
			[
				...source.opaquePackets,
				...source.transparentPackets,
				...source.reflectivePackets,
			].map((packet) => [packet.submission.id, packet.submission]),
		).values());
		for (const submission of submissions) {
			if (
				submission.source.kind === "mesh-instance" &&
				!visibleIds.has(submission.source.instanceId)
			) continue;
			this._appendViewPacket(
				{
					submission,
					sortDepth: resolveSubmissionSortDepth(submission, camera),
				},
				opaquePackets,
				transparentPackets,
				reflectivePackets,
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
		const submissions: DrawSubmission[] = [];
		const submissionIds = new Set<string>();
		const shadowCasterSubmissions: DrawSubmission[] = [];
		const shadowTransmitterSubmissions: DrawSubmission[] = [];

		for (const meshInstance of input.renderableMeshInstances) {
			const visibleInCamera =
				input.cameraVisibleMeshInstances.has(meshInstance);
			this._forEachMeshPacket(
				meshInstance,
				input.camera,
				input.deformationStates,
				input.options.packetCache ?? null,
				(packet) => {
				if (!submissionIds.has(packet.submission.id)) {
					submissionIds.add(packet.submission.id);
					submissions.push(packet.submission);
				}
				if (visibleInCamera) {
					this._appendViewPacket(
						packet,
						opaquePackets,
						transparentPackets,
						reflectivePackets
					);
				}

				if (packet.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_CASTER) {
					shadowCasterPackets.push(packet);
					shadowCasterSubmissions.push(packet.submission);
				}

				if (packet.submission.passFlags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) {
					shadowTransmitterPackets.push(packet);
					shadowTransmitterSubmissions.push(packet.submission);
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
			submissions,
			shadowCasterSubmissions,
			shadowTransmitterSubmissions,
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
		if (packet.submission.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) {
			transparentPackets.push(packet);
		} else {
			opaquePackets.push(packet);
		}
		if (packet.submission.passFlags & DRAW_PACKET_FLAG_REFLECTIVE) {
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
				if (culled.has(opaquePackets[index].submission.id)) {
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
		let source: DrawSourceRef | null = null;
		let instance: DrawInstanceBinding | null = null;

		for (const primitive of meshInstance.mesh.primitives) {
			if (primitive.visible === false) continue;
			const deformation =
				deformationStates?.get(`${meshInstance.id}:${primitive.id}`) ?? null;
			let submission = packetCache?.getReusableSubmission(
				meshInstance,
				primitive,
				deformation,
			) ?? null;
			if (!submission) {
				if (!transformStatePrepared) {
					transformStatePrepared = true;
					normalMatrix = Matrix4.normalMatrix(worldMatrix) as Matrix3Arr;
					worldScale = getMaxScaleFromMatrix(worldMatrix) || 1;
					source = {
						kind: "mesh-instance",
						instanceId: meshInstance.id,
					};
					instance = {
						renderLayers: meshInstance.renderLayers,
						worldMatrix,
						normalMatrix,
					};
				}
				submission = this._createSubmission(
					meshInstance,
					primitive,
					source!,
					instance!,
					worldScale,
					deformation,
				);
				if (packetCache) {
					submission = packetCache.storeSubmission(
						meshInstance,
						primitive,
						deformation,
						submission,
					);
				}
			}
			const sortDepth = resolveSubmissionSortDepth(submission, camera);
			visitor(packetCache
				? packetCache.getViewPacket(camera, submission, sortDepth)
				: { submission, sortDepth });
		}
	}

	private static _createSubmission(
		meshInstance: MeshInstance,
		primitive: IPrimitive,
		source: DrawSourceRef,
		instance: DrawInstanceBinding,
		worldScale: number,
		deformation: PrimitiveDeformationState | null
	): DrawSubmission {
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
			instance.worldMatrix,
			localBounds.center
		);

		return {
			id: `${meshInstance.id}:${primitive.id}`,
			source,
			geometry: resolveGeometryBinding(primitive),
			instance,
			material: {
				effective: material,
				revision: material.revision,
				pipelineKey: createMaterialPipelineKey(material),
			},
			deformation: deformation ? {
				mode: deformation.mode,
				revision: deformation.revision,
				jointPayloadKey: deformation.jointPayloadKey,
				morphPayloadKey: deformation.morphPayloadKey,
			} : NO_DEFORMATION,
			worldBounds: {
				center: {
					x: worldCenter.x,
					y: worldCenter.y,
					z: worldCenter.z,
				},
				radius: localBounds.radius * worldScale,
			},
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
			(packet) =>
				(packet.submission.instance.renderLayers & receiverLayerMask) !== 0
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
	const leftSubmission = left.submission;
	const rightSubmission = right.submission;
	const keyCompare = leftSubmission.material.pipelineKey.localeCompare(
		rightSubmission.material.pipelineKey,
	);
	if (keyCompare !== 0) return keyCompare;

	if (leftSubmission.material.effective !== rightSubmission.material.effective) {
		return leftSubmission.material.effective.name.localeCompare(
			rightSubmission.material.effective.name,
		);
	}

	if (leftSubmission.geometry.resourceKey !== rightSubmission.geometry.resourceKey) {
		return leftSubmission.id.localeCompare(rightSubmission.id);
	}

	return left.sortDepth - right.sortDepth;
}

function resolveSubmissionSortDepth(
	submission: DrawSubmission,
	camera: Camera,
): number {
	const center = submission.worldBounds.center;
	const view = camera.viewMatrix.elements;
	return -(
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

	return left.submission.id.localeCompare(right.submission.id);
}

function resolveGeometryBinding(primitive: IPrimitive): DrawGeometryBinding {
	const topology = primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
	const cached = geometryBindings.get(primitive);
	if (
		cached &&
		cached.data === primitive.geometry &&
		cached.version === (primitive.geometryVersion ?? 0) &&
		cached.topology === topology
	) {
		return cached;
	}
	const binding: DrawGeometryBinding = {
		resourceKey: primitive,
		id: primitive.id,
		data: primitive.geometry,
		version: primitive.geometryVersion ?? 0,
		topology,
	};
	geometryBindings.set(primitive, binding);
	return binding;
}

function createMaterialPipelineKey(material: Material): string {
	return [
		material.type,
		material.shading,
		material.alphaMode ?? AlphaMode.Opaque,
		material.doubleSided ? "double" : "single",
		material.depthWrite ? "depth-write" : "depth-read",
	].join(":");
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
