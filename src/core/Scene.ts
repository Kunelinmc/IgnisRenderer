import { Environment } from "./Environment";
import { Node } from "./Node";
import type { BoundingSphere } from "./types";
import { MeshInstance } from "../meshes";
import { Decal } from "../decals";
import type { SceneLight } from "../lights";
import { Camera } from "../cameras/Camera";
import { ParticleSystem } from "../particles";
import { ECSWorld } from "../ecs";
import {
	BVH,
	HybridSpatialIndex,
	isDynamicSpatialMeshInstance,
	type SpatialIndex3D,
	type SpatialIndexMode,
} from "../spatial";
import { ShadowManager, type ShadowManagerOptions } from "../lights/shadows";
import {
	doesRenderDirtyReasonInvalidateSceneBounds,
	renderDirtyReasonToMask,
	type RenderDirtyReason,
} from "../pipeline/incremental";

const ROOT_PATH = "/sceneRoot";
const DEFAULT_SCENE_BOUNDS_RADIUS = 100;

interface SpatialMeshSignature {
	mesh: MeshInstance["mesh"];
	worldBoundsVersion: number;
	dynamicState: boolean;
}

interface SceneBoundsSignature {
	mesh: MeshInstance["mesh"];
	worldBoundsVersion: number;
	visible: boolean;
}

export interface SceneOptions {
	shadows?: ShadowManagerOptions;
}

export interface SceneNodeLifecycleEvent {
	parent: Node;
	child: Node;
}

export interface SceneNodeLifecycleListener {
	nodeAttached?(event: SceneNodeLifecycleEvent): void;
	nodeDetached?(event: SceneNodeLifecycleEvent): void;
}

export class Scene {
	public readonly root: Node;
	public readonly ecs: ECSWorld;
	public readonly shadows: ShadowManager;
	public readonly environment: Environment;
	public spatial: SpatialIndex3D | null;

	private _version: number;
	private _dirtyReasonMask = 0;
	private _reparentingNodes = new WeakSet<Node>();
	private _sceneGraphDirty = false;
	private _meshInstancesCache: MeshInstance[] = [];
	private _meshInstancesCacheDirty = true;
	private _decalsCache: Decal[] = [];
	private _decalsCacheDirty = true;
	private _boundsDirty = true;
	private _boundsCache: BoundingSphere = {
		center: { x: 0, y: 0, z: 0 },
		radius: DEFAULT_SCENE_BOUNDS_RADIUS,
	};
	private _boundsScratch: BoundingSphere = {
		center: { x: 0, y: 0, z: 0 },
		radius: 0,
	};
	private _boundsSignaturesByMeshInstance = new Map<
		MeshInstance,
		SceneBoundsSignature
	>();
	private _spatialTrackedMeshInstances = new Set<MeshInstance>();
	private _spatialSignaturesByMeshInstance = new Map<MeshInstance, SpatialMeshSignature>();
	private _spatialSeenEpochByMeshInstance = new Map<MeshInstance, number>();
	private _spatialSeenEpoch = 0;
	private _spatialIndexMode: SpatialIndexMode;
	private _spatialQueryScratch: MeshInstance[] = [];
	private _nodeLifecycleListeners = new Set<SceneNodeLifecycleListener>();

	constructor(options: SceneOptions = {}) {
		this.root = new Node({
			idPrefix: "scene",
			name: "sceneRoot",
		});
		this.ecs = new ECSWorld();
		this.shadows = new ShadowManager(options.shadows);
		this.environment = new Environment();
		this.environment.on("change", () => {
			this.invalidate("unknown");
		});
		this.spatial = null;
		this._version = 0;
		this._spatialIndexMode = "bvh";

		this.root._setSceneInternal(this);
		const rootEntity = this.ecs.registerNode(this.root, null);
		this.root._setEntityIdInternal(rootEntity);
		this.ecs.setHierarchy(rootEntity, null, []);
		this.ecs.setComponent(rootEntity, "PathBinding", {
			path: ROOT_PATH,
		});
	}

	public add<T extends Node>(node: T): T {
		this.root.addChild(node);
		return node;
	}

	public remove(node: Node): boolean {
		return node.parent ? node.parent.removeChild(node) : false;
	}

	/**
	 * Subscribes to public scene graph attach and detach events.
	 *
	 * @param listener - Listener invoked after public attach or detach calls update
	 * scene ownership and ECS bindings.
	 * @returns A disposer that removes the listener.
	 * @sideEffects Stores the listener until the returned disposer is called.
	 */
	public addNodeLifecycleListener(
		listener: SceneNodeLifecycleListener
	): () => void {
		this._nodeLifecycleListeners.add(listener);
		return () => {
			this._nodeLifecycleListeners.delete(listener);
		};
	}

	public clear(): void {
		if (this.root.children.length === 0) return;
		for (const child of [...this.root.children]) {
			this.root.removeChild(child);
		}
	}

	public contains(node: Node): boolean {
		let current: Node | null = node;
		while (current) {
			if (current === this.root) return true;
			current = current.parent;
		}
		return false;
	}

	public traverse(visitor: (node: Node) => void): void {
		for (const child of this.root.children) {
			child.traverse(visitor);
		}
	}

	public getMeshInstances(): MeshInstance[] {
		if (this._meshInstancesCacheDirty) {
			this._meshInstancesCache = this.ecs.findMeshInstances();
			this._meshInstancesCacheDirty = false;
		}
		return this._meshInstancesCache;
	}

	public getDecals(): Decal[] {
		if (this._decalsCacheDirty) {
			this._decalsCache = this._collectByType(
				(node): node is Decal => node instanceof Decal
			);
			this._decalsCacheDirty = false;
		}
		return this._decalsCache;
	}

	public getLights(): SceneLight[] {
		return this.ecs.findLights();
	}

	public getCameras(): Camera[] {
		return this.ecs.findCameras();
	}

	public getParticleSystems(): ParticleSystem[] {
		return this.ecs.findParticleSystems();
	}

	public get spatialIndexMode(): SpatialIndexMode {
		return this._spatialIndexMode;
	}

	public set spatialIndexMode(mode: SpatialIndexMode) {
		this.setSpatialIndexMode(mode);
	}

	public setSpatialIndexMode(mode: SpatialIndexMode): void {
		if (mode !== "bvh" && mode !== "hybrid") {
			throw new Error(`Unsupported spatial index mode: ${mode}`);
		}
		if (this._spatialIndexMode === mode) return;
		this._spatialIndexMode = mode;
		this.spatial = null;
		this._spatialTrackedMeshInstances.clear();
		this._spatialSignaturesByMeshInstance.clear();
		this._spatialSeenEpochByMeshInstance.clear();
		this.invalidate("unknown");
	}

	public rebuildSpatialIndex(meshInstances: MeshInstance[]): SpatialIndex3D {
		if (!this.spatial) {
			const spatial = this._createSpatialIndex(meshInstances);
			this.spatial = spatial;
			this._spatialTrackedMeshInstances = new Set(meshInstances);
			this._spatialSignaturesByMeshInstance.clear();
			this._spatialSeenEpochByMeshInstance.clear();
			const seenEpoch = ++this._spatialSeenEpoch;
			for (const meshInstance of meshInstances) {
				this._spatialSignaturesByMeshInstance.set(
					meshInstance,
					createSpatialMeshSignature(meshInstance),
				);
				this._spatialSeenEpochByMeshInstance.set(meshInstance, seenEpoch);
			}
			return spatial;
		}

		const spatial = this.spatial;
		const seenEpoch = ++this._spatialSeenEpoch;
		let requiresRemovalScan = this._spatialTrackedMeshInstances.size !== meshInstances.length;

		for (const meshInstance of meshInstances) {
			if (this._spatialSeenEpochByMeshInstance.get(meshInstance) === seenEpoch) {
				requiresRemovalScan = true;
				continue;
			}
			this._spatialSeenEpochByMeshInstance.set(meshInstance, seenEpoch);
			if (!this._spatialTrackedMeshInstances.has(meshInstance)) {
				requiresRemovalScan = true;
				this._spatialTrackedMeshInstances.add(meshInstance);
				this._spatialSignaturesByMeshInstance.set(
					meshInstance,
					createSpatialMeshSignature(meshInstance),
				);
				spatial.upsert(meshInstance);
				continue;
			}

			const signature = this._spatialSignaturesByMeshInstance.get(meshInstance);
			if (!signature) {
				this._spatialSignaturesByMeshInstance.set(
					meshInstance,
					createSpatialMeshSignature(meshInstance),
				);
				spatial.upsert(meshInstance);
				continue;
			}

			if (updateSpatialMeshSignature(signature, meshInstance)) {
				spatial.markDirty(meshInstance);
			}
		}

		if (requiresRemovalScan) {
			for (const tracked of [...this._spatialTrackedMeshInstances]) {
				if (this._spatialSeenEpochByMeshInstance.get(tracked) === seenEpoch) {
					continue;
				}
				spatial.remove(tracked);
				this._spatialTrackedMeshInstances.delete(tracked);
				this._spatialSignaturesByMeshInstance.delete(tracked);
				this._spatialSeenEpochByMeshInstance.delete(tracked);
			}
		}

		return spatial;
	}

	public queryMeshInstancesInFrustum(
		camera: Camera,
		meshInstances: MeshInstance[],
	): MeshInstance[] {
		this.queryMeshInstancesInFrustumInto(
			camera,
			meshInstances,
			this._spatialQueryScratch,
		);
		return this._spatialQueryScratch.slice();
	}

	/**
	 * Writes mesh instances whose world-space bounds overlap `camera.frustum`.
	 *
	 * @param camera - Camera providing the frustum used for culling.
	 * @param meshInstances - Candidate mesh instances to synchronize into the scene spatial index.
	 * @param out - Output array cleared and filled with visible candidates.
	 * @returns The same `out` array for call chaining.
	 * @sideEffects Lazily rebuilds or updates `this.spatial` to match `meshInstances`.
	 */
	public queryMeshInstancesInFrustumInto(
		camera: Camera,
		meshInstances: MeshInstance[],
		out: MeshInstance[],
	): MeshInstance[] {
		const spatial = this.rebuildSpatialIndex(meshInstances);
		return spatial.queryFrustumInto(camera.frustum, out);
	}

	public updateWorldMatrices(): void {
		this.root.updateWorldMatrix();
		this.syncNodeToECS();
	}

	public syncNodeToECS(): void {
		if (this._sceneGraphDirty) {
			this._syncSceneGraphToECS();
			return;
		}
		this._syncNodeStateRecursive(this.root, ROOT_PATH);
		if (this._sceneGraphDirty) {
			this._syncSceneGraphToECS();
		}
	}

	private _syncSceneGraphToECS(): void {
		const activeNodes = new Set<Node>();
		const rootEntity = this.root.entityId;
		if (rootEntity === null) {
			throw new Error("Scene root entity is missing");
		}
		this._syncNodeRecursive(this.root, null, ROOT_PATH, activeNodes);

		const entities = this.ecs.query(["NodeRef"]);
		for (const entity of entities) {
			const node = this.ecs.getNodeByEntity(entity);
			if (!node || !activeNodes.has(node)) {
				this.ecs.destroyEntity(entity);
			}
		}
		this._sceneGraphDirty = false;
		this._meshInstancesCacheDirty = true;
		this._decalsCacheDirty = true;
		this._boundsDirty = true;
	}

	public syncECSToNode(): void {
		const entities = this.ecs.query(["NodeRef", "LocalTransform"]);
		for (const entity of entities) {
			this.ecs.syncEntityToNode(entity);
		}
	}

	public markNodeReparenting(node: Node, active: boolean): void {
		if (active) {
			this._reparentingNodes.add(node);
			return;
		}
		this._reparentingNodes.delete(node);
	}

	public onNodeAttachedFromAPI(parent: Node, child: Node): void {
		if (parent.scene !== this) return;
		this._setSceneRecursive(child, this);
		this._sceneGraphDirty = true;
		this._meshInstancesCacheDirty = true;
		this._decalsCacheDirty = true;
		this._boundsDirty = true;
		this.syncNodeToECS();
		this.invalidate();
		for (const listener of this._nodeLifecycleListeners) {
			listener.nodeAttached?.({ parent, child });
		}
	}

	public onNodeDetachedFromAPI(_parent: Node, child: Node): void {
		if (this._reparentingNodes.has(child)) {
			this._sceneGraphDirty = true;
			this._meshInstancesCacheDirty = true;
			this._decalsCacheDirty = true;
			this._boundsDirty = true;
			this.invalidate();
			return;
		}

		this._unregisterNodeRecursive(child);
		this._setSceneRecursive(child, null);
		this._sceneGraphDirty = true;
		this._meshInstancesCacheDirty = true;
		this._decalsCacheDirty = true;
		this._boundsDirty = true;
		this.invalidate();
		for (const listener of this._nodeLifecycleListeners) {
			listener.nodeDetached?.({ parent: _parent, child });
		}
	}

	public invalidate(reason: RenderDirtyReason = "unknown"): void {
		this._version++;
		this._dirtyReasonMask |= renderDirtyReasonToMask(reason);
		if (shouldInvalidateSceneBounds(reason)) {
			this._boundsDirty = true;
		}
	}

	public get version(): number {
		return this._version;
	}

	public get dirtyReasonMask(): number {
		return this._dirtyReasonMask;
	}

	public consumeDirtyReasonMask(): number {
		const mask = this._dirtyReasonMask;
		this._dirtyReasonMask = 0;
		return mask;
	}

	public getBounds(): BoundingSphere {
		if (!this._boundsDirty && this._haveSceneBoundsInputsChanged()) {
			this._boundsDirty = true;
		}

		if (this._boundsDirty) {
			let minX = Infinity;
			let minY = Infinity;
			let minZ = Infinity;
			let maxX = -Infinity;
			let maxY = -Infinity;
			let maxZ = -Infinity;

			for (const meshInstance of this.getMeshInstances()) {
				if (meshInstance.visible === false) continue;
				const worldBounds = meshInstance.getWorldBoundingSphere(this._boundsScratch);
				const center = worldBounds.center;
				const radius = worldBounds.radius;
				minX = Math.min(minX, center.x - radius);
				minY = Math.min(minY, center.y - radius);
				minZ = Math.min(minZ, center.z - radius);
				maxX = Math.max(maxX, center.x + radius);
				maxY = Math.max(maxY, center.y + radius);
				maxZ = Math.max(maxZ, center.z + radius);
			}

			if (minX === Infinity) {
				this._boundsCache.center.x = 0;
				this._boundsCache.center.y = 0;
				this._boundsCache.center.z = 0;
				this._boundsCache.radius = DEFAULT_SCENE_BOUNDS_RADIUS;
			} else {
				const centerX = (minX + maxX) * 0.5;
				const centerY = (minY + maxY) * 0.5;
				const centerZ = (minZ + maxZ) * 0.5;
				const sizeX = maxX - minX;
				const sizeY = maxY - minY;
				const sizeZ = maxZ - minZ;
				this._boundsCache.center.x = centerX;
				this._boundsCache.center.y = centerY;
				this._boundsCache.center.z = centerZ;
				this._boundsCache.radius =
					Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ) * 0.5;
			}
			this._boundsDirty = false;
			this._syncSceneBoundsSignatures();
		}

		return {
			center: {
				x: this._boundsCache.center.x,
				y: this._boundsCache.center.y,
				z: this._boundsCache.center.z,
			},
			radius: this._boundsCache.radius,
		};
	}

	private _haveSceneBoundsInputsChanged(): boolean {
		const meshInstances = this.getMeshInstances();
		if (this._boundsSignaturesByMeshInstance.size !== meshInstances.length) {
			return true;
		}
		for (const meshInstance of meshInstances) {
			const signature = this._boundsSignaturesByMeshInstance.get(meshInstance);
			if (
				!signature ||
				signature.mesh !== meshInstance.mesh ||
				signature.visible !== meshInstance.visible ||
				signature.worldBoundsVersion !== meshInstance.worldBoundsVersion
			) {
				return true;
			}
		}
		return false;
	}

	private _syncSceneBoundsSignatures(): void {
		this._boundsSignaturesByMeshInstance.clear();
		for (const meshInstance of this.getMeshInstances()) {
			this._boundsSignaturesByMeshInstance.set(meshInstance, {
				mesh: meshInstance.mesh,
				worldBoundsVersion: meshInstance.worldBoundsVersion,
				visible: meshInstance.visible,
			});
		}
	}

	private _collectByType<T extends Node>(predicate: (node: Node) => node is T): T[] {
		const result: T[] = [];
		this.traverse((node) => {
			if (predicate(node)) {
				result.push(node);
			}
		});
		return result;
	}

	private _syncNodeRecursive(
		node: Node,
		parentEntity: number | null,
		path: string,
		activeNodes: Set<Node>,
	): number {
		activeNodes.add(node);
		const entity = this.ecs.registerNode(node, parentEntity);
		node._setEntityIdInternal(entity);
		node._setSceneInternal(this);
		this.ecs.setExternalId(entity, node.id);
		this.ecs.setComponent(entity, "PathBinding", { path });

		const childEntities: number[] = [];
		for (const child of node.children) {
			const childEntity = this._syncNodeRecursive(
				child,
				entity,
				`${path}/${sanitizePathSegment(child.name)}_${child.id}`,
				activeNodes,
			);
			childEntities.push(childEntity);
		}
		this.ecs.setHierarchy(entity, parentEntity, childEntities);
		return entity;
	}

	private _syncNodeStateRecursive(node: Node, path: string): void {
		const entity = node.entityId;
		if (entity === null || !this.ecs.hasEntity(entity)) {
			this._sceneGraphDirty = true;
			return;
		}
		node._setSceneInternal(this);
		this.ecs.syncNodeToEntity(node, path);
		for (const child of node.children) {
			this._syncNodeStateRecursive(
				child,
				`${path}/${sanitizePathSegment(child.name)}_${child.id}`,
			);
		}
	}

	private _unregisterNodeRecursive(node: Node): void {
		for (const child of node.children) {
			this._unregisterNodeRecursive(child);
		}
		if (node.entityId !== null) {
			this.ecs.unregisterNode(node);
			node._setEntityIdInternal(null);
		}
	}

	private _setSceneRecursive(node: Node, scene: Scene | null): void {
		node._setSceneInternal(scene);
		for (const child of node.children) {
			this._setSceneRecursive(child, scene);
		}
	}

	private _createSpatialIndex(meshInstances: MeshInstance[]): SpatialIndex3D {
		if (this._spatialIndexMode === "hybrid") {
			return new HybridSpatialIndex(meshInstances);
		}
		return new BVH(meshInstances);
	}
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}

function shouldInvalidateSceneBounds(reason: RenderDirtyReason): boolean {
	return doesRenderDirtyReasonInvalidateSceneBounds(reason);
}

function createSpatialMeshSignature(
	meshInstance: MeshInstance
): SpatialMeshSignature {
	return {
		mesh: meshInstance.mesh,
		worldBoundsVersion: meshInstance.worldBoundsVersion,
		dynamicState: isDynamicSpatialMeshInstance(meshInstance),
	};
}

function updateSpatialMeshSignature(
	signature: SpatialMeshSignature,
	meshInstance: MeshInstance
): boolean {
	let changed = signature.mesh !== meshInstance.mesh;
	signature.mesh = meshInstance.mesh;
	const worldBoundsVersion = meshInstance.worldBoundsVersion;
	if (!changed && signature.worldBoundsVersion !== worldBoundsVersion) {
		changed = true;
	}
	signature.worldBoundsVersion = worldBoundsVersion;

	const dynamicState = isDynamicSpatialMeshInstance(meshInstance);
	if (!changed && signature.dynamicState !== dynamicState) {
		changed = true;
	}
	signature.dynamicState = dynamicState;

	return changed;
}
