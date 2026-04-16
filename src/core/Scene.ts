import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { Texture } from "./Texture";
import { Node } from "./Node";
import type { BoundingSphere } from "./types";
import { MeshInstance } from "../meshes";
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
import {
	renderDirtyReasonToMask,
	type RenderDirtyReason,
} from "../pipeline/incremental";

const ROOT_PATH = "/sceneRoot";
const SPATIAL_MATRIX_EPSILON = 1e-8;

interface SpatialMeshSignature {
	mesh: MeshInstance["mesh"];
	matrix: Float32Array;
	dynamicState: boolean;
}

export class Scene {
	public readonly root: Node;
	public readonly ecs: ECSWorld;
	public skybox: Texture | null;
	public spatial: SpatialIndex3D | null;

	private _version: number;
	private _dirtyReasonMask = 0;
	private _reparentingNodes = new WeakSet<Node>();
	private _spatialTrackedMeshInstances = new Set<MeshInstance>();
	private _spatialSignaturesByMeshInstance = new Map<MeshInstance, SpatialMeshSignature>();
	private _spatialIndexMode: SpatialIndexMode;

	constructor() {
		this.root = new Node({
			idPrefix: "scene",
			name: "sceneRoot",
		});
		this.ecs = new ECSWorld();
		this.skybox = null;
		this.spatial = null;
		this._version = 0;
		this._spatialIndexMode = "bvh";

		this.root._scene = this;
		const rootEntity = this.ecs.registerNode(this.root, null);
		this.root._entityId = rootEntity;
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

	public clear(): void {
		if (this.root.children.length === 0) return;
		for (const child of [...this.root.children]) {
			this.root.removeChild(child);
		}
	}

	public contains(node: Node): boolean {
		if (node === this.root) return true;
		let found = false;
		this.traverse((current) => {
			if (current === node) {
				found = true;
			}
		});
		return found;
	}

	public traverse(visitor: (node: Node) => void): void {
		for (const child of this.root.children) {
			child.traverse(visitor);
		}
	}

	public getMeshInstances(): MeshInstance[] {
		return this.ecs.findMeshInstances();
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
		this.invalidate("unknown");
	}

	public rebuildSpatialIndex(meshInstances: MeshInstance[]): SpatialIndex3D {
		if (!this.spatial) {
			const spatial = this._createSpatialIndex(meshInstances);
			this.spatial = spatial;
			this._spatialTrackedMeshInstances = new Set(meshInstances);
			this._spatialSignaturesByMeshInstance.clear();
			for (const meshInstance of meshInstances) {
				this._spatialSignaturesByMeshInstance.set(
					meshInstance,
					createSpatialMeshSignature(meshInstance),
				);
			}
			return spatial;
		}

		const spatial = this.spatial;
		const nextMeshSet = new Set(meshInstances);
		const removedMeshInstances: MeshInstance[] = [];
		for (const tracked of this._spatialTrackedMeshInstances) {
			if (!nextMeshSet.has(tracked)) {
				removedMeshInstances.push(tracked);
			}
		}

		for (const removed of removedMeshInstances) {
			spatial.remove(removed);
			this._spatialTrackedMeshInstances.delete(removed);
			this._spatialSignaturesByMeshInstance.delete(removed);
		}

		for (const meshInstance of meshInstances) {
			if (!this._spatialTrackedMeshInstances.has(meshInstance)) {
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

		return spatial;
	}

	public queryMeshInstancesInFrustum(
		camera: Camera,
		meshInstances: MeshInstance[],
	): MeshInstance[] {
		const spatial = this.rebuildSpatialIndex(meshInstances);
		return spatial.queryFrustum(camera.frustum);
	}

	public updateWorldMatrices(): void {
		this.root.updateWorldMatrix();
		this.syncNodeToECS();
	}

	public syncNodeToECS(): void {
		const activeNodes = new Set<Node>();
		const rootEntity = this.root._entityId;
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
		if (parent._scene !== this) return;
		this._setSceneRecursive(child, this);
		this.syncNodeToECS();
		this.invalidate();
	}

	public onNodeDetachedFromAPI(_parent: Node, child: Node): void {
		if (this._reparentingNodes.has(child)) {
			this.invalidate();
			return;
		}

		this._unregisterNodeRecursive(child);
		this._setSceneRecursive(child, null);
		this.invalidate();
	}

	public invalidate(reason: RenderDirtyReason = "unknown"): void {
		this._version++;
		this._dirtyReasonMask |= renderDirtyReasonToMask(reason);
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
		let min: IVector3 = { x: Infinity, y: Infinity, z: Infinity };
		let max: IVector3 = { x: -Infinity, y: -Infinity, z: -Infinity };

		for (const meshInstance of this.getMeshInstances()) {
			if (meshInstance.visible === false) continue;
			const worldBounds = meshInstance.getWorldBoundingSphere();
			const center = worldBounds.center;
			const radius = worldBounds.radius;

			min.x = Math.min(min.x, center.x - radius);
			min.y = Math.min(min.y, center.y - radius);
			min.z = Math.min(min.z, center.z - radius);
			max.x = Math.max(max.x, center.x + radius);
			max.y = Math.max(max.y, center.y + radius);
			max.z = Math.max(max.z, center.z + radius);
		}

		if (min.x === Infinity) {
			return { center: { x: 0, y: 0, z: 0 }, radius: 100 };
		}

		const center: IVector3 = {
			x: (min.x + max.x) / 2,
			y: (min.y + max.y) / 2,
			z: (min.z + max.z) / 2,
		};
		const size: IVector3 = {
			x: max.x - min.x,
			y: max.y - min.y,
			z: max.z - min.z,
		};
		const radius = Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2;

		return { center, radius };
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
		node._entityId = entity;
		node._scene = this;
		this.ecs.setExternalId(entity, node.id);
		this.ecs.setComponent(entity, "PathBinding", { path });
		this.ecs.syncNodeToEntity(node, path);

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

	private _unregisterNodeRecursive(node: Node): void {
		for (const child of node.children) {
			this._unregisterNodeRecursive(child);
		}
		if (node._entityId !== null) {
			this.ecs.unregisterNode(node);
			node._entityId = null;
		}
	}

	private _setSceneRecursive(node: Node, scene: Scene | null): void {
		node._scene = scene;
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

function hasLightType(value: unknown): value is SceneLight {
	if (!value || typeof value !== "object") return false;
	return "type" in value && "intensity" in value && "color" in value;
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}

function createSpatialMeshSignature(
	meshInstance: MeshInstance
): SpatialMeshSignature {
	return {
		mesh: meshInstance.mesh,
		matrix: captureWorldMatrix(meshInstance.worldMatrix),
		dynamicState: isDynamicSpatialMeshInstance(meshInstance),
	};
}

function updateSpatialMeshSignature(
	signature: SpatialMeshSignature,
	meshInstance: MeshInstance
): boolean {
	let changed = signature.mesh !== meshInstance.mesh;
	signature.mesh = meshInstance.mesh;
	const dynamicState = isDynamicSpatialMeshInstance(meshInstance);
	if (!changed && signature.dynamicState !== dynamicState) {
		changed = true;
	}
	signature.dynamicState = dynamicState;

	const elements = meshInstance.worldMatrix.elements;
	const matrix = signature.matrix;
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			const value = elements[row][column];
			if (
				!changed &&
				Math.abs(matrix[cursor] - value) > SPATIAL_MATRIX_EPSILON
			) {
				changed = true;
			}
			matrix[cursor] = value;
			cursor++;
		}
	}
	return changed;
}

function captureWorldMatrix(matrix: Matrix4): Float32Array {
	const result = new Float32Array(16);
	const elements = matrix.elements;
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let column = 0; column < 4; column++) {
			result[cursor++] = elements[row][column];
		}
	}
	return result;
}
