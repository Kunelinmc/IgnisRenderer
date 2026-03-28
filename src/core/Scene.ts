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
import { BVH } from "../spatial/BVH";

const ROOT_PATH = "/sceneRoot";

export class Scene {
	/** @deprecated Scene is now a compatibility facade over ECSWorld. */
	public readonly root: Node;
	public readonly ecs: ECSWorld;
	public skybox: Texture | null;
	public spatial: BVH | null;

	private _version: number;
	private _reparentingNodes = new WeakSet<Node>();

	constructor() {
		this.root = new Node({
			idPrefix: "scene",
			name: "sceneRoot",
		});
		this.ecs = new ECSWorld();
		this.skybox = null;
		this.spatial = null;
		this._version = 0;

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

	public rebuildSpatialIndex(meshInstances: MeshInstance[]): BVH {
		const spatial = this.spatial ?? new BVH();
		spatial.rebuild(meshInstances);
		this.spatial = spatial;
		return spatial;
	}

	public queryMeshInstancesInFrustum(
		camera: Camera,
		meshInstances: MeshInstance[]
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

	public invalidate(): void {
		this._version++;
	}

	public get version(): number {
		return this._version;
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
		const radius =
			Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2;

		return { center, radius };
	}

	private _collectByType<T extends Node>(
		predicate: (node: Node) => node is T
	): T[] {
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
		activeNodes: Set<Node>
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
				activeNodes
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
}

function hasLightType(value: unknown): value is SceneLight {
	if (!value || typeof value !== "object") return false;
	return "type" in value && "intensity" in value && "color" in value;
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}
