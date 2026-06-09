import { Camera } from "../cameras/Camera";
import { Matrix4 } from "../maths/Matrix4";
import { MeshInstance } from "../meshes";
import { Decal } from "../decals";
import { ParticleSystem } from "../particles";
import type { Node } from "../core/Node";
import type { SceneLight } from "../lights";
import { Light } from "../lights";
import { ComponentStore } from "./ComponentStore";
import {
	NODE_KIND,
	type ECSComponentMap,
	type ECSComponentName,
	type EntityId,
	type HierarchyComponent,
	type InteractableComponent,
	type LocalTransformComponent,
	type NameComponent,
	type NodeKind,
	type NodeKindComponent,
	type NodeRefComponent,
	type PathBindingComponent,
	type SkeletonJointComponent,
	type VisibilityComponent,
	type WorldTransformComponent,
} from "./components";

interface QueryCacheEntry {
	version: number;
	entities: EntityId[];
}

interface NodeKindLookupMap {
	[NODE_KIND.Node]: Node;
	[NODE_KIND.MeshInstance]: MeshInstance;
	[NODE_KIND.Decal]: Decal;
	[NODE_KIND.Camera]: Camera;
	[NODE_KIND.ParticleSystem]: ParticleSystem;
	[NODE_KIND.Light]: SceneLight;
}

const COMPONENT_NAMES: ECSComponentName[] = [
	"Name",
	"Visibility",
	"LocalTransform",
	"WorldTransform",
	"Hierarchy",
	"PathBinding",
	"SkeletonJoint",
	"NodeRef",
	"NodeKind",
	"Interactable",
];

export class ECSWorld {
	private _nextEntityId = 1;
	private _entityIds = new Set<EntityId>();
	private _version = 0;
	private _stores: {
		[K in ECSComponentName]: ComponentStore<ECSComponentMap[K]>;
	};
	private _queryCache = new Map<string, QueryCacheEntry>();
	private _entityByExternalId = new Map<string, EntityId>();
	private _externalIdByEntity = new Map<EntityId, string>();
	private _entityByNode = new WeakMap<Node, EntityId>();
	private _nodeByEntity = new Map<EntityId, Node>();

	constructor() {
		this._stores = {
			Name: new ComponentStore<NameComponent>(),
			Visibility: new ComponentStore<VisibilityComponent>(),
			LocalTransform: new ComponentStore<LocalTransformComponent>(),
			WorldTransform: new ComponentStore<WorldTransformComponent>(),
			Hierarchy: new ComponentStore<HierarchyComponent>(),
			PathBinding: new ComponentStore<PathBindingComponent>(),
			SkeletonJoint: new ComponentStore<SkeletonJointComponent>(),
			NodeRef: new ComponentStore<NodeRefComponent>(),
			NodeKind: new ComponentStore<NodeKindComponent>(),
			Interactable: new ComponentStore<InteractableComponent>(),
		};
	}

	public get version(): number {
		return this._version;
	}

	public createEntity(externalId?: string): EntityId {
		const entity = this._nextEntityId++;
		this._entityIds.add(entity);
		if (externalId) {
			this._entityByExternalId.set(externalId, entity);
			this._externalIdByEntity.set(entity, externalId);
		}
		this._bumpVersion();
		return entity;
	}

	public destroyEntity(entity: EntityId): boolean {
		if (!this._entityIds.has(entity)) return false;

		this._entityIds.delete(entity);
		for (const name of COMPONENT_NAMES) {
			this._stores[name].delete(entity);
		}
		const externalId = this._externalIdByEntity.get(entity);
		if (externalId) {
			this._externalIdByEntity.delete(entity);
			this._entityByExternalId.delete(externalId);
		}
		this._nodeByEntity.delete(entity);
		this._bumpVersion();
		return true;
	}

	public hasEntity(entity: EntityId): boolean {
		return this._entityIds.has(entity);
	}

	public setExternalId(entity: EntityId, externalId: string): void {
		const current = this._externalIdByEntity.get(entity);
		if (current === externalId) return;
		if (current) {
			this._entityByExternalId.delete(current);
		}
		this._externalIdByEntity.set(entity, externalId);
		this._entityByExternalId.set(externalId, entity);
		this._bumpVersion();
	}

	public getEntityByExternalId(externalId: string): EntityId | null {
		return this._entityByExternalId.get(externalId) ?? null;
	}

	public getExternalId(entity: EntityId): string | null {
		return this._externalIdByEntity.get(entity) ?? null;
	}

	public setComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
		value: ECSComponentMap[K]
	): void {
		if (!this._entityIds.has(entity)) {
			throw new Error(
				`ECSWorld.setComponent target entity "${entity}" not found`
			);
		}
		this._stores[name].set(entity, value);
		this._bumpVersion();
	}

	public getComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K
	): ECSComponentMap[K] | undefined {
		return this._stores[name].get(entity);
	}

	public hasComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K
	): boolean {
		return this._stores[name].has(entity);
	}

	public removeComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K
	): boolean {
		const removed = this._stores[name].delete(entity);
		if (removed) {
			this._bumpVersion();
		}
		return removed;
	}

	public query(required: ECSComponentName[]): EntityId[] {
		if (required.length === 0) return Array.from(this._entityIds);
		const key = required.slice().sort().join("|");
		const cached = this._queryCache.get(key);
		if (cached && cached.version === this._version) {
			return cached.entities.slice();
		}

		const sorted = required
			.slice()
			.sort(
				(a, b) =>
					this._stores[a].entities().length - this._stores[b].entities().length
			);
		const base = sorted[0];
		const entities = this._stores[base].entities();
		const result: EntityId[] = [];
		for (let i = 0; i < entities.length; i++) {
			const entity = entities[i];
			let matches = true;
			for (let j = 1; j < sorted.length; j++) {
				if (!this._stores[sorted[j]].has(entity)) {
					matches = false;
					break;
				}
			}
			if (matches) {
				result.push(entity);
			}
		}

		this._queryCache.set(key, {
			version: this._version,
			entities: result.slice(),
		});
		return result;
	}

	public registerNode(node: Node, parent: EntityId | null): EntityId {
		const existing = this._entityByNode.get(node);
		const entity = existing ?? this.createEntity(node.id);
		if (!existing) {
			this._entityByNode.set(node, entity);
			this._nodeByEntity.set(entity, node);
		}

		this.setComponent(entity, "Name", { value: node.name });
		this.setComponent(entity, "Visibility", { visible: node.visible });
		this.setComponent(
			entity,
			"LocalTransform",
			this._createLocalTransform(node)
		);
		this.setComponent(entity, "WorldTransform", {
			matrix: node.worldMatrix.clone(),
		});
		this.setComponent(entity, "Hierarchy", {
			parent,
			children: [],
		});
		this.setComponent(entity, "NodeRef", { node });
		this.setComponent(entity, "NodeKind", { kind: resolveNodeKind(node) });
		return entity;
	}

	public unregisterNode(node: Node): boolean {
		const entity = this._entityByNode.get(node);
		if (entity === undefined) return false;
		this._entityByNode.delete(node);
		this._nodeByEntity.delete(entity);
		return this.destroyEntity(entity);
	}

	public getEntityByNode(node: Node): EntityId | null {
		return this._entityByNode.get(node) ?? null;
	}

	public getNodeByEntity(entity: EntityId): Node | null {
		return this._nodeByEntity.get(entity) ?? null;
	}

	public syncNodeToEntity(node: Node, path?: string): void {
		const entity = this._entityByNode.get(node);
		if (entity === undefined) return;
		this.setComponent(entity, "Name", { value: node.name });
		this.setComponent(entity, "Visibility", { visible: node.visible });
		this.setComponent(
			entity,
			"LocalTransform",
			this._createLocalTransform(node)
		);
		this.setComponent(entity, "WorldTransform", {
			matrix: node.worldMatrix.clone(),
		});
		if (path !== undefined) {
			this.setComponent(entity, "PathBinding", { path });
		}
	}

	public syncEntityToNode(entity: EntityId): void {
		const node = this._nodeByEntity.get(entity);
		if (!node) return;
		const local = this.getComponent(entity, "LocalTransform");
		const world = this.getComponent(entity, "WorldTransform");
		const visibility = this.getComponent(entity, "Visibility");
		const name = this.getComponent(entity, "Name");
		if (local) {
			node.position.set(local.positionX, local.positionY, local.positionZ);
			node.quaternion.x = local.rotationX;
			node.quaternion.y = local.rotationY;
			node.quaternion.z = local.rotationZ;
			node.quaternion.w = local.rotationW;
			node.scale.set(local.scaleX, local.scaleY, local.scaleZ);
			node.updateLocalMatrix();
		}
		if (world) {
			world.matrix.copyTo(node.worldMatrix);
		}
		if (visibility) {
			node.visible = visibility.visible;
		}
		if (name) {
			node.name = name.value;
		}
	}

	public setHierarchy(
		entity: EntityId,
		parent: EntityId | null,
		children: EntityId[]
	): void {
		this.setComponent(entity, "Hierarchy", { parent, children });
	}

	public findMeshInstances(): MeshInstance[] {
		return this._findNodesByKind(
			NODE_KIND.MeshInstance,
			(node): node is MeshInstance => node instanceof MeshInstance
		);
	}

	public findDecals(): Decal[] {
		return this._findNodesByKind(
			NODE_KIND.Decal,
			(node): node is Decal => node instanceof Decal
		);
	}

	public findLights(): SceneLight[] {
		return this._findNodesByKind(
			NODE_KIND.Light,
			(node): node is SceneLight => node instanceof Light
		);
	}

	public findCameras(): Camera[] {
		return this._findNodesByKind(
			NODE_KIND.Camera,
			(node): node is Camera => node instanceof Camera
		);
	}

	public findParticleSystems(): ParticleSystem[] {
		return this._findNodesByKind(
			NODE_KIND.ParticleSystem,
			(node): node is ParticleSystem => node instanceof ParticleSystem
		);
	}

	private _findNodesByKind<K extends NodeKind>(
		expectedKind: K,
		isExpectedNode: (node: Node) => node is NodeKindLookupMap[K]
	): NodeKindLookupMap[K][] {
		const entities = this.query(["NodeRef", "NodeKind"]);
		const result: NodeKindLookupMap[K][] = [];
		for (const entity of entities) {
			const nodeKind = this.getComponent(entity, "NodeKind");
			if (!nodeKind || nodeKind.kind !== expectedKind) continue;
			const nodeRef = this.getComponent(entity, "NodeRef");
			if (!nodeRef) continue;
			if (isExpectedNode(nodeRef.node)) {
				result.push(nodeRef.node);
			}
		}
		return result;
	}

	private _createLocalTransform(node: Node): LocalTransformComponent {
		return {
			positionX: node.position.x,
			positionY: node.position.y,
			positionZ: node.position.z,
			rotationX: node.quaternion.x,
			rotationY: node.quaternion.y,
			rotationZ: node.quaternion.z,
			rotationW: node.quaternion.w,
			scaleX: node.scale.x,
			scaleY: node.scale.y,
			scaleZ: node.scale.z,
		};
	}

	private _bumpVersion(): void {
		this._version++;
	}
}

function resolveNodeKind(node: Node): NodeKind {
	if (node instanceof MeshInstance) return NODE_KIND.MeshInstance;
	if (node instanceof Decal) return NODE_KIND.Decal;
	if (node instanceof Camera) return NODE_KIND.Camera;
	if (node instanceof ParticleSystem) return NODE_KIND.ParticleSystem;
	if (node instanceof Light) return NODE_KIND.Light;
	return NODE_KIND.Node;
}
