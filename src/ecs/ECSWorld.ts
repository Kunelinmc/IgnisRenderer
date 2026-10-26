import { Camera } from "../cameras/Camera";
import type {
	Scene,
	SceneNodeLifecycleListener,
	SceneNodeStateChangeEvent,
} from "../core/Scene";
import { Node } from "../core/Node";
import { Decal } from "../decals";
import { Light, type SceneLight } from "../lights";
import { MeshInstance } from "../meshes";
import { ParticleSystem } from "../particles";
import { ComponentStore } from "./ComponentStore";
import {
	NODE_KIND,
	type ECSComponentMap,
	type ECSComponentName,
	type EntityId,
	type HierarchyComponent,
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
	structureVersion: number;
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

const ROOT_PATH = "/sceneRoot";
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
];
const SCENE_DERIVED_COMPONENTS = new Set<ECSComponentName>([
	"WorldTransform",
	"Hierarchy",
	"PathBinding",
	"NodeRef",
	"NodeKind",
]);
const SCENE_REQUIRED_COMPONENTS = new Set<ECSComponentName>([
	"Name",
	"Visibility",
	"LocalTransform",
	...SCENE_DERIVED_COMPONENTS,
]);

/**
 * Optional user-owned ECS projection of one scene graph.
 *
 * @internal Exposed through the `experimentalECS` namespace.
 */
export class ECSWorld {
	private readonly _scene: Scene;
	private _nextEntityId = 1;
	private _entityIds = new Set<EntityId>();
	private _version = 0;
	private _structureVersion = 0;
	private _stores: {
		[K in ECSComponentName]: ComponentStore<ECSComponentMap[K]>;
	};
	private _queryCache = new Map<string, QueryCacheEntry>();
	private _entityByExternalId = new Map<string, EntityId>();
	private _externalIdByEntity = new Map<EntityId, string>();
	private _entityByNode = new WeakMap<Node, EntityId>();
	private _nodeByEntity = new Map<EntityId, Node>();
	private _localRevisionByNode = new WeakMap<Node, number>();
	private _worldRevisionByNode = new WeakMap<Node, number>();
	private _readOnlyViews = new WeakMap<object, object>();
	private _unsubscribeScene: (() => void) | null = null;
	private _destroyed = false;

	constructor(scene: Scene) {
		this._scene = scene;
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
		};
		scene.updateWorldMatrices();
		this._projectSubtree(scene.root, ROOT_PATH);
		const listener: SceneNodeLifecycleListener = {
			nodeAttached: ({ parent, child }) => this._handleStructureChanged(parent, child),
			nodeDetached: ({ parent, child }) => this._handleStructureChanged(parent, child),
			nodeStateChanged: (event) => this._handleNodeStateChanged(event),
			transformsChanged: (nodes) => this._handleTransformsChanged(nodes),
		};
		this._unsubscribeScene = scene.addNodeLifecycleListener(listener);
	}

	public get version(): number {
		return this._version;
	}

	public createEntity(externalId?: string): EntityId {
		this._assertActive();
		return this._createEntity(externalId);
	}

	public destroyEntity(entity: EntityId): boolean {
		this._assertActive();
		if (this._nodeByEntity.has(entity)) {
			throw new Error(
				`ECSWorld cannot destroy scene-backed entity "${entity}"; detach its Node from the Scene`,
			);
		}
		return this._destroyEntityInternal(entity);
	}

	public hasEntity(entity: EntityId): boolean {
		return !this._destroyed && this._entityIds.has(entity);
	}

	/**
	 * Assigns an external identifier, transferring it from any previous owner.
	 * Empty strings are valid identifiers. The previous owner remains alive.
	 */
	public setExternalId(entity: EntityId, externalId: string): void {
		this._assertEntity(entity);
		if (this._externalIdByEntity.get(entity) === externalId) return;
		this._assignExternalId(entity, externalId);
		this._bumpDataVersion();
	}

	public getEntityByExternalId(externalId: string): EntityId | null {
		return this._destroyed ? null : this._entityByExternalId.get(externalId) ?? null;
	}

	public getExternalId(entity: EntityId): string | null {
		return this._destroyed ? null : this._externalIdByEntity.get(entity) ?? null;
	}

	public setComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
		value: ECSComponentMap[K],
	): void {
		this._assertEntity(entity);
		const node = this._nodeByEntity.get(entity);
		if (!node) {
			this._setInternalComponent(entity, name, value);
			return;
		}
		if (SCENE_DERIVED_COMPONENTS.has(name)) {
			throw new Error(
				`ECSWorld component "${name}" is read-only for scene-backed entity "${entity}"`,
			);
		}
		if (name === "Name") {
			this._writeName(node, value as ECSComponentMap["Name"]);
			return;
		}
		if (name === "Visibility") {
			this._writeVisibility(node, value as ECSComponentMap["Visibility"]);
			return;
		}
		if (name === "LocalTransform") {
			this._writeLocalTransform(node, value as ECSComponentMap["LocalTransform"]);
			return;
		}
		this._setInternalComponent(entity, name, value);
	}

	/**
	 * Returns the stored component, or a stable read-only view for scene-derived
	 * components. Nested arrays and matrices are protected; `NodeRef.node` remains
	 * the mutable authoring node. Mirrored edits must be applied with `setComponent()`.
	 */
	public getComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
	): ECSComponentMap[K] | undefined {
		if (this._destroyed) return undefined;
		const component = this._stores[name].get(entity);
		if (component && this._nodeByEntity.has(entity) && SCENE_DERIVED_COMPONENTS.has(name)) {
			return this._getReadOnlyView(component);
		}
		return component;
	}

	public hasComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
	): boolean {
		return !this._destroyed && this._stores[name].has(entity);
	}

	public removeComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
	): boolean {
		this._assertEntity(entity);
		if (this._nodeByEntity.has(entity) && SCENE_REQUIRED_COMPONENTS.has(name)) {
			throw new Error(
				`ECSWorld cannot remove required component "${name}" from scene-backed entity "${entity}"`,
			);
		}
		const removed = this._stores[name].delete(entity);
		if (removed) this._bumpStructureVersion();
		return removed;
	}

	public query(required: ECSComponentName[]): EntityId[] {
		if (this._destroyed) return [];
		if (required.length === 0) return Array.from(this._entityIds);
		const key = required.slice().sort().join("|");
		const cached = this._queryCache.get(key);
		if (cached && cached.structureVersion === this._structureVersion) {
			return cached.entities.slice();
		}
		const sorted = required.slice().sort(
			(a, b) => this._stores[a].size - this._stores[b].size,
		);
		const entities = this._stores[sorted[0]].entities();
		const result: EntityId[] = [];
		for (const entity of entities) {
			if (sorted.every((name) => this._stores[name].has(entity))) {
				result.push(entity);
			}
		}
		this._queryCache.set(key, {
			structureVersion: this._structureVersion,
			entities: result.slice(),
		});
		return result;
	}

	public getEntityByNode(node: Node): EntityId | null {
		return this._destroyed ? null : this._entityByNode.get(node) ?? null;
	}

	public getNodeByEntity(entity: EntityId): Node | null {
		return this._destroyed ? null : this._nodeByEntity.get(entity) ?? null;
	}

	public findMeshInstances(): MeshInstance[] {
		return this._findNodesByKind(
			NODE_KIND.MeshInstance,
			(node): node is MeshInstance => node instanceof MeshInstance,
		);
	}

	public findDecals(): Decal[] {
		return this._findNodesByKind(
			NODE_KIND.Decal,
			(node): node is Decal => node instanceof Decal,
		);
	}

	public findLights(): SceneLight[] {
		return this._findNodesByKind(
			NODE_KIND.Light,
			(node): node is SceneLight => node instanceof Light,
		);
	}

	public findCameras(): Camera[] {
		return this._findNodesByKind(
			NODE_KIND.Camera,
			(node): node is Camera => node instanceof Camera,
		);
	}

	public findParticleSystems(): ParticleSystem[] {
		return this._findNodesByKind(
			NODE_KIND.ParticleSystem,
			(node): node is ParticleSystem => node instanceof ParticleSystem,
		);
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		this._unsubscribeScene?.();
		this._unsubscribeScene = null;
		for (const name of COMPONENT_NAMES) this._stores[name].clear();
		this._entityIds.clear();
		this._queryCache.clear();
		this._entityByExternalId.clear();
		this._externalIdByEntity.clear();
		this._nodeByEntity.clear();
		this._entityByNode = new WeakMap<Node, EntityId>();
		this._localRevisionByNode = new WeakMap<Node, number>();
		this._worldRevisionByNode = new WeakMap<Node, number>();
		this._readOnlyViews = new WeakMap<object, object>();
	}

	private _handleStructureChanged(parent: Node, child: Node): void {
		if (this._destroyed) return;
		const entity = this._entityByNode.get(child);
		const hierarchy = entity === undefined ? undefined : this._stores.Hierarchy.get(entity);
		const previousParent = hierarchy?.parent == null
			? null
			: this._nodeByEntity.get(hierarchy.parent) ?? null;
		// Events are invalidation hints: earlier listeners may have moved the subtree again.
		if (child.scene === this._scene) {
			let subtree = child;
			while (subtree.parent && !this._entityByNode.has(subtree.parent)) {
				subtree = subtree.parent;
			}
			this._projectSubtree(subtree, this._resolvePath(subtree));
			if (subtree.parent) this._syncHierarchy(subtree.parent);
		} else {
			this._destroyProjectionSubtree(child);
		}
		if (previousParent) this._syncHierarchy(previousParent);
		this._syncHierarchy(parent);
	}

	private _handleNodeStateChanged(event: SceneNodeStateChangeEvent): void {
		if (this._destroyed) return;
		const entity = this._entityByNode.get(event.node);
		if (entity === undefined) return;
		if (event.field === "name") {
			this._syncName(entity, event.node);
			this._refreshSubtreePaths(event.node);
			return;
		}
		this._syncVisibility(entity, event.node);
	}

	private _handleTransformsChanged(nodes: readonly Node[]): void {
		if (this._destroyed) return;
		for (const node of nodes) {
			const entity = this._entityByNode.get(node);
			if (entity === undefined) continue;
			if (this._localRevisionByNode.get(node) !== node.localTransformRevision) {
				this._syncLocalTransform(entity, node);
			}
			if (this._worldRevisionByNode.get(node) !== node.worldTransformRevision) {
				this._syncWorldTransform(entity, node);
			}
		}
	}

	private _projectSubtree(node: Node, path: string): EntityId {
		const existing = this._entityByNode.get(node);
		let entity: EntityId;
		let oldParent: Node | null = null;
		if (existing !== undefined) {
			entity = existing;
			const hierarchy = this._stores.Hierarchy.get(entity);
			if (hierarchy?.parent != null) {
				oldParent = this._nodeByEntity.get(hierarchy.parent) ?? null;
			}
			this._syncName(entity, node);
			this._syncVisibility(entity, node);
			this._syncLocalTransform(entity, node);
			if (this._worldRevisionByNode.get(node) !== node.worldTransformRevision) {
				this._syncWorldTransform(entity, node);
			}
			this._syncPath(entity, path);
		} else {
			entity = this._createEntity(node.id);
			this._entityByNode.set(node, entity);
			this._nodeByEntity.set(entity, node);
			this._setInternalComponent(entity, "Name", { value: node.name });
			this._setInternalComponent(entity, "Visibility", { visible: node.visible });
			this._setInternalComponent(entity, "LocalTransform", createLocalTransform(node));
			this._setInternalComponent(entity, "WorldTransform", {
				matrix: node.worldMatrix.clone(),
			});
			this._setInternalComponent(entity, "PathBinding", { path });
			this._setInternalComponent(entity, "NodeRef", { node });
			this._setInternalComponent(entity, "NodeKind", { kind: resolveNodeKind(node) });
			this._localRevisionByNode.set(node, node.localTransformRevision);
			this._worldRevisionByNode.set(node, node.worldTransformRevision);
		}
		for (const child of node.children) {
			this._projectSubtree(
				child,
				`${path}/${sanitizePathSegment(child.name)}_${child.id}`,
			);
		}
		this._syncHierarchy(node);
		if (oldParent && oldParent !== node.parent) this._syncHierarchy(oldParent);
		return entity;
	}

	private _destroyProjectionSubtree(node: Node): void {
		for (const child of node.children) {
			if (child.scene !== this._scene) this._destroyProjectionSubtree(child);
		}
		const entity = this._entityByNode.get(node);
		if (entity === undefined) return;
		// Detached trees can be edited before their lifecycle event reaches this world.
		const hierarchy = this._stores.Hierarchy.get(entity);
		const children = hierarchy?.children;
		// Removing a projected child also unlinks it from this list.
		for (let index = (children?.length ?? 0) - 1; index >= 0; index--) {
			const child = this._nodeByEntity.get(children![index]);
			if (child && child.scene !== this._scene) this._destroyProjectionSubtree(child);
		}
		this._entityByNode.delete(node);
		this._nodeByEntity.delete(entity);
		this._destroyEntityInternal(entity);
		if (hierarchy?.parent != null) {
			const siblings = this._stores.Hierarchy.get(hierarchy.parent)?.children;
			const index = siblings?.indexOf(entity) ?? -1;
			if (index >= 0) siblings!.splice(index, 1);
		}
	}

	private _refreshSubtreePaths(node: Node): void {
		const entity = this._entityByNode.get(node);
		if (entity === undefined) return;
		this._syncPath(entity, this._resolvePath(node));
		for (const child of node.children) this._refreshSubtreePaths(child);
	}

	private _resolvePath(node: Node): string {
		if (node === this._scene.root) return ROOT_PATH;
		const parent = node.parent;
		if (!parent) return `/${sanitizePathSegment(node.name)}_${node.id}`;
		const parentEntity = this._entityByNode.get(parent);
		const parentPath = parentEntity === undefined
			? ROOT_PATH
			: this._stores.PathBinding.get(parentEntity)?.path ?? ROOT_PATH;
		return `${parentPath}/${sanitizePathSegment(node.name)}_${node.id}`;
	}

	private _syncHierarchy(node: Node): void {
		// Preserve the projected child list until detach cleanup has consumed it.
		if (node.scene !== this._scene) return;
		const entity = this._entityByNode.get(node);
		if (entity === undefined) return;
		const parent = node.parent ? this._entityByNode.get(node.parent) ?? null : null;
		const children = node.children.flatMap((child) => {
			const childEntity = this._entityByNode.get(child);
			return childEntity === undefined ? [] : [childEntity];
		});
		const current = this._stores.Hierarchy.get(entity);
		if (current) {
			for (let index = current.children.length - 1; index >= 0; index--) {
				const child = this._nodeByEntity.get(current.children[index]);
				if (child && child.scene !== this._scene) this._destroyProjectionSubtree(child);
			}
		}
		if (current && current.parent === parent && arraysEqual(current.children, children)) {
			return;
		}
		if (current) {
			current.parent = parent;
			current.children.length = 0;
			current.children.push(...children);
			this._bumpDataVersion();
			return;
		}
		this._setInternalComponent(entity, "Hierarchy", { parent, children });
	}

	private _syncName(entity: EntityId, node: Node): void {
		const current = this._stores.Name.get(entity);
		if (current?.value === node.name) return;
		if (current) {
			current.value = node.name;
			this._bumpDataVersion();
			return;
		}
		this._setInternalComponent(entity, "Name", { value: node.name });
	}

	private _syncVisibility(entity: EntityId, node: Node): void {
		const current = this._stores.Visibility.get(entity);
		if (current?.visible === node.visible) return;
		if (current) {
			current.visible = node.visible;
			this._bumpDataVersion();
			return;
		}
		this._setInternalComponent(entity, "Visibility", { visible: node.visible });
	}

	private _syncLocalTransform(entity: EntityId, node: Node): void {
		const current = this._stores.LocalTransform.get(entity);
		if (!current) {
			this._setInternalComponent(entity, "LocalTransform", createLocalTransform(node));
		} else if (copyLocalTransformFromNode(current, node)) {
			this._bumpDataVersion();
		}
		this._localRevisionByNode.set(node, node.localTransformRevision);
	}

	private _syncWorldTransform(entity: EntityId, node: Node): void {
		const current = this._stores.WorldTransform.get(entity);
		if (!current) {
			this._setInternalComponent(entity, "WorldTransform", {
				matrix: node.worldMatrix.clone(),
			});
		} else {
			node.worldMatrix.copyTo(current.matrix);
			this._bumpDataVersion();
		}
		this._worldRevisionByNode.set(node, node.worldTransformRevision);
	}

	private _syncPath(entity: EntityId, path: string): void {
		const current = this._stores.PathBinding.get(entity);
		if (current?.path === path) return;
		if (current) {
			current.path = path;
			this._bumpDataVersion();
			return;
		}
		this._setInternalComponent(entity, "PathBinding", { path });
	}

	private _writeName(node: Node, value: NameComponent): void {
		const entity = this._entityByNode.get(node)!;
		const current = this._stores.Name.get(entity);
		const nodeChanged = node.name !== value.value;
		const storeChanged = current?.value !== value.value;
		if (!nodeChanged && !storeChanged) return;
		const version = this._version;
		try {
			if (nodeChanged) node.name = value.value;
		} finally {
			// Listeners may normalize, detach, reattach, or destroy during the setter.
			const finalEntity = this.getEntityByNode(node);
			if (finalEntity !== null) {
				this._syncName(finalEntity, node);
				this._refreshSubtreePaths(node);
				if (this._version === version) this._bumpDataVersion();
			}
		}
	}

	private _writeVisibility(node: Node, value: VisibilityComponent): void {
		const entity = this._entityByNode.get(node)!;
		const current = this._stores.Visibility.get(entity);
		const nodeChanged = node.visible !== value.visible;
		const storeChanged = current?.visible !== value.visible;
		if (!nodeChanged && !storeChanged) return;
		const version = this._version;
		try {
			if (nodeChanged) node.visible = value.visible;
		} finally {
			const finalEntity = this.getEntityByNode(node);
			if (finalEntity !== null) {
				this._syncVisibility(finalEntity, node);
				if (this._version === version) this._bumpDataVersion();
			}
		}
	}

	private _writeLocalTransform(node: Node, value: LocalTransformComponent): void {
		const entity = this._entityByNode.get(node)!;
		const current = this._stores.LocalTransform.get(entity);
		const nodeChanged = !nodeMatchesLocalTransform(node, value);
		const storeChanged = !current || !localTransformsEqual(current, value);
		if (!nodeChanged && !storeChanged) return;
		const version = this._version;
		if (nodeChanged) {
			node.position.set(value.positionX, value.positionY, value.positionZ);
			node.quaternion.set(
				value.rotationX,
				value.rotationY,
				value.rotationZ,
				value.rotationW,
			);
			node.scale.set(value.scaleX, value.scaleY, value.scaleZ);
			node.updateLocalMatrix();
			node.scene?.invalidate("transform");
		}
		const finalEntity = this.getEntityByNode(node);
		if (finalEntity === null) return;
		this._syncLocalTransform(finalEntity, node);
		if (this._version === version) this._bumpDataVersion();
	}

	private _getReadOnlyView<T extends object>(value: T): T {
		// The node itself is an authoring object, not owned component storage.
		if (value instanceof Node) return value;
		const existing = this._readOnlyViews.get(value);
		if (existing) return existing as T;
		const protect = (nested: unknown): unknown => nested !== null && typeof nested === "object"
			? this._getReadOnlyView(nested)
			: nested;
		const rejectMutation = (): never => {
			throw new Error("ECSWorld scene-derived component is read-only");
		};
		const view = new Proxy(value, {
			get: (target, key, receiver) => protect(Reflect.get(target, key, receiver)),
			getOwnPropertyDescriptor: (target, key) => {
				const descriptor: PropertyDescriptor | undefined =
					Reflect.getOwnPropertyDescriptor(target, key);
				if (descriptor && "value" in descriptor) {
					descriptor.value = protect(descriptor.value);
				}
				return descriptor;
			},
			set: rejectMutation,
			defineProperty: rejectMutation,
			deleteProperty: rejectMutation,
			setPrototypeOf: rejectMutation,
			preventExtensions: rejectMutation,
		});
		this._readOnlyViews.set(value, view);
		return view;
	}

	private _findNodesByKind<K extends NodeKind>(
		expectedKind: K,
		isExpectedNode: (node: Node) => node is NodeKindLookupMap[K],
	): NodeKindLookupMap[K][] {
		const result: NodeKindLookupMap[K][] = [];
		for (const entity of this.query(["NodeRef", "NodeKind"])) {
			const nodeKind = this._stores.NodeKind.get(entity);
			const nodeRef = this._stores.NodeRef.get(entity);
			if (nodeKind?.kind === expectedKind && nodeRef && isExpectedNode(nodeRef.node)) {
				result.push(nodeRef.node);
			}
		}
		return result;
	}

	private _createEntity(externalId?: string): EntityId {
		const entity = this._nextEntityId++;
		this._entityIds.add(entity);
		if (externalId !== undefined) this._assignExternalId(entity, externalId);
		this._bumpStructureVersion();
		return entity;
	}

	private _assignExternalId(entity: EntityId, externalId: string): void {
		const current = this._externalIdByEntity.get(entity);
		if (current !== undefined) this._entityByExternalId.delete(current);
		const previousOwner = this._entityByExternalId.get(externalId);
		if (previousOwner !== undefined) this._externalIdByEntity.delete(previousOwner);
		this._externalIdByEntity.set(entity, externalId);
		this._entityByExternalId.set(externalId, entity);
	}

	private _destroyEntityInternal(entity: EntityId): boolean {
		if (!this._entityIds.delete(entity)) return false;
		for (const name of COMPONENT_NAMES) this._stores[name].delete(entity);
		const externalId = this._externalIdByEntity.get(entity);
		if (externalId !== undefined && this._entityByExternalId.get(externalId) === entity) {
			this._entityByExternalId.delete(externalId);
		}
		this._externalIdByEntity.delete(entity);
		this._nodeByEntity.delete(entity);
		this._bumpStructureVersion();
		return true;
	}

	private _setInternalComponent<K extends ECSComponentName>(
		entity: EntityId,
		name: K,
		value: ECSComponentMap[K],
	): void {
		const exists = this._stores[name].has(entity);
		this._stores[name].set(entity, value);
		if (exists) this._bumpDataVersion();
		else this._bumpStructureVersion();
	}

	private _assertEntity(entity: EntityId): void {
		this._assertActive();
		if (!this._entityIds.has(entity)) {
			throw new Error(`ECSWorld target entity "${entity}" not found`);
		}
	}

	private _assertActive(): void {
		if (this._destroyed) throw new Error("ECSWorld is destroyed");
	}

	private _bumpDataVersion(): void {
		this._version++;
	}

	private _bumpStructureVersion(): void {
		this._version++;
		this._structureVersion++;
	}
}

function createLocalTransform(node: Node): LocalTransformComponent {
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

function copyLocalTransformFromNode(target: LocalTransformComponent, node: Node): boolean {
	if (
		target.positionX === node.position.x &&
		target.positionY === node.position.y &&
		target.positionZ === node.position.z &&
		target.rotationX === node.quaternion.x &&
		target.rotationY === node.quaternion.y &&
		target.rotationZ === node.quaternion.z &&
		target.rotationW === node.quaternion.w &&
		target.scaleX === node.scale.x &&
		target.scaleY === node.scale.y &&
		target.scaleZ === node.scale.z
	) return false;
	target.positionX = node.position.x;
	target.positionY = node.position.y;
	target.positionZ = node.position.z;
	target.rotationX = node.quaternion.x;
	target.rotationY = node.quaternion.y;
	target.rotationZ = node.quaternion.z;
	target.rotationW = node.quaternion.w;
	target.scaleX = node.scale.x;
	target.scaleY = node.scale.y;
	target.scaleZ = node.scale.z;
	return true;
}

function localTransformsEqual(
	left: LocalTransformComponent,
	right: LocalTransformComponent,
): boolean {
	return left.positionX === right.positionX &&
		left.positionY === right.positionY &&
		left.positionZ === right.positionZ &&
		left.rotationX === right.rotationX &&
		left.rotationY === right.rotationY &&
		left.rotationZ === right.rotationZ &&
		left.rotationW === right.rotationW &&
		left.scaleX === right.scaleX &&
		left.scaleY === right.scaleY &&
		left.scaleZ === right.scaleZ;
}

function nodeMatchesLocalTransform(node: Node, value: LocalTransformComponent): boolean {
	return node.position.x === value.positionX &&
		node.position.y === value.positionY &&
		node.position.z === value.positionZ &&
		node.quaternion.x === value.rotationX &&
		node.quaternion.y === value.rotationY &&
		node.quaternion.z === value.rotationZ &&
		node.quaternion.w === value.rotationW &&
		node.scale.x === value.scaleX &&
		node.scale.y === value.scaleY &&
		node.scale.z === value.scaleZ;
}

function arraysEqual(left: readonly number[], right: readonly number[]): boolean {
	if (left.length !== right.length) return false;
	for (let index = 0; index < left.length; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}

function resolveNodeKind(node: Node): NodeKind {
	if (node instanceof MeshInstance) return NODE_KIND.MeshInstance;
	if (node instanceof Decal) return NODE_KIND.Decal;
	if (node instanceof Camera) return NODE_KIND.Camera;
	if (node instanceof ParticleSystem) return NODE_KIND.ParticleSystem;
	if (node instanceof Light) return NODE_KIND.Light;
	return NODE_KIND.Node;
}
