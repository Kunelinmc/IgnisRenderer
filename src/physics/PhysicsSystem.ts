import { EventEmitter } from "../core/EventEmitter";
import type { Node } from "../core/Node";
import { IdGenerator } from "../utils/IdGenerator";
import type {
	ICollisionGeometryProvider,
	BodyBinding,
	CharacterControllerDescriptor,
	CharacterControllerHandle,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsBodyHandle,
	PhysicsColliderHandle,
	PhysicsEvent,
	PhysicsEvents,
	PhysicsJointHandle,
	PhysicsStepReport,
	PhysicsTransform,
	PhysicsWorldConfig,
	PhysicsWorldStepReport,
	StepOverride,
	TransformAuthority,
} from "./types";
import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterBodyState,
	PhysicsAdapterStepResult,
} from "./IPhysicsEngineAdapter";
import { DefaultCollisionGeometryProvider } from "./DefaultCollisionGeometryProvider";
import { PhysicsBodyNode } from "./PhysicsBodyNode";
import { SimplePhysicsAdapter } from "./adapters/SimplePhysicsAdapter";
import { DefaultPhysicsSimulator } from "../simulation/physics/DefaultPhysicsSimulator";

export interface PhysicsSystemOptions {
	adapter?: IPhysicsEngineAdapter;
	geometryProvider?: ICollisionGeometryProvider;
}

interface InternalBodyBinding extends PhysicsBodyHandle {
	body: BodyBinding["body"];
	colliderIds: Set<string>;
}

interface InternalColliderBinding extends PhysicsColliderHandle {
	descriptor: ColliderDescriptor;
	shape: ColliderShape;
}

interface InternalJointBinding extends PhysicsJointHandle {
	bodyAId: string;
	bodyBId: string;
}

interface InternalControllerBinding {
	id: string;
	worldId: string;
	bodyId: string;
	descriptor: CharacterControllerDescriptor;
	handle: CharacterControllerHandle;
}

export class PhysicsSystem extends EventEmitter<PhysicsEvents> {
	private _adapter: IPhysicsEngineAdapter;
	private _geometryProvider: ICollisionGeometryProvider;
	private _simulator = new DefaultPhysicsSimulator();

	private _worldConfigById = new Map<string, PhysicsWorldConfig>();
	private _bodyById = new Map<string, InternalBodyBinding>();
	private _bodyIdByNodeId = new Map<string, string>();
	private _colliderById = new Map<string, InternalColliderBinding>();
	private _jointById = new Map<string, InternalJointBinding>();
	private _controllerById = new Map<string, InternalControllerBinding>();
	private _eventQueueByWorld = new Map<string, PhysicsEvent[]>();

	constructor(options: PhysicsSystemOptions = {}) {
		super();
		this._adapter = options.adapter ?? new SimplePhysicsAdapter();
		this._geometryProvider =
			options.geometryProvider ?? new DefaultCollisionGeometryProvider();
	}

	public async init(): Promise<void> {
		await this._adapter.init();
	}

	public initSync(): void {
		if (!this._adapter.capabilities.syncInit || !this._adapter.initSync) {
			throw new Error(
				`${this._adapter.id} adapter does not support synchronous init`
			);
		}
		this._adapter.initSync();
	}

	public createWorld(config: PhysicsWorldConfig): void {
		if (!config.worldId || config.worldId.trim().length === 0) {
			throw new Error("Physics worldId is required");
		}
		if (this._worldConfigById.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}
		this._adapter.createWorld(config);
		this._worldConfigById.set(config.worldId, {
			...config,
			gravity: config.gravity ? { ...config.gravity } : undefined,
		});
		this._eventQueueByWorld.set(config.worldId, []);
	}

	public destroyWorld(worldId: string): void {
		if (!this._worldConfigById.has(worldId)) return;

		for (const body of Array.from(this._bodyById.values())) {
			if (body.worldId === worldId) {
				this.detachBody(body);
			}
		}
		for (const joint of Array.from(this._jointById.values())) {
			if (joint.worldId === worldId) {
				this._adapter.destroyJoint(worldId, joint.id);
				this._jointById.delete(joint.id);
			}
		}
		for (const controller of Array.from(this._controllerById.values())) {
			if (controller.worldId === worldId) {
				this._adapter.destroyCharacterController(worldId, controller.id);
				this._controllerById.delete(controller.id);
			}
		}
		this._adapter.destroyWorld(worldId);
		this._worldConfigById.delete(worldId);
		this._eventQueueByWorld.delete(worldId);
	}

	public attachBody(
		node: Node | PhysicsBodyNode,
		desc?: BodyBinding
	): PhysicsBodyHandle {
		if (this._bodyIdByNodeId.has(node.id)) {
			throw new Error(`Node "${node.id}" is already bound to a physics body`);
		}

		const binding = resolveBinding(node, desc);
		this._requireWorld(binding.worldId);
		const authority = binding.authority ?? "physics";
		const bodyType = binding.body.type ?? "dynamic";
		if (authority === "animation" && bodyType === "dynamic") {
			throw new Error(
				'authority="animation" does not allow dynamic rigid bodies. Use kinematic/fixed.'
			);
		}

		const bodyId = IdGenerator.nextId("physicsBody");
		const handle: InternalBodyBinding = {
			id: bodyId,
			worldId: binding.worldId,
			node,
			authority,
			body: {
				...binding.body,
			},
			colliderIds: new Set(),
		};
		this._adapter.createBody(
			binding.worldId,
			bodyId,
			handle.body,
			this._resolveNodeTransform(node)
		);

		this._bodyById.set(bodyId, handle);
		this._bodyIdByNodeId.set(node.id, bodyId);

		for (const collider of binding.colliders ?? []) {
			this.addCollider(handle, collider);
		}

		return handle;
	}

	public detachBody(target: Node | PhysicsBodyHandle): void {
		const body = this._resolveBody(target);

		for (const colliderId of body.colliderIds) {
			this._adapter.destroyCollider(body.worldId, colliderId);
			this._colliderById.delete(colliderId);
		}
		body.colliderIds.clear();

		for (const [jointId, joint] of this._jointById) {
			if (joint.bodyAId === body.id || joint.bodyBId === body.id) {
				this._adapter.destroyJoint(joint.worldId, joint.id);
				this._jointById.delete(jointId);
			}
		}

		for (const [controllerId, controller] of this._controllerById) {
			if (controller.bodyId === body.id) {
				this._adapter.destroyCharacterController(
					controller.worldId,
					controller.id
				);
				this._controllerById.delete(controllerId);
			}
		}

		this._adapter.destroyBody(body.worldId, body.id);
		this._bodyById.delete(body.id);
		this._bodyIdByNodeId.delete(body.node.id);
	}

	public addCollider(
		target: Node | PhysicsBodyHandle,
		desc: ColliderDescriptor
	): PhysicsColliderHandle {
		const body = this._resolveBody(target);
		const shape = this._resolveColliderShape(body, desc);
		const colliderId = IdGenerator.nextId("physicsCollider");

		this._adapter.addCollider(body.worldId, body.id, colliderId, desc, shape);
		const collider: InternalColliderBinding = {
			id: colliderId,
			worldId: body.worldId,
			bodyId: body.id,
			descriptor: { ...desc },
			shape,
		};
		this._colliderById.set(colliderId, collider);
		body.colliderIds.add(colliderId);
		return collider;
	}

	public createJoint(desc: JointDescriptor): PhysicsJointHandle {
		this._assertCapability("joints");
		this._requireWorld(desc.worldId);
		const bodyA = this._resolveBodyRef(desc.bodyA);
		const bodyB = this._resolveBodyRef(desc.bodyB);
		if (bodyA.worldId !== desc.worldId || bodyB.worldId !== desc.worldId) {
			throw new Error(
				"Joint bodies must belong to the same world as descriptor.worldId"
			);
		}

		const jointId = IdGenerator.nextId("physicsJoint");
		const normalized: JointDescriptor = {
			...desc,
			bodyA: bodyA.id,
			bodyB: bodyB.id,
		};
		this._adapter.createJoint(desc.worldId, jointId, normalized);
		const handle: InternalJointBinding = {
			id: jointId,
			worldId: desc.worldId,
			bodyAId: bodyA.id,
			bodyBId: bodyB.id,
		};
		this._jointById.set(jointId, handle);
		return handle;
	}

	public createCharacterController(
		desc: CharacterControllerDescriptor
	): CharacterControllerHandle {
		this._assertCapability("characterController");
		this._requireWorld(desc.worldId);
		const body = this._resolveBodyRef(desc.body);
		if (body.worldId !== desc.worldId) {
			throw new Error(
				"Character controller body must belong to descriptor.worldId"
			);
		}

		const controllerId = IdGenerator.nextId("physicsController");
		const normalized: CharacterControllerDescriptor = {
			...desc,
			body: body.id,
		};
		this._adapter.createCharacterController(
			desc.worldId,
			controllerId,
			normalized
		);

		const handle: CharacterControllerHandle = {
			id: controllerId,
			worldId: desc.worldId,
			moveAndSlide: (direction, deltaTimeMs) => {
				return this._adapter.moveCharacterController(
					desc.worldId,
					controllerId,
					direction,
					Math.max(0, deltaTimeMs) / 1000
				);
			},
			jump: (speed) => {
				this._adapter.jumpCharacterController(
					desc.worldId,
					controllerId,
					Math.max(0, speed ?? desc.jumpSpeed ?? 0)
				);
			},
			isGrounded: () => {
				return this._adapter.isCharacterControllerGrounded(
					desc.worldId,
					controllerId
				);
			},
			setMaxSlope: (value) => {
				this._adapter.setCharacterControllerMaxSlope(
					desc.worldId,
					controllerId,
					value
				);
			},
			setStepHeight: (value) => {
				this._adapter.setCharacterControllerStepHeight(
					desc.worldId,
					controllerId,
					value
				);
			},
		};

		this._controllerById.set(controllerId, {
			id: controllerId,
			worldId: desc.worldId,
			bodyId: body.id,
			descriptor: normalized,
			handle,
		});
		return handle;
	}

	public step(deltaTimeMs: number, opts: StepOverride = {}): PhysicsStepReport {
		const worldIds = opts.worldIds ?? Array.from(this._worldConfigById.keys());
		const targetWorlds = worldIds.map((worldId) => {
			const config = this._requireWorld(worldId);
			return { worldId, config };
		});

		this._syncAnimationAuthorityBodies(
			targetWorlds.map((item) => item.worldId)
		);

		const simulationContext = {
			worlds: targetWorlds,
			stepWorld: (worldId: string, deltaSeconds: number) =>
				this._adapter.stepWorld(worldId, deltaSeconds),
		};

		this._simulator.beginFrame(simulationContext);
		const simulation = this._simulator.simulate(simulationContext, {
			deltaTimeMs,
			override: opts,
		});
		this._simulator.endFrame();

		const worldReports: PhysicsWorldStepReport[] = [];
		const events: PhysicsEvent[] = [];
		const movedBodyIds = new Set<string>();
		const dirtyWorldIds = new Set<string>();

		for (const worldResult of simulation.worldResults) {
			let activeBodies = 0;
			let sleepingBodies = 0;
			let ccdBodies = 0;
			for (const stepResult of worldResult.steps) {
				activeBodies = stepResult.activeBodies;
				sleepingBodies = stepResult.sleepingBodies;
				ccdBodies = stepResult.ccdBodies;

				this._applyPhysicsAuthorityState(
					stepResult,
					movedBodyIds,
					dirtyWorldIds,
					worldResult.worldId
				);
				for (const event of stepResult.events) {
					events.push(event);
					this._enqueueEvent(event);
					dirtyWorldIds.add(event.worldId);
				}
			}

			worldReports.push({
				worldId: worldResult.worldId,
				mode: worldResult.mode,
				substeps: worldResult.substeps,
				consumedDeltaMs: worldResult.consumedDeltaMs,
				activeBodies,
				sleepingBodies,
				ccdBodies,
			});
		}

		const dirty = movedBodyIds.size > 0 || events.length > 0;
		const report: PhysicsStepReport = {
			inputDeltaMs: Math.max(0, deltaTimeMs),
			processedDeltaMs: simulation.processedDeltaMs,
			worldReports,
			events,
			dirty,
		};

		if (dirty) {
			this.emit("dirty", {
				worldIds: Array.from(dirtyWorldIds),
				movedBodyIds: Array.from(movedBodyIds),
				events,
			});
		}
		this.emit("step", report);
		return report;
	}

	public drainEvents(worldId?: string): PhysicsEvent[] {
		if (worldId) {
			const queue = this._eventQueueByWorld.get(worldId);
			if (!queue || queue.length === 0) return [];
			const drained = [...queue];
			queue.length = 0;
			return drained;
		}
		const drained: PhysicsEvent[] = [];
		for (const queue of this._eventQueueByWorld.values()) {
			if (queue.length === 0) continue;
			drained.push(...queue);
			queue.length = 0;
		}
		return drained;
	}

	public connectRendererWakeup(renderer: {
		requestRender(): void;
	}): () => void {
		const listener = () => renderer.requestRender();
		this.on("dirty", listener);
		return () => {
			this.off("dirty", listener);
		};
	}

	private _syncAnimationAuthorityBodies(worldIds: string[]): void {
		const scoped = new Set(worldIds);
		for (const body of this._bodyById.values()) {
			if (!scoped.has(body.worldId)) continue;
			if (body.authority !== "animation") continue;
			this._adapter.setBodyTransform(
				body.worldId,
				body.id,
				this._resolveNodeTransform(body.node)
			);
		}
	}

	private _applyPhysicsAuthorityState(
		stepResult: PhysicsAdapterStepResult,
		movedBodyIds: Set<string>,
		dirtyWorldIds: Set<string>,
		worldId: string
	): void {
		for (const state of stepResult.bodyStates) {
			const body = this._bodyById.get(state.bodyId);
			if (!body || body.authority !== "physics") continue;
			if (this._applyNodeTransform(body.node, state)) {
				movedBodyIds.add(body.id);
				dirtyWorldIds.add(worldId);
			}
		}
	}

	private _applyNodeTransform(
		node: Node,
		state: PhysicsAdapterBodyState
	): boolean {
		const oldPosition = {
			x: node.position.x,
			y: node.position.y,
			z: node.position.z,
		};
		const oldRotation = [
			node.quaternion.x,
			node.quaternion.y,
			node.quaternion.z,
			node.quaternion.w,
		];

		node.position.set(
			state.transform.position.x,
			state.transform.position.y,
			state.transform.position.z
		);
		node.quaternion.x = state.transform.rotation[0];
		node.quaternion.y = state.transform.rotation[1];
		node.quaternion.z = state.transform.rotation[2];
		node.quaternion.w = state.transform.rotation[3];
		node.updateLocalMatrix();

		const moved =
			Math.abs(oldPosition.x - node.position.x) > 1e-6 ||
			Math.abs(oldPosition.y - node.position.y) > 1e-6 ||
			Math.abs(oldPosition.z - node.position.z) > 1e-6 ||
			Math.abs(oldRotation[0] - node.quaternion.x) > 1e-6 ||
			Math.abs(oldRotation[1] - node.quaternion.y) > 1e-6 ||
			Math.abs(oldRotation[2] - node.quaternion.z) > 1e-6 ||
			Math.abs(oldRotation[3] - node.quaternion.w) > 1e-6;
		return moved;
	}

	private _enqueueEvent(event: PhysicsEvent): void {
		const queue = this._eventQueueByWorld.get(event.worldId);
		if (!queue) return;
		queue.push(event);
		this.emit(event.type, event as any);
	}

	private _resolveNodeTransform(node: Node): PhysicsTransform {
		return {
			position: {
				x: node.position.x,
				y: node.position.y,
				z: node.position.z,
			},
			rotation: [
				node.quaternion.x,
				node.quaternion.y,
				node.quaternion.z,
				node.quaternion.w,
			],
		};
	}

	private _resolveColliderShape(
		body: PhysicsBodyHandle,
		desc: ColliderDescriptor
	): ColliderShape {
		if (!desc.mode || desc.mode === "explicit") {
			if (!("shape" in desc)) {
				throw new Error("Explicit collider descriptor requires shape");
			}
			return desc.shape;
		}

		const sourceNode = desc.sourceNode ?? body.node;
		if (desc.mode === "auto-fit") {
			const bounds = this._geometryProvider.getBounds(sourceNode);
			if (!bounds) {
				throw new Error(`auto-fit collider failed for node "${sourceNode.id}"`);
			}
			if (desc.shapePreference === "sphere") {
				return {
					kind: "sphere",
					radius: Math.max(0.001, bounds.sphere.radius),
				};
			}
			return {
				kind: "box",
				halfExtents: {
					x: (bounds.box.max.x - bounds.box.min.x) * 0.5,
					y: (bounds.box.max.y - bounds.box.min.y) * 0.5,
					z: (bounds.box.max.z - bounds.box.min.z) * 0.5,
				},
			};
		}

		const triangles = this._geometryProvider.getTriangles(sourceNode);
		if (!triangles) {
			throw new Error(
				`trimesh-cook collider failed for node "${sourceNode.id}"`
			);
		}
		return {
			kind: "trimesh",
			vertices: triangles.vertices,
			indices: triangles.indices,
		};
	}

	private _resolveBody(target: Node | PhysicsBodyHandle): InternalBodyBinding {
		if (isBodyHandle(target)) {
			const body = this._bodyById.get(target.id);
			if (body) return body;
			throw new Error(`Physics body "${target.id}" does not exist`);
		}
		const bodyId = this._bodyIdByNodeId.get(target.id);
		if (!bodyId) {
			throw new Error(`Node "${target.id}" is not bound to any physics body`);
		}
		const body = this._bodyById.get(bodyId);
		if (body) return body;
		throw new Error(`Physics body "${bodyId}" does not exist`);
	}

	private _resolveBodyRef(
		target: Node | PhysicsBodyHandle | string
	): InternalBodyBinding {
		if (typeof target === "string") {
			const body = this._bodyById.get(target);
			if (body) return body;
			throw new Error(`Physics body "${target}" does not exist`);
		}
		return this._resolveBody(target);
	}

	private _requireWorld(worldId: string): PhysicsWorldConfig {
		const world = this._worldConfigById.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}

	private _assertCapability(
		name: keyof IPhysicsEngineAdapter["capabilities"]
	): void {
		if (this._adapter.capabilities[name]) return;
		throw new Error(
			`Adapter "${this._adapter.id}" does not support capability "${name}"`
		);
	}
}

function resolveBinding(
	node: Node | PhysicsBodyNode,
	desc: BodyBinding | undefined
): BodyBinding {
	if (desc) return desc;
	if (node instanceof PhysicsBodyNode) {
		return node.bodyBinding;
	}
	throw new Error(
		"attachBody(node, desc) requires desc unless node is a PhysicsBodyNode"
	);
}

function isBodyHandle(value: unknown): value is PhysicsBodyHandle {
	if (!value || typeof value !== "object") return false;
	return "id" in value && "worldId" in value && "node" in value;
}
