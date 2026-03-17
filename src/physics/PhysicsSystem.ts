import type { IVector3 } from "../maths/types";
import { EventEmitter } from "../core/EventEmitter";
import type { Node } from "../core/Node";
import { IdGenerator } from "../foundation/IdGenerator";
import type {
	ICollisionGeometryProvider,
	BodyBinding,
	CharacterControllerDescriptor,
	CharacterControllerHandle,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsBodyHandle,
	PhysicsBoxCastQuery,
	PhysicsColliderHandle,
	PhysicsEvent,
	PhysicsEvents,
	PhysicsJointHandle,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
	PhysicsQueryHit,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	PhysicsStepReport,
	PhysicsTransform,
	PhysicsWorldConfig,
	PhysicsWorldStepReport,
	StepOverride,
	TransformAuthority,
	PhysicsEntityId,
} from "./types";
import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterStepResult,
} from "./IPhysicsEngineAdapter";
import { DefaultCollisionGeometryProvider } from "./DefaultCollisionGeometryProvider";
import { PhysicsBodyNode } from "./PhysicsBodyNode";
import { SimplePhysicsAdapter } from "./adapters/SimplePhysicsAdapter";
import { DefaultPhysicsSimulator } from "../simulation/physics/DefaultPhysicsSimulator";
import type { PhysicsSimulationResult } from "../simulation/physics/types";
import {
	BROADPHASE_CELL_SIZE,
	BROADPHASE_MAX_DIRTY_CELLS,
	DEFAULT_BROADPHASE_BODY_RADIUS,
	TRANSFORM_EPSILON,
} from "./constants";

export interface PhysicsSystemOptions {
	adapter?: IPhysicsEngineAdapter;
	geometryProvider?: ICollisionGeometryProvider;
}

interface InternalBodyBinding extends PhysicsBodyHandle {
	body: BodyBinding["body"];
	colliderIds: Set<string>;
	broadphaseRadius: number;
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

interface CachedBodyState {
	positionX: number;
	positionY: number;
	positionZ: number;
	rotationX: number;
	rotationY: number;
	rotationZ: number;
	rotationW: number;
	sleeping: boolean;
}

interface BroadphaseBounds {
	minX: number;
	minY: number;
	minZ: number;
	maxX: number;
	maxY: number;
	maxZ: number;
}

interface BroadphaseRuntimeState {
	dirtyBodyIds: Set<string>;
	dirtyCells: Set<string>;
	dirtyBounds: BroadphaseBounds | null;
}

interface SleepingIslandState {
	id: string;
	bodyIds: Set<string>;
	dynamicBodyCount: number;
	sleepingDynamicBodyCount: number;
}

interface CachedWorldStats {
	activeBodies: number;
	sleepingBodies: number;
	ccdBodies: number;
}

interface WorldRuntimeState {
	worldId: string;
	bodyIds: Set<string>;
	animationAuthorityBodyIds: Set<string>;
	dynamicBodyIds: Set<string>;
	bodyStateCacheById: Map<string, CachedBodyState>;
	bodyBroadphaseRadiusById: Map<string, number>;
	sleepByBodyId: Map<string, boolean>;
	ccdByBodyId: Map<string, boolean>;
	broadphase: BroadphaseRuntimeState;
	jointPairKeys: Set<string>;
	contactPairKeys: Set<string>;
	activePairKeys: Set<string>;
	islandIdByBodyId: Map<string, string>;
	islandsById: Map<string, SleepingIslandState>;
	awakeIslandIds: Set<string>;
	islandsDirty: boolean;
	forceStepNextFrame: boolean;
	controllerDirty: boolean;
	cachedStats: CachedWorldStats;
}

export class PhysicsSystem extends EventEmitter<PhysicsEvents> {
	private _adapter: IPhysicsEngineAdapter;
	private _geometryProvider: ICollisionGeometryProvider;
	private _simulator = new DefaultPhysicsSimulator();

	private _worldConfigById = new Map<string, PhysicsWorldConfig>();
	private _runtimeByWorldId = new Map<string, WorldRuntimeState>();
	private _bodyById = new Map<string, InternalBodyBinding>();
	private _bodyIdByNodeId = new Map<string, string>();
	private _colliderById = new Map<string, InternalColliderBinding>();
	private _jointById = new Map<string, InternalJointBinding>();
	private _controllerById = new Map<string, InternalControllerBinding>();
	private _eventQueueByWorld = new Map<string, PhysicsEvent[]>();
	private _entityNodeResolver:
		| ((entityId: PhysicsEntityId) => Node | null)
		| null = null;

	constructor(options: PhysicsSystemOptions = {}) {
		super();
		this._adapter = options.adapter ?? new SimplePhysicsAdapter();
		this._geometryProvider =
			options.geometryProvider ?? new DefaultCollisionGeometryProvider();
	}

	public setEntityNodeResolver(
		resolver: ((entityId: PhysicsEntityId) => Node | null) | null
	): void {
		this._entityNodeResolver = resolver;
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
		this._runtimeByWorldId.set(
			config.worldId,
			createWorldRuntime(config.worldId)
		);
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
				this._unregisterJointPair(worldId, joint.bodyAId, joint.bodyBId);
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
		this._runtimeByWorldId.delete(worldId);
		this._eventQueueByWorld.delete(worldId);
	}

	public attachBody(
		target: Node | PhysicsBodyNode | PhysicsEntityId,
		desc?: BodyBinding
	): PhysicsBodyHandle {
		const node = this._resolveNodeTarget(target);
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
			entityId: typeof target === "number" ? target : undefined,
			authority,
			body: {
				...binding.body,
			},
			colliderIds: new Set(),
			broadphaseRadius: DEFAULT_BROADPHASE_BODY_RADIUS,
		};
		this._adapter.createBody(
			binding.worldId,
			bodyId,
			handle.body,
			this._resolveNodeTransform(node)
		);

		this._bodyById.set(bodyId, handle);
		this._bodyIdByNodeId.set(node.id, bodyId);
		this._registerBodyRuntime(handle);

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
				this._unregisterJointPair(joint.worldId, joint.bodyAId, joint.bodyBId);
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
		this._unregisterBodyRuntime(body);
	}

	public addCollider(
		target: Node | PhysicsBodyHandle | PhysicsEntityId,
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
		this._updateBodyBroadphaseRadius(body, desc, shape);
		this._markWorldDirtyForStep(body.worldId);
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
		this._registerJointPair(desc.worldId, bodyA.id, bodyB.id);
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
			moveAndSlide: (direction, deltaSeconds) => {
				this._markWorldControllerDirty(desc.worldId);
				return this._adapter.moveCharacterController(
					desc.worldId,
					controllerId,
					direction,
					Math.max(0, deltaSeconds)
				);
			},
			jump: (speed) => {
				this._markWorldControllerDirty(desc.worldId);
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
		this._markWorldDirtyForStep(desc.worldId);
		return handle;
	}

	public raycast(query: PhysicsRaycastQuery): PhysicsQueryHit | null {
		this._assertCapability("query");
		const worldId = this._resolveQueryWorldId(query.worldId);
		return this._adapter.raycast(worldId, query);
	}

	public sphereCast(query: PhysicsSphereCastQuery): PhysicsQueryHit | null {
		this._assertCapability("shapeCast");
		const worldId = this._resolveQueryWorldId(query.worldId);
		return this._adapter.sphereCast(worldId, query);
	}

	public boxCast(query: PhysicsBoxCastQuery): PhysicsQueryHit | null {
		this._assertCapability("shapeCast");
		const worldId = this._resolveQueryWorldId(query.worldId);
		return this._adapter.boxCast(worldId, query);
	}

	public overlapSphere(query: PhysicsOverlapSphereQuery): PhysicsOverlapHit[] {
		this._assertCapability("query");
		const worldId = this._resolveQueryWorldId(query.worldId);
		return this._adapter.overlapSphere(worldId, query);
	}

	public overlapBox(query: PhysicsOverlapBoxQuery): PhysicsOverlapHit[] {
		this._assertCapability("query");
		const worldId = this._resolveQueryWorldId(query.worldId);
		return this._adapter.overlapBox(worldId, query);
	}

	public step(
		deltaTimeSeconds: number,
		opts: StepOverride = {}
	): PhysicsStepReport {
		const targetWorlds = this._resolveStepWorldTargets(opts);
		const worldIds = targetWorlds.map((item) => item.worldId);
		this._beginStepRuntime(worldIds);
		this._syncAnimationAuthorityBodies(worldIds);

		const simulationContext = {
			worlds: targetWorlds,
			stepWorld: (worldId: string, deltaSeconds: number) =>
				this._stepWorldWithOptimizations(worldId, deltaSeconds),
		};

		this._simulator.beginFrame(simulationContext);
		const simulation = this._simulator.simulate(simulationContext, {
			deltaTimeSeconds,
			override: opts,
		});
		this._simulator.endFrame();
		return this._finalizeStepReport(deltaTimeSeconds, simulation);
	}

	public async stepAsync(
		deltaTimeSeconds: number,
		opts: StepOverride = {}
	): Promise<PhysicsStepReport> {
		const stepWorldAsync = this._adapter.stepWorldAsync;
		const simulateAsync = this._simulator.simulateAsync;
		if (!stepWorldAsync || !simulateAsync) {
			return this.step(deltaTimeSeconds, opts);
		}

		const targetWorlds = this._resolveStepWorldTargets(opts);
		const worldIds = targetWorlds.map((item) => item.worldId);
		this._beginStepRuntime(worldIds);
		this._syncAnimationAuthorityBodies(worldIds);

		const simulationContext = {
			worlds: targetWorlds,
			stepWorld: (worldId: string, deltaSeconds: number) =>
				this._stepWorldWithOptimizationsAsync(worldId, deltaSeconds),
		};

		this._simulator.beginFrame(simulationContext);
		const simulation = await simulateAsync.call(
			this._simulator,
			simulationContext,
			{
				deltaTimeSeconds,
				override: opts,
			}
		);
		this._simulator.endFrame();
		return this._finalizeStepReport(deltaTimeSeconds, simulation);
	}

	private _resolveStepWorldTargets(opts: StepOverride) {
		const worldIds = opts.worldIds ?? Array.from(this._worldConfigById.keys());
		return worldIds.map((worldId) => {
			const config = this._requireWorld(worldId);
			return { worldId, config };
		});
	}

	private _finalizeStepReport(
		deltaTimeSeconds: number,
		simulation: PhysicsSimulationResult
	): PhysicsStepReport {
		const worldReports: PhysicsWorldStepReport[] = [];
		const events: PhysicsEvent[] = [];
		const movedBodyIds = new Set<string>();
		const dirtyWorldIds = new Set<string>();

		for (const worldResult of simulation.worldResults) {
			const runtime = this._runtimeByWorldId.get(worldResult.worldId);
			let activeBodies = runtime?.cachedStats.activeBodies ?? 0;
			let sleepingBodies = runtime?.cachedStats.sleepingBodies ?? 0;
			let ccdBodies = runtime?.cachedStats.ccdBodies ?? 0;
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
				this._setWorldCachedStats(worldResult.worldId, {
					activeBodies: stepResult.activeBodies,
					sleepingBodies: stepResult.sleepingBodies,
					ccdBodies: stepResult.ccdBodies,
				});
				for (const event of stepResult.events) {
					events.push(event);
					this._enqueueEvent(event);
					this._trackPairActivity(event);
					dirtyWorldIds.add(event.worldId);
				}
			}
			this._rebuildIslandsIfNeeded(worldResult.worldId);

			worldReports.push({
				worldId: worldResult.worldId,
				mode: worldResult.mode,
				substeps: worldResult.substeps,
				consumedDeltaSeconds: worldResult.consumedDeltaSeconds,
				activeBodies,
				sleepingBodies,
				ccdBodies,
			});
		}

		const dirty = movedBodyIds.size > 0 || events.length > 0;
		const report: PhysicsStepReport = {
			inputDeltaSeconds: Math.max(0, deltaTimeSeconds),
			processedDeltaSeconds: simulation.processedDeltaSeconds,
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

	private _beginStepRuntime(worldIds: string[]): void {
		for (const worldId of worldIds) {
			const runtime = this._runtimeByWorldId.get(worldId);
			if (!runtime) continue;
			this._refreshAnimationAuthorityIndex(runtime);
			runtime.broadphase.dirtyBodyIds.clear();
			runtime.broadphase.dirtyCells.clear();
			runtime.broadphase.dirtyBounds = null;
		}
	}

	private _stepWorldWithOptimizations(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) {
			return this._adapter.stepWorld(worldId, deltaSeconds);
		}
		if (deltaSeconds <= 0) {
			return this._buildSkippedStepResult(runtime);
		}
		if (this._canSkipWorldStep(runtime)) {
			return this._buildSkippedStepResult(runtime);
		}
		const result = this._adapter.stepWorld(worldId, deltaSeconds);
		runtime.forceStepNextFrame = false;
		runtime.controllerDirty = false;
		return result;
	}

	private async _stepWorldWithOptimizationsAsync(
		worldId: string,
		deltaSeconds: number
	): Promise<PhysicsAdapterStepResult> {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) {
			if (this._adapter.stepWorldAsync) {
				return this._adapter.stepWorldAsync(worldId, deltaSeconds);
			}
			return this._adapter.stepWorld(worldId, deltaSeconds);
		}
		if (deltaSeconds <= 0) {
			return this._buildSkippedStepResult(runtime);
		}
		if (this._canSkipWorldStep(runtime)) {
			return this._buildSkippedStepResult(runtime);
		}

		let result: PhysicsAdapterStepResult;
		if (this._adapter.stepWorldAsync) {
			result = await this._adapter.stepWorldAsync(worldId, deltaSeconds);
		} else {
			result = this._adapter.stepWorld(worldId, deltaSeconds);
		}
		runtime.forceStepNextFrame = false;
		runtime.controllerDirty = false;
		return result;
	}

	private _canSkipWorldStep(runtime: WorldRuntimeState): boolean {
		if (runtime.forceStepNextFrame) return false;
		if (runtime.controllerDirty) return false;
		if (runtime.broadphase.dirtyBodyIds.size > 0) return false;
		if (runtime.activePairKeys.size > 0) return false;
		this._rebuildIslandsIfNeeded(runtime.worldId);
		return runtime.awakeIslandIds.size === 0;
	}

	private _buildSkippedStepResult(
		runtime: WorldRuntimeState
	): PhysicsAdapterStepResult {
		return {
			bodyStates: [],
			events: [],
			activeBodies: runtime.cachedStats.activeBodies,
			sleepingBodies: runtime.cachedStats.sleepingBodies,
			ccdBodies: runtime.cachedStats.ccdBodies,
		};
	}

	private _syncAnimationAuthorityBodies(worldIds: string[]): void {
		for (const worldId of worldIds) {
			const runtime = this._runtimeByWorldId.get(worldId);
			if (!runtime) continue;

			this._refreshAnimationAuthorityIndex(runtime);
			for (const bodyId of runtime.animationAuthorityBodyIds) {
				const body = this._bodyById.get(bodyId);
				if (!body) continue;

				const cache = runtime.bodyStateCacheById.get(body.id);
				if (cache && !this._hasNodeTransformDelta(cache, body.node)) {
					continue;
				}

				const transform = this._resolveNodeTransform(body.node);
				if (cache) {
					this._markBroadphaseBodyDirtyFromCache(
						worldId,
						body.id,
						cache,
						transform,
						this._resolveBodyBroadphaseRadius(runtime, body)
					);
				}

				this._adapter.setBodyTransform(body.worldId, body.id, transform);
				this._setCachedBodyState(runtime, body.id, transform, false);
				this._setBodySleepingState(runtime, body.id, false);
				runtime.forceStepNextFrame = true;
			}
		}
	}

	private _applyPhysicsAuthorityState(
		stepResult: PhysicsAdapterStepResult,
		movedBodyIds: Set<string>,
		dirtyWorldIds: Set<string>,
		worldId: string
	): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;

		for (const state of stepResult.bodyStates) {
			const body = this._bodyById.get(state.bodyId);
			if (!body) continue;

			const cache = runtime.bodyStateCacheById.get(body.id);
			const transformChanged =
				!cache || this._hasTransformDelta(cache, state.transform);

			this._setBodySleepingState(runtime, body.id, state.sleeping);
			runtime.ccdByBodyId.set(body.id, state.ccd);

			if (transformChanged && cache) {
				this._markBroadphaseBodyDirtyFromCache(
					worldId,
					body.id,
					cache,
					state.transform,
					this._resolveBodyBroadphaseRadius(runtime, body)
				);
			}
			this._setCachedBodyState(
				runtime,
				body.id,
				state.transform,
				state.sleeping
			);

			if (body.authority !== "physics") continue;
			if (!transformChanged) continue;

			this._applyNodeTransform(body.node, state.transform);
			movedBodyIds.add(body.id);
			dirtyWorldIds.add(worldId);
		}
	}

	private _setCachedBodyState(
		runtime: WorldRuntimeState,
		bodyId: string,
		transform: PhysicsTransform,
		sleeping: boolean
	): void {
		const existing = runtime.bodyStateCacheById.get(bodyId);
		if (existing) {
			existing.positionX = transform.position.x;
			existing.positionY = transform.position.y;
			existing.positionZ = transform.position.z;
			existing.rotationX = transform.rotation[0];
			existing.rotationY = transform.rotation[1];
			existing.rotationZ = transform.rotation[2];
			existing.rotationW = transform.rotation[3];
			existing.sleeping = sleeping;
			return;
		}
		runtime.bodyStateCacheById.set(bodyId, {
			positionX: transform.position.x,
			positionY: transform.position.y,
			positionZ: transform.position.z,
			rotationX: transform.rotation[0],
			rotationY: transform.rotation[1],
			rotationZ: transform.rotation[2],
			rotationW: transform.rotation[3],
			sleeping,
		});
	}

	private _hasTransformDelta(
		cache: CachedBodyState,
		transform: PhysicsTransform
	): boolean {
		return (
			Math.abs(cache.positionX - transform.position.x) > TRANSFORM_EPSILON ||
			Math.abs(cache.positionY - transform.position.y) > TRANSFORM_EPSILON ||
			Math.abs(cache.positionZ - transform.position.z) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationX - transform.rotation[0]) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationY - transform.rotation[1]) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationZ - transform.rotation[2]) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationW - transform.rotation[3]) > TRANSFORM_EPSILON
		);
	}

	private _hasNodeTransformDelta(cache: CachedBodyState, node: Node): boolean {
		return (
			Math.abs(cache.positionX - node.position.x) > TRANSFORM_EPSILON ||
			Math.abs(cache.positionY - node.position.y) > TRANSFORM_EPSILON ||
			Math.abs(cache.positionZ - node.position.z) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationX - node.quaternion.x) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationY - node.quaternion.y) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationZ - node.quaternion.z) > TRANSFORM_EPSILON ||
			Math.abs(cache.rotationW - node.quaternion.w) > TRANSFORM_EPSILON
		);
	}

	private _markBroadphaseBodyDirtyFromCache(
		worldId: string,
		bodyId: string,
		cache: CachedBodyState,
		transform: PhysicsTransform,
		radius: number
	): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;

		const r = Math.max(0.001, radius);
		const minX = Math.min(cache.positionX, transform.position.x) - r;
		const minY = Math.min(cache.positionY, transform.position.y) - r;
		const minZ = Math.min(cache.positionZ, transform.position.z) - r;
		const maxX = Math.max(cache.positionX, transform.position.x) + r;
		const maxY = Math.max(cache.positionY, transform.position.y) + r;
		const maxZ = Math.max(cache.positionZ, transform.position.z) + r;

		runtime.broadphase.dirtyBodyIds.add(bodyId);
		this._expandBroadphaseBounds(runtime, minX, minY, minZ, maxX, maxY, maxZ);
		this._insertDirtyCells(runtime, minX, minY, minZ, maxX, maxY, maxZ);
	}

	private _expandBroadphaseBounds(
		runtime: WorldRuntimeState,
		minX: number,
		minY: number,
		minZ: number,
		maxX: number,
		maxY: number,
		maxZ: number
	): void {
		if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return;
		const bounds = runtime.broadphase.dirtyBounds;
		if (!bounds) {
			runtime.broadphase.dirtyBounds = {
				minX,
				minY,
				minZ,
				maxX,
				maxY,
				maxZ,
			};
			return;
		}
		bounds.minX = Math.min(bounds.minX, minX);
		bounds.minY = Math.min(bounds.minY, minY);
		bounds.minZ = Math.min(bounds.minZ, minZ);
		bounds.maxX = Math.max(bounds.maxX, maxX);
		bounds.maxY = Math.max(bounds.maxY, maxY);
		bounds.maxZ = Math.max(bounds.maxZ, maxZ);
	}

	private _insertDirtyCells(
		runtime: WorldRuntimeState,
		minX: number,
		minY: number,
		minZ: number,
		maxX: number,
		maxY: number,
		maxZ: number
	): void {
		if (runtime.broadphase.dirtyCells.has("*")) return;

		const minCellX = Math.floor(minX / BROADPHASE_CELL_SIZE);
		const minCellY = Math.floor(minY / BROADPHASE_CELL_SIZE);
		const minCellZ = Math.floor(minZ / BROADPHASE_CELL_SIZE);
		const maxCellX = Math.floor(maxX / BROADPHASE_CELL_SIZE);
		const maxCellY = Math.floor(maxY / BROADPHASE_CELL_SIZE);
		const maxCellZ = Math.floor(maxZ / BROADPHASE_CELL_SIZE);

		const cellCount =
			(maxCellX - minCellX + 1) *
			(maxCellY - minCellY + 1) *
			(maxCellZ - minCellZ + 1);

		if (cellCount > BROADPHASE_MAX_DIRTY_CELLS) {
			runtime.broadphase.dirtyCells.clear();
			runtime.broadphase.dirtyCells.add("*");
			return;
		}

		for (let x = minCellX; x <= maxCellX; x++) {
			for (let y = minCellY; y <= maxCellY; y++) {
				for (let z = minCellZ; z <= maxCellZ; z++) {
					runtime.broadphase.dirtyCells.add(`${x}|${y}|${z}`);
				}
			}
		}
	}

	private _setBodySleepingState(
		runtime: WorldRuntimeState,
		bodyId: string,
		sleeping: boolean
	): void {
		const previous = runtime.sleepByBodyId.get(bodyId);
		if (previous === sleeping) return;
		runtime.sleepByBodyId.set(bodyId, sleeping);

		const islandId = runtime.islandIdByBodyId.get(bodyId);
		if (!islandId) return;
		const island = runtime.islandsById.get(islandId);
		if (!island) return;

		const body = this._bodyById.get(bodyId);
		if (!body) return;
		if ((body.body.type ?? "dynamic") !== "dynamic") return;

		if (previous === true) {
			island.sleepingDynamicBodyCount = Math.max(
				0,
				island.sleepingDynamicBodyCount - 1
			);
		}
		if (sleeping) {
			island.sleepingDynamicBodyCount++;
		}

		if (island.sleepingDynamicBodyCount >= island.dynamicBodyCount) {
			runtime.awakeIslandIds.delete(islandId);
		} else {
			runtime.awakeIslandIds.add(islandId);
		}
	}

	private _trackPairActivity(event: PhysicsEvent): void {
		const runtime = this._runtimeByWorldId.get(event.worldId);
		if (!runtime) return;

		const pairKey = makeBodyPairKey(event.bodyAId, event.bodyBId);
		const active =
			event.type === "collisionBegin" ||
			event.type === "collisionStay" ||
			event.type === "triggerBegin" ||
			event.type === "triggerStay";
		if (active) runtime.activePairKeys.add(pairKey);
		else runtime.activePairKeys.delete(pairKey);

		if (!event.type.startsWith("collision")) return;
		const had = runtime.contactPairKeys.has(pairKey);
		if (active) runtime.contactPairKeys.add(pairKey);
		else runtime.contactPairKeys.delete(pairKey);
		if (had !== runtime.contactPairKeys.has(pairKey)) {
			runtime.islandsDirty = true;
		}
	}

	private _registerBodyRuntime(body: InternalBodyBinding): void {
		const runtime = this._requireRuntime(body.worldId);
		runtime.bodyIds.add(body.id);
		runtime.bodyBroadphaseRadiusById.set(body.id, body.broadphaseRadius);
		runtime.sleepByBodyId.set(body.id, false);
		runtime.ccdByBodyId.set(body.id, this._resolveBodyCcd(body));
		this._setCachedBodyState(
			runtime,
			body.id,
			this._resolveNodeTransform(body.node),
			false
		);
		if ((body.body.type ?? "dynamic") === "dynamic") {
			runtime.dynamicBodyIds.add(body.id);
		}
		if (body.authority === "animation") {
			runtime.animationAuthorityBodyIds.add(body.id);
		}
		runtime.islandsDirty = true;
		runtime.forceStepNextFrame = true;
		this._recomputeWorldCachedStats(runtime);
	}

	private _unregisterBodyRuntime(body: InternalBodyBinding): void {
		const runtime = this._runtimeByWorldId.get(body.worldId);
		if (!runtime) return;

		const wasSleeping = runtime.sleepByBodyId.get(body.id) === true;
		runtime.bodyIds.delete(body.id);
		runtime.animationAuthorityBodyIds.delete(body.id);
		runtime.dynamicBodyIds.delete(body.id);
		runtime.bodyStateCacheById.delete(body.id);
		runtime.bodyBroadphaseRadiusById.delete(body.id);
		runtime.sleepByBodyId.delete(body.id);
		runtime.ccdByBodyId.delete(body.id);

		const islandId = runtime.islandIdByBodyId.get(body.id);
		if (islandId) {
			const island = runtime.islandsById.get(islandId);
			if (island) {
				island.bodyIds.delete(body.id);
				if ((body.body.type ?? "dynamic") === "dynamic") {
					island.dynamicBodyCount = Math.max(0, island.dynamicBodyCount - 1);
					if (wasSleeping) {
						island.sleepingDynamicBodyCount = Math.max(
							0,
							island.sleepingDynamicBodyCount - 1
						);
					}
				}
				if (island.bodyIds.size === 0) {
					runtime.islandsById.delete(islandId);
					runtime.awakeIslandIds.delete(islandId);
				}
			}
		}
		runtime.islandIdByBodyId.delete(body.id);

		let removedPairs = false;
		for (const pairSet of [
			runtime.jointPairKeys,
			runtime.contactPairKeys,
			runtime.activePairKeys,
		]) {
			for (const pairKey of Array.from(pairSet)) {
				if (!pairKeyContainsBody(pairKey, body.id)) continue;
				pairSet.delete(pairKey);
				removedPairs = true;
			}
		}
		if (removedPairs) runtime.islandsDirty = true;
		runtime.islandsDirty = true;
		runtime.forceStepNextFrame = true;
		this._recomputeWorldCachedStats(runtime);
	}

	private _refreshAnimationAuthorityIndex(runtime: WorldRuntimeState): void {
		for (const bodyId of runtime.bodyIds) {
			const body = this._bodyById.get(bodyId);
			if (!body || body.worldId !== runtime.worldId) {
				runtime.animationAuthorityBodyIds.delete(bodyId);
				continue;
			}
			if (body.authority === "animation") {
				runtime.animationAuthorityBodyIds.add(bodyId);
			} else {
				runtime.animationAuthorityBodyIds.delete(bodyId);
			}
		}
	}

	private _registerJointPair(
		worldId: string,
		bodyAId: string,
		bodyBId: string
	): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;
		const pairKey = makeBodyPairKey(bodyAId, bodyBId);
		if (runtime.jointPairKeys.has(pairKey)) return;
		runtime.jointPairKeys.add(pairKey);
		runtime.islandsDirty = true;
		runtime.forceStepNextFrame = true;
	}

	private _unregisterJointPair(
		worldId: string,
		bodyAId: string,
		bodyBId: string
	): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;
		const pairKey = makeBodyPairKey(bodyAId, bodyBId);
		if (!runtime.jointPairKeys.delete(pairKey)) return;
		runtime.islandsDirty = true;
		runtime.forceStepNextFrame = true;
	}

	private _rebuildIslandsIfNeeded(worldId: string): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime || !runtime.islandsDirty) return;

		runtime.islandIdByBodyId.clear();
		runtime.islandsById.clear();
		runtime.awakeIslandIds.clear();

		const adjacency = new Map<string, Set<string>>();
		for (const bodyId of runtime.bodyIds) {
			adjacency.set(bodyId, new Set());
		}
		for (const pairKey of runtime.jointPairKeys) {
			this._appendIslandEdge(adjacency, pairKey);
		}
		for (const pairKey of runtime.contactPairKeys) {
			this._appendIslandEdge(adjacency, pairKey);
		}

		const visited = new Set<string>();
		let islandIndex = 0;

		for (const bodyId of runtime.bodyIds) {
			if (visited.has(bodyId)) continue;
			const islandId = `island:${islandIndex++}`;
			const pending = [bodyId];
			const islandBodies = new Set<string>();
			let dynamicBodyCount = 0;
			let sleepingDynamicBodyCount = 0;

			while (pending.length > 0) {
				const next = pending.pop();
				if (!next || visited.has(next)) continue;
				visited.add(next);
				islandBodies.add(next);
				runtime.islandIdByBodyId.set(next, islandId);

				const body = this._bodyById.get(next);
				if (body && (body.body.type ?? "dynamic") === "dynamic") {
					dynamicBodyCount++;
					if (runtime.sleepByBodyId.get(next) === true) {
						sleepingDynamicBodyCount++;
					}
				}

				const neighbors = adjacency.get(next);
				if (!neighbors) continue;
				for (const neighbor of neighbors) {
					if (!visited.has(neighbor)) {
						pending.push(neighbor);
					}
				}
			}

			const island: SleepingIslandState = {
				id: islandId,
				bodyIds: islandBodies,
				dynamicBodyCount,
				sleepingDynamicBodyCount,
			};
			runtime.islandsById.set(islandId, island);
			if (dynamicBodyCount > sleepingDynamicBodyCount) {
				runtime.awakeIslandIds.add(islandId);
			}
		}

		runtime.islandsDirty = false;
	}

	private _appendIslandEdge(
		adjacency: Map<string, Set<string>>,
		pairKey: string
	): void {
		const pair = splitPairKey(pairKey);
		if (!pair) return;
		const [bodyAId, bodyBId] = pair;
		const a = adjacency.get(bodyAId);
		const b = adjacency.get(bodyBId);
		if (!a || !b) return;
		a.add(bodyBId);
		b.add(bodyAId);
	}

	private _resolveBodyBroadphaseRadius(
		runtime: WorldRuntimeState,
		body: InternalBodyBinding
	): number {
		return (
			runtime.bodyBroadphaseRadiusById.get(body.id) ?? body.broadphaseRadius
		);
	}

	private _updateBodyBroadphaseRadius(
		body: InternalBodyBinding,
		desc: ColliderDescriptor,
		shape: ColliderShape
	): void {
		const radius = computeColliderBroadphaseRadius(shape, desc.offset);
		if (radius <= body.broadphaseRadius) return;
		body.broadphaseRadius = radius;
		const runtime = this._runtimeByWorldId.get(body.worldId);
		if (!runtime) return;
		runtime.bodyBroadphaseRadiusById.set(body.id, radius);
	}

	private _resolveBodyCcd(body: InternalBodyBinding): boolean {
		const world = this._worldConfigById.get(body.worldId);
		return body.body.ccd ?? world?.enableCCD ?? false;
	}

	private _markWorldDirtyForStep(worldId: string): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;
		runtime.forceStepNextFrame = true;
	}

	private _markWorldControllerDirty(worldId: string): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;
		runtime.controllerDirty = true;
		runtime.forceStepNextFrame = true;
	}

	private _setWorldCachedStats(worldId: string, stats: CachedWorldStats): void {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (!runtime) return;
		runtime.cachedStats.activeBodies = stats.activeBodies;
		runtime.cachedStats.sleepingBodies = stats.sleepingBodies;
		runtime.cachedStats.ccdBodies = stats.ccdBodies;
	}

	private _recomputeWorldCachedStats(runtime: WorldRuntimeState): void {
		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const bodyId of runtime.bodyIds) {
			if (runtime.sleepByBodyId.get(bodyId) === true) sleepingBodies++;
			if (runtime.ccdByBodyId.get(bodyId) === true) ccdBodies++;
		}
		runtime.cachedStats.sleepingBodies = sleepingBodies;
		runtime.cachedStats.activeBodies = Math.max(
			0,
			runtime.bodyIds.size - sleepingBodies
		);
		runtime.cachedStats.ccdBodies = ccdBodies;
	}

	private _applyNodeTransform(node: Node, transform: PhysicsTransform): void {
		node.position.set(
			transform.position.x,
			transform.position.y,
			transform.position.z
		);
		node.quaternion.x = transform.rotation[0];
		node.quaternion.y = transform.rotation[1];
		node.quaternion.z = transform.rotation[2];
		node.quaternion.w = transform.rotation[3];
		node.updateLocalMatrix();
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

	private _resolveBody(
		target: Node | PhysicsBodyHandle | PhysicsEntityId
	): InternalBodyBinding {
		if (typeof target === "number") {
			if (!this._entityNodeResolver) {
				throw new Error(
					"PhysicsSystem entity target requires setEntityNodeResolver()"
				);
			}
			const node = this._entityNodeResolver(target);
			if (!node) {
				throw new Error(`Entity "${target}" is not bound to a Node`);
			}
			return this._resolveBody(node);
		}

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
		target: Node | PhysicsBodyHandle | string | PhysicsEntityId
	): InternalBodyBinding {
		if (typeof target === "string") {
			const body = this._bodyById.get(target);
			if (body) return body;
			throw new Error(`Physics body "${target}" does not exist`);
		}
		return this._resolveBody(target);
	}

	private _resolveNodeTarget(
		target: Node | PhysicsBodyNode | PhysicsEntityId
	): Node | PhysicsBodyNode {
		if (typeof target !== "number") return target;
		if (!this._entityNodeResolver) {
			throw new Error(
				"attachBody(entityId, desc) requires setEntityNodeResolver()"
			);
		}
		const node = this._entityNodeResolver(target);
		if (node) return node;
		throw new Error(`Entity "${target}" is not bound to a Node`);
	}

	private _requireWorld(worldId: string): PhysicsWorldConfig {
		const world = this._worldConfigById.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}

	private _requireRuntime(worldId: string): WorldRuntimeState {
		const runtime = this._runtimeByWorldId.get(worldId);
		if (runtime) return runtime;
		throw new Error(`Physics world runtime "${worldId}" does not exist`);
	}

	private _resolveQueryWorldId(worldId?: string): string {
		if (worldId && worldId.trim().length > 0) {
			this._requireWorld(worldId);
			return worldId;
		}
		if (this._worldConfigById.size === 1) {
			return this._worldConfigById.keys().next().value;
		}
		if (this._worldConfigById.size === 0) {
			throw new Error(
				"Physics query requires an active world, but no worlds are created"
			);
		}
		throw new Error(
			"Physics query.worldId is required when multiple worlds are active"
		);
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

function createWorldRuntime(worldId: string): WorldRuntimeState {
	return {
		worldId,
		bodyIds: new Set(),
		animationAuthorityBodyIds: new Set(),
		dynamicBodyIds: new Set(),
		bodyStateCacheById: new Map(),
		bodyBroadphaseRadiusById: new Map(),
		sleepByBodyId: new Map(),
		ccdByBodyId: new Map(),
		broadphase: {
			dirtyBodyIds: new Set(),
			dirtyCells: new Set(),
			dirtyBounds: null,
		},
		jointPairKeys: new Set(),
		contactPairKeys: new Set(),
		activePairKeys: new Set(),
		islandIdByBodyId: new Map(),
		islandsById: new Map(),
		awakeIslandIds: new Set(),
		islandsDirty: false,
		forceStepNextFrame: false,
		controllerDirty: false,
		cachedStats: {
			activeBodies: 0,
			sleepingBodies: 0,
			ccdBodies: 0,
		},
	};
}

function makeBodyPairKey(left: string, right: string): string {
	return left < right ? `${left}|${right}` : `${right}|${left}`;
}

function splitPairKey(pairKey: string): [string, string] | null {
	const separator = pairKey.indexOf("|");
	if (separator <= 0 || separator >= pairKey.length - 1) return null;
	return [pairKey.slice(0, separator), pairKey.slice(separator + 1)];
}

function pairKeyContainsBody(pairKey: string, bodyId: string): boolean {
	const pair = splitPairKey(pairKey);
	if (!pair) return false;
	return pair[0] === bodyId || pair[1] === bodyId;
}

function computeColliderBroadphaseRadius(
	shape: ColliderShape,
	offset?: IVector3
): number {
	const offsetRadius = offset ? Math.hypot(offset.x, offset.y, offset.z) : 0;
	return computeShapeRadius(shape) + offsetRadius;
}

function computeShapeRadius(shape: ColliderShape): number {
	switch (shape.kind) {
		case "sphere":
			return Math.max(0.001, shape.radius);
		case "capsule":
			return Math.max(0.001, shape.radius + shape.halfHeight);
		case "cylinder":
			return Math.max(
				0.001,
				Math.sqrt(
					shape.radius * shape.radius + shape.halfHeight * shape.halfHeight
				)
			);
		case "box":
			return Math.max(
				0.001,
				Math.hypot(
					shape.halfExtents.x,
					shape.halfExtents.y,
					shape.halfExtents.z
				)
			);
		case "trimesh": {
			const vertices = shape.vertices;
			const length = vertices.length;
			if (length < 3) return 0.5;
			let maxRadiusSq = 0;
			for (let i = 0; i < length; i += 3) {
				const x = vertices[i];
				const y = vertices[i + 1];
				const z = vertices[i + 2];
				const radiusSq = x * x + y * y + z * z;
				if (radiusSq > maxRadiusSq) maxRadiusSq = radiusSq;
			}
			return Math.max(0.001, Math.sqrt(maxRadiusSq));
		}
		default:
			return 0.5;
	}
}
