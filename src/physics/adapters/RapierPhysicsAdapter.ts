import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterBodyState,
	PhysicsAdapterCapabilities,
	PhysicsAdapterStepResult,
} from "../IPhysicsEngineAdapter";
import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import { DEFAULT_GRAVITY } from "../constants";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	PhysicsBoxCastQuery,
	PhysicsEvent,
	PhysicsEventType,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
	PhysicsQueryFilter,
	PhysicsQueryHit,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
	RigidBodyType,
} from "../types";
import { SimplePhysicsAdapter } from "./SimplePhysicsAdapter";

interface RapierModuleLike {
	init?: () => Promise<void> | void;
	World?: new (gravity: unknown) => unknown;
	Vector3?: new (x: number, y: number, z: number) => unknown;
	Quaternion?: new (x: number, y: number, z: number, w: number) => unknown;
	RigidBodyDesc?: {
		dynamic?: () => unknown;
		fixed?: () => unknown;
		kinematicPositionBased?: () => unknown;
		kinematicVelocityBased?: () => unknown;
	};
	ColliderDesc?: {
		cuboid?: (x: number, y: number, z: number) => unknown;
		ball?: (radius: number) => unknown;
		capsule?: (halfHeight: number, radius: number) => unknown;
		cylinder?: (halfHeight: number, radius: number) => unknown;
		trimesh?: (vertices: Float32Array, indices: Uint32Array) => unknown;
	};
	JointData?: {
		fixed?: (
			anchorA: unknown,
			frameA: unknown,
			anchorB: unknown,
			frameB: unknown
		) => unknown;
		revolute?: (anchorA: unknown, anchorB: unknown, axis: unknown) => unknown;
		spring?: (
			restLength: number,
			stiffness: number,
			damping: number,
			anchorA: unknown,
			anchorB: unknown
		) => unknown;
	};
}

interface RapierBodyState {
	id: string;
	type: RigidBodyType;
	descriptor: RigidBodyDescriptor;
	rigidBody: any;
	sleeping: boolean;
	ccd: boolean;
	colliderIds: Set<string>;
}

interface RapierColliderState {
	id: string;
	bodyId: string;
	descriptor: ColliderDescriptor;
	shape: ColliderShape;
	trimeshBVH: TrimeshRayBVHEntry | null;
	rapierCollider: any;
	isTrigger: boolean;
	collisionMask: number;
	radius: number;
	halfExtents: IVector3;
	offset: IVector3;
}

interface RapierJointState {
	id: string;
	descriptor: JointDescriptor;
	rapierJoint?: any;
}

interface RapierControllerState {
	id: string;
	bodyId: string;
	grounded: boolean;
	offset: number;
	maxSlope: number;
	stepHeight: number;
	gravityScale: number | null;
	rapierController: any | null;
	movementCollider: any | null;
	ownsMovementCollider: boolean;
	pendingDirection: IVector3;
	pendingJumpSpeed: number;
	lastResolvedMovement: IVector3;
}

interface RapierWorldState {
	config: PhysicsWorldConfig;
	world: any;
	bodies: Map<string, RapierBodyState>;
	colliders: Map<string, RapierColliderState>;
	joints: Map<string, RapierJointState>;
	controllers: Map<string, RapierControllerState>;
	activePairs: Map<string, "collision" | "trigger">;
}

interface RapierQueryCandidate {
	body: RapierBodyState;
	collider: RapierColliderState;
	center: IVector3;
	rotation: [number, number, number, number];
}

interface RapierQueryHit {
	distance: number;
	point: IVector3;
	normal: IVector3;
}

interface TrimeshRayBVHNode {
	minX: number;
	minY: number;
	minZ: number;
	maxX: number;
	maxY: number;
	maxZ: number;
	start: number;
	end: number;
	left: TrimeshRayBVHNode | null;
	right: TrimeshRayBVHNode | null;
}

interface TrimeshRayBVHEntry {
	triangleOrder: Uint32Array;
	root: TrimeshRayBVHNode | null;
}

interface TrimeshTriangleBounds {
	minX: number;
	minY: number;
	minZ: number;
	maxX: number;
	maxY: number;
	maxZ: number;
}

interface RapierCharacterMoveResolution {
	movement: IVector3;
	grounded: boolean | null;
}

interface RapierCharacterColliderResolution {
	collider: any | null;
	owned: boolean;
}

const DEFAULT_COLLISION_MASK = 0xffffffff;

export interface RapierPhysicsAdapterOptions {
	moduleLoader?: () => Promise<unknown>;
	strict?: boolean;
	fallbackAdapter?: IPhysicsEngineAdapter;
}

export class RapierPhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id = "rapier";
	public readonly capabilities: PhysicsAdapterCapabilities;

	private _moduleLoader: () => Promise<unknown>;
	private _strict: boolean;
	private _delegate: IPhysicsEngineAdapter;
	private _loadedModule: unknown = null;
	private _rapier: RapierModuleLike | null = null;
	private _worlds = new Map<string, RapierWorldState>();

	constructor(options: RapierPhysicsAdapterOptions = {}) {
		this._moduleLoader =
			options.moduleLoader ??
			(async () => loadOptionalRapierModule());
		this._strict = options.strict ?? true;
		this._delegate =
			options.fallbackAdapter ?? new SimplePhysicsAdapter("rapier-fallback");
		this.capabilities = {
			joints: true,
			characterController: true,
			shapeCast: true,
			query: true,
			syncInit: false,
		};
	}

	public async init(): Promise<void> {
		try {
			this._loadedModule = await this._moduleLoader();
		} catch (error) {
			if (this._strict) {
				throw new Error(
					`RapierPhysicsAdapter failed to load "@dimforge/rapier3d-compat". Install it as an optional peer dependency or provide moduleLoader. Inner error: ${String(error)}`
				);
			}
			await this._delegate.init();
			return;
		}

		const rapier = resolveRapierModule(this._loadedModule);
		if (!rapier || !isRapierUsable(rapier)) {
			await this._delegate.init();
			return;
		}

		try {
			if (typeof rapier.init === "function") {
				await rapier.init();
			}
		} catch (error) {
			if (this._strict) {
				throw new Error(
					`RapierPhysicsAdapter failed to initialize the loaded Rapier module. Inner error: ${String(error)}`
				);
			}
			await this._delegate.init();
			return;
		}

		this._rapier = rapier;
	}

	public initSync(): void {
		throw new Error(
			"RapierPhysicsAdapter.initSync is not supported because Rapier requires asynchronous module loading. Use init() instead."
		);
	}

	public hasWorld(worldId: string): boolean {
		if (this._usingFallback()) return this._delegate.hasWorld(worldId);
		return this._worlds.has(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		if (this._usingFallback()) {
			this._delegate.createWorld(config);
			return;
		}

		if (this._worlds.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}

		const rapier = this._requireRapier();
		const gravity = cloneVector(config.gravity ?? DEFAULT_GRAVITY);
		const world = new rapier.World!(
			this._toRapierVector3(gravity) ?? {
				x: gravity.x,
				y: gravity.y,
				z: gravity.z,
			}
		);
		this._worlds.set(config.worldId, {
			config: {
				...config,
				gravity,
			},
			world,
			bodies: new Map(),
			colliders: new Map(),
			joints: new Map(),
			controllers: new Map(),
			activePairs: new Map(),
		});
	}

	public destroyWorld(worldId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyWorld(worldId);
			return;
		}
		const world = this._worlds.get(worldId);
		if (!world) return;
		for (const controller of world.controllers.values()) {
			this._disposeCharacterController(world, controller);
		}
		for (const bodyId of Array.from(world.bodies.keys())) {
			this.destroyBody(worldId, bodyId);
		}
		world.joints.clear();
		world.controllers.clear();
		world.colliders.clear();
		world.activePairs.clear();
		this._invoke(world.world, ["free"], [[]]);
		this._worlds.delete(worldId);
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		if (this._usingFallback()) {
			this._delegate.createBody(worldId, bodyId, descriptor, initialTransform);
			return;
		}

		const world = this._requireWorld(worldId);
		if (world.bodies.has(bodyId)) {
			throw new Error(
				`Physics body "${bodyId}" already exists in "${worldId}"`
			);
		}

		const rigidBodyDesc = this._createRigidBodyDescriptor(
			descriptor,
			initialTransform,
			world.config.allowSleep
		);
		const ccdEnabled = descriptor.ccd ?? world.config.enableCCD ?? false;
		if (ccdEnabled) {
			this._invoke(rigidBodyDesc, ["setCcdEnabled"], [[true]]);
		}
		const rigidBody = world.world.createRigidBody(rigidBodyDesc);
		if (!rigidBody) {
			throw new Error(
				`RapierPhysicsAdapter failed to create body "${bodyId}" in "${worldId}"`
			);
		}
		this._applyRigidBodyMassOverride(
			rigidBody,
			descriptor.type ?? "dynamic",
			descriptor.mass
		);

		if (descriptor.linearVelocity) {
			this._setVector3(
				rigidBody,
				["setLinvel"],
				descriptor.linearVelocity,
				false
			);
		}
		if (descriptor.angularVelocity) {
			this._setVector3(
				rigidBody,
				["setAngvel"],
				descriptor.angularVelocity,
				false
			);
		}

		world.bodies.set(bodyId, {
			id: bodyId,
			type: descriptor.type ?? "dynamic",
			descriptor: { ...descriptor },
			rigidBody,
			sleeping: false,
			ccd: ccdEnabled,
			colliderIds: new Set(),
		});
	}

	public destroyBody(worldId: string, bodyId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyBody(worldId, bodyId);
			return;
		}

		const world = this._requireWorld(worldId);
		const body = world.bodies.get(bodyId);
		if (!body) return;

		for (const colliderId of body.colliderIds) {
			const collider = world.colliders.get(colliderId);
			if (collider) {
				this._invoke(
					world.world,
					["removeCollider"],
					[
						[collider.rapierCollider],
						[collider.rapierCollider, true],
						[collider.rapierCollider, true, true],
					]
				);
			}
			world.colliders.delete(colliderId);
		}
		body.colliderIds.clear();

		for (const [jointId, joint] of world.joints) {
			const descriptor = joint.descriptor;
			const bodyAId = resolveJointBodyId(descriptor.bodyA);
			const bodyBId = resolveJointBodyId(descriptor.bodyB);
			if (bodyAId === bodyId || bodyBId === bodyId) {
				this.destroyJoint(worldId, jointId);
			}
		}
		for (const [controllerId, controller] of world.controllers) {
			if (controller.bodyId === bodyId) {
				this._disposeCharacterController(world, controller);
				world.controllers.delete(controllerId);
			}
		}

		this._invoke(
			world.world,
			["removeRigidBody"],
			[[body.rigidBody], [body.rigidBody, true]]
		);
		world.bodies.delete(bodyId);
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		if (this._usingFallback()) {
			this._delegate.setBodyTransform(worldId, bodyId, transform);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		const translation = cloneVector(transform.position);
		const rotation = cloneQuaternion(transform.rotation);
		const useKinematicSetters = body.type === "kinematic";

		if (useKinematicSetters) {
			this._setVector3(
				body.rigidBody,
				["setNextKinematicTranslation", "setTranslation"],
				translation,
				true
			);
			this._setQuaternion(
				body.rigidBody,
				["setNextKinematicRotation", "setRotation"],
				rotation,
				true
			);
		} else {
			this._setVector3(body.rigidBody, ["setTranslation"], translation, true);
			this._setQuaternion(body.rigidBody, ["setRotation"], rotation, true);
		}
		body.sleeping = false;
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		if (this._usingFallback()) {
			this._delegate.setBodyLinearVelocity(worldId, bodyId, velocity);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		this._setVector3(body.rigidBody, ["setLinvel"], velocity, true);
		body.sleeping = false;
	}

	public setAngularVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		if (this._usingFallback()) {
			this._delegate.setAngularVelocity(worldId, bodyId, velocity);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		this._setVector3(
			body.rigidBody,
			["setAngvel", "setAngularVelocity"],
			velocity,
			true
		);
		body.sleeping = false;
	}

	public applyForce(worldId: string, bodyId: string, force: IVector3): void {
		if (this._usingFallback()) {
			this._delegate.applyForce(worldId, bodyId, force);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		this._setVector3(
			body.rigidBody,
			["addForce", "applyForce"],
			force,
			true
		);
		body.sleeping = false;
	}

	public applyTorque(worldId: string, bodyId: string, torque: IVector3): void {
		if (this._usingFallback()) {
			this._delegate.applyTorque(worldId, bodyId, torque);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		this._setVector3(
			body.rigidBody,
			["addTorque", "applyTorque"],
			torque,
			true
		);
		body.sleeping = false;
	}

	public applyImpulse(
		worldId: string,
		bodyId: string,
		impulse: IVector3
	): void {
		if (this._usingFallback()) {
			this._delegate.applyImpulse(worldId, bodyId, impulse);
			return;
		}
		const body = this._requireBody(worldId, bodyId);
		this._setVector3(
			body.rigidBody,
			["applyImpulse", "addImpulse"],
			impulse,
			true
		);
		body.sleeping = false;
	}

	public addCollider(
		worldId: string,
		bodyId: string,
		colliderId: string,
		descriptor: ColliderDescriptor,
		shape: ColliderShape
	): void {
		if (this._usingFallback()) {
			this._delegate.addCollider(
				worldId,
				bodyId,
				colliderId,
				descriptor,
				shape
			);
			return;
		}
		const world = this._requireWorld(worldId);
		if (world.colliders.has(colliderId)) {
			throw new Error(
				`Physics collider "${colliderId}" already exists in "${worldId}"`
			);
		}
		const body = this._requireBody(worldId, bodyId);
		const colliderDesc = this._createColliderDescriptor(shape);
		const offset = cloneVector(descriptor.offset ?? { x: 0, y: 0, z: 0 });
		if (descriptor.isTrigger === true) {
			this._invoke(colliderDesc, ["setSensor"], [[true]]);
		}
		this._setVector3(colliderDesc, ["setTranslation"], offset);
		if (descriptor.material) {
			if (Number.isFinite(descriptor.material.friction)) {
				this._invoke(
					colliderDesc,
					["setFriction"],
					[[descriptor.material.friction]]
				);
			}
			if (Number.isFinite(descriptor.material.restitution)) {
				this._invoke(
					colliderDesc,
					["setRestitution"],
					[[descriptor.material.restitution]]
				);
			}
			if (Number.isFinite(descriptor.material.density)) {
				this._invoke(
					colliderDesc,
					["setDensity"],
					[[descriptor.material.density]]
				);
			}
		}

		const rapierCollider = world.world.createCollider(
			colliderDesc,
			body.rigidBody
		);
		if (!rapierCollider) {
			throw new Error(
				`RapierPhysicsAdapter failed to create collider "${colliderId}" in "${worldId}"`
			);
		}

		const collider: RapierColliderState = {
			id: colliderId,
			bodyId,
			descriptor,
			shape,
			trimeshBVH: buildTrimeshRayBVH(shape, descriptor),
			rapierCollider,
			isTrigger: descriptor.isTrigger === true,
			collisionMask: DEFAULT_COLLISION_MASK,
			radius: computeShapeRadius(shape),
			halfExtents: computeShapeHalfExtents(shape),
			offset,
		};
		world.colliders.set(colliderId, collider);
		body.colliderIds.add(colliderId);
		this._applyNativeCollisionMask(collider);
	}

	public destroyCollider(worldId: string, colliderId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyCollider(worldId, colliderId);
			return;
		}
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) return;

		this._invoke(
			world.world,
			["removeCollider"],
			[
				[collider.rapierCollider],
				[collider.rapierCollider, true],
				[collider.rapierCollider, true, true],
			]
		);
		world.colliders.delete(colliderId);
		const body = world.bodies.get(collider.bodyId);
		body?.colliderIds.delete(colliderId);
	}

	public setColliderSensor(
		worldId: string,
		colliderId: string,
		isSensor: boolean
	): void {
		if (this._usingFallback()) {
			this._delegate.setColliderSensor(worldId, colliderId, isSensor);
			return;
		}
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) {
			throw new Error(
				`Physics collider "${colliderId}" does not exist in "${worldId}"`
			);
		}
		collider.isTrigger = isSensor === true;
		this._invoke(
			collider.rapierCollider,
			["setSensor"],
			[[collider.isTrigger], [collider.isTrigger, true]]
		);
	}

	public setCollisionMask(
		worldId: string,
		colliderId: string,
		mask: number
	): void {
		if (this._usingFallback()) {
			this._delegate.setCollisionMask(worldId, colliderId, mask);
			return;
		}
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) {
			throw new Error(
				`Physics collider "${colliderId}" does not exist in "${worldId}"`
			);
		}
		collider.collisionMask = sanitizeCollisionMask(mask);
		this._applyNativeCollisionMask(collider);
	}

	public createJoint(
		worldId: string,
		jointId: string,
		descriptor: JointDescriptor
	): void {
		if (this._usingFallback()) {
			this._delegate.createJoint(worldId, jointId, descriptor);
			return;
		}

		const world = this._requireWorld(worldId);
		if (world.joints.has(jointId)) {
			throw new Error(
				`Physics joint "${jointId}" already exists in "${worldId}"`
			);
		}

		const bodyAId = resolveJointBodyId(descriptor.bodyA);
		const bodyBId = resolveJointBodyId(descriptor.bodyB);
		const bodyA = this._requireBody(worldId, bodyAId);
		const bodyB = this._requireBody(worldId, bodyBId);
		let rapierJoint: any = undefined;

		const jointData = this._createJointData(descriptor);
		if (jointData) {
			try {
				rapierJoint = world.world.createImpulseJoint(
					jointData,
					bodyA.rigidBody,
					bodyB.rigidBody,
					descriptor.collisionEnabled ?? false
				);
			} catch {}
		}

		world.joints.set(jointId, {
			id: jointId,
			descriptor,
			rapierJoint,
		});
	}

	public destroyJoint(worldId: string, jointId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyJoint(worldId, jointId);
			return;
		}
		const world = this._requireWorld(worldId);
		const joint = world.joints.get(jointId);
		if (!joint) return;
		if (joint.rapierJoint) {
			this._invoke(
				world.world,
				["removeImpulseJoint"],
				[[joint.rapierJoint, true], [joint.rapierJoint]]
			);
		}
		world.joints.delete(jointId);
	}

	public createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void {
		if (this._usingFallback()) {
			this._delegate.createCharacterController(
				worldId,
				controllerId,
				descriptor
			);
			return;
		}
		const world = this._requireWorld(worldId);
		if (world.controllers.has(controllerId)) {
			throw new Error(
				`Character controller "${controllerId}" already exists in "${worldId}"`
			);
		}
		const bodyId = resolveBodyId(descriptor.body);
		const body = this._requireBody(worldId, bodyId);
		const controller: RapierControllerState = {
			id: controllerId,
			bodyId,
			grounded: false,
			offset: Math.max(
				0.001,
				Math.min(
					Math.max(0, descriptor.radius) * 0.1 || 0.01,
					0.05
				)
			),
			maxSlope: Math.max(0, descriptor.maxSlope ?? 60),
			stepHeight: Math.max(0, descriptor.stepHeight ?? 0.3),
			gravityScale:
				Number.isFinite(descriptor.gravityScale) ?
					Number(descriptor.gravityScale)
				:	null,
			rapierController: null,
			movementCollider: null,
			ownsMovementCollider: false,
			pendingDirection: { x: 0, y: 0, z: 0 },
			pendingJumpSpeed: 0,
			lastResolvedMovement: { x: 0, y: 0, z: 0 },
		};

		this._bindRapierCharacterController(world, body, controller, descriptor);
		world.controllers.set(controllerId, controller);
	}

	public destroyCharacterController(
		worldId: string,
		controllerId: string
	): void {
		if (this._usingFallback()) {
			this._delegate.destroyCharacterController(worldId, controllerId);
			return;
		}
		const world = this._requireWorld(worldId);
		const controller = world.controllers.get(controllerId);
		if (controller) {
			this._disposeCharacterController(world, controller);
		}
		world.controllers.delete(controllerId);
	}

	public moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		_deltaSeconds: number
	): CharacterMoveResult {
		if (this._usingFallback()) {
			return this._delegate.moveCharacterController(
				worldId,
				controllerId,
				direction,
				_deltaSeconds
			);
		}
		const controller = this._requireController(worldId, controllerId);
		controller.pendingDirection = cloneVector(direction);
		return {
			grounded: controller.grounded,
			moved: cloneVector(direction),
		};
	}

	public jumpCharacterController(
		worldId: string,
		controllerId: string,
		speed: number
	): void {
		if (this._usingFallback()) {
			this._delegate.jumpCharacterController(worldId, controllerId, speed);
			return;
		}
		const controller = this._requireController(worldId, controllerId);
		controller.pendingJumpSpeed = Math.max(0, speed);
	}

	public isCharacterControllerGrounded(
		worldId: string,
		controllerId: string
	): boolean {
		if (this._usingFallback()) {
			return this._delegate.isCharacterControllerGrounded(
				worldId,
				controllerId
			);
		}
		const controller = this._requireController(worldId, controllerId);
		return controller.grounded;
	}

	public setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		if (this._usingFallback()) {
			this._delegate.setCharacterControllerMaxSlope(
				worldId,
				controllerId,
				value
			);
			return;
		}
		const controller = this._requireController(worldId, controllerId);
		controller.maxSlope = Math.max(0, value);
		this._syncRapierCharacterControllerSettings(controller);
	}

	public setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		if (this._usingFallback()) {
			this._delegate.setCharacterControllerStepHeight(
				worldId,
				controllerId,
				value
			);
			return;
		}
		const controller = this._requireController(worldId, controllerId);
		controller.stepHeight = Math.max(0, value);
		this._syncRapierCharacterControllerSettings(controller);
	}

	public raycast(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.raycast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const ray = normalizeDirection(query.direction);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return null;

		let bestHit: PhysicsQueryHit | null = null;
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			const hit = intersectRayWithCollider(
				query.origin,
				ray,
				maxDistance,
				candidate
			);
			if (!hit) continue;
			const hitResult = toQueryHit(worldId, candidate, hit, maxDistance);
			if (!bestHit || hitResult.distance < bestHit.distance) {
				bestHit = hitResult;
			}
		}
		return bestHit;
	}

	public raycastAll(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit[] {
		if (this._usingFallback()) {
			return this._delegate.raycastAll(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const ray = normalizeDirection(query.direction);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return [];

		const hits: PhysicsQueryHit[] = [];
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			const hit = intersectRayWithCollider(
				query.origin,
				ray,
				maxDistance,
				candidate
			);
			if (!hit) continue;
			hits.push(toQueryHit(worldId, candidate, hit, maxDistance));
		}
		hits.sort((left, right) => left.distance - right.distance);
		return truncateHits(hits, query.maxHits);
	}

	public sphereCast(
		worldId: string,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.sphereCast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const ray = normalizeDirection(query.direction);
		const castRadius = Math.max(0, query.radius);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return null;

		let bestHit: PhysicsQueryHit | null = null;
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			const hit = intersectSphereCastWithCollider(
				query.center,
				ray,
				castRadius,
				maxDistance,
				candidate
			);
			if (!hit) continue;
			const hitResult = toQueryHit(worldId, candidate, hit, maxDistance);
			if (!bestHit || hitResult.distance < bestHit.distance) {
				bestHit = hitResult;
			}
		}
		return bestHit;
	}

	public boxCast(
		worldId: string,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.boxCast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const ray = normalizeDirection(query.direction);
		const castHalfExtents = sanitizeHalfExtents(query.halfExtents);
		const queryRotation = sanitizeQueryRotation(query.rotation);
		const castBroadphaseExtents = toOrientedBoundsExtents(
			castHalfExtents,
			queryRotation
		);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return null;

		let bestHit: PhysicsQueryHit | null = null;
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			const hit = intersectBoxCastWithCollider(
				query.center,
				ray,
				castBroadphaseExtents,
				maxDistance,
				candidate
			);
			if (!hit) continue;
			const hitResult = toQueryHit(worldId, candidate, hit, maxDistance);
			if (!bestHit || hitResult.distance < bestHit.distance) {
				bestHit = hitResult;
			}
		}
		return bestHit;
	}

	public overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallback()) {
			return this._delegate.overlapSphere(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const radius = Math.max(0, query.radius);
		const hits: PhysicsOverlapHit[] = [];

		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			if (!intersectsSphereWithCollider(query.center, radius, candidate)) {
				continue;
			}
			hits.push(toOverlapHit(worldId, candidate));
		}

		return truncateHits(hits, query.maxHits);
	}

	public overlapBox(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallback()) {
			return this._delegate.overlapBox(worldId, query);
		}
		const world = this._requireWorld(worldId);
		const halfExtents = sanitizeHalfExtents(query.halfExtents);
		const queryRotation = sanitizeQueryRotation(query.rotation);
		const queryBroadphaseExtents = toOrientedBoundsExtents(
			halfExtents,
			queryRotation
		);
		const queryMin = Vector3.sub(query.center, queryBroadphaseExtents);
		const queryMax = Vector3.add(query.center, queryBroadphaseExtents);
		const hits: PhysicsOverlapHit[] = [];

		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			if (!intersectsBoxWithCollider(queryMin, queryMax, candidate)) {
				continue;
			}
			hits.push(toOverlapHit(worldId, candidate));
		}

		return truncateHits(hits, query.maxHits);
	}

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		if (this._usingFallback()) {
			return this._delegate.stepWorld(worldId, deltaSeconds);
		}
		const world = this._requireWorld(worldId);
		const dt = Math.max(0, deltaSeconds);

		this._applyCharacterControllerInputs(world, dt);
		this._setWorldStepDelta(world, dt);
		if (dt > 0) {
			this._invokeOrThrow(
				world.world,
				["step"],
				[[]],
				`Rapier world "${worldId}" does not expose step()`
			);
		}

		const events = this._resolveCollisions(world);
		const bodyStates: PhysicsAdapterBodyState[] = [];
		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const body of world.bodies.values()) {
			const transform = this._readBodyTransform(body.rigidBody);
			const sleeping = this._readBodySleeping(body);
			const ccd = this._readBodyCcd(body);
			body.sleeping = sleeping;
			if (sleeping) sleepingBodies++;
			if (ccd) ccdBodies++;
			bodyStates.push({
				bodyId: body.id,
				transform,
				sleeping,
				ccd,
			});
		}

		return {
			bodyStates,
			events,
			activeBodies: world.bodies.size - sleepingBodies,
			sleepingBodies,
			ccdBodies,
		};
	}

	private _getQueryCandidates(
		world: RapierWorldState,
		filter?: PhysicsQueryFilter
	): RapierQueryCandidate[] {
		const includeBodyIds = toSet(filter?.includeBodyIds);
		const excludeBodyIds = toSet(filter?.excludeBodyIds);
		const includeColliderIds = toSet(filter?.includeColliderIds);
		const excludeColliderIds = toSet(filter?.excludeColliderIds);
		const includeTriggers = filter?.includeTriggers ?? true;

		const candidates: RapierQueryCandidate[] = [];
		for (const collider of world.colliders.values()) {
			if (!includeTriggers && collider.isTrigger) continue;
			if (includeBodyIds && !includeBodyIds.has(collider.bodyId)) continue;
			if (excludeBodyIds?.has(collider.bodyId)) continue;
			if (includeColliderIds && !includeColliderIds.has(collider.id)) continue;
			if (excludeColliderIds?.has(collider.id)) continue;
			const body = world.bodies.get(collider.bodyId);
			if (!body) continue;
			const bodyTransform = this._readBodyTransform(body.rigidBody);
			const rotation = sanitizeQueryRotation(bodyTransform.rotation);
			const rotatedOffset = rotateVectorByQuaternion(collider.offset, rotation);
			const center = Vector3.add(
				bodyTransform.position,
				rotatedOffset
			);
			candidates.push({
				body,
				collider,
				center,
				rotation,
			});
		}
		return candidates;
	}

	private _resolveCollisions(world: RapierWorldState): PhysicsEvent[] {
		const events: PhysicsEvent[] = [];
		const nowSeconds = Date.now() / 1000;
		const currentPairs = new Map<string, "collision" | "trigger">();
		const colliders = Array.from(world.colliders.values());

		for (let i = 0; i < colliders.length; i++) {
			for (let j = i + 1; j < colliders.length; j++) {
				const left = colliders[i];
				const right = colliders[j];
				if (left.bodyId === right.bodyId) continue;
				if (!canCollidersInteract(left, right)) continue;

				const leftBody = world.bodies.get(left.bodyId);
				const rightBody = world.bodies.get(right.bodyId);
				if (!leftBody || !rightBody) continue;

				const leftCenter = Vector3.add(
					this._readBodyPosition(leftBody.rigidBody),
					left.offset
				);
				const rightCenter = Vector3.add(
					this._readBodyPosition(rightBody.rigidBody),
					right.offset
				);
				const distance = Math.hypot(
					leftCenter.x - rightCenter.x,
					leftCenter.y - rightCenter.y,
					leftCenter.z - rightCenter.z
				);
				const overlap = left.radius + right.radius - distance;
				if (overlap <= 0) continue;

				const pairKey = makePairKey(left.bodyId, right.bodyId);
				const activeBefore = world.activePairs.has(pairKey);
				const eventKind =
					left.isTrigger || right.isTrigger ? "trigger" : "collision";
				currentPairs.set(pairKey, eventKind);
				const eventType = resolvePairEventType(eventKind, activeBefore);

				events.push({
					type: eventType,
					worldId: world.config.worldId,
					bodyAId: left.bodyId,
					bodyBId: right.bodyId,
					timestampSeconds: nowSeconds,
				});
			}
		}

		for (const [previous, previousKind] of world.activePairs) {
			if (currentPairs.has(previous)) continue;
			const [bodyAId, bodyBId] = previous.split("|");
			events.push({
				type: previousKind === "collision" ? "collisionEnd" : "triggerEnd",
				worldId: world.config.worldId,
				bodyAId,
				bodyBId,
				timestampSeconds: nowSeconds,
			});
		}

		world.activePairs = currentPairs;
		this._resolveGroundedControllers(world);
		return events;
	}

	private _resolveOverlap(
		leftBody: RapierBodyState,
		rightBody: RapierBodyState,
		leftCenter: IVector3,
		rightCenter: IVector3,
		overlap: number
	): void {
		const separationAxis = Vector3.sub(leftCenter, rightCenter);
		const length = Vector3.length(separationAxis) || 1;
		const normal = Vector3.scale(separationAxis, 1 / length);
		const separation = overlap * 0.5;

		if (leftBody.type === "dynamic") {
			const position = this._readBodyPosition(leftBody.rigidBody);
			this._setVector3(
				leftBody.rigidBody,
				["setTranslation"],
				{
					x: position.x + normal.x * separation,
					y: position.y + normal.y * separation,
					z: position.z + normal.z * separation,
				},
				true
			);
		}
		if (rightBody.type === "dynamic") {
			const position = this._readBodyPosition(rightBody.rigidBody);
			this._setVector3(
				rightBody.rigidBody,
				["setTranslation"],
				{
					x: position.x - normal.x * separation,
					y: position.y - normal.y * separation,
					z: position.z - normal.z * separation,
				},
				true
			);
		}
	}

	private _applyCharacterControllerInputs(
		world: RapierWorldState,
		deltaSeconds: number
	): void {
		for (const controller of world.controllers.values()) {
			const body = world.bodies.get(controller.bodyId);
			if (!body) continue;

			const current = this._readBodyPosition(body.rigidBody);
			const desiredMovement = {
				x: controller.pendingDirection.x * deltaSeconds,
				y: controller.pendingDirection.y * deltaSeconds,
				z: controller.pendingDirection.z * deltaSeconds,
			};
			if (controller.gravityScale !== null) {
				const gravity = world.config.gravity ?? DEFAULT_GRAVITY;
				if (body.type === "dynamic") {
					const gravityDelta = (controller.gravityScale - 1) * deltaSeconds;
					if (Math.abs(gravityDelta) > 1e-8) {
						const linearVelocity = this._readBodyLinearVelocity(body.rigidBody);
						this._setVector3(
							body.rigidBody,
							["setLinvel"],
							{
								x: linearVelocity.x + gravity.x * gravityDelta,
								y: linearVelocity.y + gravity.y * gravityDelta,
								z: linearVelocity.z + gravity.z * gravityDelta,
							},
							true
						);
					}
				} else {
					desiredMovement.x += gravity.x * controller.gravityScale * deltaSeconds;
					desiredMovement.y += gravity.y * controller.gravityScale * deltaSeconds;
					desiredMovement.z += gravity.z * controller.gravityScale * deltaSeconds;
				}
			}
			if (controller.pendingJumpSpeed > 0) {
				desiredMovement.y += controller.pendingJumpSpeed * deltaSeconds;
			}

			let resolvedMovement = cloneVector(desiredMovement);
			if (controller.rapierController && controller.movementCollider) {
				const resolution = this._computeRapierCharacterControllerMovement(
					controller,
					desiredMovement
				);
				resolvedMovement = resolution.movement;
				if (resolution.grounded !== null) {
					controller.grounded = resolution.grounded;
				}
			}

			const next = {
				x: current.x + resolvedMovement.x,
				y: current.y + resolvedMovement.y,
				z: current.z + resolvedMovement.z,
			};
			if (body.type === "kinematic") {
				this._setVector3(
					body.rigidBody,
					["setNextKinematicTranslation", "setTranslation"],
					next,
					true
				);
			} else {
				this._setVector3(body.rigidBody, ["setTranslation"], next, true);
			}

			if (controller.pendingJumpSpeed > 0) {
				const linvel = this._readBodyLinearVelocity(body.rigidBody);
				this._setVector3(
					body.rigidBody,
					["setLinvel"],
					{
						x: linvel.x,
						y: controller.pendingJumpSpeed,
						z: linvel.z,
					},
					true
				);
				if (body.type === "kinematic") {
					this._setVector3(
						body.rigidBody,
						["setNextKinematicTranslation", "setTranslation"],
						{
							x: next.x,
							y: next.y,
							z: next.z,
						},
						true
					);
				}
			}

			controller.lastResolvedMovement = cloneVector(resolvedMovement);
			controller.pendingDirection = { x: 0, y: 0, z: 0 };
			controller.pendingJumpSpeed = 0;
		}
	}

	private _resolveGroundedControllers(world: RapierWorldState): void {
		for (const controller of world.controllers.values()) {
			if (controller.rapierController) continue;
			const body = world.bodies.get(controller.bodyId);
			if (!body) continue;
			const position = this._readBodyPosition(body.rigidBody);
			controller.grounded = position.y <= controller.stepHeight;
		}
	}

	private _bindRapierCharacterController(
		world: RapierWorldState,
		body: RapierBodyState,
		controller: RapierControllerState,
		descriptor: CharacterControllerDescriptor
	): void {
		const creator = (world.world as {
			createCharacterController?: (offset: number) => unknown;
		}).createCharacterController;
		if (typeof creator !== "function") return;

		let rapierController: any = null;
		try {
			rapierController = creator.call(world.world, controller.offset);
		} catch {
			try {
				rapierController = creator.call(world.world, 0.01);
			} catch {
				rapierController = null;
			}
		}
		if (!rapierController) return;

		const colliderResolution = this._resolveCharacterMovementCollider(
			world,
			body,
			descriptor
		);
		if (!colliderResolution.collider) {
			this._invoke(rapierController, ["free"], [[]]);
			return;
		}

		controller.rapierController = rapierController;
		controller.movementCollider = colliderResolution.collider;
		controller.ownsMovementCollider = colliderResolution.owned;
		this._syncRapierCharacterControllerSettings(controller);
	}

	private _resolveCharacterMovementCollider(
		world: RapierWorldState,
		body: RapierBodyState,
		descriptor: CharacterControllerDescriptor
	): RapierCharacterColliderResolution {
		for (const colliderId of body.colliderIds) {
			const collider = world.colliders.get(colliderId);
			if (collider?.rapierCollider && !collider.isTrigger) {
				return {
					collider: collider.rapierCollider,
					owned: false,
				};
			}
		}

		for (const colliderId of body.colliderIds) {
			const collider = world.colliders.get(colliderId);
			if (collider?.rapierCollider) {
				return {
					collider: collider.rapierCollider,
					owned: false,
				};
			}
		}

		const radius = Math.max(0.001, descriptor.radius);
		const halfHeight = Math.max(0, descriptor.height * 0.5 - radius);
		const characterShape: ColliderShape = {
			kind: "capsule",
			halfHeight,
			radius,
		};
		let colliderDesc: any = null;
		try {
			colliderDesc = this._createColliderDescriptor(characterShape);
		} catch {
			try {
				colliderDesc = this._createColliderDescriptor({
					kind: "sphere",
					radius,
				});
			} catch {
				colliderDesc = null;
			}
		}
		if (!colliderDesc) {
			return {
				collider: null,
				owned: false,
			};
		}
		this._invoke(colliderDesc, ["setSensor"], [[true]]);
		let rapierCollider: unknown = null;
		try {
			rapierCollider = world.world.createCollider(colliderDesc, body.rigidBody);
		} catch {
			rapierCollider = null;
		}
		return {
			collider: rapierCollider ?? null,
			owned: rapierCollider ? true : false,
		};
	}

	private _syncRapierCharacterControllerSettings(
		controller: RapierControllerState
	): void {
		if (!controller.rapierController) return;
		const rapierController = controller.rapierController;
		const maxSlopeRadians = toRadians(controller.maxSlope);
		const stepHeight = Math.max(0, controller.stepHeight);

		this._invoke(rapierController, ["setOffset"], [[controller.offset]]);
		this._invoke(
			rapierController,
			["setApplyImpulsesToDynamicBodies"],
			[[true]]
		);
		this._invoke(
			rapierController,
			["setMaxSlopeClimbAngle"],
			[[maxSlopeRadians]]
		);
		this._invoke(
			rapierController,
			["setMinSlopeSlideAngle"],
			[[maxSlopeRadians]]
		);

		if (stepHeight > 0) {
			const minWidth = Math.max(0.001, stepHeight * 0.5);
			this._invoke(
				rapierController,
				["enableAutostep", "setAutostep"],
				[
					[stepHeight, minWidth, true],
					[stepHeight, minWidth, false],
					[stepHeight, minWidth],
					[stepHeight],
				]
			);
			this._invoke(
				rapierController,
				["enableSnapToGround", "setSnapToGround"],
				[[stepHeight]]
			);
			return;
		}

		this._invoke(rapierController, ["disableAutostep"], [[]]);
		this._invoke(rapierController, ["disableSnapToGround"], [[]]);
	}

	private _computeRapierCharacterControllerMovement(
		controller: RapierControllerState,
		desiredMovement: IVector3
	): RapierCharacterMoveResolution {
		if (!controller.rapierController || !controller.movementCollider) {
			return {
				movement: cloneVector(desiredMovement),
				grounded: null,
			};
		}

		const rapierController = controller.rapierController;
		const movementCollider = controller.movementCollider;
		this._syncRapierCharacterControllerSettings(controller);
		const desired = cloneVector(desiredMovement);
		const rapierDesired = this._toRapierVector3(desired);

		const computed = this._invoke(
			rapierController,
			["computeColliderMovement"],
			[
				[movementCollider, desired],
				[movementCollider, rapierDesired ?? desired],
				[movementCollider, desired, undefined, undefined, undefined, undefined],
				[
					movementCollider,
					rapierDesired ?? desired,
					undefined,
					undefined,
					undefined,
					undefined,
				],
			]
		);

		let movement = cloneVector(desiredMovement);
		if (computed) {
			const computedMovement = this._readFromGetter(
				rapierController,
				"computedMovement"
			);
			if (computedMovement !== undefined) {
				movement = readVector3(computedMovement);
			}
		}

		const computedGrounded = this._readFromGetter(
			rapierController,
			"computedGrounded"
		);
		if (typeof computedGrounded === "boolean") {
			return {
				movement,
				grounded: computedGrounded,
			};
		}
		const isGrounded = this._readFromGetter(rapierController, "isGrounded");
		return {
			movement,
			grounded: typeof isGrounded === "boolean" ? isGrounded : null,
		};
	}

	private _disposeCharacterController(
		world: RapierWorldState,
		controller: RapierControllerState
	): void {
		if (controller.ownsMovementCollider && controller.movementCollider) {
			this._invoke(
				world.world,
				["removeCollider"],
				[
					[controller.movementCollider],
					[controller.movementCollider, true],
					[controller.movementCollider, true, true],
				]
			);
		}
		if (controller.rapierController) {
			this._invoke(controller.rapierController, ["free"], [[]]);
		}
		controller.rapierController = null;
		controller.movementCollider = null;
		controller.ownsMovementCollider = false;
	}

	private _createRigidBodyDescriptor(
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform,
		defaultCanSleep?: boolean
	): any {
		const rapier = this._requireRapier();
		const bodyType = descriptor.type ?? "dynamic";
		const bodyDescFactory = rapier.RigidBodyDesc;
		if (!bodyDescFactory) {
			throw new Error("Loaded Rapier module is missing RigidBodyDesc");
		}

		let bodyDesc: any = null;
		switch (bodyType) {
			case "dynamic":
				bodyDesc = bodyDescFactory.dynamic?.();
				break;
			case "fixed":
				bodyDesc = bodyDescFactory.fixed?.();
				break;
			case "kinematic":
				bodyDesc =
					bodyDescFactory.kinematicPositionBased?.() ??
					bodyDescFactory.kinematicVelocityBased?.();
				break;
			default:
				bodyDesc = bodyDescFactory.dynamic?.();
				break;
		}
		if (!bodyDesc) {
			throw new Error(
				"Loaded Rapier module could not create a rigid body descriptor"
			);
		}

		this._setVector3(bodyDesc, ["setTranslation"], initialTransform.position);
		this._setQuaternion(bodyDesc, ["setRotation"], initialTransform.rotation);

		if (bodyType === "dynamic" && Number.isFinite(descriptor.mass)) {
			const requestedMass = Number(descriptor.mass);
			if (requestedMass > 0) {
				this._invoke(
					bodyDesc,
					["setAdditionalMass", "setMass"],
					[
						[requestedMass],
						[requestedMass, true],
						[requestedMass, false],
					]
				);
			}
		}

		if (Number.isFinite(descriptor.linearDamping)) {
			this._invoke(
				bodyDesc,
				["setLinearDamping"],
				[[descriptor.linearDamping]]
			);
		}
		if (Number.isFinite(descriptor.angularDamping)) {
			this._invoke(
				bodyDesc,
				["setAngularDamping"],
				[[descriptor.angularDamping]]
			);
		}
		const canSleep = descriptor.canSleep ?? defaultCanSleep;
		if (typeof canSleep === "boolean") {
			this._invoke(bodyDesc, ["setCanSleep"], [[canSleep]]);
		}
		if (descriptor.lockTranslations) {
			const [x, y, z] = descriptor.lockTranslations;
			this._invoke(bodyDesc, ["setEnabledTranslations"], [[!x, !y, !z]]);
			this._invoke(bodyDesc, ["restrictTranslations"], [[x, y, z]]);
		}
		if (descriptor.lockRotations) {
			const [x, y, z] = descriptor.lockRotations;
			this._invoke(bodyDesc, ["setEnabledRotations"], [[!x, !y, !z]]);
			this._invoke(bodyDesc, ["restrictRotations"], [[x, y, z]]);
		}

		return bodyDesc;
	}

	private _applyRigidBodyMassOverride(
		rigidBody: any,
		bodyType: RigidBodyType,
		mass: number | undefined
	): void {
		if (bodyType !== "dynamic" || !Number.isFinite(mass)) return;
		const requestedMass = Number(mass);
		if (requestedMass <= 0) return;
		this._invoke(
			rigidBody,
			["setAdditionalMass", "setMass"],
			[
				[requestedMass, true],
				[requestedMass, false],
				[requestedMass],
			]
		);
	}

	private _createColliderDescriptor(shape: ColliderShape): any {
		const rapier = this._requireRapier();
		const colliderFactory = rapier.ColliderDesc;
		if (!colliderFactory) {
			throw new Error("Loaded Rapier module is missing ColliderDesc");
		}
		let descriptor: any = null;
		switch (shape.kind) {
			case "box":
				descriptor = colliderFactory.cuboid?.(
					shape.halfExtents.x,
					shape.halfExtents.y,
					shape.halfExtents.z
				);
				break;
			case "sphere":
				descriptor = colliderFactory.ball?.(shape.radius);
				break;
			case "capsule":
				descriptor = colliderFactory.capsule?.(shape.halfHeight, shape.radius);
				break;
			case "cylinder":
				descriptor = colliderFactory.cylinder?.(shape.halfHeight, shape.radius);
				break;
			case "trimesh": {
				const vertices =
					shape.vertices instanceof Float32Array ?
						shape.vertices
					:	new Float32Array(shape.vertices);
				const indices =
					shape.indices instanceof Uint32Array ?
						shape.indices
					:	new Uint32Array(shape.indices);
				descriptor = colliderFactory.trimesh?.(vertices, indices);
				break;
			}
			default:
				throw new Error(
					`Unsupported collider shape kind "${(shape as { kind?: unknown }).kind}"`
				);
		}
		if (!descriptor) {
			throw new Error(
				`Loaded Rapier module failed to build collider descriptor for "${shape.kind}"`
			);
		}
		return descriptor;
	}

	private _createJointData(descriptor: JointDescriptor): any {
		const rapier = this._requireRapier();
		const jointDataFactory = rapier.JointData;
		if (!jointDataFactory) return null;

		const anchorA = descriptor.anchorA ?? { x: 0, y: 0, z: 0 };
		const anchorB = descriptor.anchorB ?? { x: 0, y: 0, z: 0 };
		const identity = { x: 0, y: 0, z: 0, w: 1 };

		switch (descriptor.type) {
			case "fixed":
				return jointDataFactory.fixed?.(
					this._toRapierVector3(anchorA) ?? anchorA,
					this._toRapierQuaternion(identity) ?? identity,
					this._toRapierVector3(anchorB) ?? anchorB,
					this._toRapierQuaternion(identity) ?? identity
				);
			case "hinge": {
				const axis = normalizeDirection(
					descriptor.axis ?? { x: 0, y: 1, z: 0 }
				);
				return jointDataFactory.revolute?.(
					this._toRapierVector3(anchorA) ?? anchorA,
					this._toRapierVector3(anchorB) ?? anchorB,
					this._toRapierVector3(axis) ?? axis
				);
			}
			case "spring": {
				const restLength = 0;
				const stiffness = Math.max(0, descriptor.stiffness ?? 50);
				const damping = Math.max(0, descriptor.damping ?? 2);
				return jointDataFactory.spring?.(
					restLength,
					stiffness,
					damping,
					this._toRapierVector3(anchorA) ?? anchorA,
					this._toRapierVector3(anchorB) ?? anchorB
				);
			}
			default:
				return null;
		}
	}

	private _readBodyTransform(rigidBody: any): PhysicsTransform {
		const position = this._readBodyPosition(rigidBody);
		const rotationLike = this._readFromGetter(rigidBody, "rotation");
		const rotation = readQuaternion(rotationLike);
		return {
			position,
			rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
		};
	}

	private _readBodyPosition(rigidBody: any): IVector3 {
		const value = this._readFromGetter(rigidBody, "translation");
		return readVector3(value);
	}

	private _readBodyLinearVelocity(rigidBody: any): IVector3 {
		const value = this._readFromGetter(rigidBody, "linvel");
		return readVector3(value);
	}

	private _readBodySleeping(body: RapierBodyState): boolean {
		const value = this._readFromGetter(body.rigidBody, "isSleeping");
		if (typeof value === "boolean") return value;
		if (body.type !== "dynamic") return false;
		const velocity = this._readBodyLinearVelocity(body.rigidBody);
		const speed = Math.hypot(velocity.x, velocity.y, velocity.z);
		return speed < 0.001;
	}

	private _readBodyCcd(body: RapierBodyState): boolean {
		const value = this._readFromGetter(body.rigidBody, "isCcdEnabled");
		return typeof value === "boolean" ? value : body.ccd;
	}

	private _setWorldStepDelta(
		world: RapierWorldState,
		deltaSeconds: number
	): void {
		if (deltaSeconds <= 0) return;
		const raw = world.world as {
			timestep?: number;
			integrationParameters?: {
				dt?: number;
			};
		};
		if (typeof raw.timestep === "number") {
			raw.timestep = deltaSeconds;
			return;
		}
		if (
			raw.integrationParameters &&
			typeof raw.integrationParameters === "object"
		) {
			if (typeof raw.integrationParameters.dt === "number") {
				raw.integrationParameters.dt = deltaSeconds;
				return;
			}
		}
		this._invoke(world.world, ["setTimestep"], [[deltaSeconds]]);
	}

	private _toRapierVector3(value: IVector3): unknown {
		const VectorCtor = this._rapier?.Vector3;
		if (typeof VectorCtor === "function") {
			try {
				return new VectorCtor(value.x, value.y, value.z);
			} catch {}
		}
		return { x: value.x, y: value.y, z: value.z };
	}

	private _toRapierQuaternion(value: {
		x: number;
		y: number;
		z: number;
		w: number;
	}): unknown {
		const QuaternionCtor = this._rapier?.Quaternion;
		if (typeof QuaternionCtor === "function") {
			try {
				return new QuaternionCtor(value.x, value.y, value.z, value.w);
			} catch {}
		}
		return {
			x: value.x,
			y: value.y,
			z: value.z,
			w: value.w,
		};
	}

	private _setVector3(
		target: any,
		methodNames: string[],
		value: IVector3,
		wakeUp?: boolean
	): void {
		const plain = { x: value.x, y: value.y, z: value.z };
		const rapierValue = this._toRapierVector3(value);
		const args: unknown[][] = [];
		if (wakeUp !== undefined) {
			args.push([plain, wakeUp]);
			if (rapierValue) args.push([rapierValue, wakeUp]);
			args.push([value.x, value.y, value.z, wakeUp]);
			args.push([plain]);
			if (rapierValue) args.push([rapierValue]);
			args.push([value.x, value.y, value.z]);
		} else {
			args.push([value.x, value.y, value.z]);
			args.push([plain]);
			if (rapierValue) args.push([rapierValue]);
		}
		this._invoke(target, methodNames, args);
	}

	private _setQuaternion(
		target: any,
		methodNames: string[],
		value:
			| [number, number, number, number]
			| {
					x: number;
					y: number;
					z: number;
					w: number;
			  },
		wakeUp?: boolean
	): void {
		const plain =
			Array.isArray(value) ?
				{
					x: value[0],
					y: value[1],
					z: value[2],
					w: value[3],
				}
			:	value;
		const rapierValue = this._toRapierQuaternion(plain);
		const args: unknown[][] = [];
		if (wakeUp !== undefined) {
			args.push([plain, wakeUp]);
			if (rapierValue) args.push([rapierValue, wakeUp]);
		}
		args.push([plain]);
		if (rapierValue) args.push([rapierValue]);
		if (wakeUp !== undefined) {
			args.push([plain.x, plain.y, plain.z, plain.w, wakeUp]);
		}
		args.push([plain.x, plain.y, plain.z, plain.w]);
		this._invoke(target, methodNames, args);
	}

	private _invoke(
		target: any,
		methodNames: string[],
		argVariants: unknown[][]
	): boolean {
		if (!target || typeof target !== "object") return false;
		for (const methodName of methodNames) {
			const fn = (target as Record<string, unknown>)[methodName];
			if (typeof fn !== "function") continue;
			for (const args of argVariants) {
				try {
					(fn as (...callArgs: unknown[]) => unknown).apply(target, args);
					return true;
				} catch {}
			}
		}
		return false;
	}

	private _invokeOrThrow(
		target: any,
		methodNames: string[],
		argVariants: unknown[][],
		errorMessage: string
	): void {
		if (this._invoke(target, methodNames, argVariants)) return;
		throw new Error(errorMessage);
	}

	private _applyNativeCollisionMask(collider: RapierColliderState): void {
		const interactionGroups = collisionMaskToInteractionGroups(
			collider.collisionMask
		);
		this._invoke(
			collider.rapierCollider,
			["setCollisionGroups"],
			[[interactionGroups]]
		);
	}

	private _readFromGetter(target: any, getterName: string): unknown {
		if (!target || typeof target !== "object") return undefined;
		const member = (target as Record<string, unknown>)[getterName];
		if (typeof member === "function") {
			try {
				return (member as () => unknown).call(target);
			} catch {
				return undefined;
			}
		}
		return member;
	}

	private _usingFallback(): boolean {
		return this._rapier === null;
	}

	private _requireRapier(): RapierModuleLike {
		if (this._rapier) return this._rapier;
		throw new Error(
			"RapierPhysicsAdapter is not initialized with a usable Rapier module"
		);
	}

	private _requireWorld(worldId: string): RapierWorldState {
		const world = this._worlds.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}

	private _requireBody(worldId: string, bodyId: string): RapierBodyState {
		const world = this._requireWorld(worldId);
		const body = world.bodies.get(bodyId);
		if (body) return body;
		throw new Error(`Physics body "${bodyId}" does not exist in "${worldId}"`);
	}

	private _requireController(
		worldId: string,
		controllerId: string
	): RapierControllerState {
		const world = this._requireWorld(worldId);
		const controller = world.controllers.get(controllerId);
		if (controller) return controller;
		throw new Error(
			`Character controller "${controllerId}" does not exist in "${worldId}"`
		);
	}
}

function resolveRapierModule(loaded: unknown): RapierModuleLike | null {
	if (!loaded || typeof loaded !== "object") return null;
	const direct = loaded as RapierModuleLike & { default?: unknown };
	if (isRapierUsable(direct)) return direct;
	const fallback = direct.default;
	if (!fallback || typeof fallback !== "object") return null;
	return isRapierUsable(fallback as RapierModuleLike) ?
			(fallback as RapierModuleLike)
		:	null;
}

function isRapierUsable(module: RapierModuleLike): boolean {
	if (!module || typeof module !== "object") return false;
	if (typeof module.World !== "function") return false;
	if (!module.RigidBodyDesc || !module.ColliderDesc) return false;
	if (typeof module.RigidBodyDesc.dynamic !== "function") return false;
	if (typeof module.ColliderDesc.ball !== "function") return false;
	return true;
}

async function loadOptionalRapierModule(): Promise<unknown> {
	try {
		// @ts-expect-error - Allow dynamic import of Rapier module, which may not exist
		return await import("@dimforge/rapier3d-compat");
	} catch (packageImportError) {
		try {
			const rapierEsModulePath =
				"/node_modules/@dimforge/rapier3d-compat/rapier.es.js";
			return await import(
				/* @vite-ignore */
				rapierEsModulePath
			);
		} catch (viteDevUrlImportError) {
			throw new Error(
				`Failed package import (${String(packageImportError)}), and failed Vite dev URL fallback (${String(viteDevUrlImportError)})`
			);
		}
	}
}

function resolveBodyId(value: CharacterControllerDescriptor["body"]): string {
	if (typeof value === "string") return value;
	if (value && typeof value === "object" && "id" in value) {
		return String((value as { id: string }).id);
	}
	return "";
}

function resolveJointBodyId(
	value: JointDescriptor["bodyA"] | JointDescriptor["bodyB"]
): string {
	if (typeof value === "string") return value;
	if (value && typeof value === "object" && "id" in value) {
		return String((value as { id: string }).id);
	}
	return "";
}

function cloneVector(source: IVector3): IVector3 {
	return new Vector3().copy(source);
}

function cloneQuaternion(
	value: [number, number, number, number]
): [number, number, number, number] {
	return [value[0], value[1], value[2], value[3]];
}

function readVector3(value: unknown): IVector3 {
	if (!value || typeof value !== "object") return { x: 0, y: 0, z: 0 };
	const v = value as { x?: number; y?: number; z?: number };
	return {
		x: Number.isFinite(v.x) ? v.x : 0,
		y: Number.isFinite(v.y) ? v.y : 0,
		z: Number.isFinite(v.z) ? v.z : 0,
	};
}

function readQuaternion(value: unknown): {
	x: number;
	y: number;
	z: number;
	w: number;
} {
	if (!value || typeof value !== "object") {
		return { x: 0, y: 0, z: 0, w: 1 };
	}
	const q = value as { x?: number; y?: number; z?: number; w?: number };
	return {
		x: Number.isFinite(q.x) ? q.x : 0,
		y: Number.isFinite(q.y) ? q.y : 0,
		z: Number.isFinite(q.z) ? q.z : 0,
		w: Number.isFinite(q.w) ? q.w : 1,
	};
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
			const length =
				Array.isArray(vertices) ? vertices.length : vertices.length;
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

function computeShapeHalfExtents(shape: ColliderShape): IVector3 {
	switch (shape.kind) {
		case "sphere": {
			const radius = Math.max(0.001, shape.radius);
			return { x: radius, y: radius, z: radius };
		}
		case "capsule": {
			const radius = Math.max(0.001, shape.radius);
			return {
				x: radius,
				y: radius + Math.max(0, shape.halfHeight),
				z: radius,
			};
		}
		case "cylinder": {
			const radius = Math.max(0.001, shape.radius);
			return {
				x: radius,
				y: Math.max(0.001, shape.halfHeight),
				z: radius,
			};
		}
		case "box":
			return {
				x: Math.max(0.001, Math.abs(shape.halfExtents.x)),
				y: Math.max(0.001, Math.abs(shape.halfExtents.y)),
				z: Math.max(0.001, Math.abs(shape.halfExtents.z)),
			};
		case "trimesh": {
			const vertices = shape.vertices;
			const length =
				Array.isArray(vertices) ? vertices.length : vertices.length;
			if (length < 3) {
				return { x: 0.5, y: 0.5, z: 0.5 };
			}
			let minX = Infinity;
			let minY = Infinity;
			let minZ = Infinity;
			let maxX = -Infinity;
			let maxY = -Infinity;
			let maxZ = -Infinity;
			for (let i = 0; i < length; i += 3) {
				const x = vertices[i];
				const y = vertices[i + 1];
				const z = vertices[i + 2];
				if (x < minX) minX = x;
				if (y < minY) minY = y;
				if (z < minZ) minZ = z;
				if (x > maxX) maxX = x;
				if (y > maxY) maxY = y;
				if (z > maxZ) maxZ = z;
			}
			return {
				x: Math.max(0.001, (maxX - minX) * 0.5),
				y: Math.max(0.001, (maxY - minY) * 0.5),
				z: Math.max(0.001, (maxZ - minZ) * 0.5),
			};
		}
		default:
			return { x: 0.5, y: 0.5, z: 0.5 };
	}
}

function toQueryHit(
	worldId: string,
	candidate: RapierQueryCandidate,
	hit: RapierQueryHit,
	maxDistance: number
): PhysicsQueryHit {
	return {
		worldId,
		bodyId: candidate.body.id,
		colliderId: candidate.collider.id,
		point: cloneVector(hit.point),
		normal: cloneVector(hit.normal),
		distance: hit.distance,
		fraction:
			maxDistance > 0 ?
				Math.min(1, Math.max(0, hit.distance / maxDistance))
			:	0,
		isTrigger: candidate.collider.isTrigger,
	};
}

function toOverlapHit(
	worldId: string,
	candidate: RapierQueryCandidate
): PhysicsOverlapHit {
	return {
		worldId,
		bodyId: candidate.body.id,
		colliderId: candidate.collider.id,
		isTrigger: candidate.collider.isTrigger,
	};
}

function truncateHits<T>(hits: T[], maxHits?: number): T[] {
	if (!Number.isFinite(maxHits) || maxHits === undefined) return hits;
	const max = Math.max(0, Math.floor(maxHits));
	if (max === 0) return [];
	if (hits.length <= max) return hits;
	return hits.slice(0, max);
}

function normalizeDirection(direction: IVector3): IVector3 {
	const length = Vector3.length(direction);
	if (length <= 1e-8) {
		throw new Error("Physics query direction must be non-zero");
	}
	return Vector3.normalize(direction);
}

function sanitizeMaxDistance(maxDistance?: number): number {
	if (maxDistance === undefined) return Infinity;
	if (!Number.isFinite(maxDistance)) return Infinity;
	return Math.max(0, maxDistance);
}

function sanitizeHalfExtents(halfExtents: IVector3): IVector3 {
	return {
		x: Math.max(0, Math.abs(halfExtents.x)),
		y: Math.max(0, Math.abs(halfExtents.y)),
		z: Math.max(0, Math.abs(halfExtents.z)),
	};
}

function sanitizeQueryRotation(
	rotation?: [number, number, number, number]
): [number, number, number, number] {
	if (!rotation) return [0, 0, 0, 1];
	return normalizeQuaternion(rotation);
}

function normalizeQuaternion(
	rotation: [number, number, number, number]
): [number, number, number, number] {
	const x = Number.isFinite(rotation[0]) ? rotation[0] : 0;
	const y = Number.isFinite(rotation[1]) ? rotation[1] : 0;
	const z = Number.isFinite(rotation[2]) ? rotation[2] : 0;
	const w = Number.isFinite(rotation[3]) ? rotation[3] : 1;
	const length = Math.hypot(x, y, z, w);
	if (length <= 1e-8) return [0, 0, 0, 1];
	const invLength = 1 / length;
	return [x * invLength, y * invLength, z * invLength, w * invLength];
}

function toOrientedBoundsExtents(
	halfExtents: IVector3,
	rotation: [number, number, number, number]
): IVector3 {
	const matrix = quaternionToMatrix3(rotation);
	return {
		x:
			Math.abs(matrix[0]) * halfExtents.x +
			Math.abs(matrix[1]) * halfExtents.y +
			Math.abs(matrix[2]) * halfExtents.z,
		y:
			Math.abs(matrix[3]) * halfExtents.x +
			Math.abs(matrix[4]) * halfExtents.y +
			Math.abs(matrix[5]) * halfExtents.z,
		z:
			Math.abs(matrix[6]) * halfExtents.x +
			Math.abs(matrix[7]) * halfExtents.y +
			Math.abs(matrix[8]) * halfExtents.z,
	};
}

function toSet(values?: string[]): Set<string> | null {
	if (!values || values.length === 0) return null;
	return new Set(values);
}

function sanitizeCollisionMask(mask: number): number {
	if (!Number.isFinite(mask)) return DEFAULT_COLLISION_MASK;
	return Math.floor(mask) >>> 0;
}

function decodeCollisionFilter(mask: number): { group: number; filter: number } {
	const sanitized = sanitizeCollisionMask(mask);
	const lowBits = sanitized & 0xffff;
	const highBits = (sanitized >>> 16) & 0xffff;
	if (highBits === 0) {
		return {
			group: lowBits,
			filter: lowBits,
		};
	}
	return {
		group: highBits,
		filter: lowBits,
	};
}

function collisionMaskToInteractionGroups(mask: number): number {
	const filter = decodeCollisionFilter(mask);
	return (((filter.group & 0xffff) << 16) | (filter.filter & 0xffff)) >>> 0;
}

function canCollidersInteract(
	left: RapierColliderState,
	right: RapierColliderState
): boolean {
	const leftFilter = decodeCollisionFilter(left.collisionMask);
	const rightFilter = decodeCollisionFilter(right.collisionMask);
	return (
		(leftFilter.group & rightFilter.filter) !== 0 &&
		(rightFilter.group & leftFilter.filter) !== 0
	);
}

function intersectRayWithCollider(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	candidate: RapierQueryCandidate
): RapierQueryHit | null {
	switch (candidate.collider.shape.kind) {
		case "box": {
			const min = {
				x: candidate.center.x - candidate.collider.halfExtents.x,
				y: candidate.center.y - candidate.collider.halfExtents.y,
				z: candidate.center.z - candidate.collider.halfExtents.z,
			};
			const max = {
				x: candidate.center.x + candidate.collider.halfExtents.x,
				y: candidate.center.y + candidate.collider.halfExtents.y,
				z: candidate.center.z + candidate.collider.halfExtents.z,
			};
			return intersectRayAabb(origin, direction, maxDistance, min, max);
		}
		case "trimesh": {
			if (shouldUseExactMeshRaycast(candidate)) {
				return intersectRayTrimesh(
					origin,
					direction,
					maxDistance,
					candidate.center,
					candidate.rotation,
					candidate.collider.shape.vertices,
					candidate.collider.shape.indices,
					candidate.collider.trimeshBVH
				);
			}
			return intersectRaySphere(
				origin,
				direction,
				maxDistance,
				candidate.center,
				candidate.collider.radius
			);
		}
		default:
			return intersectRaySphere(
				origin,
				direction,
				maxDistance,
				candidate.center,
				candidate.collider.radius
			);
	}
}

function intersectSphereCastWithCollider(
	origin: IVector3,
	direction: IVector3,
	radius: number,
	maxDistance: number,
	candidate: RapierQueryCandidate
): RapierQueryHit | null {
	switch (candidate.collider.shape.kind) {
		case "box": {
			const expandedHalfExtents = {
				x: candidate.collider.halfExtents.x + radius,
				y: candidate.collider.halfExtents.y + radius,
				z: candidate.collider.halfExtents.z + radius,
			};
			const min = Vector3.sub(candidate.center, expandedHalfExtents);
			const max = Vector3.add(candidate.center, expandedHalfExtents);
			const hit = intersectRayAabb(origin, direction, maxDistance, min, max);
			if (!hit) return null;
			hit.point = Vector3.sub(hit.point, Vector3.scale(hit.normal, radius));
			return hit;
		}
		default: {
			const hit = intersectRaySphere(
				origin,
				direction,
				maxDistance,
				candidate.center,
				candidate.collider.radius + radius
			);
			if (!hit) return null;
			hit.point = Vector3.sub(hit.point, Vector3.scale(hit.normal, radius));
			return hit;
		}
	}
}

function intersectBoxCastWithCollider(
	origin: IVector3,
	direction: IVector3,
	castHalfExtents: IVector3,
	maxDistance: number,
	candidate: RapierQueryCandidate
): RapierQueryHit | null {
	const expandedHalfExtents = {
		x: candidate.collider.halfExtents.x + castHalfExtents.x,
		y: candidate.collider.halfExtents.y + castHalfExtents.y,
		z: candidate.collider.halfExtents.z + castHalfExtents.z,
	};
	const min = Vector3.sub(candidate.center, expandedHalfExtents);
	const max = Vector3.add(candidate.center, expandedHalfExtents);
	return intersectRayAabb(origin, direction, maxDistance, min, max);
}

function intersectsSphereWithCollider(
	center: IVector3,
	radius: number,
	candidate: RapierQueryCandidate
): boolean {
	switch (candidate.collider.shape.kind) {
		case "box": {
			const min = {
				x: candidate.center.x - candidate.collider.halfExtents.x,
				y: candidate.center.y - candidate.collider.halfExtents.y,
				z: candidate.center.z - candidate.collider.halfExtents.z,
			};
			const max = {
				x: candidate.center.x + candidate.collider.halfExtents.x,
				y: candidate.center.y + candidate.collider.halfExtents.y,
				z: candidate.center.z + candidate.collider.halfExtents.z,
			};
			return intersectsSphereAabb(center, radius, min, max);
		}
		default: {
			const delta = Vector3.sub(center, candidate.center);
			const radii = radius + candidate.collider.radius;
			return Vector3.dot(delta, delta) <= radii * radii;
		}
	}
}

function intersectsBoxWithCollider(
	queryMin: IVector3,
	queryMax: IVector3,
	candidate: RapierQueryCandidate
): boolean {
	switch (candidate.collider.shape.kind) {
		case "box": {
			const min = Vector3.sub(candidate.center, candidate.collider.halfExtents);
			const max = Vector3.add(candidate.center, candidate.collider.halfExtents);
			return intersectsAabb(queryMin, queryMax, min, max);
		}
		default:
			return intersectsSphereAabb(
				candidate.center,
				candidate.collider.radius,
				queryMin,
				queryMax
			);
	}
}

function intersectRaySphere(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	center: IVector3,
	radius: number
): RapierQueryHit | null {
	const radiusClamped = Math.max(0.001, radius);
	const oc = Vector3.sub(origin, center);
	const b = Vector3.dot(oc, direction);
	const c = Vector3.dot(oc, oc) - radiusClamped * radiusClamped;
	if (c > 0 && b > 0) return null;

	const discriminant = b * b - c;
	if (discriminant < 0) return null;
	const sqrtDiscriminant = Math.sqrt(discriminant);
	let distance = -b - sqrtDiscriminant;
	if (distance < 0) distance = 0;
	if (distance > maxDistance) return null;

	const point = Vector3.add(origin, Vector3.scale(direction, distance));
	const normalCandidate = Vector3.sub(point, center);
	const normalLength = Math.hypot(
		normalCandidate.x,
		normalCandidate.y,
		normalCandidate.z
	);
	const normal =
		normalLength > 1e-8 ?
			Vector3.scale(normalCandidate, 1 / normalLength)
		:	Vector3.scale(direction, -1);
	return { distance, point, normal };
}

function intersectRayAabb(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	min: IVector3,
	max: IVector3
): RapierQueryHit | null {
	let entry = 0;
	let exit = maxDistance;
	let hitAxis = -1;
	let hitNormalSign = 0;

	for (let axis = 0; axis < 3; axis++) {
		const o = getAxis(origin, axis);
		const d = getAxis(direction, axis);
		const minAxis = getAxis(min, axis);
		const maxAxis = getAxis(max, axis);

		if (Math.abs(d) <= 1e-8) {
			if (o < minAxis || o > maxAxis) return null;
			continue;
		}

		let t1 = (minAxis - o) / d;
		let t2 = (maxAxis - o) / d;
		let normalSign = d > 0 ? -1 : 1;
		if (t1 > t2) {
			const temp = t1;
			t1 = t2;
			t2 = temp;
			normalSign = d > 0 ? 1 : -1;
		}

		if (t1 > entry) {
			entry = t1;
			hitAxis = axis;
			hitNormalSign = normalSign;
		}
		exit = Math.min(exit, t2);
		if (entry > exit) return null;
	}

	if (entry < 0 || entry > maxDistance) return null;
	const point = Vector3.add(origin, Vector3.scale(direction, entry));
	const normal =
		hitAxis >= 0 ?
			axisVector(hitAxis, hitNormalSign)
		:	Vector3.scale(direction, -1);
	return {
		distance: entry,
		point,
		normal,
	};
}

function shouldUseExactMeshRaycast(candidate: RapierQueryCandidate): boolean {
	const descriptor = candidate.collider.descriptor;
	if (descriptor.mode !== "mesh") return false;
	if (descriptor.backendPreference === "approx") return false;
	return (descriptor.narrowphase ?? "face-bvh") === "face-bvh";
}

function buildTrimeshRayBVH(
	shape: ColliderShape,
	descriptor: ColliderDescriptor
): TrimeshRayBVHEntry | null {
	if (shape.kind !== "trimesh") return null;
	if (descriptor.mode !== "mesh") return null;
	if (descriptor.backendPreference === "approx") return null;
	if ((descriptor.narrowphase ?? "face-bvh") !== "face-bvh") return null;

	const triangleCount = Math.floor(shape.indices.length / 3);
	if (triangleCount <= 0) return null;
	const triangleOrder = new Uint32Array(triangleCount);
	const boundsByTriangle = new Array<TrimeshTriangleBounds>(triangleCount);

	for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
		triangleOrder[triangleIndex] = triangleIndex;
		boundsByTriangle[triangleIndex] = computeTrimeshTriangleBounds(
			shape.vertices,
			shape.indices,
			triangleIndex
		);
	}

	return {
		triangleOrder,
		root: buildTrimeshRayBVHNode(boundsByTriangle, triangleOrder, 0, triangleCount),
	};
}

function buildTrimeshRayBVHNode(
	boundsByTriangle: TrimeshTriangleBounds[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): TrimeshRayBVHNode | null {
	const count = end - start;
	if (count <= 0) return null;
	const bounds = computeTrimeshNodeBounds(boundsByTriangle, triangleOrder, start, end);
	if (count <= 8) {
		return {
			...bounds,
			start,
			end,
			left: null,
			right: null,
		};
	}

	const axis = resolveTrimeshSplitAxis(boundsByTriangle, triangleOrder, start, end);
	sortTriangleOrderRangeByAxis(boundsByTriangle, triangleOrder, start, end, axis);
	const middle = start + Math.floor(count * 0.5);

	return {
		...bounds,
		start,
		end,
		left: buildTrimeshRayBVHNode(boundsByTriangle, triangleOrder, start, middle),
		right: buildTrimeshRayBVHNode(boundsByTriangle, triangleOrder, middle, end),
	};
}

function computeTrimeshTriangleBounds(
	vertices: Float32Array | number[],
	indices: Uint32Array | number[],
	triangleIndex: number
): TrimeshTriangleBounds {
	const base = triangleIndex * 3;
	const index0 = Number(indices[base]) * 3;
	const index1 = Number(indices[base + 1]) * 3;
	const index2 = Number(indices[base + 2]) * 3;

	const x0 = Number(vertices[index0]);
	const y0 = Number(vertices[index0 + 1]);
	const z0 = Number(vertices[index0 + 2]);
	const x1 = Number(vertices[index1]);
	const y1 = Number(vertices[index1 + 1]);
	const z1 = Number(vertices[index1 + 2]);
	const x2 = Number(vertices[index2]);
	const y2 = Number(vertices[index2 + 1]);
	const z2 = Number(vertices[index2 + 2]);

	return {
		minX: Math.min(x0, x1, x2),
		minY: Math.min(y0, y1, y2),
		minZ: Math.min(z0, z1, z2),
		maxX: Math.max(x0, x1, x2),
		maxY: Math.max(y0, y1, y2),
		maxZ: Math.max(z0, z1, z2),
	};
}

function computeTrimeshNodeBounds(
	boundsByTriangle: TrimeshTriangleBounds[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): TrimeshTriangleBounds {
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (let index = start; index < end; index++) {
		const bounds = boundsByTriangle[triangleOrder[index]];
		if (bounds.minX < minX) minX = bounds.minX;
		if (bounds.minY < minY) minY = bounds.minY;
		if (bounds.minZ < minZ) minZ = bounds.minZ;
		if (bounds.maxX > maxX) maxX = bounds.maxX;
		if (bounds.maxY > maxY) maxY = bounds.maxY;
		if (bounds.maxZ > maxZ) maxZ = bounds.maxZ;
	}

	return { minX, minY, minZ, maxX, maxY, maxZ };
}

function resolveTrimeshSplitAxis(
	boundsByTriangle: TrimeshTriangleBounds[],
	triangleOrder: Uint32Array,
	start: number,
	end: number
): 0 | 1 | 2 {
	let minCx = Infinity;
	let minCy = Infinity;
	let minCz = Infinity;
	let maxCx = -Infinity;
	let maxCy = -Infinity;
	let maxCz = -Infinity;

	for (let index = start; index < end; index++) {
		const bounds = boundsByTriangle[triangleOrder[index]];
		const cx = (bounds.minX + bounds.maxX) * 0.5;
		const cy = (bounds.minY + bounds.maxY) * 0.5;
		const cz = (bounds.minZ + bounds.maxZ) * 0.5;
		if (cx < minCx) minCx = cx;
		if (cy < minCy) minCy = cy;
		if (cz < minCz) minCz = cz;
		if (cx > maxCx) maxCx = cx;
		if (cy > maxCy) maxCy = cy;
		if (cz > maxCz) maxCz = cz;
	}

	const extentX = maxCx - minCx;
	const extentY = maxCy - minCy;
	const extentZ = maxCz - minCz;
	if (extentX >= extentY && extentX >= extentZ) return 0;
	if (extentY >= extentX && extentY >= extentZ) return 1;
	return 2;
}

function sortTriangleOrderRangeByAxis(
	boundsByTriangle: TrimeshTriangleBounds[],
	triangleOrder: Uint32Array,
	start: number,
	end: number,
	axis: 0 | 1 | 2
): void {
	const range = Array.from(triangleOrder.slice(start, end));
	range.sort((left, right) => {
		const leftCentroid = readTriangleCentroidAxis(boundsByTriangle[left], axis);
		const rightCentroid = readTriangleCentroidAxis(boundsByTriangle[right], axis);
		return leftCentroid - rightCentroid;
	});
	for (let index = start; index < end; index++) {
		triangleOrder[index] = range[index - start];
	}
}

function readTriangleCentroidAxis(
	bounds: TrimeshTriangleBounds,
	axis: 0 | 1 | 2
): number {
	if (axis === 0) return (bounds.minX + bounds.maxX) * 0.5;
	if (axis === 1) return (bounds.minY + bounds.maxY) * 0.5;
	return (bounds.minZ + bounds.maxZ) * 0.5;
}

function intersectRayTrimesh(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	center: IVector3,
	rotation: [number, number, number, number],
	vertices: Float32Array | number[],
	indices: Uint32Array | number[],
	bvh: TrimeshRayBVHEntry | null
): RapierQueryHit | null {
	const localOrigin = rotateVectorByInverseQuaternion(
		Vector3.sub(origin, center),
		rotation
	);
	const localDirection = rotateVectorByInverseQuaternion(direction, rotation);
	let bestDistance = Infinity;
	let bestNormal: IVector3 | null = null;
	const visitRange = (
		start: number,
		end: number,
		triangleOrder: Uint32Array
	): void => {
		for (let cursor = start; cursor < end; cursor++) {
			const triangleIndex = triangleOrder[cursor];
			const base = triangleIndex * 3;
			const index0 = Number(indices[base]) * 3;
			const index1 = Number(indices[base + 1]) * 3;
			const index2 = Number(indices[base + 2]) * 3;

			const v0x = Number(vertices[index0]);
			const v0y = Number(vertices[index0 + 1]);
			const v0z = Number(vertices[index0 + 2]);
			const v1x = Number(vertices[index1]);
			const v1y = Number(vertices[index1 + 1]);
			const v1z = Number(vertices[index1 + 2]);
			const v2x = Number(vertices[index2]);
			const v2y = Number(vertices[index2 + 1]);
			const v2z = Number(vertices[index2 + 2]);

			const hitDistance = intersectRayTriangleDistance(
				localOrigin,
				localDirection,
				v0x,
				v0y,
				v0z,
				v1x,
				v1y,
				v1z,
				v2x,
				v2y,
				v2z
			);
			if (hitDistance === null || hitDistance > maxDistance) continue;
			if (hitDistance >= bestDistance) continue;

			const edge1x = v1x - v0x;
			const edge1y = v1y - v0y;
			const edge1z = v1z - v0z;
			const edge2x = v2x - v0x;
			const edge2y = v2y - v0y;
			const edge2z = v2z - v0z;
			const nx = edge1y * edge2z - edge1z * edge2y;
			const ny = edge1z * edge2x - edge1x * edge2z;
			const nz = edge1x * edge2y - edge1y * edge2x;
			const nLength = Math.hypot(nx, ny, nz);
			bestNormal =
				nLength > 1e-8 ?
					{ x: nx / nLength, y: ny / nLength, z: nz / nLength }
				:	{ x: -localDirection.x, y: -localDirection.y, z: -localDirection.z };
			bestDistance = hitDistance;
		}
	};

	if (bvh?.root) {
		const stack: TrimeshRayBVHNode[] = [bvh.root];
		while (stack.length > 0) {
			const node = stack.pop();
			if (!node) continue;
			const nodeDistance = intersectRayBounds(
				localOrigin,
				localDirection,
				maxDistance,
				node.minX,
				node.minY,
				node.minZ,
				node.maxX,
				node.maxY,
				node.maxZ
			);
			if (nodeDistance === null || nodeDistance > bestDistance) continue;
			if (!node.left && !node.right) {
				visitRange(node.start, node.end, bvh.triangleOrder);
				continue;
			}
			if (node.left) stack.push(node.left);
			if (node.right) stack.push(node.right);
		}
	} else {
		const triangleCount = Math.floor(indices.length / 3);
		const triangleOrder = new Uint32Array(triangleCount);
		for (let index = 0; index < triangleCount; index++) {
			triangleOrder[index] = index;
		}
		visitRange(0, triangleCount, triangleOrder);
	}

	if (!Number.isFinite(bestDistance) || !bestNormal) return null;
	const worldNormal = rotateVectorByQuaternion(bestNormal, rotation);
	const worldNormalLength = Math.hypot(
		worldNormal.x,
		worldNormal.y,
		worldNormal.z
	);
	const resolvedNormal =
		worldNormalLength > 1e-8 ?
			{
				x: worldNormal.x / worldNormalLength,
				y: worldNormal.y / worldNormalLength,
				z: worldNormal.z / worldNormalLength,
			}
		:	{
				x: -direction.x,
				y: -direction.y,
				z: -direction.z,
			};
	return {
		distance: bestDistance,
		point: {
			x: origin.x + direction.x * bestDistance,
			y: origin.y + direction.y * bestDistance,
			z: origin.z + direction.z * bestDistance,
		},
		normal: resolvedNormal,
	};
}

function intersectRayTriangleDistance(
	origin: IVector3,
	direction: IVector3,
	v0x: number,
	v0y: number,
	v0z: number,
	v1x: number,
	v1y: number,
	v1z: number,
	v2x: number,
	v2y: number,
	v2z: number
): number | null {
	const epsilon = 1e-8;
	const edge1x = v1x - v0x;
	const edge1y = v1y - v0y;
	const edge1z = v1z - v0z;
	const edge2x = v2x - v0x;
	const edge2y = v2y - v0y;
	const edge2z = v2z - v0z;

	const hx = direction.y * edge2z - direction.z * edge2y;
	const hy = direction.z * edge2x - direction.x * edge2z;
	const hz = direction.x * edge2y - direction.y * edge2x;
	const a = edge1x * hx + edge1y * hy + edge1z * hz;
	if (Math.abs(a) < epsilon) return null;

	const f = 1 / a;
	const sx = origin.x - v0x;
	const sy = origin.y - v0y;
	const sz = origin.z - v0z;
	const u = f * (sx * hx + sy * hy + sz * hz);
	if (u < 0 || u > 1) return null;

	const qx = sy * edge1z - sz * edge1y;
	const qy = sz * edge1x - sx * edge1z;
	const qz = sx * edge1y - sy * edge1x;
	const v = f * (direction.x * qx + direction.y * qy + direction.z * qz);
	if (v < 0 || u + v > 1) return null;

	const t = f * (edge2x * qx + edge2y * qy + edge2z * qz);
	if (t < 0) return null;
	return t;
}

function intersectRayBounds(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	minX: number,
	minY: number,
	minZ: number,
	maxX: number,
	maxY: number,
	maxZ: number
): number | null {
	let tMin = 0;
	let tMax = maxDistance;

	const axisTests: Array<[number, number, number, number]> = [
		[origin.x, direction.x, minX, maxX],
		[origin.y, direction.y, minY, maxY],
		[origin.z, direction.z, minZ, maxZ],
	];

	for (const [originAxis, directionAxis, minAxis, maxAxis] of axisTests) {
		if (Math.abs(directionAxis) <= 1e-8) {
			if (originAxis < minAxis || originAxis > maxAxis) return null;
			continue;
		}
		let t1 = (minAxis - originAxis) / directionAxis;
		let t2 = (maxAxis - originAxis) / directionAxis;
		if (t1 > t2) {
			const temp = t1;
			t1 = t2;
			t2 = temp;
		}
		if (t1 > tMin) tMin = t1;
		if (t2 < tMax) tMax = t2;
		if (tMin > tMax) return null;
	}

	if (tMin < 0) tMin = 0;
	if (tMin > maxDistance) return null;
	return tMin;
}

function rotateVectorByQuaternion(
	vector: IVector3,
	rotation: [number, number, number, number]
): IVector3 {
	const x = rotation[0];
	const y = rotation[1];
	const z = rotation[2];
	const w = rotation[3];
	const tx = 2 * (y * vector.z - z * vector.y);
	const ty = 2 * (z * vector.x - x * vector.z);
	const tz = 2 * (x * vector.y - y * vector.x);
	return {
		x: vector.x + w * tx + (y * tz - z * ty),
		y: vector.y + w * ty + (z * tx - x * tz),
		z: vector.z + w * tz + (x * ty - y * tx),
	};
}

function rotateVectorByInverseQuaternion(
	vector: IVector3,
	rotation: [number, number, number, number]
): IVector3 {
	return rotateVectorByQuaternion(vector, [
		-rotation[0],
		-rotation[1],
		-rotation[2],
		rotation[3],
	]);
}

function intersectsSphereAabb(
	center: IVector3,
	radius: number,
	min: IVector3,
	max: IVector3
): boolean {
	const clampedX = Math.max(min.x, Math.min(max.x, center.x));
	const clampedY = Math.max(min.y, Math.min(max.y, center.y));
	const clampedZ = Math.max(min.z, Math.min(max.z, center.z));
	const dx = center.x - clampedX;
	const dy = center.y - clampedY;
	const dz = center.z - clampedZ;
	return dx * dx + dy * dy + dz * dz <= radius * radius;
}

function intersectsAabb(
	minA: IVector3,
	maxA: IVector3,
	minB: IVector3,
	maxB: IVector3
): boolean {
	return (
		minA.x <= maxB.x &&
		maxA.x >= minB.x &&
		minA.y <= maxB.y &&
		maxA.y >= minB.y &&
		minA.z <= maxB.z &&
		maxA.z >= minB.z
	);
}

function getAxis(v: IVector3, axis: number): number {
	switch (axis) {
		case 0:
			return v.x;
		case 1:
			return v.y;
		default:
			return v.z;
	}
}

function axisVector(axis: number, sign: number): IVector3 {
	if (axis === 0) return { x: sign, y: 0, z: 0 };
	if (axis === 1) return { x: 0, y: sign, z: 0 };
	return { x: 0, y: 0, z: sign };
}

function quaternionToMatrix3(
	rotation: [number, number, number, number]
): [number, number, number, number, number, number, number, number, number] {
	const x = rotation[0];
	const y = rotation[1];
	const z = rotation[2];
	const w = rotation[3];
	const xx = x * x;
	const yy = y * y;
	const zz = z * z;
	const xy = x * y;
	const xz = x * z;
	const yz = y * z;
	const wx = w * x;
	const wy = w * y;
	const wz = w * z;
	return [
		1 - 2 * (yy + zz),
		2 * (xy - wz),
		2 * (xz + wy),
		2 * (xy + wz),
		1 - 2 * (xx + zz),
		2 * (yz - wx),
		2 * (xz - wy),
		2 * (yz + wx),
		1 - 2 * (xx + yy),
	];
}

function toRadians(degrees: number): number {
	if (!Number.isFinite(degrees)) return 0;
	return (degrees * Math.PI) / 180;
}

function makePairKey(left: string, right: string): string {
	return left < right ? `${left}|${right}` : `${right}|${left}`;
}

function resolvePairEventType(
	prefix: "collision" | "trigger",
	activeBefore: boolean
): PhysicsEventType {
	if (activeBefore) {
		return prefix === "collision" ? "collisionStay" : "triggerStay";
	}
	return prefix === "collision" ? "collisionBegin" : "triggerBegin";
}
