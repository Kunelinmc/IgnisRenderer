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
	PhysicsQueryFilter,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
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

interface AmmoModuleLike {
	btDefaultCollisionConfiguration?: new () => unknown;
	btCollisionDispatcher?: new (config: unknown) => unknown;
	btDbvtBroadphase?: new () => unknown;
	btSequentialImpulseConstraintSolver?: new () => unknown;
	btDiscreteDynamicsWorld?: new (
		dispatcher: unknown,
		broadphase: unknown,
		solver: unknown,
		collisionConfiguration: unknown
	) => unknown;
	btVector3?: new (x: number, y: number, z: number) => unknown;
	btQuaternion?: new (x: number, y: number, z: number, w: number) => unknown;
	btTransform?: new () => unknown;
	btDefaultMotionState?: new (transform: unknown) => unknown;
	btRigidBodyConstructionInfo?: new (
		mass: number,
		motionState: unknown,
		shape: unknown,
		localInertia: unknown
	) => unknown;
	btRigidBody?: new (info: unknown) => unknown;
	btSphereShape?: new (radius: number) => unknown;
	btBoxShape?: new (halfExtents: unknown) => unknown;
	btCapsuleShape?: new (radius: number, height: number) => unknown;
	btCylinderShape?: new (halfExtents: unknown) => unknown;
	btCompoundShape?: new () => unknown;
	btFixedConstraint?: new (
		bodyA: unknown,
		bodyB: unknown,
		frameInA: unknown,
		frameInB: unknown
	) => unknown;
	btHingeConstraint?: new (...args: unknown[]) => unknown;
	btPoint2PointConstraint?: new (
		bodyA: unknown,
		bodyB: unknown,
		pivotInA: unknown,
		pivotInB: unknown
	) => unknown;
	btGeneric6DofSpringConstraint?: new (
		bodyA: unknown,
		bodyB: unknown,
		frameInA: unknown,
		frameInB: unknown,
		useLinearReferenceFrameA?: boolean
	) => unknown;
	ClosestRayResultCallback?: new (from: unknown, to: unknown) => unknown;
	btCollisionWorld_ClosestRayResultCallback?: new (
		from: unknown,
		to: unknown
	) => unknown;
	ClosestConvexResultCallback?: new (from: unknown, to: unknown) => unknown;
	btCollisionWorld_ClosestConvexResultCallback?: new (
		from: unknown,
		to: unknown
	) => unknown;
	destroy?: (target: unknown) => void;
}

interface AmmoBodyState {
	id: string;
	type: RigidBodyType;
	mass: number;
	rigidBody: any;
	motionState: any;
	constructionInfo: any;
	shape: any;
	transform: PhysicsTransform;
	ccd: boolean;
	colliderIds: Set<string>;
}

interface AmmoColliderState {
	id: string;
	bodyId: string;
	descriptor: ColliderDescriptor;
	shape: ColliderShape;
	ammoShape: any;
	childTransform: any;
	isTrigger: boolean;
	collisionMask: number;
	radius: number;
	halfExtents: IVector3;
	offset: IVector3;
}

interface AmmoJointState {
	id: string;
	descriptor: JointDescriptor;
	constraint: any;
	ownedObjects: unknown[];
}

interface AmmoWorldState {
	world: any;
	collisionConfig: any;
	dispatcher: any;
	broadphase: any;
	solver: any;
	bodies: Map<string, AmmoBodyState>;
	colliders: Map<string, AmmoColliderState>;
	joints: Map<string, AmmoJointState>;
}

interface AmmoQueryCandidate {
	body: AmmoBodyState;
	collider: AmmoColliderState;
	center: IVector3;
}

interface AmmoQueryHit {
	distance: number;
	point: IVector3;
	normal: IVector3;
}

const DEFAULT_COLLISION_MASK = 0xffffffff;

export interface AmmoPhysicsAdapterOptions {
	moduleLoader?: () => Promise<unknown>;
	strict?: boolean;
	fallbackAdapter?: IPhysicsEngineAdapter;
}

export class AmmoPhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id = "ammo";
	public readonly capabilities: PhysicsAdapterCapabilities;

	private _moduleLoader: () => Promise<unknown>;
	private _strict: boolean;
	private _delegate: IPhysicsEngineAdapter;
	private _loadedModule: unknown = null;
	private _ammo: AmmoModuleLike | null = null;
	private _worlds = new Map<string, AmmoWorldState>();

	constructor(options: AmmoPhysicsAdapterOptions = {}) {
		this._moduleLoader =
			options.moduleLoader ?? (async () => loadOptionalModule("ammo.js"));
		this._strict = options.strict ?? true;
		this._delegate =
			options.fallbackAdapter ?? new SimplePhysicsAdapter("ammo-fallback");
		this.capabilities = {
			...this._delegate.capabilities,
			syncInit: false,
		};
	}

	public async init(): Promise<void> {
		try {
			this._loadedModule = await this._moduleLoader();
		} catch (error) {
			if (this._strict) {
				throw new Error(
					`AmmoPhysicsAdapter failed to load "ammo.js". Install it as an optional peer dependency or provide moduleLoader. Inner error: ${String(error)}`
				);
			}
			await this._delegate.init();
			return;
		}

		const ammo = await resolveAmmoModule(this._loadedModule);
		if (!ammo || !isAmmoUsable(ammo)) {
			if (this._strict) {
				throw new Error(
					"AmmoPhysicsAdapter loaded module is not a usable Ammo runtime. Pass strict: false to fallback."
				);
			}
			await this._delegate.init();
			return;
		}

		this._ammo = ammo;
		await this._delegate.init();
	}

	public initSync(): void {
		throw new Error(
			"AmmoPhysicsAdapter.initSync is not supported because Ammo usually requires asynchronous module loading. Use init() instead."
		);
	}

	public hasWorld(worldId: string): boolean {
		if (this._worlds.has(worldId)) return true;
		return this._delegate.hasWorld(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		this._delegate.createWorld(config);
		if (this._usingFallback()) return;
		if (this._worlds.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}

		try {
			const collisionConfig = this._newAmmo("btDefaultCollisionConfiguration");
			const dispatcher = this._newAmmo(
				"btCollisionDispatcher",
				collisionConfig
			);
			const broadphase = this._newAmmo("btDbvtBroadphase");
			const solver = this._newAmmo("btSequentialImpulseConstraintSolver");
			const world = this._newAmmo(
				"btDiscreteDynamicsWorld",
				dispatcher,
				broadphase,
				solver,
				collisionConfig
			);
			this._setWorldGravity(world, config.gravity ?? DEFAULT_GRAVITY);

			this._worlds.set(config.worldId, {
				world,
				collisionConfig,
				dispatcher,
				broadphase,
				solver,
				bodies: new Map(),
				colliders: new Map(),
				joints: new Map(),
			});
		} catch (error) {
			this._delegate.destroyWorld(config.worldId);
			throw new Error(
				`AmmoPhysicsAdapter failed to create world "${config.worldId}". Inner error: ${String(error)}`
			);
		}
	}

	public destroyWorld(worldId: string): void {
		if (!this._usingFallback()) {
			const world = this._worlds.get(worldId);
			if (world) {
				for (const jointId of Array.from(world.joints.keys())) {
					this.destroyJoint(worldId, jointId);
				}
				for (const bodyId of Array.from(world.bodies.keys())) {
					this._destroyAmmoBody(world, bodyId);
				}
				this._destroyAmmoObject(world.world);
				this._destroyAmmoObject(world.solver);
				this._destroyAmmoObject(world.broadphase);
				this._destroyAmmoObject(world.dispatcher);
				this._destroyAmmoObject(world.collisionConfig);
				this._worlds.delete(worldId);
			}
		}
		this._delegate.destroyWorld(worldId);
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		this._delegate.createBody(worldId, bodyId, descriptor, initialTransform);
		if (this._usingFallback()) return;

		const world = this._requireWorld(worldId);
		if (world.bodies.has(bodyId)) {
			throw new Error(
				`Physics body "${bodyId}" already exists in "${worldId}"`
			);
		}

		try {
			const type = descriptor.type ?? "dynamic";
			const mass = resolveBodyMass(type, descriptor.mass);
			const shape =
				this._tryNewAmmo("btCompoundShape") ?? this._createFallbackShape();
			const transform = this._createAmmoTransform(initialTransform);
			const motionState = this._newAmmo("btDefaultMotionState", transform);
			const localInertia = this._createAmmoVector3({ x: 0, y: 0, z: 0 });
			if (mass > 0) {
				this._invoke(shape, ["calculateLocalInertia"], [[mass, localInertia]]);
			}

			const constructionInfo = this._newAmmo(
				"btRigidBodyConstructionInfo",
				mass,
				motionState,
				shape,
				localInertia
			);
			const rigidBody = this._newAmmo("btRigidBody", constructionInfo);
			const ccd = descriptor.ccd ?? false;

			this._applyBodyFlags(rigidBody, type);
			if (ccd) this._applyBodyCcd(rigidBody, 0.5);
			if (descriptor.linearVelocity) {
				this._setVector3(
					rigidBody,
					["setLinearVelocity"],
					descriptor.linearVelocity
				);
			}
			if (descriptor.angularVelocity) {
				this._setVector3(
					rigidBody,
					["setAngularVelocity"],
					descriptor.angularVelocity
				);
			}
			this._invoke(world.world, ["addRigidBody"], [[rigidBody]]);
			world.bodies.set(bodyId, {
				id: bodyId,
				type,
				mass,
				rigidBody,
				motionState,
				constructionInfo,
				shape,
				transform: cloneTransform(initialTransform),
				ccd,
				colliderIds: new Set(),
			});

			this._destroyAmmoObject(localInertia);
			this._destroyAmmoObject(transform);
		} catch (error) {
			this._delegate.destroyBody(worldId, bodyId);
			throw new Error(
				`AmmoPhysicsAdapter failed to create body "${bodyId}" in "${worldId}". Inner error: ${String(error)}`
			);
		}
	}

	public destroyBody(worldId: string, bodyId: string): void {
		if (!this._usingFallback()) {
			const world = this._worlds.get(worldId);
			if (world) this._destroyAmmoBody(world, bodyId);
		}
		this._delegate.destroyBody(worldId, bodyId);
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		this._delegate.setBodyTransform(worldId, bodyId, transform);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setBodyTransform(body, transform, true);
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		this._delegate.setBodyLinearVelocity(worldId, bodyId, velocity);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(body.rigidBody, ["setLinearVelocity"], velocity);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
	}

	public setAngularVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		this._delegate.setAngularVelocity(worldId, bodyId, velocity);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(body.rigidBody, ["setAngularVelocity"], velocity);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
	}

	public applyForce(worldId: string, bodyId: string, force: IVector3): void {
		this._delegate.applyForce(worldId, bodyId, force);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(
			body.rigidBody,
			["applyCentralForce", "applyForce"],
			force
		);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
	}

	public applyTorque(worldId: string, bodyId: string, torque: IVector3): void {
		this._delegate.applyTorque(worldId, bodyId, torque);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(body.rigidBody, ["applyTorque"], torque);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
	}

	public applyImpulse(
		worldId: string,
		bodyId: string,
		impulse: IVector3
	): void {
		this._delegate.applyImpulse(worldId, bodyId, impulse);
		if (this._usingFallback()) return;
		const world = this._worlds.get(worldId);
		const body = world?.bodies.get(bodyId);
		if (!body) return;
		this._setVector3(
			body.rigidBody,
			["applyCentralImpulse", "applyImpulse"],
			impulse
		);
		this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
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
		const ammoShape = this._createAmmoColliderShape(shape);
		const offset = cloneVector(descriptor.offset ?? { x: 0, y: 0, z: 0 });
		const isTrigger = descriptor.isTrigger === true;
		let childTransform: any = null;
		let attachedToBody = false;
		const hadColliders = body.colliderIds.size > 0;

		try {
			if (this._isCompoundShape(body.shape)) {
				childTransform = this._createAmmoOffsetTransform(offset);
				this._invokeOrThrow(
					body.shape,
					["addChildShape"],
					[[childTransform, ammoShape]],
					`Ammo body "${bodyId}" does not expose addChildShape()`
				);
				attachedToBody = true;
			} else if (!hadColliders) {
				if (!isZeroVector(offset)) {
					const compound = this._tryNewAmmo("btCompoundShape");
					if (compound) {
						childTransform = this._createAmmoOffsetTransform(offset);
						this._invokeOrThrow(
							compound,
							["addChildShape"],
							[[childTransform, ammoShape]],
							`Ammo body "${bodyId}" does not expose addChildShape()`
						);
						const previousShape = body.shape;
						body.shape = compound;
						this._invoke(body.rigidBody, ["setCollisionShape"], [[compound]]);
						this._destroyAmmoObject(previousShape);
						attachedToBody = true;
					}
				}
				if (!attachedToBody) {
					const previousShape = body.shape;
					body.shape = ammoShape;
					this._invoke(body.rigidBody, ["setCollisionShape"], [[ammoShape]]);
					this._destroyAmmoObject(previousShape);
					attachedToBody = true;
				}
			} else {
				const compound = this._tryNewAmmo("btCompoundShape");
				if (compound) {
					const identity = this._newAmmo("btTransform");
					this._invoke(identity, ["setIdentity"], [[]]);
					this._invoke(compound, ["addChildShape"], [[identity, body.shape]]);
					this._destroyAmmoObject(identity);

					childTransform = this._createAmmoOffsetTransform(offset);
					this._invokeOrThrow(
						compound,
						["addChildShape"],
						[[childTransform, ammoShape]],
						`Ammo body "${bodyId}" does not expose addChildShape()`
					);
					body.shape = compound;
					this._invoke(body.rigidBody, ["setCollisionShape"], [[compound]]);
					attachedToBody = true;
				}
			}

			this._applyBodyMaterial(body.rigidBody, descriptor);
			this._refreshBodyMassProperties(body);
			this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);

			const collider: AmmoColliderState = {
				id: colliderId,
				bodyId,
				descriptor,
				shape,
				ammoShape,
				childTransform,
				isTrigger,
				collisionMask: DEFAULT_COLLISION_MASK,
				radius: computeShapeRadius(shape),
				halfExtents: computeShapeHalfExtents(shape),
				offset,
			};
			world.colliders.set(colliderId, collider);
			body.colliderIds.add(colliderId);
		} catch (error) {
			if (!attachedToBody) this._destroyAmmoObject(ammoShape);
			if (childTransform) this._destroyAmmoObject(childTransform);
			throw new Error(
				`AmmoPhysicsAdapter failed to create collider "${colliderId}" in "${worldId}". Inner error: ${String(error)}`
			);
		}
	}

	public destroyCollider(worldId: string, colliderId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyCollider(worldId, colliderId);
			return;
		}
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) return;
		const body = world.bodies.get(collider.bodyId);
		let detachedShape = true;
		if (body) {
			if (this._isCompoundShape(body.shape)) {
				detachedShape = this._invoke(
					body.shape,
					["removeChildShape"],
					[[collider.ammoShape]]
				);
			}
			body.colliderIds.delete(colliderId);

			if (
				!this._isCompoundShape(body.shape) &&
				body.shape === collider.ammoShape
			) {
				const replacement = this._createFallbackShape();
				body.shape = replacement;
				this._invoke(body.rigidBody, ["setCollisionShape"], [[replacement]]);
			} else if (
				this._isCompoundShape(body.shape) &&
				body.colliderIds.size === 0
			) {
				const oldShape = body.shape;
				const replacement = this._createFallbackShape();
				body.shape = replacement;
				this._invoke(body.rigidBody, ["setCollisionShape"], [[replacement]]);
				if (oldShape !== collider.ammoShape) this._destroyAmmoObject(oldShape);
			}
			this._refreshBodyMassProperties(body);
		}
		world.colliders.delete(colliderId);
		if (collider.childTransform)
			this._destroyAmmoObject(collider.childTransform);
		if (detachedShape && (!body || body.shape !== collider.ammoShape)) {
			this._destroyAmmoObject(collider.ammoShape);
		}
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
		const world = this._worlds.get(worldId);
		const collider = world?.colliders.get(colliderId);
		if (!collider) return;
		collider.isTrigger = isSensor === true;
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
		const world = this._worlds.get(worldId);
		const collider = world?.colliders.get(colliderId);
		if (!collider) return;
		collider.collisionMask = sanitizeCollisionMask(mask);
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
		const nativeJoint = this._createAmmoJoint(bodyA, bodyB, descriptor);
		if (nativeJoint.constraint) {
			const disableCollision = !(descriptor.collisionEnabled ?? false);
			this._invoke(
				world.world,
				["addConstraint"],
				[[nativeJoint.constraint, disableCollision], [nativeJoint.constraint]]
			);
		}
		world.joints.set(jointId, {
			id: jointId,
			descriptor,
			constraint: nativeJoint.constraint,
			ownedObjects: nativeJoint.ownedObjects,
		});
	}

	public destroyJoint(worldId: string, jointId: string): void {
		if (this._usingFallback()) {
			this._delegate.destroyJoint(worldId, jointId);
			return;
		}
		const world = this._worlds.get(worldId);
		const joint = world?.joints.get(jointId);
		if (joint) {
			if (joint.constraint) {
				this._invoke(
					world?.world,
					["removeConstraint"],
					[[joint.constraint], [joint.constraint, true]]
				);
				this._destroyAmmoObject(joint.constraint);
			}
			for (const target of joint.ownedObjects) {
				this._destroyAmmoObject(target);
			}
			world?.joints.delete(jointId);
		}
	}

	public createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void {
		this._delegate.createCharacterController(worldId, controllerId, descriptor);
	}

	public destroyCharacterController(
		worldId: string,
		controllerId: string
	): void {
		this._delegate.destroyCharacterController(worldId, controllerId);
	}

	public moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		deltaSeconds: number
	): CharacterMoveResult {
		return this._delegate.moveCharacterController(
			worldId,
			controllerId,
			direction,
			deltaSeconds
		);
	}

	public jumpCharacterController(
		worldId: string,
		controllerId: string,
		speed: number
	): void {
		this._delegate.jumpCharacterController(worldId, controllerId, speed);
	}

	public isCharacterControllerGrounded(
		worldId: string,
		controllerId: string
	): boolean {
		return this._delegate.isCharacterControllerGrounded(worldId, controllerId);
	}

	public setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		this._delegate.setCharacterControllerMaxSlope(worldId, controllerId, value);
	}

	public setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		this._delegate.setCharacterControllerStepHeight(
			worldId,
			controllerId,
			value
		);
	}

	public raycast(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.raycast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return (
			this._raycastNative(worldId, world, query) ??
			this._raycastApprox(worldId, world, query)
		);
	}

	public raycastAll(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit[] {
		if (this._usingFallback()) {
			return this._delegate.raycastAll(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return this._raycastAllApprox(worldId, world, query);
	}

	public sphereCast(
		worldId: string,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.sphereCast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return (
			this._sphereCastNative(worldId, world, query) ??
			this._sphereCastApprox(worldId, world, query)
		);
	}

	public boxCast(
		worldId: string,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		if (this._usingFallback()) {
			return this._delegate.boxCast(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return (
			this._boxCastNative(worldId, world, query) ??
			this._boxCastApprox(worldId, world, query)
		);
	}

	public overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallback()) {
			return this._delegate.overlapSphere(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return this._overlapSphereApprox(worldId, world, query);
	}

	public overlapBox(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[] {
		if (this._usingFallback()) {
			return this._delegate.overlapBox(worldId, query);
		}
		const world = this._requireWorld(worldId);
		return this._overlapBoxApprox(worldId, world, query);
	}

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		if (this._usingFallback()) {
			return this._delegate.stepWorld(worldId, deltaSeconds);
		}
		const world = this._worlds.get(worldId);
		if (!world) return this._delegate.stepWorld(worldId, deltaSeconds);

		const dt = Math.max(0, deltaSeconds);
		if (dt > 0) {
			this._invokeOrThrow(
				world.world,
				["stepSimulation"],
				[[dt, 1, dt], [dt, 1], [dt]],
				`Ammo world "${worldId}" does not expose stepSimulation()`
			);
		}

		const bodyStates: PhysicsAdapterBodyState[] = [];
		for (const body of world.bodies.values()) {
			const transform = this._readBodyTransform(body);
			const sleeping = this._readBodySleeping(body);
			bodyStates.push({
				bodyId: body.id,
				transform,
				sleeping,
				ccd: body.ccd,
			});
			this._delegate.setBodyTransform(worldId, body.id, transform);
		}

		const delegateResult = this._delegate.stepWorld(worldId, 0);
		const seen = new Set(bodyStates.map((item) => item.bodyId));
		for (const state of delegateResult.bodyStates) {
			if (seen.has(state.bodyId)) continue;
			bodyStates.push(state);
		}

		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const state of bodyStates) {
			if (state.sleeping) sleepingBodies++;
			if (state.ccd) ccdBodies++;
		}

		return {
			bodyStates,
			events: delegateResult.events,
			activeBodies: bodyStates.length - sleepingBodies,
			sleepingBodies,
			ccdBodies,
		};
	}

	private _destroyAmmoBody(world: AmmoWorldState, bodyId: string): void {
		const body = world.bodies.get(bodyId);
		if (!body) return;
		for (const [jointId, joint] of world.joints) {
			const bodyAId = resolveJointBodyId(joint.descriptor.bodyA);
			const bodyBId = resolveJointBodyId(joint.descriptor.bodyB);
			if (bodyAId !== bodyId && bodyBId !== bodyId) continue;
			this.destroyJoint(joint.descriptor.worldId, jointId);
		}
		for (const colliderId of Array.from(body.colliderIds)) {
			const collider = world.colliders.get(colliderId);
			if (!collider) continue;
			world.colliders.delete(colliderId);
			if (collider.childTransform)
				this._destroyAmmoObject(collider.childTransform);
			if (collider.ammoShape && collider.ammoShape !== body.shape) {
				this._destroyAmmoObject(collider.ammoShape);
			}
		}
		body.colliderIds.clear();
		this._invoke(world.world, ["removeRigidBody"], [[body.rigidBody]]);
		this._destroyAmmoObject(body.rigidBody);
		this._destroyAmmoObject(body.motionState);
		this._destroyAmmoObject(body.constructionInfo);
		this._destroyAmmoObject(body.shape);
		world.bodies.delete(bodyId);
	}

	private _setBodyTransform(
		body: AmmoBodyState,
		transform: PhysicsTransform,
		wakeUp: boolean
	): void {
		const ammoTransform = this._createAmmoTransform(transform);
		this._invoke(body.rigidBody, ["setWorldTransform"], [[ammoTransform]]);
		this._invoke(body.motionState, ["setWorldTransform"], [[ammoTransform]]);
		if (wakeUp) this._invoke(body.rigidBody, ["activate"], [[true], [1], []]);
		body.transform = cloneTransform(transform);
		this._destroyAmmoObject(ammoTransform);
	}

	private _readBodyTransform(body: AmmoBodyState): PhysicsTransform {
		let rawTransform: any = null;
		let ownedTemp: any = null;
		const temp = this._newAmmo("btTransform");
		this._invoke(temp, ["setIdentity"], [[]]);
		if (this._invoke(body.motionState, ["getWorldTransform"], [[temp]])) {
			rawTransform = temp;
			ownedTemp = temp;
		} else {
			this._destroyAmmoObject(temp);
		}

		if (!rawTransform) {
			const direct = this._readFromGetter(body.rigidBody, "getWorldTransform");
			if (direct && typeof direct === "object") rawTransform = direct;
		}
		if (!rawTransform) return cloneTransform(body.transform);

		const origin = this._readFromGetter(rawTransform, "getOrigin");
		const rotation = this._readFromGetter(rawTransform, "getRotation");
		const transform: PhysicsTransform = {
			position: readAmmoVector3(origin),
			rotation: toQuaternionTuple(rotation),
		};
		body.transform = cloneTransform(transform);
		if (ownedTemp) this._destroyAmmoObject(ownedTemp);
		return transform;
	}

	private _readBodySleeping(body: AmmoBodyState): boolean {
		if (body.type !== "dynamic") return false;
		const isActive = this._readFromGetter(body.rigidBody, "isActive");
		if (typeof isActive === "boolean") return !isActive;
		const isSleeping = this._readFromGetter(body.rigidBody, "isSleeping");
		if (typeof isSleeping === "boolean") return isSleeping;
		const velocity = readAmmoVector3(
			this._readFromGetter(body.rigidBody, "getLinearVelocity")
		);
		const speed = Math.hypot(velocity.x, velocity.y, velocity.z);
		return speed < 0.001;
	}

	private _applyBodyFlags(rigidBody: any, type: RigidBodyType): void {
		const CF_KINEMATIC_OBJECT = 2;
		const CF_NO_CONTACT_RESPONSE = 4;
		const DISABLE_DEACTIVATION = 4;
		const rawFlags = this._readFromGetter(rigidBody, "getCollisionFlags");
		let flags = Number.isFinite(rawFlags as number) ? Number(rawFlags) : 0;
		flags |= CF_NO_CONTACT_RESPONSE;
		if (type === "kinematic") {
			flags |= CF_KINEMATIC_OBJECT;
			this._invoke(rigidBody, ["setActivationState"], [[DISABLE_DEACTIVATION]]);
		}
		this._invoke(rigidBody, ["setCollisionFlags"], [[flags]]);
	}

	private _applyBodyCcd(rigidBody: any, radius: number): void {
		this._invoke(rigidBody, ["setCcdMotionThreshold"], [[1e-7]]);
		this._invoke(
			rigidBody,
			["setCcdSweptSphereRadius"],
			[[Math.max(0.01, radius)]]
		);
	}

	private _setWorldGravity(world: any, gravity: IVector3): void {
		const g = this._createAmmoVector3(gravity);
		this._invoke(world, ["setGravity"], [[g], [gravity]]);
		this._destroyAmmoObject(g);
	}

	private _createAmmoColliderShape(shape: ColliderShape): any {
		switch (shape.kind) {
			case "sphere":
				return this._newAmmo("btSphereShape", Math.max(0.001, shape.radius));
			case "box": {
				const halfExtents = this._createAmmoVector3({
					x: Math.max(0.001, Math.abs(shape.halfExtents.x)),
					y: Math.max(0.001, Math.abs(shape.halfExtents.y)),
					z: Math.max(0.001, Math.abs(shape.halfExtents.z)),
				});
				const colliderShape = this._newAmmo("btBoxShape", halfExtents);
				this._destroyAmmoObject(halfExtents);
				return colliderShape;
			}
			case "capsule": {
				const ammo = this._requireAmmo();
				if (typeof ammo.btCapsuleShape === "function") {
					return new ammo.btCapsuleShape(
						Math.max(0.001, shape.radius),
						Math.max(0, shape.halfHeight) * 2
					);
				}
				return this._newAmmo("btSphereShape", Math.max(0.001, shape.radius));
			}
			case "cylinder": {
				const ammo = this._requireAmmo();
				const halfExtents = this._createAmmoVector3({
					x: Math.max(0.001, shape.radius),
					y: Math.max(0.001, shape.halfHeight),
					z: Math.max(0.001, shape.radius),
				});
				const colliderShape =
					typeof ammo.btCylinderShape === "function" ?
						new ammo.btCylinderShape(halfExtents)
					:	this._newAmmo("btBoxShape", halfExtents);
				this._destroyAmmoObject(halfExtents);
				return colliderShape;
			}
			case "trimesh":
				return this._createFallbackShape();
			default:
				return this._createFallbackShape();
		}
	}

	private _isCompoundShape(shape: any): boolean {
		if (!shape || typeof shape !== "object") return false;
		const count = this._readFromGetter(shape, "getNumChildShapes");
		return typeof count === "number" && Number.isFinite(count);
	}

	private _createAmmoOffsetTransform(offset: IVector3): any {
		return this._createAmmoTransform({
			position: offset,
			rotation: [0, 0, 0, 1],
		});
	}

	private _refreshBodyMassProperties(body: AmmoBodyState): void {
		if (body.type !== "dynamic" || body.mass <= 0) return;
		const inertia = this._createAmmoVector3({ x: 0, y: 0, z: 0 });
		this._invoke(body.shape, ["calculateLocalInertia"], [[body.mass, inertia]]);
		this._invoke(body.rigidBody, ["setMassProps"], [[body.mass, inertia]]);
		this._invoke(body.rigidBody, ["updateInertiaTensor"], [[]]);
		this._destroyAmmoObject(inertia);
	}

	private _applyBodyMaterial(
		rigidBody: any,
		descriptor: ColliderDescriptor
	): void {
		const material = descriptor.material;
		if (!material) return;
		if (Number.isFinite(material.friction)) {
			this._invoke(rigidBody, ["setFriction"], [[material.friction]]);
		}
		if (Number.isFinite(material.restitution)) {
			this._invoke(rigidBody, ["setRestitution"], [[material.restitution]]);
		}
	}

	private _createAmmoJoint(
		bodyA: AmmoBodyState,
		bodyB: AmmoBodyState,
		descriptor: JointDescriptor
	): { constraint: any; ownedObjects: unknown[] } {
		const ownedObjects: unknown[] = [];
		const anchorA = cloneVector(descriptor.anchorA ?? { x: 0, y: 0, z: 0 });
		const anchorB = cloneVector(descriptor.anchorB ?? { x: 0, y: 0, z: 0 });
		const axis = normalizeOptionalDirection(descriptor.axis, {
			x: 0,
			y: 1,
			z: 0,
		});

		if (descriptor.type === "fixed") {
			const frameA = this._createAmmoOffsetTransform(anchorA);
			const frameB = this._createAmmoOffsetTransform(anchorB);
			ownedObjects.push(frameA, frameB);
			const fixed =
				this._tryNewAmmo(
					"btFixedConstraint",
					bodyA.rigidBody,
					bodyB.rigidBody,
					frameA,
					frameB
				) ??
				this._tryCreatePointConstraint(
					bodyA.rigidBody,
					bodyB.rigidBody,
					anchorA,
					anchorB,
					ownedObjects
				);
			return { constraint: fixed, ownedObjects };
		}

		if (descriptor.type === "hinge") {
			const pivotA = this._createAmmoVector3(anchorA);
			const pivotB = this._createAmmoVector3(anchorB);
			const axisVector = this._createAmmoVector3(axis);
			ownedObjects.push(pivotA, pivotB, axisVector);
			const hinge =
				this._tryNewAmmo(
					"btHingeConstraint",
					bodyA.rigidBody,
					bodyB.rigidBody,
					pivotA,
					pivotB,
					axisVector,
					axisVector,
					true
				) ??
				this._tryNewAmmo(
					"btHingeConstraint",
					bodyA.rigidBody,
					bodyB.rigidBody,
					pivotA,
					pivotB,
					axisVector,
					axisVector
				) ??
				this._tryCreatePointConstraint(
					bodyA.rigidBody,
					bodyB.rigidBody,
					anchorA,
					anchorB,
					ownedObjects
				);
			if (hinge && descriptor.limits) {
				this._invoke(
					hinge,
					["setLimit"],
					[[descriptor.limits[0], descriptor.limits[1]]]
				);
			}
			return { constraint: hinge, ownedObjects };
		}

		const frameA = this._createAmmoOffsetTransform(anchorA);
		const frameB = this._createAmmoOffsetTransform(anchorB);
		ownedObjects.push(frameA, frameB);
		const spring = this._tryNewAmmo(
			"btGeneric6DofSpringConstraint",
			bodyA.rigidBody,
			bodyB.rigidBody,
			frameA,
			frameB,
			true
		);
		if (spring) {
			const stiffness = Math.max(0, descriptor.stiffness ?? 25);
			const damping = Math.max(0, descriptor.damping ?? 3);
			this._invoke(spring, ["enableSpring"], [[0, true]]);
			this._invoke(spring, ["setStiffness"], [[0, stiffness]]);
			this._invoke(spring, ["setDamping"], [[0, damping]]);
			if (descriptor.limits) {
				const lower = this._createAmmoVector3({
					x: descriptor.limits[0],
					y: descriptor.limits[0],
					z: descriptor.limits[0],
				});
				const upper = this._createAmmoVector3({
					x: descriptor.limits[1],
					y: descriptor.limits[1],
					z: descriptor.limits[1],
				});
				ownedObjects.push(lower, upper);
				this._invoke(spring, ["setLinearLowerLimit"], [[lower]]);
				this._invoke(spring, ["setLinearUpperLimit"], [[upper]]);
			}
			return { constraint: spring, ownedObjects };
		}

		return {
			constraint: this._tryCreatePointConstraint(
				bodyA.rigidBody,
				bodyB.rigidBody,
				anchorA,
				anchorB,
				ownedObjects
			),
			ownedObjects,
		};
	}

	private _tryCreatePointConstraint(
		bodyA: any,
		bodyB: any,
		anchorA: IVector3,
		anchorB: IVector3,
		ownedObjects: unknown[]
	): any {
		const pivotA = this._createAmmoVector3(anchorA);
		const pivotB = this._createAmmoVector3(anchorB);
		ownedObjects.push(pivotA, pivotB);
		return this._tryNewAmmo(
			"btPoint2PointConstraint",
			bodyA,
			bodyB,
			pivotA,
			pivotB
		);
	}

	private _createFallbackShape(): any {
		const ammo = this._requireAmmo();
		if (typeof ammo.btSphereShape === "function") {
			return new ammo.btSphereShape(0.5);
		}
		const halfExtents = this._createAmmoVector3({ x: 0.5, y: 0.5, z: 0.5 });
		const shape = this._newAmmo("btBoxShape", halfExtents);
		this._destroyAmmoObject(halfExtents);
		return shape;
	}

	private _createAmmoVector3(value: IVector3): any {
		const Ctor = this._requireAmmoCtor("btVector3");
		return new Ctor(value.x, value.y, value.z);
	}

	private _createAmmoQuaternion(value: [number, number, number, number]): any {
		const ammo = this._requireAmmo();
		if (typeof ammo.btQuaternion === "function") {
			return new ammo.btQuaternion(value[0], value[1], value[2], value[3]);
		}
		return { x: value[0], y: value[1], z: value[2], w: value[3] };
	}

	private _createAmmoTransform(transform: PhysicsTransform): any {
		const ammoTransform = this._newAmmo("btTransform");
		this._invoke(ammoTransform, ["setIdentity"], [[]]);
		const origin = this._createAmmoVector3(transform.position);
		const rotation = this._createAmmoQuaternion(transform.rotation);
		this._invoke(ammoTransform, ["setOrigin"], [[origin]]);
		this._invoke(ammoTransform, ["setRotation"], [[rotation]]);
		this._destroyAmmoObject(origin);
		this._destroyAmmoObject(rotation);
		return ammoTransform;
	}

	private _raycastNative(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
		const direction = normalizeDirection(query.direction);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return null;

		const from = this._createAmmoVector3(query.origin);
		const to = this._createAmmoVector3(
			Vector3.add(query.origin, Vector3.scale(direction, maxDistance))
		);
		const callback =
			this._tryNewAmmo("ClosestRayResultCallback", from, to) ??
			this._tryNewAmmo("btCollisionWorld_ClosestRayResultCallback", from, to);
		if (!callback) {
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			return null;
		}

		const invoked = this._invoke(
			world.world,
			["rayTest"],
			[[from, to, callback]]
		);
		if (!invoked) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			return null;
		}

		const hasHit = this._readHitStatus(callback);
		if (!hasHit) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			return null;
		}

		const collisionObject =
			this._readFromGetter(callback, "get_m_collisionObject") ??
			this._readFromGetter(callback, "m_collisionObject");
		const body = this._resolveHitBody(world, collisionObject);
		const bodyId = body?.id;
		if (!bodyId) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			return null;
		}

		const collider = this._chooseHitCollider(
			world,
			bodyId,
			query.filter,
			readAmmoVector3(this._readFromGetter(callback, "get_m_hitPointWorld"))
		);
		if (!collider) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			return null;
		}
		const point = readAmmoVector3(
			this._readFromGetter(callback, "get_m_hitPointWorld") ??
				this._readFromGetter(callback, "m_hitPointWorld")
		);
		const normal = readAmmoVector3(
			this._readFromGetter(callback, "get_m_hitNormalWorld") ??
				this._readFromGetter(callback, "m_hitNormalWorld")
		);
		const fractionRaw =
			this._readFromGetter(callback, "get_m_closestHitFraction") ??
			this._readFromGetter(callback, "m_closestHitFraction");
		const fraction =
			typeof fractionRaw === "number" && Number.isFinite(fractionRaw) ?
				Math.min(1, Math.max(0, fractionRaw))
			:	Math.min(
					1,
					Math.max(
						0,
						Vector3.length(Vector3.sub(point, query.origin)) / maxDistance
					)
				);
		const distance = fraction * maxDistance;

		this._destroyAmmoObject(callback);
		this._destroyAmmoObject(from);
		this._destroyAmmoObject(to);
		return {
			worldId,
			bodyId,
			colliderId: collider.id,
			point,
			normal:
				Vector3.length(normal) > 1e-8 ? Vector3.normalize(normal) : normal,
			distance,
			fraction,
			isTrigger: collider.isTrigger,
		};
	}

	private _sphereCastNative(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
		const radius = Math.max(0, query.radius);
		const shape = this._tryNewAmmo("btSphereShape", Math.max(0.001, radius));
		if (!shape) return null;
		const result = this._convexCastNative(
			worldId,
			world,
			shape,
			query.center,
			query.direction,
			query.maxDistance,
			query.filter
		);
		this._destroyAmmoObject(shape);
		return result;
	}

	private _boxCastNative(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		const halfExtents = sanitizeHalfExtents(query.halfExtents);
		const extents = this._createAmmoVector3({
			x: Math.max(0.001, halfExtents.x),
			y: Math.max(0.001, halfExtents.y),
			z: Math.max(0.001, halfExtents.z),
		});
		const shape = this._tryNewAmmo("btBoxShape", extents);
		this._destroyAmmoObject(extents);
		if (!shape) return null;
		const result = this._convexCastNative(
			worldId,
			world,
			shape,
			query.center,
			query.direction,
			query.maxDistance,
			query.filter
		);
		this._destroyAmmoObject(shape);
		return result;
	}

	private _convexCastNative(
		worldId: string,
		world: AmmoWorldState,
		shape: any,
		origin: IVector3,
		direction: IVector3,
		maxDistanceRaw: number | undefined,
		filter?: PhysicsQueryFilter
	): PhysicsQueryHit | null {
		const ray = normalizeDirection(direction);
		const maxDistance = sanitizeMaxDistance(maxDistanceRaw);
		if (maxDistance <= 0) return null;

		const from = this._createAmmoTransform({
			position: origin,
			rotation: [0, 0, 0, 1],
		});
		const to = this._createAmmoTransform({
			position: Vector3.add(origin, Vector3.scale(ray, maxDistance)),
			rotation: [0, 0, 0, 1],
		});
		const fromVector = this._createAmmoVector3(origin);
		const toVector = this._createAmmoVector3(
			Vector3.add(origin, Vector3.scale(ray, maxDistance))
		);
		const callback =
			this._tryNewAmmo("ClosestConvexResultCallback", fromVector, toVector) ??
			this._tryNewAmmo(
				"btCollisionWorld_ClosestConvexResultCallback",
				fromVector,
				toVector
			);
		if (!callback) {
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			this._destroyAmmoObject(fromVector);
			this._destroyAmmoObject(toVector);
			return null;
		}

		const invoked = this._invoke(
			world.world,
			["convexSweepTest"],
			[
				[shape, from, to, callback, 0],
				[shape, from, to, callback],
			]
		);
		if (!invoked || !this._readHitStatus(callback)) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			this._destroyAmmoObject(fromVector);
			this._destroyAmmoObject(toVector);
			return null;
		}

		const collisionObject =
			this._readFromGetter(callback, "get_m_hitCollisionObject") ??
			this._readFromGetter(callback, "m_hitCollisionObject");
		const body = this._resolveHitBody(world, collisionObject);
		const bodyId = body?.id;
		if (!bodyId) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			this._destroyAmmoObject(fromVector);
			this._destroyAmmoObject(toVector);
			return null;
		}
		const point = readAmmoVector3(
			this._readFromGetter(callback, "get_m_hitPointWorld") ??
				this._readFromGetter(callback, "m_hitPointWorld")
		);
		const collider = this._chooseHitCollider(world, bodyId, filter, point);
		if (!collider) {
			this._destroyAmmoObject(callback);
			this._destroyAmmoObject(from);
			this._destroyAmmoObject(to);
			this._destroyAmmoObject(fromVector);
			this._destroyAmmoObject(toVector);
			return null;
		}
		const normal = readAmmoVector3(
			this._readFromGetter(callback, "get_m_hitNormalWorld") ??
				this._readFromGetter(callback, "m_hitNormalWorld")
		);
		const fractionRaw =
			this._readFromGetter(callback, "get_m_closestHitFraction") ??
			this._readFromGetter(callback, "m_closestHitFraction");
		const fraction =
			typeof fractionRaw === "number" && Number.isFinite(fractionRaw) ?
				Math.min(1, Math.max(0, fractionRaw))
			:	Math.min(
					1,
					Math.max(0, Vector3.length(Vector3.sub(point, origin)) / maxDistance)
				);
		const result: PhysicsQueryHit = {
			worldId,
			bodyId,
			colliderId: collider.id,
			point,
			normal:
				Vector3.length(normal) > 1e-8 ? Vector3.normalize(normal) : normal,
			distance: fraction * maxDistance,
			fraction,
			isTrigger: collider.isTrigger,
		};
		this._destroyAmmoObject(callback);
		this._destroyAmmoObject(from);
		this._destroyAmmoObject(to);
		this._destroyAmmoObject(fromVector);
		this._destroyAmmoObject(toVector);
		return result;
	}

	private _raycastApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
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

	private _raycastAllApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit[] {
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

	private _sphereCastApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null {
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

	private _boxCastApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsBoxCastQuery
	): PhysicsQueryHit | null {
		const ray = normalizeDirection(query.direction);
		const castHalfExtents = sanitizeHalfExtents(query.halfExtents);
		const maxDistance = sanitizeMaxDistance(query.maxDistance);
		if (maxDistance <= 0) return null;

		let bestHit: PhysicsQueryHit | null = null;
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			const hit = intersectBoxCastWithCollider(
				query.center,
				ray,
				castHalfExtents,
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

	private _overlapSphereApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
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

	private _overlapBoxApprox(
		worldId: string,
		world: AmmoWorldState,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[] {
		const halfExtents = sanitizeHalfExtents(query.halfExtents);
		const queryMin = Vector3.sub(query.center, halfExtents);
		const queryMax = Vector3.add(query.center, halfExtents);
		const hits: PhysicsOverlapHit[] = [];
		for (const candidate of this._getQueryCandidates(world, query.filter)) {
			if (!intersectsBoxWithCollider(queryMin, queryMax, candidate)) {
				continue;
			}
			hits.push(toOverlapHit(worldId, candidate));
		}
		return truncateHits(hits, query.maxHits);
	}

	private _getQueryCandidates(
		world: AmmoWorldState,
		filter?: PhysicsQueryFilter
	): AmmoQueryCandidate[] {
		const includeBodyIds = toSet(filter?.includeBodyIds);
		const excludeBodyIds = toSet(filter?.excludeBodyIds);
		const includeColliderIds = toSet(filter?.includeColliderIds);
		const excludeColliderIds = toSet(filter?.excludeColliderIds);
		const includeTriggers = filter?.includeTriggers ?? true;
		const candidates: AmmoQueryCandidate[] = [];
		for (const collider of world.colliders.values()) {
			if (!includeTriggers && collider.isTrigger) continue;
			if (includeBodyIds && !includeBodyIds.has(collider.bodyId)) continue;
			if (excludeBodyIds?.has(collider.bodyId)) continue;
			if (includeColliderIds && !includeColliderIds.has(collider.id)) continue;
			if (excludeColliderIds?.has(collider.id)) continue;
			const body = world.bodies.get(collider.bodyId);
			if (!body) continue;
			candidates.push({
				body,
				collider,
				center: Vector3.add(body.transform.position, collider.offset),
			});
		}
		return candidates;
	}

	private _readHitStatus(callback: any): boolean {
		const variants = [
			this._readFromGetter(callback, "hasHit"),
			this._readFromGetter(callback, "get_m_hasHit"),
			this._readFromGetter(callback, "m_hasHit"),
		];
		for (const value of variants) {
			if (typeof value === "boolean") return value;
			if (typeof value === "number") return value !== 0;
		}
		return false;
	}

	private _resolveHitBody(
		world: AmmoWorldState,
		collisionObject: unknown
	): AmmoBodyState | null {
		if (!collisionObject) return null;
		for (const body of world.bodies.values()) {
			if (body.rigidBody === collisionObject) return body;
		}
		return null;
	}

	private _chooseHitCollider(
		world: AmmoWorldState,
		bodyId: string,
		filter: PhysicsQueryFilter | undefined,
		hitPoint: IVector3
	): AmmoColliderState | null {
		const includeColliderIds = toSet(filter?.includeColliderIds);
		const excludeColliderIds = toSet(filter?.excludeColliderIds);
		const includeTriggers = filter?.includeTriggers ?? true;
		let selected: AmmoColliderState | null = null;
		let bestDistance = Infinity;
		for (const colliderId of world.bodies.get(bodyId)?.colliderIds ?? []) {
			const collider = world.colliders.get(colliderId);
			if (!collider) continue;
			if (!includeTriggers && collider.isTrigger) continue;
			if (includeColliderIds && !includeColliderIds.has(collider.id)) continue;
			if (excludeColliderIds?.has(collider.id)) continue;
			const body = world.bodies.get(bodyId);
			if (!body) continue;
			const center = Vector3.add(body.transform.position, collider.offset);
			const distance = Vector3.length(Vector3.sub(hitPoint, center));
			if (distance < bestDistance) {
				bestDistance = distance;
				selected = collider;
			}
		}
		return selected;
	}

	private _setVector3(target: any, methods: string[], value: IVector3): void {
		const vector = this._createAmmoVector3(value);
		this._invoke(target, methods, [
			[vector],
			[value.x, value.y, value.z],
			[value],
		]);
		this._destroyAmmoObject(vector);
	}

	private _newAmmo(name: string, ...args: unknown[]): any {
		const Ctor = this._requireAmmoCtor(name);
		return new Ctor(...args);
	}

	private _tryNewAmmo(name: string, ...args: unknown[]): any {
		const ammo = this._ammo as Record<string, unknown> | null;
		if (!ammo) return null;
		const Ctor = ammo[name];
		if (typeof Ctor !== "function") return null;
		try {
			return new (Ctor as new (...ctorArgs: unknown[]) => any)(...args);
		} catch {
			return null;
		}
	}

	private _requireAmmoCtor(name: string): new (...args: unknown[]) => any {
		const ammo = this._requireAmmo();
		const member = (ammo as Record<string, unknown>)[name];
		if (typeof member !== "function") {
			throw new Error(`Loaded Ammo module is missing constructor ${name}`);
		}
		return member as new (...args: unknown[]) => any;
	}

	private _destroyAmmoObject(target: unknown): void {
		if (!target) return;
		const destroy = this._ammo?.destroy;
		if (typeof destroy !== "function") return;
		try {
			destroy(target);
		} catch {}
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
		return this._ammo === null;
	}

	private _requireAmmo(): AmmoModuleLike {
		if (this._ammo) return this._ammo;
		throw new Error(
			"AmmoPhysicsAdapter is not initialized with a usable Ammo module"
		);
	}

	private _requireBody(worldId: string, bodyId: string): AmmoBodyState {
		const world = this._requireWorld(worldId);
		const body = world.bodies.get(bodyId);
		if (body) return body;
		throw new Error(`Physics body "${bodyId}" does not exist in "${worldId}"`);
	}

	private _requireWorld(worldId: string): AmmoWorldState {
		const world = this._worlds.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}
}

async function resolveAmmoModule(
	loaded: unknown
): Promise<AmmoModuleLike | null> {
	for (const candidate of collectAmmoCandidates(loaded)) {
		const value = await materializeAmmoCandidate(candidate);
		if (value && isAmmoUsable(value)) return value;
	}
	for (const candidate of collectAmmoCandidates(loaded)) {
		const value = await materializeAmmoCandidate(candidate);
		if (value) return value;
	}
	return null;
}

function collectAmmoCandidates(loaded: unknown): unknown[] {
	const queue: unknown[] = [loaded];
	if (loaded && typeof loaded === "object") {
		const source = loaded as { default?: unknown; Ammo?: unknown };
		queue.push(source.default);
		queue.push(source.Ammo);
	}
	return queue;
}

async function materializeAmmoCandidate(
	candidate: unknown
): Promise<AmmoModuleLike | null> {
	if (!candidate) return null;
	if (typeof candidate === "function") {
		try {
			const resolved = await Promise.resolve(
				(candidate as () => Promise<unknown> | unknown)()
			);
			if (resolved && typeof resolved === "object") {
				return resolved as AmmoModuleLike;
			}
		} catch {}
		return null;
	}
	return typeof candidate === "object" ? (candidate as AmmoModuleLike) : null;
}

function isAmmoUsable(module: AmmoModuleLike): boolean {
	if (!module || typeof module !== "object") return false;
	return (
		typeof module.btDefaultCollisionConfiguration === "function" &&
		typeof module.btCollisionDispatcher === "function" &&
		typeof module.btDbvtBroadphase === "function" &&
		typeof module.btSequentialImpulseConstraintSolver === "function" &&
		typeof module.btDiscreteDynamicsWorld === "function" &&
		typeof module.btVector3 === "function" &&
		typeof module.btTransform === "function" &&
		typeof module.btDefaultMotionState === "function" &&
		typeof module.btRigidBodyConstructionInfo === "function" &&
		typeof module.btRigidBody === "function" &&
		(typeof module.btSphereShape === "function" ||
			typeof module.btBoxShape === "function")
	);
}

function resolveBodyMass(type: RigidBodyType, mass?: number): number {
	if (type !== "dynamic") return 0;
	if (Number.isFinite(mass) && Number(mass) > 0) return Number(mass);
	return 1;
}

function cloneVector(source: IVector3): IVector3 {
	return { x: source.x, y: source.y, z: source.z };
}

function cloneTransform(transform: PhysicsTransform): PhysicsTransform {
	return {
		position: cloneVector(transform.position),
		rotation: [
			transform.rotation[0],
			transform.rotation[1],
			transform.rotation[2],
			transform.rotation[3],
		],
	};
}

function resolveJointBodyId(value: JointDescriptor["bodyA"]): string {
	if (typeof value === "string") return value;
	if (value && typeof value === "object" && "id" in value) {
		return String((value as { id: string }).id);
	}
	return "";
}

function isZeroVector(value: IVector3): boolean {
	return (
		Math.abs(value.x) <= 1e-8 &&
		Math.abs(value.y) <= 1e-8 &&
		Math.abs(value.z) <= 1e-8
	);
}

function normalizeOptionalDirection(
	value: IVector3 | undefined,
	fallback: IVector3
): IVector3 {
	if (!value) return cloneVector(fallback);
	const length = Vector3.length(value);
	if (length <= 1e-8) return cloneVector(fallback);
	return Vector3.normalize(value);
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
	candidate: AmmoQueryCandidate,
	hit: AmmoQueryHit,
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
	candidate: AmmoQueryCandidate
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

function toSet(values?: string[]): Set<string> | null {
	if (!values || values.length === 0) return null;
	return new Set(values);
}

function sanitizeCollisionMask(mask: number): number {
	if (!Number.isFinite(mask)) return DEFAULT_COLLISION_MASK;
	return Math.floor(mask) >>> 0;
}

function intersectRayWithCollider(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	candidate: AmmoQueryCandidate
): AmmoQueryHit | null {
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
	candidate: AmmoQueryCandidate
): AmmoQueryHit | null {
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
	candidate: AmmoQueryCandidate
): AmmoQueryHit | null {
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
	candidate: AmmoQueryCandidate
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
	candidate: AmmoQueryCandidate
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
): AmmoQueryHit | null {
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
): AmmoQueryHit | null {
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
			normalSign *= -1;
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
	return { distance: entry, point, normal };
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

function getAxis(vector: IVector3, axis: number): number {
	switch (axis) {
		case 0:
			return vector.x;
		case 1:
			return vector.y;
		default:
			return vector.z;
	}
}

function axisVector(axis: number, sign: number): IVector3 {
	if (axis === 0) return { x: sign, y: 0, z: 0 };
	if (axis === 1) return { x: 0, y: sign, z: 0 };
	return { x: 0, y: 0, z: sign };
}

function readAmmoComponent(
	value: unknown,
	key: string,
	fallback: number
): number {
	if (!value || typeof value !== "object") return fallback;
	const source = value as Record<string, unknown>;
	const direct = source[key];
	if (typeof direct === "number" && Number.isFinite(direct)) return direct;
	if (typeof direct === "function") {
		try {
			const result = (direct as () => unknown).call(value);
			if (typeof result === "number" && Number.isFinite(result)) return result;
		} catch {}
	}
	const getter = source[`get${key.toUpperCase()}`];
	if (typeof getter === "function") {
		try {
			const result = (getter as () => unknown).call(value);
			if (typeof result === "number" && Number.isFinite(result)) return result;
		} catch {}
	}
	return fallback;
}

function readAmmoVector3(value: unknown): IVector3 {
	return {
		x: readAmmoComponent(value, "x", 0),
		y: readAmmoComponent(value, "y", 0),
		z: readAmmoComponent(value, "z", 0),
	};
}

function toQuaternionTuple(value: unknown): [number, number, number, number] {
	return [
		readAmmoComponent(value, "x", 0),
		readAmmoComponent(value, "y", 0),
		readAmmoComponent(value, "z", 0),
		readAmmoComponent(value, "w", 1),
	];
}

function loadOptionalModule(moduleName: string): Promise<unknown> {
	const importer = new Function("m", "return import(m)") as (
		modulePath: string
	) => Promise<unknown>;
	return importer(moduleName);
}
