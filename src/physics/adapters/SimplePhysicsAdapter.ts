import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import { DEFAULT_GRAVITY } from "../constants";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	PhysicsBoxCastQuery,
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
	PhysicsEvent,
	PhysicsEventType,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
	RigidBodyType,
} from "../types";
import type {
	IPhysicsEngineAdapter,
	PhysicsAdapterBodyState,
	PhysicsAdapterCapabilities,
	PhysicsAdapterStepResult,
} from "../IPhysicsEngineAdapter";

interface SimpleBodyState {
	id: string;
	type: RigidBodyType;
	mass: number;
	canSleep: boolean;
	linearDamping: number;
	angularDamping: number;
	lockTranslations: [boolean, boolean, boolean] | null;
	lockRotations: [boolean, boolean, boolean] | null;
	transform: PhysicsTransform;
	linearVelocity: IVector3;
	angularVelocity: IVector3;
	accumulatedForce: IVector3;
	accumulatedTorque: IVector3;
	sleeping: boolean;
	ccd: boolean;
	colliderIds: Set<string>;
}

interface SimpleColliderState {
	id: string;
	bodyId: string;
	descriptor: ColliderDescriptor;
	shape: ColliderShape;
	isTrigger: boolean;
	collisionMask: number;
	radius: number;
	halfExtents: IVector3;
	offset: IVector3;
}

interface SimpleControllerState {
	id: string;
	bodyId: string;
	grounded: boolean;
	maxSlope: number;
	stepHeight: number;
	gravityScale: number | null;
	pendingDirection: IVector3;
	pendingJumpSpeed: number;
}

interface SimpleWorldState {
	config: PhysicsWorldConfig;
	bodies: Map<string, SimpleBodyState>;
	colliders: Map<string, SimpleColliderState>;
	joints: Map<string, JointDescriptor>;
	controllers: Map<string, SimpleControllerState>;
	activePairs: Map<string, "collision" | "trigger">;
}

interface SimpleQueryCandidate {
	body: SimpleBodyState;
	collider: SimpleColliderState;
	center: IVector3;
}

interface SimpleQueryHit {
	distance: number;
	point: IVector3;
	normal: IVector3;
}

const DEFAULT_COLLISION_MASK = 0xffffffff;

export class SimplePhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id: string;
	public readonly capabilities: PhysicsAdapterCapabilities;
	private _worlds = new Map<string, SimpleWorldState>();

	constructor(id: string = "simple") {
		this.id = id;
		this.capabilities = {
			joints: true,
			characterController: true,
			shapeCast: true,
			query: true,
			syncInit: true,
		};
	}

	public async init(): Promise<void> {}

	public initSync(): void {}

	public hasWorld(worldId: string): boolean {
		return this._worlds.has(worldId);
	}

	public createWorld(config: PhysicsWorldConfig): void {
		if (this._worlds.has(config.worldId)) {
			throw new Error(`Physics world "${config.worldId}" already exists`);
		}
		this._worlds.set(config.worldId, {
			config: {
				...config,
				gravity: cloneVector(config.gravity ?? DEFAULT_GRAVITY),
			},
			bodies: new Map(),
			colliders: new Map(),
			joints: new Map(),
			controllers: new Map(),
			activePairs: new Map(),
		});
	}

	public destroyWorld(worldId: string): void {
		this._worlds.delete(worldId);
	}

	public createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void {
		const world = this._requireWorld(worldId);
		if (world.bodies.has(bodyId)) {
			throw new Error(
				`Physics body "${bodyId}" already exists in "${worldId}"`
			);
		}
		world.bodies.set(bodyId, {
			id: bodyId,
			type: descriptor.type ?? "dynamic",
			mass: resolveBodyMass(descriptor),
			canSleep: descriptor.canSleep ?? world.config.allowSleep ?? true,
			linearDamping: sanitizeDamping(descriptor.linearDamping),
			angularDamping: sanitizeDamping(descriptor.angularDamping),
			lockTranslations: sanitizeLockAxes(descriptor.lockTranslations),
			lockRotations: sanitizeLockAxes(descriptor.lockRotations),
			transform: cloneTransform(initialTransform),
			linearVelocity: cloneVector(
				descriptor.linearVelocity ?? { x: 0, y: 0, z: 0 }
			),
			angularVelocity: cloneVector(
				descriptor.angularVelocity ?? { x: 0, y: 0, z: 0 }
			),
			accumulatedForce: { x: 0, y: 0, z: 0 },
			accumulatedTorque: { x: 0, y: 0, z: 0 },
			sleeping: false,
			ccd: descriptor.ccd ?? world.config.enableCCD ?? false,
			colliderIds: new Set(),
		});
	}

	public destroyBody(worldId: string, bodyId: string): void {
		const world = this._requireWorld(worldId);
		const body = world.bodies.get(bodyId);
		if (!body) return;
		for (const colliderId of body.colliderIds) {
			world.colliders.delete(colliderId);
		}
		body.colliderIds.clear();
		world.bodies.delete(bodyId);

		for (const [controllerId, controller] of world.controllers) {
			if (controller.bodyId === bodyId) {
				world.controllers.delete(controllerId);
			}
		}
	}

	public setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void {
		const body = this._requireBody(worldId, bodyId);
		body.transform = cloneTransform(transform);
		body.sleeping = false;
	}

	public setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		const body = this._requireBody(worldId, bodyId);
		body.linearVelocity = cloneVector(velocity);
		body.sleeping = false;
	}

	public setAngularVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void {
		const body = this._requireBody(worldId, bodyId);
		body.angularVelocity = cloneVector(velocity);
		body.sleeping = false;
	}

	public applyForce(worldId: string, bodyId: string, force: IVector3): void {
		const body = this._requireBody(worldId, bodyId);
		if (body.type !== "dynamic") return;
		body.accumulatedForce.x += force.x;
		body.accumulatedForce.y += force.y;
		body.accumulatedForce.z += force.z;
		body.sleeping = false;
	}

	public applyTorque(worldId: string, bodyId: string, torque: IVector3): void {
		const body = this._requireBody(worldId, bodyId);
		if (body.type !== "dynamic") return;
		body.accumulatedTorque.x += torque.x;
		body.accumulatedTorque.y += torque.y;
		body.accumulatedTorque.z += torque.z;
		body.sleeping = false;
	}

	public applyImpulse(
		worldId: string,
		bodyId: string,
		impulse: IVector3
	): void {
		const body = this._requireBody(worldId, bodyId);
		if (body.type !== "dynamic") return;
		const invMass = 1 / body.mass;
		body.linearVelocity.x += impulse.x * invMass;
		body.linearVelocity.y += impulse.y * invMass;
		body.linearVelocity.z += impulse.z * invMass;
		body.sleeping = false;
	}

	public addCollider(
		worldId: string,
		bodyId: string,
		colliderId: string,
		descriptor: ColliderDescriptor,
		shape: ColliderShape
	): void {
		const world = this._requireWorld(worldId);
		if (world.colliders.has(colliderId)) {
			throw new Error(
				`Physics collider "${colliderId}" already exists in "${worldId}"`
			);
		}
		const body = this._requireBody(worldId, bodyId);
		const collider: SimpleColliderState = {
			id: colliderId,
			bodyId,
			descriptor,
			shape,
			isTrigger: descriptor.isTrigger === true,
			collisionMask: DEFAULT_COLLISION_MASK,
			radius: computeShapeRadius(shape),
			halfExtents: computeShapeHalfExtents(shape),
			offset: cloneVector(descriptor.offset ?? { x: 0, y: 0, z: 0 }),
		};
		world.colliders.set(colliderId, collider);
		body.colliderIds.add(colliderId);
	}

	public destroyCollider(worldId: string, colliderId: string): void {
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) return;
		world.colliders.delete(colliderId);
		const body = world.bodies.get(collider.bodyId);
		body?.colliderIds.delete(colliderId);
	}

	public setColliderSensor(
		worldId: string,
		colliderId: string,
		isSensor: boolean
	): void {
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) {
			throw new Error(
				`Physics collider "${colliderId}" does not exist in "${worldId}"`
			);
		}
		collider.isTrigger = isSensor === true;
	}

	public setCollisionMask(
		worldId: string,
		colliderId: string,
		mask: number
	): void {
		const world = this._requireWorld(worldId);
		const collider = world.colliders.get(colliderId);
		if (!collider) {
			throw new Error(
				`Physics collider "${colliderId}" does not exist in "${worldId}"`
			);
		}
		collider.collisionMask = sanitizeCollisionMask(mask);
	}

	public createJoint(
		worldId: string,
		jointId: string,
		descriptor: JointDescriptor
	): void {
		const world = this._requireWorld(worldId);
		if (world.joints.has(jointId)) {
			throw new Error(
				`Physics joint "${jointId}" already exists in "${worldId}"`
			);
		}
		world.joints.set(jointId, descriptor);
	}

	public destroyJoint(worldId: string, jointId: string): void {
		const world = this._requireWorld(worldId);
		world.joints.delete(jointId);
	}

	public createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void {
		const world = this._requireWorld(worldId);
		if (world.controllers.has(controllerId)) {
			throw new Error(
				`Character controller "${controllerId}" already exists in "${worldId}"`
			);
		}
		const bodyId = resolveBodyId(descriptor.body);
		this._requireBody(worldId, bodyId);
		world.controllers.set(controllerId, {
			id: controllerId,
			bodyId,
			grounded: false,
			maxSlope: Math.max(0, descriptor.maxSlope ?? 60),
			stepHeight: Math.max(0, descriptor.stepHeight ?? 0.3),
			gravityScale:
				Number.isFinite(descriptor.gravityScale) ?
					Number(descriptor.gravityScale)
				:	null,
			pendingDirection: { x: 0, y: 0, z: 0 },
			pendingJumpSpeed: 0,
		});
	}

	public destroyCharacterController(
		worldId: string,
		controllerId: string
	): void {
		const world = this._requireWorld(worldId);
		world.controllers.delete(controllerId);
	}

	public moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		_deltaSeconds: number
	): CharacterMoveResult {
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
		const controller = this._requireController(worldId, controllerId);
		controller.pendingJumpSpeed = Math.max(0, speed);
	}

	public isCharacterControllerGrounded(
		worldId: string,
		controllerId: string
	): boolean {
		const controller = this._requireController(worldId, controllerId);
		return controller.grounded;
	}

	public setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		const controller = this._requireController(worldId, controllerId);
		controller.maxSlope = Math.max(0, value);
	}

	public setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void {
		const controller = this._requireController(worldId, controllerId);
		controller.stepHeight = Math.max(0, value);
	}

	public raycast(
		worldId: string,
		query: PhysicsRaycastQuery
	): PhysicsQueryHit | null {
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
		const world = this._requireWorld(worldId);
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

	public overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[] {
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
		const world = this._requireWorld(worldId);
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

	public stepWorld(
		worldId: string,
		deltaSeconds: number
	): PhysicsAdapterStepResult {
		const world = this._requireWorld(worldId);
		const dt = Math.max(0, deltaSeconds);
		const gravity = world.config.gravity ?? DEFAULT_GRAVITY;

		this._applyCharacterControllerInputs(world, dt);
		for (const body of world.bodies.values()) {
			if (body.type !== "dynamic") continue;
			if (dt > 0) {
				const invMass = 1 / body.mass;
				body.linearVelocity.x +=
					(gravity.x + body.accumulatedForce.x * invMass) * dt;
				body.linearVelocity.y +=
					(gravity.y + body.accumulatedForce.y * invMass) * dt;
				body.linearVelocity.z +=
					(gravity.z + body.accumulatedForce.z * invMass) * dt;
				body.angularVelocity.x += body.accumulatedTorque.x * invMass * dt;
				body.angularVelocity.y += body.accumulatedTorque.y * invMass * dt;
				body.angularVelocity.z += body.accumulatedTorque.z * invMass * dt;
				body.accumulatedForce.x = 0;
				body.accumulatedForce.y = 0;
				body.accumulatedForce.z = 0;
				body.accumulatedTorque.x = 0;
				body.accumulatedTorque.y = 0;
				body.accumulatedTorque.z = 0;

				const linearDampingFactor = computeDampingFactor(
					body.linearDamping,
					dt
				);
				const angularDampingFactor = computeDampingFactor(
					body.angularDamping,
					dt
				);
				body.linearVelocity.x *= linearDampingFactor;
				body.linearVelocity.y *= linearDampingFactor;
				body.linearVelocity.z *= linearDampingFactor;
				body.angularVelocity.x *= angularDampingFactor;
				body.angularVelocity.y *= angularDampingFactor;
				body.angularVelocity.z *= angularDampingFactor;

				if (body.lockTranslations) {
					if (body.lockTranslations[0]) body.linearVelocity.x = 0;
					if (body.lockTranslations[1]) body.linearVelocity.y = 0;
					if (body.lockTranslations[2]) body.linearVelocity.z = 0;
				}
				if (body.lockRotations) {
					if (body.lockRotations[0]) body.angularVelocity.x = 0;
					if (body.lockRotations[1]) body.angularVelocity.y = 0;
					if (body.lockRotations[2]) body.angularVelocity.z = 0;
				}
			}
			if (!body.lockTranslations?.[0]) {
				body.transform.position.x += body.linearVelocity.x * dt;
			}
			if (!body.lockTranslations?.[1]) {
				body.transform.position.y += body.linearVelocity.y * dt;
			}
			if (!body.lockTranslations?.[2]) {
				body.transform.position.z += body.linearVelocity.z * dt;
			}
		}

		const events = this._resolveCollisions(world);
		const bodyStates: PhysicsAdapterBodyState[] = [];
		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const body of world.bodies.values()) {
			const linearSpeed = Math.hypot(
				body.linearVelocity.x,
				body.linearVelocity.y,
				body.linearVelocity.z
			);
			const angularSpeed = Math.hypot(
				body.angularVelocity.x,
				body.angularVelocity.y,
				body.angularVelocity.z
			);
			body.sleeping =
				body.type === "dynamic" &&
				body.canSleep &&
				linearSpeed < 0.001 &&
				angularSpeed < 0.001;
			if (body.sleeping) sleepingBodies++;
			if (body.ccd) ccdBodies++;

			bodyStates.push({
				bodyId: body.id,
				transform: cloneTransform(body.transform),
				sleeping: body.sleeping,
				ccd: body.ccd,
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

	private _applyCharacterControllerInputs(
		world: SimpleWorldState,
		deltaSeconds: number
	): void {
		const gravity = world.config.gravity ?? DEFAULT_GRAVITY;
		for (const controller of world.controllers.values()) {
			const body = world.bodies.get(controller.bodyId);
			if (!body) continue;

			if (!body.lockTranslations?.[0]) {
				body.transform.position.x +=
					controller.pendingDirection.x * deltaSeconds;
			}
			if (!body.lockTranslations?.[2]) {
				body.transform.position.z +=
					controller.pendingDirection.z * deltaSeconds;
			}
			if (controller.pendingJumpSpeed > 0 && !body.lockTranslations?.[1]) {
				body.linearVelocity.y = controller.pendingJumpSpeed;
			}
			if (controller.gravityScale !== null) {
				if (body.type === "dynamic") {
					const gravityDelta = (controller.gravityScale - 1) * deltaSeconds;
					body.linearVelocity.x += gravity.x * gravityDelta;
					body.linearVelocity.y += gravity.y * gravityDelta;
					body.linearVelocity.z += gravity.z * gravityDelta;
				} else {
					if (!body.lockTranslations?.[0]) {
						body.transform.position.x +=
							gravity.x * controller.gravityScale * deltaSeconds;
					}
					if (!body.lockTranslations?.[1]) {
						body.transform.position.y +=
							gravity.y * controller.gravityScale * deltaSeconds;
					}
					if (!body.lockTranslations?.[2]) {
						body.transform.position.z +=
							gravity.z * controller.gravityScale * deltaSeconds;
					}
				}
			}
			controller.pendingDirection = { x: 0, y: 0, z: 0 };
			controller.pendingJumpSpeed = 0;
		}
	}

	private _resolveCollisions(world: SimpleWorldState): PhysicsEvent[] {
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
					leftBody.transform.position,
					left.offset
				);
				const rightCenter = Vector3.add(
					rightBody.transform.position,
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

				if (!left.isTrigger && !right.isTrigger) {
					this._resolveOverlap(
						leftBody,
						rightBody,
						leftCenter,
						rightCenter,
						overlap
					);
				}
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
		leftBody: SimpleBodyState,
		rightBody: SimpleBodyState,
		leftCenter: IVector3,
		rightCenter: IVector3,
		overlap: number
	): void {
		const separationAxis = Vector3.sub(leftCenter, rightCenter);
		const length = Vector3.length(separationAxis) || 1;
		const normal = Vector3.scale(separationAxis, 1 / length);
		const separation = overlap * 0.5;

		if (leftBody.type === "dynamic") {
			leftBody.transform.position.x += normal.x * separation;
			leftBody.transform.position.y += normal.y * separation;
			leftBody.transform.position.z += normal.z * separation;
		}
		if (rightBody.type === "dynamic") {
			rightBody.transform.position.x -= normal.x * separation;
			rightBody.transform.position.y -= normal.y * separation;
			rightBody.transform.position.z -= normal.z * separation;
		}
	}

	private _resolveGroundedControllers(world: SimpleWorldState): void {
		for (const controller of world.controllers.values()) {
			const body = world.bodies.get(controller.bodyId);
			if (!body) continue;
			controller.grounded = body.transform.position.y <= controller.stepHeight;
		}
	}

	private _getQueryCandidates(
		world: SimpleWorldState,
		filter?: PhysicsQueryFilter
	): SimpleQueryCandidate[] {
		const includeBodyIds = toSet(filter?.includeBodyIds);
		const excludeBodyIds = toSet(filter?.excludeBodyIds);
		const includeColliderIds = toSet(filter?.includeColliderIds);
		const excludeColliderIds = toSet(filter?.excludeColliderIds);
		const includeTriggers = filter?.includeTriggers ?? true;

		const candidates: SimpleQueryCandidate[] = [];
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

	private _requireWorld(worldId: string): SimpleWorldState {
		const world = this._worlds.get(worldId);
		if (world) return world;
		throw new Error(`Physics world "${worldId}" does not exist`);
	}

	private _requireBody(worldId: string, bodyId: string): SimpleBodyState {
		const world = this._requireWorld(worldId);
		const body = world.bodies.get(bodyId);
		if (body) return body;
		throw new Error(`Physics body "${bodyId}" does not exist in "${worldId}"`);
	}

	private _requireController(
		worldId: string,
		controllerId: string
	): SimpleControllerState {
		const world = this._requireWorld(worldId);
		const controller = world.controllers.get(controllerId);
		if (controller) return controller;
		throw new Error(
			`Character controller "${controllerId}" does not exist in "${worldId}"`
		);
	}
}

function resolveBodyId(value: CharacterControllerDescriptor["body"]): string {
	if (typeof value === "string") return value;
	if (value && typeof value === "object" && "id" in value) {
		return String((value as { id: string }).id);
	}
	return "";
}

function cloneVector(source: IVector3): IVector3 {
	return new Vector3().copy(source);
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

function resolveBodyMass(descriptor: RigidBodyDescriptor): number {
	if ((descriptor.type ?? "dynamic") !== "dynamic") return 1;
	if (Number.isFinite(descriptor.mass) && Number(descriptor.mass) > 0) {
		return Number(descriptor.mass);
	}
	return 1;
}

function sanitizeDamping(value: number | undefined): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(0, Number(value));
}

function sanitizeLockAxes(
	value: [boolean, boolean, boolean] | undefined
): [boolean, boolean, boolean] | null {
	if (!value) return null;
	return [value[0] === true, value[1] === true, value[2] === true];
}

function computeDampingFactor(damping: number, deltaSeconds: number): number {
	if (deltaSeconds <= 0 || damping <= 0) return 1;
	return Math.max(0, 1 - damping * deltaSeconds);
}

function sanitizeCollisionMask(mask: number): number {
	if (!Number.isFinite(mask)) return DEFAULT_COLLISION_MASK;
	return Math.floor(mask) >>> 0;
}

function canCollidersInteract(
	left: SimpleColliderState,
	right: SimpleColliderState
): boolean {
	return (left.collisionMask & right.collisionMask) !== 0;
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
	candidate: SimpleQueryCandidate,
	hit: SimpleQueryHit,
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
	candidate: SimpleQueryCandidate
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

function intersectRayWithCollider(
	origin: IVector3,
	direction: IVector3,
	maxDistance: number,
	candidate: SimpleQueryCandidate
): SimpleQueryHit | null {
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
	candidate: SimpleQueryCandidate
): SimpleQueryHit | null {
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
	candidate: SimpleQueryCandidate
): SimpleQueryHit | null {
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
	candidate: SimpleQueryCandidate
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
	candidate: SimpleQueryCandidate
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
): SimpleQueryHit | null {
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
): SimpleQueryHit | null {
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
