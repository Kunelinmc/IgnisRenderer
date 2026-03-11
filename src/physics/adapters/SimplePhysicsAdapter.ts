import type { IVector3 } from "../../maths/types";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
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
	transform: PhysicsTransform;
	linearVelocity: IVector3;
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
	radius: number;
	offset: IVector3;
}

interface SimpleControllerState {
	id: string;
	bodyId: string;
	grounded: boolean;
	maxSlope: number;
	stepHeight: number;
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

const DEFAULT_GRAVITY: IVector3 = { x: 0, y: -9.8, z: 0 };

export class SimplePhysicsAdapter implements IPhysicsEngineAdapter {
	public readonly id: string;
	public readonly capabilities: PhysicsAdapterCapabilities;
	private _worlds = new Map<string, SimpleWorldState>();

	constructor(id: string = "simple") {
		this.id = id;
		this.capabilities = {
			joints: true,
			characterController: true,
			shapeCast: false,
			query: false,
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
			transform: cloneTransform(initialTransform),
			linearVelocity: cloneVector(
				descriptor.linearVelocity ?? { x: 0, y: 0, z: 0 }
			),
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
			radius: computeShapeRadius(shape),
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
			body.linearVelocity.x += gravity.x * dt;
			body.linearVelocity.y += gravity.y * dt;
			body.linearVelocity.z += gravity.z * dt;
			body.transform.position.x += body.linearVelocity.x * dt;
			body.transform.position.y += body.linearVelocity.y * dt;
			body.transform.position.z += body.linearVelocity.z * dt;
		}

		const events = this._resolveCollisions(world);
		const bodyStates: PhysicsAdapterBodyState[] = [];
		let sleepingBodies = 0;
		let ccdBodies = 0;
		for (const body of world.bodies.values()) {
			const speed = Math.hypot(
				body.linearVelocity.x,
				body.linearVelocity.y,
				body.linearVelocity.z
			);
			body.sleeping = speed < 0.001 && body.type === "dynamic";
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
		for (const controller of world.controllers.values()) {
			const body = world.bodies.get(controller.bodyId);
			if (!body) continue;

			body.transform.position.x += controller.pendingDirection.x * deltaSeconds;
			body.transform.position.z += controller.pendingDirection.z * deltaSeconds;
			if (controller.pendingJumpSpeed > 0) {
				body.linearVelocity.y = controller.pendingJumpSpeed;
			}
			controller.pendingDirection = { x: 0, y: 0, z: 0 };
			controller.pendingJumpSpeed = 0;
		}
	}

	private _resolveCollisions(world: SimpleWorldState): PhysicsEvent[] {
		const events: PhysicsEvent[] = [];
		const nowMs = Date.now();
		const currentPairs = new Map<string, "collision" | "trigger">();
		const colliders = Array.from(world.colliders.values());

		for (let i = 0; i < colliders.length; i++) {
			for (let j = i + 1; j < colliders.length; j++) {
				const left = colliders[i];
				const right = colliders[j];
				if (left.bodyId === right.bodyId) continue;

				const leftBody = world.bodies.get(left.bodyId);
				const rightBody = world.bodies.get(right.bodyId);
				if (!leftBody || !rightBody) continue;

				const leftCenter = addVec3(leftBody.transform.position, left.offset);
				const rightCenter = addVec3(rightBody.transform.position, right.offset);
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
					timestampMs: nowMs,
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
				timestampMs: nowMs,
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
		const nx = leftCenter.x - rightCenter.x;
		const ny = leftCenter.y - rightCenter.y;
		const nz = leftCenter.z - rightCenter.z;
		const length = Math.hypot(nx, ny, nz) || 1;
		const normal = { x: nx / length, y: ny / length, z: nz / length };
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

function addVec3(left: IVector3, right: IVector3): IVector3 {
	return {
		x: left.x + right.x,
		y: left.y + right.y,
		z: left.z + right.z,
	};
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
