import type { IVector3 } from "../maths/types";
import type {
	CharacterControllerDescriptor,
	CharacterMoveResult,
	PhysicsBoxCastQuery,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapHit,
	PhysicsOverlapSphereQuery,
	PhysicsQueryHit,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsEvent,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
} from "./types";

export interface PhysicsAdapterCapabilities {
	joints: boolean;
	characterController: boolean;
	shapeCast: boolean;
	query: boolean;
	syncInit: boolean;
}

export interface PhysicsAdapterBodyState {
	bodyId: string;
	transform: PhysicsTransform;
	sleeping: boolean;
	ccd: boolean;
}

export interface PhysicsAdapterStepResult {
	bodyStates: PhysicsAdapterBodyState[];
	events: PhysicsEvent[];
	activeBodies: number;
	sleepingBodies: number;
	ccdBodies: number;
}

export interface IPhysicsEngineAdapter {
	readonly id: string;
	readonly capabilities: PhysicsAdapterCapabilities;
	init(): Promise<void>;
	initSync?(): void;
	createWorld(config: PhysicsWorldConfig): void;
	destroyWorld(worldId: string): void;
	hasWorld(worldId: string): boolean;
	createBody(
		worldId: string,
		bodyId: string,
		descriptor: RigidBodyDescriptor,
		initialTransform: PhysicsTransform
	): void;
	destroyBody(worldId: string, bodyId: string): void;
	setBodyTransform(
		worldId: string,
		bodyId: string,
		transform: PhysicsTransform
	): void;
	setBodyLinearVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void;
	setAngularVelocity(
		worldId: string,
		bodyId: string,
		velocity: IVector3
	): void;
	applyForce(worldId: string, bodyId: string, force: IVector3): void;
	applyTorque(worldId: string, bodyId: string, torque: IVector3): void;
	applyImpulse(worldId: string, bodyId: string, impulse: IVector3): void;
	addCollider(
		worldId: string,
		bodyId: string,
		colliderId: string,
		descriptor: ColliderDescriptor,
		shape: ColliderShape
	): void;
	destroyCollider(worldId: string, colliderId: string): void;
	setColliderSensor(
		worldId: string,
		colliderId: string,
		isSensor: boolean
	): void;
	setCollisionMask(worldId: string, colliderId: string, mask: number): void;
	createJoint(
		worldId: string,
		jointId: string,
		descriptor: JointDescriptor
	): void;
	destroyJoint(worldId: string, jointId: string): void;
	createCharacterController(
		worldId: string,
		controllerId: string,
		descriptor: CharacterControllerDescriptor
	): void;
	destroyCharacterController(worldId: string, controllerId: string): void;
	moveCharacterController(
		worldId: string,
		controllerId: string,
		direction: IVector3,
		deltaSeconds: number
	): CharacterMoveResult;
	jumpCharacterController(
		worldId: string,
		controllerId: string,
		speed: number
	): void;
	isCharacterControllerGrounded(worldId: string, controllerId: string): boolean;
	setCharacterControllerMaxSlope(
		worldId: string,
		controllerId: string,
		value: number
	): void;
	setCharacterControllerStepHeight(
		worldId: string,
		controllerId: string,
		value: number
	): void;
	raycast(worldId: string, query: PhysicsRaycastQuery): PhysicsQueryHit | null;
	raycastAsync?(
		worldId: string,
		query: PhysicsRaycastQuery
	): Promise<PhysicsQueryHit | null>;
	raycastAll(worldId: string, query: PhysicsRaycastQuery): PhysicsQueryHit[];
	raycastAllAsync?(
		worldId: string,
		query: PhysicsRaycastQuery
	): Promise<PhysicsQueryHit[]>;
	sphereCast(
		worldId: string,
		query: PhysicsSphereCastQuery
	): PhysicsQueryHit | null;
	sphereCastAsync?(
		worldId: string,
		query: PhysicsSphereCastQuery
	): Promise<PhysicsQueryHit | null>;
	boxCast(worldId: string, query: PhysicsBoxCastQuery): PhysicsQueryHit | null;
	boxCastAsync?(
		worldId: string,
		query: PhysicsBoxCastQuery
	): Promise<PhysicsQueryHit | null>;
	overlapSphere(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): PhysicsOverlapHit[];
	overlapSphereAsync?(
		worldId: string,
		query: PhysicsOverlapSphereQuery
	): Promise<PhysicsOverlapHit[]>;
	overlapBox(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): PhysicsOverlapHit[];
	overlapBoxAsync?(
		worldId: string,
		query: PhysicsOverlapBoxQuery
	): Promise<PhysicsOverlapHit[]>;
	stepWorld(worldId: string, deltaSeconds: number): PhysicsAdapterStepResult;
	stepWorldAsync?(
		worldId: string,
		deltaSeconds: number
	): Promise<PhysicsAdapterStepResult>;
}
