import type { Node } from "../core/Node";
import type { IVector3 } from "../maths/types";

export type QuaternionTuple = [number, number, number, number];

export type PhysicsWorldId = string;

export type PhysicsStepMode = "fixed" | "variable";

export type TransformAuthority = "physics" | "animation";
export type PhysicsEntityId = number;

export type RigidBodyType = "dynamic" | "kinematic" | "fixed";

export type JointType = "fixed" | "hinge" | "spring";

export type PhysicsEventType =
	| "collisionBegin"
	| "collisionStay"
	| "collisionEnd"
	| "triggerBegin"
	| "triggerStay"
	| "triggerEnd";

export interface PhysicsTransform {
	position: IVector3;
	rotation: QuaternionTuple;
}

export interface PhysicsStepConfig {
	mode?: PhysicsStepMode;
	fixedDeltaSeconds?: number;
	maxSubsteps?: number;
	maxDeltaSeconds?: number;
}

export interface PhysicsWorldConfig extends PhysicsStepConfig {
	worldId: PhysicsWorldId;
	gravity?: IVector3;
	allowSleep?: boolean;
	enableCCD?: boolean;
}

export interface RigidBodyDescriptor {
	type?: RigidBodyType;
	mass?: number;
	linearDamping?: number;
	angularDamping?: number;
	canSleep?: boolean;
	ccd?: boolean;
	linearVelocity?: IVector3;
	angularVelocity?: IVector3;
	lockTranslations?: [boolean, boolean, boolean];
	lockRotations?: [boolean, boolean, boolean];
}

export interface PhysicsMaterialDescriptor {
	friction?: number;
	restitution?: number;
	density?: number;
}

export interface ColliderShapeBox {
	kind: "box";
	halfExtents: IVector3;
}

export interface ColliderShapeSphere {
	kind: "sphere";
	radius: number;
}

export interface ColliderShapeCapsule {
	kind: "capsule";
	radius: number;
	halfHeight: number;
}

export interface ColliderShapeCylinder {
	kind: "cylinder";
	radius: number;
	halfHeight: number;
}

export interface ColliderShapeTrimesh {
	kind: "trimesh";
	vertices: Float32Array | number[];
	indices: Uint32Array | number[];
}

export type ColliderShape =
	| ColliderShapeBox
	| ColliderShapeSphere
	| ColliderShapeCapsule
	| ColliderShapeCylinder
	| ColliderShapeTrimesh;

export interface ColliderDescriptorBase {
	isTrigger?: boolean;
	offset?: IVector3;
	material?: PhysicsMaterialDescriptor;
}

export interface ExplicitColliderDescriptor extends ColliderDescriptorBase {
	mode?: "explicit";
	shape: ColliderShape;
}

export interface AutoFitColliderDescriptor extends ColliderDescriptorBase {
	mode: "auto-fit";
	shapePreference?: "box" | "sphere";
	sourceNode?: Node;
}

export interface TrimeshCookColliderDescriptor extends ColliderDescriptorBase {
	mode: "trimesh-cook";
	sourceNode?: Node;
}

export type ColliderDescriptor =
	| ExplicitColliderDescriptor
	| AutoFitColliderDescriptor
	| TrimeshCookColliderDescriptor;

export interface BodyBinding {
	worldId: PhysicsWorldId;
	body: RigidBodyDescriptor;
	authority?: TransformAuthority;
	colliders?: ColliderDescriptor[];
}

export interface JointDescriptor {
	worldId: PhysicsWorldId;
	type: JointType;
	bodyA: string | Node | PhysicsBodyHandle | PhysicsEntityId;
	bodyB: string | Node | PhysicsBodyHandle | PhysicsEntityId;
	anchorA?: IVector3;
	anchorB?: IVector3;
	axis?: IVector3;
	limits?: [number, number];
	stiffness?: number;
	damping?: number;
	collisionEnabled?: boolean;
}

export interface CharacterControllerDescriptor {
	worldId: PhysicsWorldId;
	body: string | Node | PhysicsBodyHandle | PhysicsEntityId;
	radius: number;
	height: number;
	stepHeight?: number;
	maxSlope?: number;
	jumpSpeed?: number;
	gravityScale?: number;
}

export interface CharacterMoveResult {
	grounded: boolean;
	moved: IVector3;
}

export interface PhysicsCollisionEvent {
	type: PhysicsEventType;
	worldId: PhysicsWorldId;
	bodyAId: string;
	bodyBId: string;
	timestampSeconds: number;
}

export type PhysicsEvent = PhysicsCollisionEvent;

export interface PhysicsQueryFilter {
	includeTriggers?: boolean;
	includeBodyIds?: string[];
	excludeBodyIds?: string[];
	includeColliderIds?: string[];
	excludeColliderIds?: string[];
}

export interface PhysicsQueryBase {
	worldId?: PhysicsWorldId;
	filter?: PhysicsQueryFilter;
	maxHits?: number;
}

export interface PhysicsRaycastQuery extends PhysicsQueryBase {
	origin: IVector3;
	direction: IVector3;
	maxDistance?: number;
}

export interface PhysicsSphereCastQuery extends PhysicsQueryBase {
	center: IVector3;
	radius: number;
	direction: IVector3;
	maxDistance?: number;
}

export interface PhysicsBoxCastQuery extends PhysicsQueryBase {
	center: IVector3;
	halfExtents: IVector3;
	direction: IVector3;
	maxDistance?: number;
}

export interface PhysicsOverlapSphereQuery extends PhysicsQueryBase {
	center: IVector3;
	radius: number;
}

export interface PhysicsOverlapBoxQuery extends PhysicsQueryBase {
	center: IVector3;
	halfExtents: IVector3;
}

export interface PhysicsQueryHit {
	worldId: PhysicsWorldId;
	bodyId: string;
	colliderId: string;
	point: IVector3;
	normal: IVector3;
	distance: number;
	fraction: number;
	isTrigger: boolean;
}

export interface PhysicsOverlapHit {
	worldId: PhysicsWorldId;
	bodyId: string;
	colliderId: string;
	isTrigger: boolean;
}

export interface PhysicsWorldStepReport {
	worldId: PhysicsWorldId;
	mode: PhysicsStepMode;
	substeps: number;
	consumedDeltaSeconds: number;
	activeBodies: number;
	sleepingBodies: number;
	ccdBodies: number;
}

export interface PhysicsStepReport {
	inputDeltaSeconds: number;
	processedDeltaSeconds: number;
	worldReports: PhysicsWorldStepReport[];
	events: PhysicsEvent[];
	dirty: boolean;
}

export interface StepOverride extends PhysicsStepConfig {
	worldIds?: PhysicsWorldId[];
}

export interface PhysicsBodyHandle {
	readonly id: string;
	readonly worldId: PhysicsWorldId;
	readonly node: Node;
	readonly entityId?: PhysicsEntityId;
	authority: TransformAuthority;
}

export interface PhysicsColliderHandle {
	readonly id: string;
	readonly worldId: PhysicsWorldId;
	readonly bodyId: string;
}

export interface PhysicsJointHandle {
	readonly id: string;
	readonly worldId: PhysicsWorldId;
}

export interface CharacterControllerHandle {
	readonly id: string;
	readonly worldId: PhysicsWorldId;
	moveAndSlide(direction: IVector3, deltaSeconds: number): CharacterMoveResult;
	jump(speed?: number): void;
	isGrounded(): boolean;
	setMaxSlope(value: number): void;
	setStepHeight(value: number): void;
}

export interface PhysicsEvents {
	collisionBegin: [PhysicsCollisionEvent];
	collisionStay: [PhysicsCollisionEvent];
	collisionEnd: [PhysicsCollisionEvent];
	triggerBegin: [PhysicsCollisionEvent];
	triggerStay: [PhysicsCollisionEvent];
	triggerEnd: [PhysicsCollisionEvent];
	dirty: [
		{
			worldIds: PhysicsWorldId[];
			movedBodyIds: string[];
			events: PhysicsEvent[];
		},
	];
	step: [PhysicsStepReport];
	[key: string]: any[];
}

export interface CollisionGeometryBounds {
	box: {
		min: IVector3;
		max: IVector3;
	};
	sphere: {
		center: IVector3;
		radius: number;
	};
}

export interface CollisionGeometryTriangles {
	vertices: Float32Array;
	indices: Uint32Array;
}

export interface ICollisionGeometryProvider {
	getBounds(node: Node): CollisionGeometryBounds | null;
	getTriangles(node: Node): CollisionGeometryTriangles | null;
}
