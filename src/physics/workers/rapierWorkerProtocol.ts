import type { IVector3 } from "../../maths/types";
import type {
	CharacterControllerDescriptor,
	ColliderDescriptor,
	ColliderShape,
	JointDescriptor,
	PhysicsMaterialDescriptor,
	PhysicsBoxCastQuery,
	PhysicsOverlapBoxQuery,
	PhysicsOverlapSphereQuery,
	PhysicsRaycastQuery,
	PhysicsSphereCastQuery,
	PhysicsTransform,
	PhysicsWorldConfig,
	RigidBodyDescriptor,
} from "../types";

export type RapierWorkerCommand =
	| {
			type: "createWorld";
			config: PhysicsWorldConfig;
	  }
	| {
			type: "destroyWorld";
			worldId: string;
	  }
	| {
			type: "createBody";
			worldId: string;
			bodyId: string;
			descriptor: RigidBodyDescriptor;
			initialTransform: PhysicsTransform;
	  }
	| {
			type: "destroyBody";
			worldId: string;
			bodyId: string;
	  }
	| {
			type: "setBodyTransform";
			worldId: string;
			bodyId: string;
			transform: PhysicsTransform;
	  }
	| {
			type: "setBodyLinearVelocity";
			worldId: string;
			bodyId: string;
			velocity: IVector3;
	  }
	| {
			type: "setAngularVelocity";
			worldId: string;
			bodyId: string;
			velocity: IVector3;
	  }
	| {
			type: "applyForce";
			worldId: string;
			bodyId: string;
			force: IVector3;
	  }
	| {
			type: "applyTorque";
			worldId: string;
			bodyId: string;
			torque: IVector3;
	  }
	| {
			type: "applyImpulse";
			worldId: string;
			bodyId: string;
			impulse: IVector3;
	  }
	| {
			type: "addCollider";
			worldId: string;
			bodyId: string;
			colliderId: string;
			descriptor: ColliderDescriptor;
			shape: ColliderShape;
	  }
	| {
			type: "destroyCollider";
			worldId: string;
			colliderId: string;
	  }
	| {
			type: "setColliderSensor";
			worldId: string;
			colliderId: string;
			isSensor: boolean;
	  }
	| {
			type: "setCollisionMask";
			worldId: string;
			colliderId: string;
			mask: number;
	  }
	| {
			type: "setColliderMaterial";
			worldId: string;
			colliderId: string;
			material: Partial<
				Pick<PhysicsMaterialDescriptor, "friction" | "restitution">
			>;
	  }
	| {
			type: "createJoint";
			worldId: string;
			jointId: string;
			descriptor: JointDescriptor;
	  }
	| {
			type: "destroyJoint";
			worldId: string;
			jointId: string;
	  }
	| {
			type: "createCharacterController";
			worldId: string;
			controllerId: string;
			descriptor: CharacterControllerDescriptor;
	  }
	| {
			type: "moveCharacterController";
			worldId: string;
			controllerId: string;
			direction: IVector3;
			deltaSeconds: number;
	  }
	| {
			type: "destroyCharacterController";
			worldId: string;
			controllerId: string;
	  }
	| {
			type: "jumpCharacterController";
			worldId: string;
			controllerId: string;
			speed: number;
	  }
	| {
			type: "setCharacterControllerMaxSlope";
			worldId: string;
			controllerId: string;
			value: number;
	  }
	| {
			type: "setCharacterControllerStepHeight";
			worldId: string;
			controllerId: string;
			value: number;
	  };

export type RapierWorkerRequest =
	| {
			type: "stepWorld";
			worldId: string;
			deltaSeconds: number;
	  }
	| {
			type: "moveCharacterController";
			worldId: string;
			controllerId: string;
			direction: IVector3;
			deltaSeconds: number;
	  }
	| {
			type: "isCharacterControllerGrounded";
			worldId: string;
			controllerId: string;
	  }
	| {
			type: "raycast";
			worldId: string;
			query: PhysicsRaycastQuery;
	  }
	| {
			type: "raycastAll";
			worldId: string;
			query: PhysicsRaycastQuery;
	  }
	| {
			type: "sphereCast";
			worldId: string;
			query: PhysicsSphereCastQuery;
	  }
	| {
			type: "boxCast";
			worldId: string;
			query: PhysicsBoxCastQuery;
	  }
	| {
			type: "overlapSphere";
			worldId: string;
			query: PhysicsOverlapSphereQuery;
	  }
	| {
			type: "overlapBox";
			worldId: string;
			query: PhysicsOverlapBoxQuery;
	  };

export type RapierWorkerTaskPayload =
	| {
			type: "init";
			strict: boolean;
	  }
	| {
			type: "dispatch";
			commands: RapierWorkerCommand[];
			request?: RapierWorkerRequest;
	  };
