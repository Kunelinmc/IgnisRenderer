import type { Matrix4 } from "../maths/Matrix4";
import type { Node } from "../core/Node";

export type EntityId = number;

export interface NameComponent {
	value: string;
}

export interface VisibilityComponent {
	visible: boolean;
}

export interface LocalTransformComponent {
	positionX: number;
	positionY: number;
	positionZ: number;
	rotationX: number;
	rotationY: number;
	rotationZ: number;
	rotationW: number;
	scaleX: number;
	scaleY: number;
	scaleZ: number;
}

export interface WorldTransformComponent {
	matrix: Matrix4;
}

export interface HierarchyComponent {
	parent: EntityId | null;
	children: EntityId[];
}

export interface PathBindingComponent {
	path: string;
}

export interface SkeletonJointComponent {
	skeletonId: string;
	jointIndex: number;
}

export interface NodeRefComponent {
	node: Node;
}

export const NODE_KIND = {
	Node: "node",
	MeshInstance: "meshInstance",
	Decal: "decal",
	Camera: "camera",
	ParticleSystem: "particleSystem",
	Light: "light",
} as const;

export type NodeKind = (typeof NODE_KIND)[keyof typeof NODE_KIND];

export interface NodeKindComponent {
	kind: NodeKind;
}

export type ECSComponentMap = {
	Name: NameComponent;
	Visibility: VisibilityComponent;
	LocalTransform: LocalTransformComponent;
	WorldTransform: WorldTransformComponent;
	Hierarchy: HierarchyComponent;
	PathBinding: PathBindingComponent;
	SkeletonJoint: SkeletonJointComponent;
	NodeRef: NodeRefComponent;
	NodeKind: NodeKindComponent;
};

export type ECSComponentName = keyof ECSComponentMap;
