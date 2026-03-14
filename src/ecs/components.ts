import type { Matrix4 } from '../maths/Matrix4'

export type EntityId = number

export interface NameComponent {
	value: string
}

export interface VisibilityComponent {
	visible: boolean
}

export interface LocalTransformComponent {
	positionX: number
	positionY: number
	positionZ: number
	rotationX: number
	rotationY: number
	rotationZ: number
	rotationW: number
	scaleX: number
	scaleY: number
	scaleZ: number
}

export interface WorldTransformComponent {
	matrix: Matrix4
}

export interface HierarchyComponent {
	parent: EntityId | null
	children: EntityId[]
}

export interface PathBindingComponent {
	path: string
}

export interface SkeletonJointComponent {
	skeletonId: string
	jointIndex: number
}

export interface NodeRefComponent {
	node: object
}

export interface NodeKindComponent {
	kind: string
}

export type ECSComponentMap = {
	Name: NameComponent
	Visibility: VisibilityComponent
	LocalTransform: LocalTransformComponent
	WorldTransform: WorldTransformComponent
	Hierarchy: HierarchyComponent
	PathBinding: PathBindingComponent
	SkeletonJoint: SkeletonJointComponent
	NodeRef: NodeRefComponent
	NodeKind: NodeKindComponent
}

export type ECSComponentName = keyof ECSComponentMap
