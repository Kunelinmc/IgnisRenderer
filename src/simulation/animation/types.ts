import type { BoundingSphere, IPrimitiveGeometry } from "../../core/types";
import type { Skeleton } from "../../animation/Skeleton";
import { defineTransientKey } from "../../foundation/TransientStore";

export const ANIMATION_SIM_DELTA_TIME_MS_KEY =
	defineTransientKey<number>("pipeline:animation-delta-time-ms");
export const ANIMATION_RUNTIME_POSE_KEY =
	defineTransientKey<AnimationPoseState[]>("pipeline:animation-pose");
export const ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY =
	defineTransientKey<DeformedGeometryMap>(
		"pipeline:animation-software-deformed-geometry"
	);
export const ANIMATION_WEBGPU_JOINT_MATRICES_KEY =
	defineTransientKey<JointMatrixMap>("pipeline:animation-webgpu-joint-matrices");
export const ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY =
	defineTransientKey<MorphWeightMap>("pipeline:animation-webgpu-morph-weights");
export const ANIMATION_DEFORMATION_STATES_KEY =
	defineTransientKey<PrimitiveDeformationMap>(
		"pipeline:animation-deformation-states"
	);

export interface AnimationPoseState {
	path: string;
	translation?: [number, number, number];
	rotation?: [number, number, number, number];
	scale?: [number, number, number];
}

export interface DeformedGeometryOverride {
	positions?: Float32Array;
	normals?: Float32Array;
	tangents?: Float32Array;
}

/**
 * Current-frame deformation metadata consumed by prepared-scene construction.
 *
 * @internal Owned by the animation and pipeline subsystems. Backends should use
 * their existing joint, morph, or software-geometry payloads instead.
 */
export interface PrimitiveDeformationState {
	readonly packetId: string;
	readonly revision: number;
	readonly localBounds: BoundingSphere;
}

export interface AnimationWebGPUJointState {
	skeleton: Skeleton;
	matrices: Float32Array;
}

export interface AnimationWebGPUMorphState {
	packetId: string;
	weights: Float32Array;
	targetCount: number;
}

export type DeformedGeometryMap = Map<string, DeformedGeometryOverride>;

export type JointMatrixMap = Map<string, AnimationWebGPUJointState>;

export type MorphWeightMap = Map<string, AnimationWebGPUMorphState>;

export type PrimitiveDeformationMap = Map<string, PrimitiveDeformationState>;
