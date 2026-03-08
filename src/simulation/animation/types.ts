import type { IPrimitiveGeometry } from "../../core/types";
import type { Skeleton } from "../../animation/Skeleton";

export const ANIMATION_SIM_DELTA_TIME_MS_KEY =
	"pipeline:animation-delta-time-ms";
export const ANIMATION_RUNTIME_POSE_KEY = "pipeline:animation-pose";
export const ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY =
	"pipeline:animation-software-deformed-geometry";
export const ANIMATION_WEBGPU_JOINT_MATRICES_KEY =
	"pipeline:animation-webgpu-joint-matrices";
export const ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY =
	"pipeline:animation-webgpu-morph-weights";

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

export interface AnimationWebGPUJointState {
	skeleton: Skeleton;
	matrices: Float32Array;
}

export interface AnimationWebGPUMorphState {
	primitiveId: string;
	weights: Float32Array;
	targetCount: number;
}

export type DeformedGeometryMap = Map<string, DeformedGeometryOverride>;

export type JointMatrixMap = Map<string, AnimationWebGPUJointState>;

export type MorphWeightMap = Map<string, AnimationWebGPUMorphState>;
