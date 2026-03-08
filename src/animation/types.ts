export type AnimationTrackTargetType = "node" | "material" | "morph";

export type AnimationInterpolation = "step" | "linear" | "cubic";

export type NodeTrackProperty = "translation" | "rotation" | "scale";
export type MaterialTrackProperty =
	| "opacity"
	| "baseColor"
	| "emissiveIntensity"
	| "emissive";

export interface AnimationTrackBinding {
	targetType: AnimationTrackTargetType;
	targetPath: string;
	property: NodeTrackProperty | MaterialTrackProperty | "weights";
	morphTargetIndex?: number;
}

export interface AnimationSampleValue {
	binding: AnimationTrackBinding;
	value: number[];
	weight: number;
	additive: boolean;
}

export interface AnimationRootMotionOptions {
	enabled?: boolean;
	trackPath?: string;
}

export interface AnimationLayerMask {
	include?: string[];
	exclude?: string[];
}

export interface AnimationTransitionCondition {
	parameter: string;
	operator: ">" | ">=" | "<" | "<=" | "==" | "!=" | "trigger";
	value?: number | boolean;
}

export interface AnimationTransition {
	from: string;
	to: string;
	duration: number;
	hasExitTime?: boolean;
	exitTime?: number;
	conditions?: AnimationTransitionCondition[];
}

export interface BlendTreeChildWeight {
	clipName: string;
	weight: number;
}

export type AnimationParameterType = "float" | "bool" | "trigger";

export interface AnimationParameterDefinition {
	name: string;
	type: AnimationParameterType;
	defaultValue?: number | boolean;
}

export type AnimationMotionDefinition =
	| { type: "clip"; clipName: string }
	| { type: "blendtree1d"; treeName: string }
	| { type: "blendtree-direct"; treeName: string };

export interface AnimationStateDefinition {
	name: string;
	motion: AnimationMotionDefinition;
	speed?: number;
	loop?: boolean;
}
