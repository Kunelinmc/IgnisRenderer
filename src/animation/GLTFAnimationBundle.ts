import type { AnimationClip } from "./AnimationClip";
import type { Skeleton } from "./Skeleton";

export interface MorphBindingEntry {
	path: string;
	instance: any;
	targetCount: number;
}

export interface GLTFAnimationBundle {
	clips: AnimationClip[];
	skeletons: Skeleton[];
	morphBindings: MorphBindingEntry[];
	nodePathMap: Record<string, string>;
}
