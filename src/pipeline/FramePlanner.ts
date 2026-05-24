import {
	type FramePass,
	type PreparedScene,
	type ResolvedFeatureState,
} from "./types";
import { type ResolvedPostProcessState } from "./PostProcessController";
import { hasParticleShadowCasters } from "./ParticleShadowVolume";
import { hasPostProcessExecutionPasses } from "../postprocess/PostProcessPipeline";

const FRAME_PASS_ORDER: FramePass["stage"][] = [
	"animation-sim",
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"postprocess",
];

export class FramePlanner {
	public static build(
		frame: PreparedScene,
		features: ResolvedFeatureState,
		postProcess: ResolvedPostProcessState,
		executors?: Partial<Record<FramePass["stage"], FramePass["executor"]>>
	): FramePass[] {
		return FRAME_PASS_ORDER.map((stage) => ({
			stage,
			executor: executors?.[stage] ?? "backend",
			enabled: shouldEnablePass(stage, frame, features, postProcess),
		}));
	}
}

function shouldEnablePass(
	stage: FramePass["stage"],
	frame: PreparedScene,
	features: ResolvedFeatureState,
	postProcess: ResolvedPostProcessState
): boolean {
	switch (stage) {
		case "animation-sim":
			return frame.hasActiveAnimations;
		case "particle-sim":
			return (frame.particleSystems?.length ?? 0) > 0;
		case "shadow":
			return (
				features.enableShadows &&
				(frame.shadowCasterPackets.length > 0 ||
					frame.shadowTransmitterPackets.length > 0 ||
					hasParticleShadowCasters(frame.particleSystems))
			);
		case "reflection":
			return features.enableReflection && frame.reflectivePackets.length > 0;
		case "main-opaque":
			return true;
		case "main-transparent":
			return frame.transparentPackets.length > 0;
		case "particles":
			return (frame.particleSystems?.length ?? 0) > 0;
		case "postprocess":
			return hasPostProcessExecutionPasses(postProcess);
		default:
			return false;
	}
}
