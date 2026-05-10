import {
	type FramePass,
	type PreparedScene,
	type ResolvedFeatureState,
} from "./types";
import {
	isFogPostProcessEnabled,
	type ResolvedPostProcessState,
} from "./PostProcess";
import { hasParticleShadowCasters } from "./ParticleShadowVolume";

const FRAME_PASS_ORDER: FramePass["stage"][] = [
	"animation-sim",
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"gamma",
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
		case "ssao":
			return postProcess.enabled.ssao;
		case "ssgi":
			return postProcess.enabled.ssgi;
		case "taa":
			return postProcess.enabled.taa;
		case "ssr":
			return postProcess.enabled.ssr;
		case "volumetric":
			return postProcess.enabled.volumetric;
		case "fog":
			return isFogPostProcessEnabled(postProcess);
		case "motion-blur":
			return postProcess.enabled["motion-blur"];
		case "dof":
			return postProcess.enabled.dof;
		case "bloom":
			return postProcess.enabled.bloom;
		case "tonemap":
			return postProcess.enabled.tonemap;
		case "color-filter":
			return postProcess.enabled["color-filter"];
		case "fxaa":
			return postProcess.enabled.fxaa;
		case "gamma":
			return postProcess.enabled.gamma;
		default:
			return false;
	}
}
