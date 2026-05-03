import {
	isFogPostProcessEnabled,
	type FramePass,
	type PreparedScene,
	type ResolvedFeatureState,
} from "./types";
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
		executors?: Partial<Record<FramePass["stage"], FramePass["executor"]>>
	): FramePass[] {
		return FRAME_PASS_ORDER.map((stage) => ({
			stage,
			executor: executors?.[stage] ?? "backend",
			enabled: shouldEnablePass(stage, frame, features),
		}));
	}
}

function shouldEnablePass(
	stage: FramePass["stage"],
	frame: PreparedScene,
	features: ResolvedFeatureState
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
			return features.enableSSAO;
		case "ssgi":
			return features.enableSSGI;
		case "taa":
			return features.enableTAA;
		case "ssr":
			return features.enableSSR;
		case "volumetric":
			return features.enableVolumetric;
		case "fog":
			return isFogPostProcessEnabled(features);
		case "motion-blur":
			return features.enableMotionBlur;
		case "dof":
			return features.enableDOF;
		case "bloom":
			return features.enableBloom;
		case "tonemap":
			return features.enableToneMapping !== false;
		case "color-filter":
			return features.enableColorFilter;
		case "fxaa":
			return features.enableFXAA;
		case "gamma":
			return features.enableGamma;
		default:
			return false;
	}
}
