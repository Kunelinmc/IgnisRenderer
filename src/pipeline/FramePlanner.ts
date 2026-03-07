import type { FramePass, PreparedScene, ResolvedFeatureState } from "./types";

const FRAME_PASS_ORDER: FramePass["stage"][] = [
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"taa",
	"ssr",
	"volumetric",
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
		case "particle-sim":
			return (frame.particleSystems?.length ?? 0) > 0;
		case "shadow":
			return features.enableShadows && frame.shadowCasterPackets.length > 0;
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
		case "taa":
			return features.enableTAA;
		case "ssr":
			return features.enableSSR;
		case "volumetric":
			return features.enableVolumetric;
		case "fxaa":
			return features.enableFXAA;
		case "gamma":
			return features.enableGamma;
		default:
			return false;
	}
}
