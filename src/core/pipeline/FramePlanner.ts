import type { FramePass, PreparedScene, ResolvedFeatureState } from "./types";

const FRAME_PASS_ORDER: FramePass["stage"][] = [
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"ssao",
	"volumetric",
	"fxaa",
	"gamma",
];

export class FramePlanner {
	public static build(
		frame: PreparedScene,
		features: ResolvedFeatureState
	): FramePass[] {
		return FRAME_PASS_ORDER.map((stage) => ({
			stage,
			executor: stage === "shadow" ? "shared" : "backend",
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
		case "shadow":
			return features.enableShadows && frame.shadowCasterPackets.length > 0;
		case "reflection":
			return features.enableReflection && frame.reflectivePackets.length > 0;
		case "main-opaque":
			return true;
		case "main-transparent":
			return frame.transparentPackets.length > 0;
		case "ssao":
			return features.enableSSAO;
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
