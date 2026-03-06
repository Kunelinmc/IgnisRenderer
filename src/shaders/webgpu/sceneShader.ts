import WEBGPU_SCENE_SHADER_CONSTANTS from "./parts/constants.wgsl?raw";
import WEBGPU_SCENE_SHADER_DEFINITIONS from "./parts/definitions.wgsl?raw";
import WEBGPU_SCENE_SHADER_UTILS from "./parts/utils.wgsl?raw";
import WEBGPU_SCENE_VERTEX_STAGE from "./parts/vertexStage.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PRELUDE from "./parts/fragmentPrelude.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PHONG from "./parts/fragmentPhong.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PBR_SETUP from "./parts/fragmentPbrSetup.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PBR_DIRECTIONAL from "./parts/fragmentPbrDirectional.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PBR_POINT from "./parts/fragmentPbrPoint.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PBR_SPOT from "./parts/fragmentPbrSpot.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_PBR_AMBIENT from "./parts/fragmentPbrAmbient.wgsl?raw";
import WEBGPU_SCENE_FRAGMENT_SINGLE_TARGET from "./parts/fragmentSingleTarget.wgsl?raw";

export const WEBGPU_SCENE_SHADER = [
	WEBGPU_SCENE_SHADER_CONSTANTS,
	WEBGPU_SCENE_SHADER_DEFINITIONS,
	WEBGPU_SCENE_SHADER_UTILS,
	WEBGPU_SCENE_VERTEX_STAGE,
	WEBGPU_SCENE_FRAGMENT_PRELUDE,
	WEBGPU_SCENE_FRAGMENT_PHONG,
	WEBGPU_SCENE_FRAGMENT_PBR_SETUP,
	WEBGPU_SCENE_FRAGMENT_PBR_DIRECTIONAL,
	WEBGPU_SCENE_FRAGMENT_PBR_POINT,
	WEBGPU_SCENE_FRAGMENT_PBR_SPOT,
	WEBGPU_SCENE_FRAGMENT_PBR_AMBIENT,
	WEBGPU_SCENE_FRAGMENT_SINGLE_TARGET,
].join("\n\n");
