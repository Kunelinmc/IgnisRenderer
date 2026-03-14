import { loadWebGLShaderPart } from "./shaderSource";

const SCENE_VERTEX_SHADER_SOURCE = await loadWebGLShaderPart("sceneVertex");
const SCENE_FRAGMENT_SHADER_TEMPLATE = await loadWebGLShaderPart(
	"sceneFragment"
);

export interface WebGLSceneLightLimits {
	maxDirectionalLights: number;
	maxPointLights: number;
	maxSpotLights: number;
}

export interface WebGLSceneShaderSource {
	vertex: string;
	fragment: string;
}

function replaceLightLimit(
	source: string,
	placeholder: string,
	value: number
): string {
	return source.replaceAll(placeholder, String(Math.max(0, value | 0)));
}

export function createWebGLSceneShaderSource(
	limits: WebGLSceneLightLimits
): WebGLSceneShaderSource {
	const withDirectional = replaceLightLimit(
		SCENE_FRAGMENT_SHADER_TEMPLATE,
		"__MAX_DIRECTIONAL_LIGHTS__",
		limits.maxDirectionalLights
	);
	const withPoint = replaceLightLimit(
		withDirectional,
		"__MAX_POINT_LIGHTS__",
		limits.maxPointLights
	);
	const fragment = replaceLightLimit(
		withPoint,
		"__MAX_SPOT_LIGHTS__",
		limits.maxSpotLights
	);
	return {
		vertex: SCENE_VERTEX_SHADER_SOURCE,
		fragment,
	};
}
