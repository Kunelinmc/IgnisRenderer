import { type CompositeShaderSource, createInlineCompositeShaderSource } from "../runtime";
import {
	loadWebGLShaderPart,
	loadWebGLShaderPartComposite,
} from "./shaderSource";

const SCENE_VERTEX_SHADER_SOURCE = await loadWebGLShaderPart("sceneVertex");
const SCENE_FRAGMENT_SHADER_TEMPLATE = await loadWebGLShaderPart(
	"sceneFragment"
);
const SCENE_VERTEX_SHADER_COMPOSITE = await loadWebGLShaderPartComposite(
	"sceneVertex"
);
const SCENE_FRAGMENT_SHADER_COMPOSITE_TEMPLATE = await loadWebGLShaderPartComposite(
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

export interface WebGLSceneCompositeShaderSource {
	vertex: CompositeShaderSource;
	fragment: CompositeShaderSource;
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

export function createWebGLSceneCompositeShaderSource(
	limits: WebGLSceneLightLimits
): WebGLSceneCompositeShaderSource {
	const shader = createWebGLSceneShaderSource(limits);
	return {
		vertex: SCENE_VERTEX_SHADER_COMPOSITE,
		fragment: createInlineCompositeShaderSource(
			shader.fragment,
			"./parts/sceneFragment.glsl",
			"template"
		),
	};
}
