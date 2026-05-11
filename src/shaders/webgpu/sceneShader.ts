import { composeCompositeShaderSources, type CompositeShaderSource } from "../runtime";
import { loadSceneShaderPart, loadSceneShaderPartComposite } from "./shaderSource";

const SCENE_PARTS = [
	"lightData",
	"constants",
	"definitions",
	"utils",
	"vertexStage",
	"fragmentPrelude",
	"fragmentPhong",
	"fragmentPbrSetup",
	"fragmentPbrDirectional",
	"fragmentPbrPoint",
	"fragmentPbrSpot",
	"fragmentPbrAmbient",
	"fragmentGBuffer",
	"fragmentSingleTarget",
] as const;

let _sceneShaderPromise: Promise<string> | null = null;
let _sceneShaderCompositePromise: Promise<CompositeShaderSource> | null = null;

export function getWebGPUSceneShader(): Promise<string> {
	if (!_sceneShaderPromise) {
		_sceneShaderPromise = Promise.all(
			SCENE_PARTS.map((part) => loadSceneShaderPart(part))
		).then((parts) => parts.join("\n\n"));
	}
	return _sceneShaderPromise;
}

export function getWebGPUSceneShaderComposite(): Promise<CompositeShaderSource> {
	if (!_sceneShaderCompositePromise) {
		_sceneShaderCompositePromise = Promise.all(
			SCENE_PARTS.map((part) => loadSceneShaderPartComposite(part))
		).then((parts) =>
			composeCompositeShaderSources(
				parts.map((part) => ({
					code: part.code,
					sourceMap: part.sourceMap,
					sourcePath: part.sourceMap.segments[0]?.sourcePath ?? "<scene-part>",
					kind: "template",
				})),
				"\n\n",
				"template"
			)
		);
	}
	return _sceneShaderCompositePromise;
}
