import { loadSceneShaderPart } from "./shaderSource";

const SCENE_PARTS = [
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
	"fragmentSingleTarget",
] as const;

let _sceneShaderPromise: Promise<string> | null = null;

export function getWebGPUSceneShader(): Promise<string> {
	if (!_sceneShaderPromise) {
		_sceneShaderPromise = Promise.all(
			SCENE_PARTS.map((part) => loadSceneShaderPart(part))
		).then((parts) => parts.join("\n\n"));
	}
	return _sceneShaderPromise;
}
