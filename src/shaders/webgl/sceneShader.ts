import {
	createWebGLShaderSourceFactory,
	type WebGLSceneCompositeShaderSource,
	type WebGLSceneLightLimits,
	type WebGLSceneShaderSource,
	type WebGLShaderSourceFactory,
} from "./WebGLShaderSourceFactory";

let _defaultFactory: WebGLShaderSourceFactory | null = null;
let _defaultPrepared = false;

function getDefaultFactory(): WebGLShaderSourceFactory {
	if (!_defaultFactory) {
		_defaultFactory = createWebGLShaderSourceFactory();
	}
	return _defaultFactory;
}

export async function prepareDefaultWebGLSceneShaderSourceFactory():
	Promise<void> {
	if (_defaultPrepared) {
		return;
	}
	await getDefaultFactory().prepareSceneParts();
	_defaultPrepared = true;
}

export function createWebGLSceneShaderSource(
	limits: WebGLSceneLightLimits
): WebGLSceneShaderSource {
	if (!_defaultPrepared) {
		throw new Error(
			"WebGL scene shader source factory is not prepared. " +
				"Migration hint: await prepareDefaultWebGLSceneShaderSourceFactory() before calling createWebGLSceneShaderSource()."
		);
	}
	return getDefaultFactory().createSceneShaderSource(limits);
}

export function createWebGLSceneCompositeShaderSource(
	limits: WebGLSceneLightLimits
): WebGLSceneCompositeShaderSource {
	if (!_defaultPrepared) {
		throw new Error(
			"WebGL scene shader source factory is not prepared. " +
				"Migration hint: await prepareDefaultWebGLSceneShaderSourceFactory() before calling createWebGLSceneCompositeShaderSource()."
		);
	}
	return getDefaultFactory().createSceneCompositeShaderSource(limits);
}

export type {
	WebGLSceneCompositeShaderSource,
	WebGLSceneLightLimits,
	WebGLSceneShaderSource,
	WebGLShaderSourceFactory,
} from "./WebGLShaderSourceFactory";
