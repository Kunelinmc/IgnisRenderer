import type { ShaderTargetMode } from "../../materials/ShaderMaterial";

import type { WebGLSceneSamplerLayout } from "./WebGLSceneSamplerLayout";
import type { WebGLSceneUniforms } from "./WebGLSceneProgramUniforms";

/** @internal Reflected WebGL scene program shared by scene binders and passes. */
export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: WebGLSceneUniforms;
	targetMode?: ShaderTargetMode;
	colorOutputCount?: 1 | 3 | 5;
	samplerLayout: WebGLSceneSamplerLayout;
}
