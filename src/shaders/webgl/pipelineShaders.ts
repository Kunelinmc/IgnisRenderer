import { loadWebGLShaderPart } from "./shaderSource";

export const WEBGL_SKYBOX_VERTEX_SHADER = await loadWebGLShaderPart(
	"skyboxVertex"
);
export const WEBGL_SKYBOX_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"skyboxFragment"
);
export const WEBGL_PRESENT_VERTEX_SHADER = await loadWebGLShaderPart(
	"presentVertex"
);
export const WEBGL_PRESENT_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"presentFragment"
);
export const WEBGL_PARTICLE_VERTEX_SHADER = await loadWebGLShaderPart(
	"particleVertex"
);
export const WEBGL_PARTICLE_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"particleFragment"
);
export const WEBGL_FXAA_VERTEX_SHADER = WEBGL_PRESENT_VERTEX_SHADER;
export const WEBGL_SHADOW_DEPTH_VERTEX_SHADER = await loadWebGLShaderPart(
	"shadowDepthVertex"
);
export const WEBGL_SHADOW_DEPTH_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"shadowDepthFragment"
);
export const WEBGL_COPY_VERTEX_SHADER = WEBGL_PRESENT_VERTEX_SHADER;
export const WEBGL_COPY_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"copyFragment"
);
export const WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"postProcessStubFragment"
);
export const WEBGL_FXAA_FRAGMENT_SHADER = await loadWebGLShaderPart(
	"fxaaFragment"
);
