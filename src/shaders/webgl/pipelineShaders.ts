import {
	createWebGLShaderSourceFactory,
	type WebGLShaderSourceFactory,
	type WebGLShaderPart,
} from "./WebGLShaderSourceFactory";

const PIPELINE_PARTS: readonly WebGLShaderPart[] = [
	"environmentVertex",
	"environmentFragment",
	"presentVertex",
	"presentFragment",
	"particleVertex",
	"particleFragment",
	"shadowDepthVertex",
	"shadowDepthFragment",
	"copyFragment",
	"postProcessStubFragment",
	"toneMappingFragment",
	"colorFilterFragment",
	"fxaaFragment",
	"bloomFragment",
	"interactionOutlineFragment",
	"motionBlurFragment",
	"fogFragment",
	"dofFragment",
	"taaFragment",
	"ssaoRawFragment",
	"ssaoBlurFragment",
	"ssaoCombineFragment",
];

let _defaultFactory: WebGLShaderSourceFactory | null = null;
let _defaultPrepared = false;

function getDefaultFactory(): WebGLShaderSourceFactory {
	if (!_defaultFactory) {
		_defaultFactory = createWebGLShaderSourceFactory();
	}
	return _defaultFactory;
}

export async function prepareDefaultWebGLPipelineShaderSourceFactory():
	Promise<void> {
	if (_defaultPrepared) {
		return;
	}
	await getDefaultFactory().prepareParts(PIPELINE_PARTS);
	_defaultPrepared = true;
}

export function getPreparedWebGLPipelineShaderPart(part: WebGLShaderPart): string {
	if (!_defaultPrepared) {
		throw new Error(
			"WebGL pipeline shader source factory is not prepared. " +
				"Migration hint: await prepareDefaultWebGLPipelineShaderSourceFactory() before requesting pipeline shader parts."
		);
	}
	return getDefaultFactory().getRawPart(part);
}

export type {
	WebGLShaderPart,
	WebGLShaderSourceFactory,
} from "./WebGLShaderSourceFactory";
