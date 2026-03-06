struct DirectionalLightData {
	direction: vec4<f32>,
	color: vec4<f32>,
}

struct PointLightData {
	positionRange: vec4<f32>,
	color: vec4<f32>,
}

struct SpotLightData {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
}

struct ShadowData {
	viewProjection: mat4x4<f32>,
	paramsA: vec4<f32>,
	paramsB: vec4<f32>,
}

struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	skyboxBasisRight: vec4<f32>,
	skyboxBasisUp: vec4<f32>,
	skyboxBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, 4>,
	pointLights: array<PointLightData, 4>,
	spotLights: array<SpotLightData, 4>,
	directionalShadows: array<ShadowData, 4>,
	spotShadows: array<ShadowData, 4>,
	shAmbientCoeffs: array<vec4<f32>, 16>,
}

struct ModelUniforms {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
	baseColorFactor: vec4<f32>,
	emissiveFactor: vec4<f32>,
	surfaceParams0: vec4<f32>,
	surfaceParams1: vec4<f32>,
	surfaceParams2: vec4<f32>,
	surfaceParams3: vec4<f32>,
	specularColorFactor: vec4<f32>,
	phongAmbientShininess: vec4<f32>,
	phongSpecularShading: vec4<f32>,
	sheenColorClearcoatNormalScale: vec4<f32>,
	attenuationColor: vec4<f32>,
	materialFlags: vec4<f32>,
	textureTransformA: array<vec4<f32>, 14>,
	textureTransformB: array<vec4<f32>, 14>,
}

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec4<f32>,
	@location(4) uv2: vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
	@location(4) uv2: vec2<f32>,
	@location(5) currentClip: vec4<f32>,
	@location(6) prevClip: vec4<f32>,
}

struct SceneFragmentOutput {
	@location(0) sceneColor: vec4<f32>,
	@location(1) gAlbedoAlpha: vec4<f32>,
	@location(2) gNormalRoughMetal: vec4<f32>,
	@location(3) gEmissiveOcclusion: vec4<f32>,
	@location(4) gMotionDepth: vec4<f32>,
}

struct RefractionResult {
	direction: vec3<f32>,
	valid: f32,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowAtlas: texture_depth_2d;
@group(0) @binding(2) var envSpecularTexture: texture_2d<f32>;
@group(0) @binding(3) var envSpecularSampler: sampler;

@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(1) @binding(1) var baseColorTexture: texture_2d<f32>;
@group(1) @binding(2) var baseColorSampler: sampler;
@group(1) @binding(3) var metallicRoughnessTexture: texture_2d<f32>;
@group(1) @binding(4) var metallicRoughnessSampler: sampler;
@group(1) @binding(5) var normalTexture: texture_2d<f32>;
@group(1) @binding(6) var normalSampler: sampler;
@group(1) @binding(7) var emissiveTexture: texture_2d<f32>;
@group(1) @binding(8) var emissiveSampler: sampler;
@group(1) @binding(9) var occlusionTexture: texture_2d<f32>;
@group(1) @binding(10) var occlusionSampler: sampler;
@group(1) @binding(11) var specularTexture: texture_2d<f32>;
@group(1) @binding(12) var specularSampler: sampler;
@group(1) @binding(13) var specularColorTexture: texture_2d<f32>;
@group(1) @binding(14) var specularColorSampler: sampler;
@group(1) @binding(15) var clearcoatTexture: texture_2d<f32>;
@group(1) @binding(16) var clearcoatSampler: sampler;
@group(1) @binding(17) var clearcoatRoughnessTexture: texture_2d<f32>;
@group(1) @binding(18) var clearcoatRoughnessSampler: sampler;
@group(1) @binding(19) var clearcoatNormalTexture: texture_2d<f32>;
@group(1) @binding(20) var clearcoatNormalSampler: sampler;
@group(1) @binding(21) var sheenColorTexture: texture_2d<f32>;
@group(1) @binding(22) var sheenColorSampler: sampler;
@group(1) @binding(23) var sheenRoughnessTexture: texture_2d<f32>;
@group(1) @binding(24) var sheenRoughnessSampler: sampler;
@group(1) @binding(25) var transmissionTexture: texture_2d<f32>;
@group(1) @binding(26) var transmissionSampler: sampler;
@group(1) @binding(27) var thicknessTexture: texture_2d<f32>;
@group(1) @binding(28) var thicknessSampler: sampler;
