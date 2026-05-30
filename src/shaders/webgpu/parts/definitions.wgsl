#import <ignis/webgpu/constants>
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	pointLights: array<PointLightData, __WEBGPU_MAX_POINT_LIGHTS__>,
	spotLights: array<SpotLightData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
	shAmbientCoeffs: array<vec4<f32>, __WEBGPU_SH_COEFFICIENT_COUNT__>,
	reflectionProbes: array<ReflectionProbeData, __WEBGPU_MAX_REFLECTION_PROBES__>,
	localLightProbeCounts: vec4<f32>,
	localLightProbeWorldToProbeRow0: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeWorldToProbeRow1: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeWorldToProbeRow2: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeDataA: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeDataB: array<vec4<f32>, __WEBGPU_MAX_LOCAL_LIGHT_PROBES__>,
	localLightProbeSHAmbientCoeffs:
		array<vec4<f32>, __WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT__>,
	areaLightCounts: vec4<f32>,
	areaLights: array<AreaLightData, __WEBGPU_MAX_AREA_LIGHTS__>,
}

struct ClusterGridParams {
	screenWidth: u32,
	screenHeight: u32,
	tilesX: u32,
	tilesY: u32,
	zSlices: u32,
	clusterCount: u32,
	near: f32,
	far: f32,
	logScale: f32,
	logBias: f32,
	lightCount: u32,
	reserved1: u32,
}

struct ClusterLightRecord {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
	packedFlags: u32,
	shadowIndex: u32,
	reserved0: u32,
	reserved1: u32,
}

struct ClusterHeader {
	offset: u32,
	count: u32,
	flags: u32,
	reserved: u32,
}

struct ClusterLightBuffer {
	lights: array<ClusterLightRecord>,
}

struct ClusterHeaderBuffer {
	headers: array<ClusterHeader>,
}

struct ClusterLightIndexList {
	indices: array<u32>,
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
	anisotropyParams: vec4<f32>,
	anisotropyTextureTransformA: vec4<f32>,
	anisotropyTextureTransformB: vec4<f32>,
	materialFlags: vec4<f32>,
	textureTransformA: array<vec4<f32>, __WEBGPU_TEXTURE_SLOT_COUNT__>,
	textureTransformB: array<vec4<f32>, __WEBGPU_TEXTURE_SLOT_COUNT__>,
}

struct AnimationParams {
	jointCount: f32,
	morphTargetCount: f32,
	prevJointOffset: f32,
	prevMorphOffset: f32,
}

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv0: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec4<f32>,
	@location(4) uv1: vec2<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
	@location(9) uv2: vec2<f32>,
	@location(10) uv3: vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv0: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
	@location(4) uv1: vec2<f32>,
	@location(5) currentClip: vec4<f32>,
	@location(6) prevClip: vec4<f32>,
	@location(7) uv2: vec2<f32>,
	@location(8) uv3: vec2<f32>,
}

struct SceneFragmentOutput {
	@location(0) sceneColor: vec4<f32>,
	@location(1) gAlbedoAlpha: vec4<f32>,
	@location(2) gNormalRoughMetal: vec4<f32>,
	@location(3) gEmissiveOcclusion: vec4<f32>,
	@location(4) gMotionDepth: vec4<f32>,
}

struct GBufferFragmentOutput {
	@location(0) gAlbedoAlpha: vec4<f32>,
	@location(1) gNormalRoughMetal: vec4<f32>,
	@location(2) gEmissiveOcclusion: vec4<f32>,
	@location(3) gMotionDepth: vec4<f32>,
	@location(4) gSpecular: vec4<f32>,
	@location(5) gCoatSheen: vec4<f32>,
	@location(6) gSheenReflectance: vec4<f32>,
}

struct SceneFragmentOITOutput {
	@location(0) accum: vec4<f32>,
	@location(1) reveal: vec4<f32>,
}

struct RefractionResult {
	direction: vec3<f32>,
	valid: f32,
}

struct FogUniforms {
	fogParams0: vec4<f32>,
	fogParams1: vec4<f32>,
}

struct ParticleShadowVolumeBuffer {
	data: array<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var shadowAtlas: texture_depth_2d;
@group(0) @binding(2) var envSpecularTexture: texture_2d<f32>;
@group(0) @binding(3) var envSpecularSampler: sampler;
@group(0) @binding(4) var envSpecularFallbackTexture: texture_2d<f32>;
@group(0) @binding(5) var envSpecularFallbackSampler: sampler;
@group(0) @binding(6) var<uniform> fog: FogUniforms;
@group(0) @binding(7) var<storage, read> particleShadowVolumes:
	ParticleShadowVolumeBuffer;
@group(0) @binding(8) var shadowTransmittanceAtlas: texture_2d<f32>;
@group(0) @binding(9) var brdfLUTTexture: texture_2d<f32>;

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
@group(1) @binding(29) var iridescenceTexture: texture_2d<f32>;
@group(1) @binding(31) var iridescenceThicknessTexture: texture_2d<f32>;
@group(1) @binding(30) var<uniform> animationParams: AnimationParams;
@group(1) @binding(32) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(33) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(34) var<storage, read> morphPositionDeltas: array<vec4<f32>>;
@group(1) @binding(35) var<storage, read> morphNormalDeltas: array<vec4<f32>>;
@group(1) @binding(37) var anisotropyTexture: texture_2d<f32>;

@group(2) @binding(0) var<uniform> clusterGrid: ClusterGridParams;
@group(2) @binding(1) var<storage, read> clusterLights: ClusterLightBuffer;
@group(2) @binding(2) var<storage, read> clusterHeaders: ClusterHeaderBuffer;
@group(2) @binding(3) var<storage, read> clusterIndices: ClusterLightIndexList;
