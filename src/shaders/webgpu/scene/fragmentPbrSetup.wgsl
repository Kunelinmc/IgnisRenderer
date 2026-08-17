var mrSample = vec4<f32>(1.0);
if (hasPBRTexture(PBR_TEXTURE_METALLIC_ROUGHNESS_MAP)) {
	mrSample = sampleLinearTexture(
		metallicRoughnessTexture,
		metallicRoughnessSampler,
		TEX_METALLIC_ROUGHNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var occlusionSample = vec4<f32>(1.0);
if (hasPBRTexture(PBR_TEXTURE_OCCLUSION_MAP)) {
	occlusionSample = sampleLinearTexture(
		occlusionTexture,
		occlusionSampler,
		TEX_OCCLUSION,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var specularSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_SPECULAR) &&
	model.specularColorFactor.a > EPSILON &&
	max(max(model.specularColorFactor.r, model.specularColorFactor.g),
		model.specularColorFactor.b) > EPSILON &&
	hasPBRTexture(PBR_TEXTURE_SPECULAR_MAP)
) {
	specularSample = sampleLinearTexture(
		specularTexture,
		specularSampler,
		TEX_SPECULAR,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var specularColorSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_SPECULAR) &&
	model.specularColorFactor.a > EPSILON &&
	max(max(model.specularColorFactor.r, model.specularColorFactor.g),
		model.specularColorFactor.b) > EPSILON &&
	hasPBRTexture(PBR_TEXTURE_SPECULAR_COLOR_MAP)
) {
	specularColorSample = sampleColorTexture(
		specularColorTexture,
		specularColorSampler,
		TEX_SPECULAR_COLOR,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var clearcoatSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
	hasPBRTexture(PBR_TEXTURE_CLEARCOAT_MAP)
) {
	clearcoatSample = sampleLinearTexture(
		clearcoatTexture,
		clearcoatSampler,
		TEX_CLEARCOAT,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var clearcoatRoughnessSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
	hasPBRTexture(PBR_TEXTURE_CLEARCOAT_ROUGHNESS_MAP)
) {
	clearcoatRoughnessSample = sampleLinearTexture(
		clearcoatRoughnessTexture,
		clearcoatRoughnessSampler,
		TEX_CLEARCOAT_ROUGHNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var sheenColorSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_SHEEN) &&
	hasPBRTexture(PBR_TEXTURE_SHEEN_COLOR_MAP)
) {
	sheenColorSample = sampleColorTexture(
		sheenColorTexture,
		sheenColorSampler,
		TEX_SHEEN_COLOR,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var sheenRoughnessSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_SHEEN) &&
	hasPBRTexture(PBR_TEXTURE_SHEEN_ROUGHNESS_MAP)
) {
	sheenRoughnessSample = sampleLinearTexture(
		sheenRoughnessTexture,
		sheenRoughnessSampler,
		TEX_SHEEN_ROUGHNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var transmissionSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_TRANSMISSION) &&
	hasPBRTexture(PBR_TEXTURE_TRANSMISSION_MAP)
) {
	transmissionSample = sampleLinearTexture(
		transmissionTexture,
		transmissionSampler,
		TEX_TRANSMISSION,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var thicknessSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_TRANSMISSION) &&
	hasPBRTexture(PBR_TEXTURE_THICKNESS_MAP)
) {
	thicknessSample = sampleLinearTexture(
		thicknessTexture,
		transmissionSampler,
		TEX_THICKNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var iridescenceSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_IRIDESCENCE) &&
	hasPBRTexture(PBR_TEXTURE_IRIDESCENCE_MAP)
) {
	iridescenceSample = sampleLinearTexture(
		iridescenceTexture,
		transmissionSampler,
		TEX_IRIDESCENCE,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}
var iridescenceThicknessSample = vec4<f32>(1.0);
if (
	hasPBRFeature(PBR_FEATURE_IRIDESCENCE) &&
	hasPBRTexture(PBR_TEXTURE_IRIDESCENCE_THICKNESS_MAP)
) {
	iridescenceThicknessSample = sampleLinearTexture(
		iridescenceThicknessTexture,
		transmissionSampler,
		TEX_IRIDESCENCE_THICKNESS,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	);
}

var normalSample = vec3<f32>(0.5, 0.5, 1.0);
if (hasPBRTexture(PBR_TEXTURE_NORMAL_MAP)) {
	normalSample = sampleLinearTexture(
		normalTexture,
		normalSampler,
		TEX_NORMAL,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	).rgb;
}
var clearcoatNormalSample = vec3<f32>(0.5, 0.5, 1.0);
if (
	hasPBRFeature(PBR_FEATURE_CLEARCOAT) &&
	hasPBRTexture(PBR_TEXTURE_CLEARCOAT_NORMAL_MAP)
) {
	clearcoatNormalSample = sampleLinearTexture(
		clearcoatNormalTexture,
		clearcoatNormalSampler,
		TEX_CLEARCOAT_NORMAL,
		input.uv0,
		input.uv1,
		input.uv2,
		input.uv3
	).rgb;
}

let roughness = clamp(model.surfaceParams0.x * mrSample.g, 0.04, 1.0);
let metalness = clamp(model.surfaceParams0.y * mrSample.b, 0.0, 1.0);
let reflectance = clamp(model.surfaceParams0.z, 0.0, 1.0);
let occlusion = clamp(
	1.0 + model.surfaceParams1.x * (occlusionSample.r - 1.0),
	0.0,
	1.0
);
let clearcoat = clamp(model.surfaceParams1.z * clearcoatSample.r, 0.0, 1.0);
let clearcoatRoughness = clamp(
	model.surfaceParams1.w * clearcoatRoughnessSample.g,
	0.04,
	1.0
);
let sheenColor = model.sheenColorClearcoatNormalScale.rgb * sheenColorSample.rgb;
let sheenRoughness = clamp(model.surfaceParams2.x * sheenRoughnessSample.a, 0.0, 1.0);
let transmission = clamp(model.surfaceParams2.y * transmissionSample.r, 0.0, 1.0);
let ior = max(model.surfaceParams2.z, 1.0);
let thickness = max(model.surfaceParams2.w * thicknessSample.g, 0.0);
let attenuationDistance = model.surfaceParams3.x;
let iridescence = clamp(model.surfaceParams3.y * iridescenceSample.r, 0.0, 1.0);
let iridescenceIor = max(model.surfaceParams3.z, 1.0);
let iridescenceThickness = max(
	mix(
		model.surfaceParams3.w,
		model.attenuationColor.a,
		iridescenceThicknessSample.g
	),
	0.0
);
let attenuationColor = clamp(model.attenuationColor.rgb, vec3<f32>(0.0001), vec3<f32>(1.0));

let specularFactor = clamp(model.specularColorFactor.a * specularSample.a, 0.0, 1.0);
let specularColor = clamp(
	model.specularColorFactor.rgb * specularColorSample.rgb,
	vec3<f32>(0.0),
	vec3<f32>(1.0)
);

var pbrNormal = applyNormalMap(
	normal,
	input.worldTangent,
	normalSample,
	model.surfaceParams1.y
);
let pbrShadowNormal = pbrNormal;
if (doubleSided && dot(pbrNormal, viewDir) < 0.0) {
	pbrNormal = -pbrNormal;
}

var clearcoatNormal = applyNormalMap(
	pbrNormal,
	input.worldTangent,
	clearcoatNormalSample,
	model.sheenColorClearcoatNormalScale.a
);
if (doubleSided && dot(clearcoatNormal, viewDir) < 0.0) {
	clearcoatNormal = -clearcoatNormal;
}

let albedo = clamp(baseColor, vec3<f32>(0.0), vec3<f32>(1.0));
let baseF0 = 0.16 * reflectance * reflectance;
let f0Norm = min(vec3<f32>(baseF0) * specularColor * specularFactor, vec3<f32>(1.0));
let realF0 = mix(f0Norm, albedo, vec3<f32>(metalness));
let nDotV = max(dot(pbrNormal, viewDir), PBR_MIN_NDOTV);
let energyCompensation =
	resolveSpecularEnergyCompensation(nDotV, roughness, realF0);
let anisotropyData = resolveAnisotropyDirection(
	input.uv0,
	input.uv1,
	input.uv2,
	input.uv3
);
let anisotropyStrength = anisotropyData.z;
let anisotropyTangent = resolveAnisotropyTangent(
	pbrNormal,
	input.worldTangent,
	anisotropyData.xy
);
let anisotropyBitangent = safeNormalize(
	cross(pbrNormal, anisotropyTangent),
	fallbackTangentFromNormal(pbrNormal)
);
let opaquePBRSurface = OpaquePBRSurfaceInput(
	pbrNormal,
	viewDir,
	albedo,
	realF0,
	roughness,
	metalness,
	transmission,
	energyCompensation,
	iridescence,
	iridescenceIor,
	iridescenceThickness,
	anisotropyTangent,
	anisotropyBitangent,
	anisotropyStrength,
	clearcoat,
	clearcoatRoughness,
	clearcoatNormal,
	sheenColor,
	sheenRoughness
);
let reflectionDir = select(
	reflectViewDirection(pbrNormal, viewDir),
	resolveAnisotropicReflectionDirection(
		pbrNormal,
		viewDir,
		anisotropyBitangent,
		roughness,
		anisotropyStrength
	),
	anisotropyStrength > EPSILON
);
let maxSheenColor = max(max(sheenColor.x, sheenColor.y), sheenColor.z);

var volumeAttenuation = vec3<f32>(1.0);
if (thickness > 0.0 && attenuationDistance > 0.0) {
	let absorb = -log(attenuationColor) / attenuationDistance;
	volumeAttenuation = exp(-absorb * thickness);
}

var directLight = vec3<f32>(0.0);
