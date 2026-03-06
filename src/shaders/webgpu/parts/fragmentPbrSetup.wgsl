let mrSample = sampleLinearTexture(
	metallicRoughnessTexture,
	metallicRoughnessSampler,
	TEX_METALLIC_ROUGHNESS,
	input.uv,
	input.uv2
);
let occlusionSample = sampleLinearTexture(
	occlusionTexture,
	occlusionSampler,
	TEX_OCCLUSION,
	input.uv,
	input.uv2
);
let specularSample = sampleLinearTexture(
	specularTexture,
	specularSampler,
	TEX_SPECULAR,
	input.uv,
	input.uv2
);
let specularColorSample = sampleColorTexture(
	specularColorTexture,
	specularColorSampler,
	TEX_SPECULAR_COLOR,
	input.uv,
	input.uv2
);
let clearcoatSample = sampleLinearTexture(
	clearcoatTexture,
	clearcoatSampler,
	TEX_CLEARCOAT,
	input.uv,
	input.uv2
);
let clearcoatRoughnessSample = sampleLinearTexture(
	clearcoatRoughnessTexture,
	clearcoatRoughnessSampler,
	TEX_CLEARCOAT_ROUGHNESS,
	input.uv,
	input.uv2
);
let sheenColorSample = sampleColorTexture(
	sheenColorTexture,
	sheenColorSampler,
	TEX_SHEEN_COLOR,
	input.uv,
	input.uv2
);
let sheenRoughnessSample = sampleLinearTexture(
	sheenRoughnessTexture,
	sheenRoughnessSampler,
	TEX_SHEEN_ROUGHNESS,
	input.uv,
	input.uv2
);
let transmissionSample = sampleLinearTexture(
	transmissionTexture,
	transmissionSampler,
	TEX_TRANSMISSION,
	input.uv,
	input.uv2
);
let thicknessSample = sampleLinearTexture(
	thicknessTexture,
	thicknessSampler,
	TEX_THICKNESS,
	input.uv,
	input.uv2
);

let normalSample = sampleLinearTexture(
	normalTexture,
	normalSampler,
	TEX_NORMAL,
	input.uv,
	input.uv2
).rgb;
let clearcoatNormalSample = sampleLinearTexture(
	clearcoatNormalTexture,
	clearcoatNormalSampler,
	TEX_CLEARCOAT_NORMAL,
	input.uv,
	input.uv2
).rgb;

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
let reflectionDir = reflectViewDirection(pbrNormal, viewDir);
let maxSheenColor = max(max(sheenColor.x, sheenColor.y), sheenColor.z);

var volumeAttenuation = vec3<f32>(1.0);
if (thickness > 0.0 && attenuationDistance > 0.0) {
	let absorb = -log(attenuationColor) / attenuationDistance;
	volumeAttenuation = exp(-absorb * thickness);
}

var directLight = vec3<f32>(0.0);
