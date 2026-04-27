let shAmbientEnabled = useSHAmbient();
var ambientColor = frame.ambientColor.rgb;
if (ambientColor.x + ambientColor.y + ambientColor.z == 0.0) {
	ambientColor = vec3<f32>(PBR_AMBIENT_FALLBACK_LINEAR);
}

var diffuseAmbient = ambientColor;
var specularAmbientRadiance = ambientColor / PI;
if (shAmbientEnabled) {
	let localSelection = selectTopTwoLocalLightProbes(input.worldPosition);
	let globalDiffuseAmbient = calculateIrradianceFromSH(pbrNormal);
	let localDiffuseAmbient = sampleBlendedLocalLightProbeIrradiance(
		localSelection,
		pbrNormal
	);
	diffuseAmbient = mix(
		globalDiffuseAmbient,
		localDiffuseAmbient.rgb,
		localDiffuseAmbient.w
	) / 255.0;

	let globalSpecularAmbient = sampleSHRadiance(reflectionDir);
	let localSpecularAmbient = sampleBlendedLocalLightProbeRadiance(
		localSelection,
		reflectionDir
	);
	specularAmbientRadiance = mix(
		globalSpecularAmbient,
		localSpecularAmbient.rgb,
		localSpecularAmbient.w
	) / 255.0;
}

let fAmbient = fresnelSchlick(nDotV, realF0);
let refractionResult = refractViewDirection(viewDir, pbrNormal, ior);
let isTIR = transmission > 0.0 && refractionResult.valid < 0.5;
let effectiveFAmbient = select(fAmbient, vec3<f32>(1.0), isTIR);
let kdAmbient =
	(vec3<f32>(1.0) - effectiveFAmbient) *
	(1.0 - metalness) *
	(1.0 - transmission);
let ktAmbient =
	(vec3<f32>(1.0) - effectiveFAmbient) *
	(1.0 - metalness) *
	transmission;

let ccAmbientFresnel = select(0.0, fresnelSchlickScalar(nDotV, 0.04), clearcoat > 0.0);
let clearcoatAmbientAttenuation = 1.0 - ccAmbientFresnel * clearcoat;
let baseAmbientAttenuation =
	vec3<f32>(clearcoatAmbientAttenuation) * (vec3<f32>(1.0) - sheenColor * 0.5);

var ambientLight = diffuseAmbient * albedo * kdAmbient * baseAmbientAttenuation;
if (transmission > 0.0 && refractionResult.valid > 0.5) {
	var transmissionRadiance = ambientColor;
	if (hasEnvSpecular()) {
		transmissionRadiance = sampleEnvironmentSpecular(
			refractionResult.direction,
			roughness,
			input.worldPosition
		);
	} else if (shAmbientEnabled) {
		let localSelection = selectTopTwoLocalLightProbes(input.worldPosition);
		let globalTransmissionRadiance = sampleSHRadiance(refractionResult.direction);
		let localTransmissionRadiance = sampleBlendedLocalLightProbeRadiance(
			localSelection,
			refractionResult.direction
		);
		transmissionRadiance = mix(
			globalTransmissionRadiance,
			localTransmissionRadiance.rgb,
			localTransmissionRadiance.w
		) / 255.0;
	}
	ambientLight +=
		transmissionRadiance *
		albedo *
		ktAmbient *
		volumeAttenuation *
		clearcoatAmbientAttenuation;
}

let specularAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - roughness) * 0.5);
let clearcoatAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - clearcoatRoughness) * 0.5);
if (hasEnvSpecular()) {
	let prefiltered = sampleEnvironmentSpecular(
		reflectionDir,
		roughness,
		input.worldPosition
	);
	let brdf = sampleBRDFLUT(nDotV, roughness);
	ambientLight +=
		prefiltered *
		(effectiveFAmbient * brdf.x + vec3<f32>(brdf.y)) *
		clearcoatAmbientAttenuation;

	let clearcoatNdotV = max(dot(clearcoatNormal, viewDir), PBR_MIN_NDOTV);
	let clearcoatReflectionDir = reflectViewDirection(clearcoatNormal, viewDir);
	let clearcoatPrefiltered = sampleEnvironmentSpecular(
		clearcoatReflectionDir,
		clearcoatRoughness,
		input.worldPosition
	);
	let clearcoatBrdf = sampleBRDFLUT(clearcoatNdotV, clearcoatRoughness);
	ambientLight +=
		clearcoatPrefiltered *
		(ccAmbientFresnel * clearcoatBrdf.x + clearcoatBrdf.y) *
		clearcoat;
} else {
	ambientLight +=
		specularAmbientRadiance *
		effectiveFAmbient *
		specularAmbientFactor *
		clearcoatAmbientAttenuation;
	ambientLight +=
		specularAmbientRadiance *
		ccAmbientFresnel *
		clearcoatAmbientFactor *
		clearcoat;
}

if (maxSheenColor > 0.0) {
	let sheenAmbientFactor = max(PBR_SPEC_FALLBACK, (1.0 - max(sheenRoughness, 0.04)) * 0.5);
	ambientLight +=
		specularAmbientRadiance *
		sheenColor *
		sheenAmbientFactor *
		clearcoatAmbientAttenuation;
}

ambientLight *= occlusion;

let finalLinear = max(directLight + ambientLight + emissive, vec3<f32>(0.0));
let outputAlpha = resolveTransmissionAlpha(alpha, transmission, nDotV, realF0);
return buildSceneOutput(
	finalLinear,
	outputAlpha,
	albedo,
	pbrNormal,
	roughness,
	metalness,
	emissive,
	occlusion,
	motion,
	linearDepth
);
}
