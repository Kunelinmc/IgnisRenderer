let shAmbientEnabled = useSHAmbient();
var ambientColor = frame.ambientColor.rgb;

var diffuseAmbient = ambientColor / PI;
var specularAmbientRadiance = vec3<f32>(0.0);
if (shAmbientEnabled) {
	diffuseAmbient =
		sampleDiffuseProbeIrradiance(input.worldPosition, pbrNormal) / (255.0 * PI);

	let localSelection = selectTopTwoLocalLightProbes(input.worldPosition);
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

let fAmbient = resolveIridescenceFresnel(
	nDotV,
	realF0,
	iridescence,
	iridescenceThickness,
	iridescenceIor
);
let refractionResult = refractViewDirection(viewDir, pbrNormal, ior);
let isTIR = transmission > 0.0 && refractionResult.valid < 0.5;
let effectiveFAmbient = select(fAmbient, vec3<f32>(1.0), isTIR);
var kdAmbient = vec3<f32>((1.0 - metalness) * (1.0 - transmission));
if (shAmbientEnabled) {
	kdAmbient =
		diffuseFresnelWeight(effectiveFAmbient, iridescence) *
		(1.0 - metalness) *
		(1.0 - transmission);
}
let ktAmbient =
	(vec3<f32>(1.0) - effectiveFAmbient) *
	(1.0 - metalness) *
	transmission;

let ccAmbientFresnel = select(0.0, fresnelSchlickScalar(nDotV, 0.04), clearcoat > 0.0);
let diffuseCcAmbientFresnel = select(
	select(0.0, 0.04, clearcoat > 0.0),
	ccAmbientFresnel,
	shAmbientEnabled
);
let clearcoatDiffuseAttenuation = 1.0 - diffuseCcAmbientFresnel * clearcoat;
let clearcoatSpecularAttenuation = 1.0 - ccAmbientFresnel * clearcoat;
let baseAmbientAttenuation =
	vec3<f32>(clearcoatDiffuseAttenuation) * (vec3<f32>(1.0) - sheenColor * 0.5);

var ambientLight = diffuseAmbient * albedo * kdAmbient * baseAmbientAttenuation;
if (
	includeTransmissionBackground &&
	transmission > 0.0 &&
	refractionResult.valid > 0.5
) {
	var transmissionRadiance = vec3<f32>(0.0);
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
		clearcoatSpecularAttenuation;
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
	let splitSumFresnel = select(
		realF0,
		effectiveFAmbient,
		iridescence > EPSILON || isTIR
	);
	ambientLight +=
		prefiltered *
		(splitSumFresnel * brdf.x + vec3<f32>(brdf.y)) *
		energyCompensation *
		clearcoatSpecularAttenuation;

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
		clearcoatSpecularAttenuation;
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
		clearcoatSpecularAttenuation;
}

ambientLight *= occlusion;

let finalLinear = max(directLight + ambientLight + emissive, vec3<f32>(0.0));
let outputAlpha = resolveTransmissionAlpha(
	alpha,
	transmission,
	nDotV,
	realF0,
	iridescence,
	iridescenceThickness,
	iridescenceIor
);
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

fn resolveTransmissionAlpha(
	baseAlpha: f32,
	transmission: f32,
	nDotV: f32,
	f0: vec3<f32>,
	iridescence: f32,
	iridescenceThickness: f32,
	iridescenceIor: f32
) -> f32 {
	let clampedTransmission = clamp(transmission, 0.0, 1.0);
	if (clampedTransmission <= EPSILON) {
		return clamp(baseAlpha, 0.0, 1.0);
	}

	let fresnel = resolveIridescenceFresnel(
		nDotV,
		f0,
		iridescence,
		iridescenceThickness,
		iridescenceIor
	);
	let fresnelAverage = clamp(
		(fresnel.x + fresnel.y + fresnel.z) * (1.0 / 3.0),
		0.0,
		1.0
	);
	let floorAlpha = max(0, fresnelAverage);
	let blended =
		baseAlpha * (1.0 - clampedTransmission) +
		floorAlpha * clampedTransmission;
	return clamp(max(floorAlpha, blended), 0.0, 1.0);
}

fn refractViewDirection(v: vec3<f32>, n: vec3<f32>, ior: f32) -> RefractionResult {
	let cosThetaI = dot(v, n);
	let outside = cosThetaI > 0.0;
	let eta = select(ior, 1.0 / max(ior, 1.0), outside);
	let refractNormal = select(-n, n, outside);
	let absCosThetaI = abs(cosThetaI);
	let sin2ThetaT = eta * eta * (1.0 - absCosThetaI * absCosThetaI);

	if (sin2ThetaT > 1.0) {
		return RefractionResult(vec3<f32>(0.0), 0.0);
	}

	let cosThetaT = sqrt(max(1.0 - sin2ThetaT, 0.0));
	let refraction = eta * -v + (eta * absCosThetaI - cosThetaT) * refractNormal;
	return RefractionResult(safeNormalize(refraction, -v), 1.0);
}
